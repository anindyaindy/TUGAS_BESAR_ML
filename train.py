import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

from src.utils.reproducibility import set_seed
from src.utils.class_weights import get_pos_weight
from src.utils.logging_helper import CSVLogger, CheckpointSaver
from src.utils.plotting import plot_training_history, plot_confusion_matrix, plot_prediction_grid
from src.dataset.dataloader import create_dataloader
from src.models.efficientnet import build_efficientnet_b3
from src.models.xception import build_xception

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    
    all_targets = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_targets.extend(labels.squeeze(1).cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
            total += labels.size(0)
            
    val_loss = running_loss / total
    if len(all_targets) == 0:
        return val_loss, 0.0, 0.0, 0.5, [], []
        
    val_acc = sum(1 for t, p in zip(all_targets, all_preds) if t == p) / len(all_targets)
    
    try:
        val_f1 = f1_score(all_targets, all_preds, zero_division=0)
        val_auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        val_f1 = 0.0
        val_auc = 0.5
        
    return val_loss, val_acc, val_f1, val_auc, all_targets, all_preds

def main():
    parser = argparse.ArgumentParser(description="DeepFake Face Classification Training Pipeline")
    parser.add_argument('--model', type=str, default='efficientnet_b3', choices=['efficientnet_b3', 'xception'],
                        help="Model architecture to train (default: efficientnet_b3)")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Input batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--img_size', type=int, default=None, help="Input image resolution")
    args = parser.parse_args()

    # Set default resolutions if not provided
    if args.img_size is None:
        args.img_size = 300 if args.model == 'efficientnet_b3' else 299

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup dataloaders
    print("Loading dataloaders...")
    train_dir = os.path.join('dataset', 'train')
    val_dir = os.path.join('dataset', 'val')
    test_dir = os.path.join('dataset', 'test')

    # Check if split dirs exist, print warning if empty
    for path in [train_dir, val_dir, test_dir]:
        if not os.path.exists(path) or not os.listdir(path):
            print(f"Warning: Directory {path} does not exist or is empty. Run downloader first.")
            return

    # Disable multiprocessing workers if on CPU to avoid warnings
    num_workers = 0 if device.type == 'cpu' else 4
    train_loader = create_dataloader(train_dir, args.batch_size, args.img_size, is_training=True, num_workers=num_workers)
    val_loader = create_dataloader(val_dir, args.batch_size, args.img_size, is_training=False, num_workers=num_workers)
    test_loader = create_dataloader(test_dir, args.batch_size, args.img_size, is_training=False, num_workers=num_workers)

    # Build Model
    print(f"Building {args.model} model...")
    if args.model == 'efficientnet_b3':
        model = build_efficientnet_b3(pretrained=True)
    else:
        model = build_xception(pretrained=True)
    model = model.to(device)

    # Compute Class Weights
    train_labels = train_loader.dataset.labels
    pos_weight = get_pos_weight(train_labels).to(device)
    print(f"Calculated class imbalance pos_weight: {pos_weight.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Setup Logging & Checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('visualisasi_plots', exist_ok=True)
    log_path = os.path.join('visualisasi_plots', f'training_log_{args.model}.csv')
    logger = CSVLogger(log_path)
    saver = CheckpointSaver('checkpoints', args.model)

    print("Starting training loop...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_auc, _, _ = validate(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} AUC: {val_auc:.4f}")
              
          # Log metrics to CSV
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, val_f1, val_auc, current_lr)
        saver.save_if_best(model, val_acc, epoch)

    # Final test set evaluation
    print("\nEvaluating best model on test set...")
    best_weights = os.path.join('checkpoints', f'best_{args.model}.pth')
    if os.path.exists(best_weights):
        model.load_state_dict(torch.load(best_weights, map_location=device))
        
    test_loss, test_acc, test_f1, test_auc, test_y_true, test_y_pred = validate(model, test_loader, criterion, device)
    
    print("\n--- Test Set Metrics ---")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    print(f"ROC-AUC: {test_auc:.4f}")

    # Generate plotting results
    print("\nGenerating final plots...")
    plot_training_history(log_path, 'visualisasi_plots')
    plot_confusion_matrix(test_y_true, test_y_pred, 'visualisasi_plots')
    plot_prediction_grid(model, test_loader, device, 'visualisasi_plots')
    print("Plots generated successfully in 'visualisasi_plots/'")

if __name__ == '__main__':
    main()
