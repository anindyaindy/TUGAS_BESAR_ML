import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_training_history(csv_path, save_dir):
    """
    Plots training and validation loss and accuracy curves over epochs.
    """
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Loss Curve
    axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss', color='b', marker='o')
    axes[0].plot(df['epoch'], df['val_loss'], label='Val Loss', color='r', marker='x')
    axes[0].set_title('Training vs Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # Accuracy Curve
    axes[1].plot(df['epoch'], df['train_acc'], label='Train Acc', color='b', marker='o')
    axes[1].plot(df['epoch'], df['val_acc'], label='Val Acc', color='r', marker='x')
    axes[1].set_title('Training vs Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_dir):
    """
    Plots raw and normalized confusion matrices.
    """
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels = ['REAL', 'FAKE']
    
    # Raw Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title('Raw Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Normalized Confusion Matrix
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title('Normalized Confusion Matrix')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

def plot_prediction_grid(model, dataloader, device, save_dir):
    """
    Generates a 4x4 visual prediction grid using 16 batch images,
    coloring correct predictions in green and incorrect predictions in red.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    images_list = []
    labels_list = []
    preds_list = []
    probs_list = []
    
    # Extract at least 16 images for visualization
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs_device = imgs.to(device)
            outputs = model(imgs_device)
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            images_list.append(imgs.numpy())
            labels_list.append(lbls.numpy())
            preds_list.append(preds)
            probs_list.append(probs)
            
            if len(np.concatenate(images_list, axis=0)) >= 16:
                break
                
    all_images = np.concatenate(images_list, axis=0)[:16]
    all_labels = np.concatenate(labels_list, axis=0)[:16]
    all_preds = np.concatenate(preds_list, axis=0)[:16]
    all_probs = np.concatenate(probs_list, axis=0)[:16]
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    # ImageNet normalization stats
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    for i in range(16):
        if i >= len(all_images):
            break
        ax = axes[i // 4, i % 4]
        # Denormalize for plotting
        img = all_images[i] * std + mean
        img = np.clip(img, 0, 1)
        img = np.transpose(img, (1, 2, 0))
        
        ax.imshow(img)
        ax.axis('off')
        
        true_label_str = 'REAL' if all_labels[i] == 0 else 'FAKE'
        pred_label_str = 'REAL' if all_preds[i] == 0 else 'FAKE'
        prob = all_probs[i] if all_preds[i] == 1 else (1.0 - all_probs[i])
        
        title_color = 'green' if all_labels[i] == all_preds[i] else 'red'
        
        # Border box
        for spine in ax.spines.values():
            spine.set_color(title_color)
            spine.set_linewidth(3)
            
        ax.set_title(f"T: {true_label_str} | P: {pred_label_str}\nConf: {prob*100:.1f}%", color=title_color, fontsize=10)
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_grid.png'), dpi=150)
    plt.close()
