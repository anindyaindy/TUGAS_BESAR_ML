import os
import csv
import torch

class CSVLogger:
    """
    Saves metrics of each training epoch to a designated CSV file.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.headers = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1', 'val_auc', 'lr']
        # Set up CSV header on initialization if file is new
        if not os.path.exists(file_path):
            dir_name = os.path.dirname(file_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(file_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, val_f1, val_auc, lr):
        """
        Appends epoch metrics to the CSV file.
        """
        with open(self.file_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, val_f1, val_auc, lr])


class CheckpointSaver:
    """
    Monitors validation performance and saves the best model checkpoint weights.
    """
    def __init__(self, save_dir, model_name):
        self.save_dir = save_dir
        self.model_name = model_name
        self.best_metric = -1.0
        os.makedirs(self.save_dir, exist_ok=True)

    def save_if_best(self, model, metric, epoch):
        """
        Saves the model weights if the validation metric is higher than the previous best.
        """
        if metric > self.best_metric:
            self.best_metric = metric
            save_path = os.path.join(self.save_dir, f'best_{self.model_name}.pth')
            # Save the model state dictionary
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch}: New best validation metric {metric:.4f}. Model saved to {save_path}")
            return True
        return False
