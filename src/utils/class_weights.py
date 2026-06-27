import torch
import numpy as np

def get_pos_weight(labels):
    """
    Calculates the pos_weight factor for PyTorch's BCEWithLogitsLoss:
    pos_weight = total_negative_samples / total_positive_samples
    Helps address class imbalance during model training.
    """
    labels = np.array(labels)
    num_neg = np.sum(labels == 0)
    num_pos = np.sum(labels == 1)
    
    # Safely handle zero division
    if num_pos == 0:
        return torch.tensor([1.0], dtype=torch.float32)
        
    pos_weight = num_neg / num_pos
    return torch.tensor([pos_weight], dtype=torch.float32)
