import torch
from torch.utils.data import Dataset
from PIL import Image

class FaceDataset(Dataset):
    """
    Custom PyTorch Dataset for loading face images (REAL/FAKE) from file paths.
    """
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load image and convert to RGB
        img = Image.open(path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        # Return label as a PyTorch Float Tensor (necessary for BCEWithLogitsLoss)
        return img, torch.tensor(label, dtype=torch.float32)
