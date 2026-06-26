import os
import glob
from torch.utils.data import DataLoader
from src.dataset.face_dataset import FaceDataset
from src.dataset.transforms import get_train_transforms, get_valid_transforms

def create_dataloader(split_dir, batch_size, img_size, is_training=False, num_workers=4):
    """
    Scans the REAL and FAKE directories under split_dir,
    and returns a PyTorch DataLoader.
    """
    real_pattern = os.path.join(split_dir, 'REAL', '*')
    fake_pattern = os.path.join(split_dir, 'FAKE', '*')
    
    real_paths = glob.glob(real_pattern)
    fake_paths = glob.glob(fake_pattern)
    
    # Filter to ensure only valid images are processed
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    real_paths = [p for p in real_paths if os.path.splitext(p)[1].lower() in valid_exts]
    fake_paths = [p for p in fake_paths if os.path.splitext(p)[1].lower() in valid_exts]
    
    file_paths = real_paths + fake_paths
    # 0 = REAL, 1 = FAKE
    labels = [0] * len(real_paths) + [1] * len(fake_paths)
    
    # Select appropriate transforms
    transform = get_train_transforms(img_size) if is_training else get_valid_transforms(img_size)
    
    dataset = FaceDataset(file_paths, labels, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return dataloader
