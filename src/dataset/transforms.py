import torchvision.transforms as T

def get_train_transforms(img_size):
    """
    Returns training transforms containing Resize, RandomHorizontalFlip,
    RandomRotation, ColorJitter, ToTensor, and ImageNet Normalization.
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_valid_transforms(img_size):
    """
    Returns validation/test transforms containing Resize, ToTensor,
    and ImageNet Normalization. No random augmentations applied.
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
