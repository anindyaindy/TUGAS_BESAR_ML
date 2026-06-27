import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Sets random seed for random, numpy, PyTorch, and CUDA backends
    to ensure deterministic and reproducible experimental runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Lock CUDA convolution algorithms to run deterministically
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
