import timm
import torch.nn as nn

def build_xception(pretrained=True):
    """
    Builds and returns the Xception architecture with a single binary logit output class.
    """
    model = timm.create_model('xception', pretrained=pretrained, num_classes=1)
    return model
