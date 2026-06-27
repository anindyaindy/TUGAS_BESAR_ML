import timm
import torch.nn as nn

def build_efficientnet_b3(pretrained=True):
    """
    Builds and returns the EfficientNet-B3 architecture with a single binary logit output class.
    """
    model = timm.create_model('efficientnet_b3', pretrained=pretrained, num_classes=1)
    return model
