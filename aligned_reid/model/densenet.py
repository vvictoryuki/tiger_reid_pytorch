import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tm 
import torch.utils.checkpoint as cp
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo


__all__ = ['densenet121', 'densenet169', 'densenet201', 'densenet161']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

def densenet121(pretrained=False):
    """Constructs a Densenet-121 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = tm.densenet121(pretrained = pretrained).features
    return 1024, model


def densenet161(pretrained=False):
    """Constructs a Densenet-161 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = tm.densenet161(pretrained = pretrained).features
    return 2208, model


def densenet169(pretrained=False):
    """Constructs a Densenet-169 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = tm.densenet169(pretrained = pretrained).features
    return 1664, model


def densenet201(pretrained=False):
    """Constructs a Densenet-201 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = tm.densenet201(pretrained = pretrained).features
    return 1920, model
