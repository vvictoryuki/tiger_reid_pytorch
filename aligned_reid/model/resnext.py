import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
import sys

import torchvision.models as tm

def resnext101_32x8d(pretrained=True) :

    model = tm.resnext101_32x8d(pretrained=pretrained).features
    return 2048, model
