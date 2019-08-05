import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as tm

__all__ = [ 'vgg11', 'vgg13', 'vgg16', 'vgg19' ]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
}

class VGG(nn.Module) :

    def __init__(self, model_name) :
        super(VGG, self).__init__()

        self.features = self._make_layers(config[model_name])


    def _make_layers(self, layer_list) :

        input_channels = 3

        layers = []

        for index, layer in enumerate(layer_list) :

            if layer == 'm' :
                layers += [
                    nn.MaxPool2d(
                        kernel_size = 2,
                        stride = 2
                    )
                ]
            else :
                output_channels = int(layer)
                layers += [
                    nn.Conv2d(input_channels, output_channels, 3, 1, 1, bias = False),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(True)
                ]
                input_channels = output_channels

        return nn.Sequential(*layers)

    def forward(self, x) :
        output = self.features(x)
        return output

def vgg11(pretrained=False) :
    """Constructs a VGG-11 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = tm.vgg11_bn(pretrained=pretrained).features
    return 512, model

def vgg13(pretrained=False) :
    """Constructs a VGG-13 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = tm.vgg13_bn(pretrained=pretrained).features
    return 512, model

def vgg16(pretrained=False) :
    """Constructs a VGG-16 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = tm.vgg16_bn(pretrained=pretrained).features
    return 512, model

def vgg19(pretrained=False) :
    """Constructs a VGG-19 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = tm.vgg19_bn(pretrained=pretrained).features
    return 512, model

