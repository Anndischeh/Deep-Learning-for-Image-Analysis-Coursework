import torch
import torch.nn as nn
from torchsummary import summary
from Utils.config import device
from Utils.config import image_resize

def flattened_size(image_resize, architecture):
    """
    Calculates the flattened size of the feature map after CNN layers.

    Args:
        image_resize (int): Size of the input image.
        architecture (list): CNN architecture.

    Returns:
        int: Flattened size.
    """
    res = image_resize
    for layer in architecture:
        if isinstance(layer, tuple):
            kernel_size, _, stride, padding = layer
            res = int((res + 2 * padding - kernel_size) / stride + 1)
        elif isinstance(layer, list):
            for _ in range(layer[-1]):
                for sublayer in layer[:-1]:
                    kernel_size, _, stride, padding = sublayer
                    res = int((res + 2 * padding - kernel_size) / stride + 1)
        elif isinstance(layer, str):
            res = int((res - 2) / 2 + 1)
    return res


class CNN_block(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels, stride, padding):
        super(CNN_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.leaky_relu(self.batch_norm(self.conv(x)))


class YOLOv1(nn.Module):
    def __init__(self, architecture, S, B, C, in_channels=3, image_resize=448, linear_size=4096):
        # If we ever want to train with grayscale images, set in_channels=1
        super(YOLOv1, self).__init__()
        self.in_channels = in_channels # temporary use for layer creation
        self.cnn_layers = self.CNN_layers(architecture)
        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*(flattened_size(image_resize, architecture))**2, linear_size),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(linear_size, S*S*(C+B*5))
        )

    def CNN_layers(self, architecture):
        layers = []
        for layer in architecture:
            if layer == 'maxpool':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if type(layer) == tuple:
                layers.append(CNN_block(self.in_channels, *layer))
                self.in_channels = layer[1]
            if type(layer) == list:
                temp = []
                for sub_layer in layer[:-1]:
                    temp.append(CNN_block(self.in_channels, *sub_layer))
                    self.in_channels = sub_layer[1]
                layers.append(nn.Sequential(*(temp*layer[-1])))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.cnn_layers(x)
        # x = torch.flatten(x, start_dim=1) # necessary?
        x = self.fcs(x)
        return x