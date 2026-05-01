# /models/ResNet.py

"""Implements the Base Resnet model."""

from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn


class Model(nn.Module):
    """Base model for all the custom models implemented using pytorch."""
    def __init__(self, num_classes: int, input_shape: Tuple = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.pretrained = False

    def model_details(self):
        print(f"classes : {self.num_classes} pretrained : {self.pretrained}")


class BasicBlock(nn.Module):
    """Resnet Basic Block."""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, activation: nn.functional, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class Bottleneck(nn.Module):
    """Resnet Bottleneck Block."""
    expansion = 4

    def __init__(self, in_planes: int, planes: int, activation: nn.functional, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.activation = activation
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class BaseResNet(Model):
    """Resnet Base Model Class."""
    def __init__(
        self,
        block: nn.Module,
        num_blocks: list[int],
        activation: nn.functional,
        num_classes: int,
        in_channels: int,  # Added for consistency
        *args,
        **kwargs,
    ):
        super().__init__(num_classes, *args, **kwargs)
        self.in_planes = 64
        self.activation = activation

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # === THIS IS THE FIX (PART 1) ===
        # Add an adaptive pooling layer to handle variable image sizes.
        # The output size (1, 1) means it will produce a feature map of size 1x1.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: nn.Module, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for curr_stride in strides:
            layers.append(block(self.in_planes, planes, self.activation, curr_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # === THIS IS THE FIX (PART 2) ===
        # Replace the hardcoded pooling with the adaptive pooling layer.
        out = self.avgpool(out)
        
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# --- The specific ResNet classes now pass `in_channels` ---
# Note: The BaseResNet __init__ was missing in_channels in your provided code,
# I've re-added it from our previous fix to ensure it works.

class ResNet18(BaseResNet):
    """Resnet 18 Model Class."""
    def __init__(self, configs):
        super().__init__(
            block=BasicBlock,
            num_blocks=[2, 2, 2, 2],
            activation=F.relu,
            num_classes=configs.num_classes,
            in_channels=configs.in_channels
        )
        self.__class__.__name__ = "Resnet18"

class ResNet34(BaseResNet):
    """Resnet 34 Model Class."""
    def __init__(self, configs):
        super().__init__(
            block=BasicBlock,
            num_blocks=[3, 4, 6, 3],
            activation=F.relu,
            num_classes=configs.num_classes,
            in_channels=configs.in_channels,
        )
        self.__class__.__name__ = "Resnet34"

class ResNet10(BaseResNet):
    """Resnet 10 Model Class."""
    def __init__(self, configs):
        super().__init__(
            block=BasicBlock,
            num_blocks=[1, 1, 1, 1],
            activation=F.relu,
            num_classes=configs.num_classes,
            in_channels=configs.in_channels,
        )
        self.__class__.__name__ = "ResNet10"

class ResNet22(BaseResNet):
    """Resnet 22 Model Class."""
    def __init__(self, configs):
        super().__init__(
            block=BasicBlock,
            num_blocks=[2, 3, 3, 2],
            activation=F.relu,
            num_classes=configs.num_classes,
            in_channels=configs.in_channels,
        )
        self.__class__.__name__ = "ResNet22"