import inspect
import sys

from torch import nn

from .CNN import FedAvgCNN
from .ResNet import ResNet10, ResNet18, ResNet22, ResNet34
from .ResNetDim import (
    ResNet10_0_5x,
    ResNet10_2x,
    ResNet18_0_5x,
    ResNet18_0_25x,
    ResNet18_0_125x,
    ResNet18_0_0625x,
    ResNet18_2x,
    ResNet34_0_5x,
    ResNet34_2x,
)
from .VGG import VGG8, VGG11
from .MobileNetV2 import MobileNetV2
from .EfficientNet import EfficientNetB0
from .TextModels import LSTMModel
# Automatically create a list of all classes or functions imported in this file
MODELS = sorted([
    name
    for name, obj in sys.modules[__name__].__dict__.items()
    if inspect.isclass(obj) and issubclass(obj, nn.Module) and name != "BaseHeadSplit"
])
print(f"MODELS = {MODELS}")

print(f"{MODELS = }")


class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()
        self.base = base
        self.head = head

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out
