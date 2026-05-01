# /models/EfficientNet.py

from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetB0(nn.Module):
    """
    EfficientNetB0 model, adapted for custom number of classes and input channels.
    """
    def __init__(self, configs):
        super(EfficientNetB0, self).__init__()
        num_classes = configs.num_classes
        in_channels = configs.in_channels
        
        # Load a pretrained EfficientNetB0
        # We will use this as a feature extractor and modify it.
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # 1. Modify the first convolutional layer for different input channels if needed
        if in_channels != 3:
            original_first_layer = self.model.features[0][0]
            self.model.features[0][0] = nn.Conv2d(
                in_channels,
                original_first_layer.out_channels,
                kernel_size=original_first_layer.kernel_size,
                stride=original_first_layer.stride,
                padding=original_first_layer.padding,
                bias=False
            )

        # 2. Modify the final classifier layer for the correct number of classes
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

        self.__class__.__name__ = "EfficientNetB0"

    def forward(self, x):
        return self.model(x)