# /models/VGG.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration dictionary for VGG variants.
# Now includes VGG8, designed as a lighter alternative.
cfg = {
    # For 32x32 images like CIFAR.
    'VGG8_CIFAR':  [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'VGG11_CIFAR': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],

    # For larger images like TinyImageNet (64x64) or ImageNet (224x224).
    'VGG8_IMAGENET':  [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'], # Same as CIFAR version, adaptive pooling handles size.
    'VGG11_IMAGENET': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name: str, num_classes: int, in_channels: int = 3, batch_norm: bool = True):
        super(VGG, self).__init__()
        self.vgg_name = vgg_name
        self.features = self._make_layers(cfg[vgg_name], in_channels, batch_norm)

        # --- UPDATED STABLE CLASSIFIER ---
        # The VGG8 will use a slightly deeper but still stable classifier.
        # The VGG11 will use the simplest, most stable single-layer classifier.
        if 'VGG8' in self.vgg_name:
            self.classifier = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Linear(512, num_classes)
            )
        else: # VGG11 and others
             self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg_list: list, in_channels: int, batch_norm: bool):
        layers = []
        for x in cfg_list:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1)]
                if batch_norm:
                    layers += [nn.BatchNorm2d(x)]
                layers += [nn.ReLU(inplace=True)]
                in_channels = x
        
        # Add an adaptive pool at the end to guarantee a 1x1 output.
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)


class VGG8(nn.Module):
    """
    Lighter VGG8 Model that intelligently selects its configuration based on
    the dataset being used to ensure stability and performance.
    """
    def __init__(self, configs):
        super(VGG8, self).__init__()
        if configs.dataset.lower() in ['cifar10', 'cifar100']:
            vgg_name = 'VGG8_CIFAR'
        else:
            vgg_name = 'VGG8_IMAGENET'

        self.vgg = VGG(
            vgg_name=vgg_name,
            num_classes=configs.num_classes,
            in_channels=configs.in_channels,
            batch_norm=True
        )
        self.__class__.__name__ = "VGG8"

    def forward(self, x):
        return self.vgg(x)


class VGG11(nn.Module):
    """
    VGG11 Model that intelligently selects its configuration based on
    the dataset being used to ensure stability and performance.
    """
    def __init__(self, configs):
        super(VGG11, self).__init__()
        if configs.dataset.lower() in ['cifar10', 'cifar100']:
            vgg_name = 'VGG11_CIFAR'
        else:
            vgg_name = 'VGG11_IMAGENET'

        self.vgg = VGG(
            vgg_name=vgg_name,
            num_classes=configs.num_classes,
            in_channels=configs.in_channels,
            batch_norm=True
        )
        self.__class__.__name__ = "VGG11"

    def forward(self, x):
        return self.vgg(x)