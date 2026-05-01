from typing import Callable, List, Optional, Type, Union

import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torchvision.utils import _log_api_usage_once


class ResNetDim(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        dim=64,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(block, layers, num_classes=num_classes)
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = dim
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, dim, layers[0])
        self.layer2 = self._make_layer(
            block, dim * 2, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, dim * 4, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, dim * 8, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


# ========================================================================================================


def ResNet10_2x(configs):
    return ResNetDim(
        block=BasicBlock,
        layers=[1, 1, 1, 1],
        num_classes=configs.num_classes,
        dim=64 * 2,
    )


def ResNet10_0_5x(configs):
    return ResNetDim(
        block=BasicBlock,
        layers=[1, 1, 1, 1],
        num_classes=configs.num_classes,
        dim=64 // 2,
    )


# ========================================================================================================


def ResNet18_0_0625x(configs):
    return ResNetDim(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=configs.num_classes,
        dim=64 // 16,
    )


def ResNet18_0_125x(configs):
    return ResNetDim(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=configs.num_classes,
        dim=64 // 8,
    )


def ResNet18_0_25x(configs):
    return ResNetDim(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=configs.num_classes,
        dim=64 // 4,
    )


def ResNet18_0_5x(configs):
    return ResNetDim(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=configs.num_classes,
        dim=64 // 2,
    )


def ResNet18_2x(configs):
    return ResNetDim(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=configs.num_classes,
        dim=64 * 2,
    )


# ========================================================================================================


def ResNet34_0_5x(configs):
    return ResNetDim(
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        num_classes=configs.num_classes,
        dim=64 // 2,
    )


def ResNet34_2x(configs):
    return ResNetDim(
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        num_classes=configs.num_classes,
        dim=64 * 2,
    )
