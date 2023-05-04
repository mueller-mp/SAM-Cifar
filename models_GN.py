# adapted to GroupNorm from own codebase (homura - ASAM)
# ResNet variants
from functools import partial
from typing import Callable, Optional, Type, Union

import torch
from torch import nn
from torchvision import models

from homura.modules.attention import AttentionPool2d
from homura.vision.models import MODEL_REGISTRY
from homura.vision.models._utils import SELayer, conv1x1, conv3x3, init_parameters


def initialization(module: nn.Module,
                   use_zero_init: bool):
    init_parameters(module)
    if use_zero_init:
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in module.modules():
            if isinstance(m, BasicBlockGN):
                nn.init.constant_(m.norm2.weight, 0)


class BasicBlockGN(nn.Module):
    expansion = 1

    def __init__(self,
                 in_planes: int,
                 planes: int,
                 stride: int,
                 groups: int,
                 width_per_group: int,
                 norm: Optional[Type[nn.GroupNorm]],
                 act: Callable[[torch.Tensor], torch.Tensor],
                 n_groups_norm: int = 1,

                 ):
        super().__init__()
        planes = int(planes * (width_per_group / 16)) * groups
        self.conv1 = conv3x3(in_planes, planes, stride, bias=norm is None)
        self.conv2 = conv3x3(planes, planes, bias=norm is None)
        self.act = act
        self.norm1 = nn.Identity() if norm is None else (norm(planes) if 'Batch' in str(norm) else norm(planes//8, planes))
        self.norm2 = nn.Identity() if norm is None else (norm(planes) if 'Batch' in str(norm) else norm(planes//8, planes))

        self.downsample = nn.Identity()
        if in_planes != planes:
            self.downsample = nn.Sequential(conv1x1(in_planes, planes, stride=stride, bias=norm is None),
                                            nn.Identity() if norm is None else (norm(planes) if 'Batch' in str(norm) else norm(in_planes//8, in_planes)))
    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += self.downsample(x)
        out = self.act(out)

        return out


class PreactBasicBlockGN(BasicBlockGN):
    def __init__(self,
                 in_planes: int,
                 planes: int,
                 stride: int,
                 groups: int,
                 width_per_group: int,
                 norm: Optional[Type[nn.GroupNorm]],
                 act: Callable[[torch.Tensor], torch.Tensor],
                 n_groups_norm: int = 1,
                 ):
        super().__init__(in_planes, planes, stride, groups, width_per_group, norm, act)
        self.norm1 = nn.Identity() if norm is None else (norm(in_planes) if 'Batch' in str(norm) else norm(in_planes//8, in_planes))        
        if in_planes != planes:
            self.downsample = conv1x1(in_planes, planes, stride=stride, bias=norm is None)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)

        out += self.downsample(x)
        return out

    
# for resnext
class BottleneckGN(nn.Module):
    expansion = 4

    def __init__(self,
                 in_planes: int,
                 planes: int,
                 stride: int,
                 groups: int,
                 width_per_group: int,
                 norm: Optional[Type[nn.BatchNorm2d]],
                 act: Callable[[torch.Tensor], torch.Tensor]
                 ):
        super().__init__()
        raise NotImplementedError
    
class ResNetGN(nn.Module):
    """ResNet for CIFAR data. For ImageNet classification, use `torchvision`'s.
    """

    def __init__(self,
                 block: Type[Union[BasicBlockGN, BottleneckGN]],
                 num_classes: int,
                 layer_depth: int,
                 width: int = 16,
                 widen_factor: int = 1,
                 in_channels: int = 3,
                 groups: int = 1,
                 width_per_group: int = 16,
                 norm: Optional[Type[nn.GroupNorm]] = nn.GroupNorm,
                 act: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                 preact: bool = False,
                 final_pool: Callable[[torch.Tensor], torch.Tensor] = nn.AdaptiveAvgPool2d(1),
                 initializer: Optional[Callable[[nn.Module], None]] = None
                 ):
        super(ResNetGN, self).__init__()
        self.inplane = width
        self.groups = groups
        self.norm = norm
        self.width_per_group = width_per_group
        self.preact = preact
        self.n_groups_norm = width*widen_factor//2
        self.conv1 = conv3x3(in_channels, width, stride=1, bias=norm is None)
        self.norm1 = nn.Identity() if norm is None else (norm(4 * width * block.expansion * widen_factor if self.preact
                                                             else width) if 'Batch' in str(norm) else norm(width//8, 4 * width * block.expansion * widen_factor if self.preact else width))
        self.act = act
        self.layer1 = self._make_layer(block, width * widen_factor, layer_depth=layer_depth, stride=1)
        self.layer2 = self._make_layer(block, width * 2 * widen_factor, layer_depth=layer_depth, stride=2)
        self.layer3 = self._make_layer(block, width * 4 * widen_factor, layer_depth=layer_depth, stride=2)
        self.final_pool = final_pool
        self.fc = nn.Linear(4 * width * block.expansion * widen_factor, num_classes)
        if initializer is None:
            initialization(self, False)
        else:
            initializer(self)

    def _make_layer(self,
                    block: Type[Union[BasicBlockGN, BottleneckGN]],
                    planes: int,
                    layer_depth: int,
                    stride: int,
                    ) -> nn.Sequential:
        layers = []
        for i in range(layer_depth):
            layers.append(
                block(self.inplane, planes, stride if i == 0 else 1,
                      self.groups, self.width_per_group, self.norm, self.act, n_groups_norm=self.n_groups_norm)
            )
            if i == 0:
                self.inplane = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        if not self.preact:
            x = self.norm1(x)
            x = self.act(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.preact:
            x = self.norm1(x)
            x = self.act(x)

        x = self.final_pool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x


def resnetGN(num_classes: int,
           depth: int,
           in_channels: int = 3,
           norm: Optional[Type[nn.GroupNorm]] = nn.GroupNorm,
           act: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
           block: Type[BasicBlockGN] = BasicBlockGN,
           **kwargs
           ) -> ResNetGN:
    f"resnet-{depth}"
    assert (depth - 2) % 6 == 0
    layer_depth = (depth - 2) // 6
    return ResNetGN(block, num_classes, layer_depth, in_channels=in_channels, norm=norm, act=act, **kwargs)


def wide_resnetGN(num_classes: int,
                depth: int,
                widen_factor: int,
                in_channels: int = 3,
                norm: Optional[Type[nn.GroupNorm]] = nn.GroupNorm,
                act: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                block: Type[BasicBlockGN] = PreactBasicBlockGN,
                **kwargs
                ) -> ResNetGN:
    f"wideresnet-{depth}-{widen_factor}"
    assert (depth - 4) % 6 == 0
    layer_depth = (depth - 4) // 6
    return ResNetGN(block, num_classes, layer_depth, in_channels=in_channels,
                  widen_factor=widen_factor, norm=norm, act=act, preact=True, **kwargs)


@MODEL_REGISTRY.register
def wrn28_10_GN(num_classes: int = 10,
             in_channels: int = 3
             ) -> ResNetGN:
    """ WideResNet by Zagoruyko&Komodakis 17
    """
    return wide_resnetGN(num_classes, 28, 10, in_channels)

