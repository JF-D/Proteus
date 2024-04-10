from typing import Union, List, Dict, Any, cast

import torch
import torch.nn as nn
import torch.distributed as dist

from .vgg import make_layers, cfgs
from .mpu.layers import ColumnParallelLinear, RowParallelLinear


class VGGFeature(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def _vgg_feature(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any):
    model = VGGFeature(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model

class VGGClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            # nn.Linear(512 * 7 * 7, 4096),
            ColumnParallelLinear(512 * 7 * 7, 4096, input_is_dp=True, gather_output=False),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(4096, 4096),
            RowParallelLinear(4096, 4096, input_is_parallel=True, scatter_to_dp=True),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(4095, num_classes),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class VGG_MP(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = _vgg_feature('vgg19', 'E', False, False, False)
        self.classifier = VGGClassifier(num_classes)
        self.linear = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.linear(x)
        return x

def get_vgg_mp(cuda=True):
    model = VGG_MP()
    model.train()
    if cuda:
        model.cuda()
    return model
