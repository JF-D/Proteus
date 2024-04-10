import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .mpu.layers import ColumnParallelLinear, RowParallelLinear


class Classifier(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # nn.Linear(256 * 6 * 6, 4096),
            ColumnParallelLinear(256 * 6 * 6,
                                 4096,
                                 input_is_dp=True,
                                 gather_output=False),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # nn.Linear(4096, 4096),
            RowParallelLinear(4096,
                              4096,
                              input_is_parallel=True,
                              scatter_to_dp=True),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.classifier(x)
        return x


class Features(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class AlexNet_MP(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet_MP, self).__init__()
        self.features = Features()
        self.classifier = Classifier(num_classes)
        self.linear = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.linear(x)
        return x


def get_alexnet_mp(cuda=True):
    model = AlexNet_MP()
    model.train()
    if cuda:
        model.cuda()
    return model
