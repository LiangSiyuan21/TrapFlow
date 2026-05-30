import numpy as np
import torch.nn as nn

from attacks.modules.base_net import BaseNet
from attacks.modules.blocks import MyConv1dPadSame, MyMaxPool1dPadSame


def conv_block8(in_channels: int, out_channels: int, kernel_size: int, stride: int,
                index: int) -> nn.Sequential:
    return nn.Sequential(
        MyConv1dPadSame(in_channels, out_channels, kernel_size, stride),
        nn.BatchNorm1d(out_channels),
        nn.ELU() if index == 0 else nn.ReLU(),

        MyConv1dPadSame(out_channels, out_channels, kernel_size, stride),
        nn.BatchNorm1d(out_channels),
        nn.ELU() if index == 0 else nn.ReLU(),

        MyMaxPool1dPadSame(8, 1),
        nn.Dropout(0.1),
    )


def conv_block5(in_channels: int, out_channels: int, kernel_size: int, stride: int,
                index: int, pooling=True) -> nn.Sequential:
    return nn.Sequential(
        MyConv1dPadSame(in_channels, out_channels, kernel_size, stride),
        nn.BatchNorm1d(out_channels),
        nn.ReLU() if index == 0 else nn.ELU(),
        MyMaxPool1dPadSame(8, 1) if pooling else nn.Identity(),
        nn.Dropout(0.1),
    )


class DFNet(BaseNet):
    def __init__(self, length: int, num_classes: int = 100, in_channels: int = 1):
        super(DFNet, self).__init__(length, num_classes, in_channels)

        self.cnn_layers = nn.Sequential(
            conv_block8(in_channels, 32, 8, 1, 0),
            conv_block8(32, 64, 8, 1, 1),
            conv_block8(64, 128, 8, 1, 2),
            conv_block5(128, 256, 8, 1, 3),
        )

        self.linear = nn.Sequential(
            nn.Linear(256 * self.linear_input(), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc = nn.Linear(512, self.num_classes)

    def forward(self, x):
        out = self.cnn_layers(x)
        out = out.reshape(out.size(0), -1)
        out = self.linear(out)
        out = self.fc(out)
        return out

    def linear_input(self):
        res = self.length
        for i in range(4):
            res = int(np.ceil(res / 8))
        return res

    def get_features(self, x):
        out = self.cnn_layers(x)
        out = out.reshape(out.size(0), -1)
        features = self.linear(out)
        return features

class DFBackbone(BaseNet):
    def __init__(self, length: int, in_channels: int = 1):
        super(DFBackbone, self).__init__(length, in_channels)

        self.cnn_layers = nn.Sequential(
            conv_block8(in_channels, 32, 8, 1, 0),
            conv_block8(32, 64, 8, 1, 1),
            conv_block8(64, 128, 8, 1, 2),
            conv_block5(128, 256, 8, 1, 3),
        )

    def forward(self, x):
        out = self.cnn_layers(x)
        return out
