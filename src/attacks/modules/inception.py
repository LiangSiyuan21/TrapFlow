import numpy as np
import torch.nn as nn

from attacks.modules.base_net import BaseNet
from attacks.modules.blocks import (MyMaxPool1dPadSame, MyMaxPool2dPadSame, MyConv1dPadSame,
                                    Inception_Block_1d_V1, Inception_Block_2d_V1)


def conv_block8(in_channels: int, out_channels: int, num_kernels: int) -> nn.Sequential:
    return nn.Sequential(
        Inception_Block_1d_V1(in_channels, out_channels, num_kernels),
        nn.BatchNorm1d(out_channels),
        nn.GELU(),

        MyConv1dPadSame(out_channels, out_channels, 8, 1),
        nn.BatchNorm1d(out_channels),
        nn.GELU(),

        MyMaxPool1dPadSame(4, 1),
        nn.Dropout(0.1),
    )


def conv2d_block8(in_channels: int, out_channels: int, w_kernal: int, num_kernels: int) -> nn.Sequential:
    return nn.Sequential(
        Inception_Block_2d_V1(in_channels, out_channels, w_kernal, 0, num_kernels),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),

        Inception_Block_2d_V1(out_channels, out_channels, 1, 0, num_kernels),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),

        MyMaxPool2dPadSame(4, 1),
        nn.Dropout(0.1),
    )


def conv_block5(in_channels: int, out_channels: int, num_kernels: int) -> nn.Sequential:
    return nn.Sequential(
        MyConv1dPadSame(in_channels, out_channels, 8, 1),
        nn.BatchNorm1d(out_channels),
        nn.GELU(),

        MyMaxPool1dPadSame(4, 1),
        nn.Dropout(0.1),
    )


def conv2d_block5(in_channels: int, out_channels: int, w_kernal: int, num_kernels: int) -> nn.Sequential:
    return nn.Sequential(
        Inception_Block_2d_V1(in_channels, out_channels, w_kernal, 0, num_kernels),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),

        MyMaxPool2dPadSame(4, 1),
        nn.Dropout(0.1),
    )


class InceptionNet(BaseNet):
    def __init__(self, length: int, num_classes: int = 100, in_channels: int = 1, num_kernels: int = 6):
        super().__init__(length, num_classes, in_channels)
        self.num_kernel = num_kernels
        self.in_channels = in_channels

        self.features = nn.Sequential(
            Inception_Block_2d_V1(in_channels, 32, 2, 0, 5),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
            nn.GELU())

        self.cnn_layers = nn.Sequential(
            conv_block8(32, 64, num_kernels),
            conv_block8(64, 128, num_kernels),
            conv_block8(128, 256, num_kernels),
            conv_block5(256, 512, num_kernels),
        )

        self.linear = nn.Sequential(
            nn.Linear(512 * self.linear_input(), 512),
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
        if x.dim() == 3:
            x = x.reshape(x.size(0), self.in_channels, 2, -1)
        out = self.features(x)
        out = out.view(out.size(0), 32, -1)
        out = self.cnn_layers(out)
        out = out.reshape(out.size(0), -1)
        out = self.linear(out)
        out = self.fc(out)
        return out

    def linear_input(self):
        res = self.length
        for i in range(4):
            res = int(np.ceil(res / 4))
        return res


if __name__ == '__main__':
    from torchsummary import summary

    model = InceptionNet(1800, 100, 4, 3).to('cuda')
    summary(model, (8, 1800))

    # from attacks.modules.df import DFNet
    #
    # model = DFNet(1024, 100, 2)
    # summary(model, (2, 1024))
