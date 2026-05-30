import numpy as np
import torch.nn as nn

from attacks.modules.base_net import BaseNet
from attacks.modules.df import DFBackbone


class TFNet(BaseNet):
    def __init__(self, length: int, num_classes: int = 100, in_channels: int = 1, emb_dim: int = 64):
        super(TFNet, self).__init__(length, num_classes, in_channels)
        self.emb_dim = emb_dim
        self.backbone = DFBackbone(length, in_channels)
        self.fc = nn.Linear(256 * self.linear_input(), self.emb_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    def linear_input(self):
        res = self.length
        for i in range(4):
            res = int(np.ceil(res / 8))
        return res
