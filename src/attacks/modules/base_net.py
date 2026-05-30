import torch
import torch.nn as nn


class BaseNet(nn.Module):
    def __init__(self, length: int, num_classes: int = 100, in_channels: int = 1, *args, **kwargs):
        super(BaseNet, self).__init__()
        self.length = length
        self.num_classes = num_classes
        self.in_channels = in_channels

    def forward(self, x: torch.tensor):
        raise NotImplementedError
