import math
import torch.nn as nn

from attacks.modules.blocks import Inception_Block_1d_V1, Inception_Block_2d_V1, Inception_Block_2d_SEB


class RFNet(nn.Module):

    def __init__(self, num_classes: int = 100, in_channel: int = 1, init_weights: bool = True):
        super(RFNet, self).__init__()
        cfg = {
            'N': [128, 128, 'M', 256, 256, 'M', 512]
        }
        self.first_layer_in_channel = in_channel
        self.first_layer_out_channel = 32
        self.first_layer = self.make_first_layers(in_channels=in_channel)
        self.features = self.make_layers(cfg['N'] + [num_classes], in_channels=self.first_layer_out_channel)
        self.classifier = nn.AdaptiveAvgPool1d(1)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        if x.dim() == 3:
            x = x.reshape(x.size(0), self.first_layer_in_channel, 2, -1)  # add channel dimension
        x = self.first_layer(x)
        x = x.view(x.size(0), self.first_layer_out_channel, -1)
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x

    @staticmethod
    def make_layers(cfg, in_channels=32):
        layers = []

        for i, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool1d(3), nn.Dropout(0.3)]
            else:
                conv1d = nn.Conv1d(in_channels, v, kernel_size=3, stride=1, padding=1)
                layers += [conv1d, nn.BatchNorm1d(v, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]
                in_channels = v

        return nn.Sequential(*layers)

    @staticmethod
    def make_first_layers(in_channels=1, out_channel=32):
        layers = []
        conv2d1 = nn.Conv2d(in_channels, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1))
        layers += [conv2d1, nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

        conv2d2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1))
        layers += [conv2d2, nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

        layers += [nn.MaxPool2d((1, 3)), nn.Dropout(0.1)]

        conv2d3 = nn.Conv2d(out_channel, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
        layers += [conv2d3, nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

        conv2d4 = nn.Conv2d(64, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
        layers += [conv2d4, nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

        layers += [nn.MaxPool2d((2, 2)), nn.Dropout(0.1)]

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class RFNet2(nn.Module):

    def __init__(self, num_classes: int = 100, in_channel: int = 1, init_weights: bool = True):
        super(RFNet2, self).__init__()
        cfg = {
            'N': [128, 128, 'M', 256, 256, 'M', 512]
        }
        self.first_layer_in_channel = in_channel
        self.first_layer_out_channel = 64
        self.first_layer = self.make_first_layers(in_channels=in_channel, out_channel=self.first_layer_out_channel)
        self.features = self.make_layers(cfg['N'] + [num_classes], in_channels=self.first_layer_out_channel)
        self.classifier = nn.AdaptiveAvgPool1d(1)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        if x.dim() == 3:
            x = x.reshape(x.size(0), self.first_layer_in_channel, 2, -1)  # add channel dimension
            # x = x.unsqueeze(1)
        x = self.first_layer(x)
        x = x.view(x.size(0), self.first_layer_out_channel, -1)
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x

    @staticmethod
    def make_layers(cfg, in_channels=32):
        layers = []

        for i, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool1d(3), nn.Dropout(0.3)]
            else:
                conv1d = Inception_Block_1d_V1(in_channels, v, 5)
                layers += [conv1d, nn.BatchNorm1d(v, eps=1e-05, momentum=0.1, affine=True), nn.GELU()]
                in_channels = v

        return nn.Sequential(*layers)

    @staticmethod
    def make_first_layers(in_channels=1, out_channel=64):
        layers = []
        # conv2d1 = nn.Conv2d(in_channels, 32, kernel_size=(2, 6), stride=1, padding=(0, 1))
        conv2d1 = Inception_Block_2d_V1(in_channels, 32, 2, 0, 5)
        layers += [conv2d1, nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True), nn.GELU()]

        conv2d2 = nn.Conv2d(32, 32, kernel_size=(1, 6), stride=1, padding=(0, 1))
        layers += [conv2d2, nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True), nn.GELU()]

        layers += [nn.MaxPool2d((1, 3)), nn.Dropout(0.1)]

        conv2d3 = nn.Conv2d(32, out_channel, kernel_size=(1, 6), stride=1, padding=(0, 1))
        layers += [conv2d3, nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.GELU()]

        conv2d4 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, 6), stride=1, padding=(0, 1))
        layers += [conv2d4, nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.GELU()]

        layers += [nn.MaxPool2d((1, 3)), nn.Dropout(0.1)]

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class RFNet3(RFNet2):

    def __init__(self, num_classes: int = 100, in_channel: int = 1, init_weights: bool = True):
        super(RFNet3, self).__init__()
        cfg = {
            'N': [128, 128, 'M', 256, 256, 'M', 512, 512, 'M']
        }
        self.first_layer_in_channel = in_channel
        self.first_layer_out_channel = 64
        self.first_layer = self.make_first_layers(in_channels=in_channel, out_channel=self.first_layer_out_channel)
        self.features = self.make_layers(cfg['N'], in_channels=self.first_layer_out_channel)
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.linear = nn.Sequential(
            nn.Linear(512 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.7),
        )

        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        if x.dim() == 3:
            x = x.reshape(x.size(0), self.first_layer_in_channel, 2, -1)  # add channel dimension
            # x = x.unsqueeze(1)
        x = self.first_layer(x)
        x = x.view(x.size(0), self.first_layer_out_channel, -1)
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(self.linear(x))
        # x = x.view(x.size(0), -1)
        return x


class RFNet4(RFNet2):

    def __init__(self, num_classes: int = 100, in_channel: int = 1,
                 num_kernels_1d: int = 4, num_kernels_2d: int = 4, init_weights: bool = True):
        super(RFNet4, self).__init__()
        cfg = {
            'N': [128, 128, 'A', 256, 256, 'A', 512]
        }
        self.first_layer_in_channel = in_channel
        self.first_layer_out_channel = 64
        self.first_layer = self.make_first_layers(in_channels=in_channel, out_channel=self.first_layer_out_channel,
                                                  num_kernels=num_kernels_2d)
        self.features = self.make_layers(cfg['N'] + [num_classes], in_channels=self.first_layer_out_channel,
                                         num_kernels=num_kernels_1d)
        self.classifier = nn.AdaptiveAvgPool1d(1)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        if x.dim() == 3:
            x = x.reshape(x.size(0), self.first_layer_in_channel, 2, -1)  # add channel dimension
            # x = x.unsqueeze(1)
        x = self.first_layer(x)
        x = x.view(x.size(0), self.first_layer_out_channel, -1)
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x

    @staticmethod
    def make_layers(cfg, in_channels=32, num_kernels=5):
        layers = []

        for i, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool1d(3), nn.Dropout(0.1)]
            elif v == 'A':
                layers += [nn.AvgPool1d(3), nn.Dropout(0.1)]
            else:
                conv1d = Inception_Block_1d_V1(in_channels, v, num_kernels)
                layers += [conv1d, nn.BatchNorm1d(v, eps=1e-05, momentum=0.1, affine=True), nn.GELU()]
                in_channels = v

        return nn.Sequential(*layers)

    @staticmethod
    def make_first_layers(in_channels=1, out_channel=64, num_kernels=5):
        layers = []
        # conv2d1 = nn.Conv2d(in_channels, 32, kernel_size=(2, 6), stride=1, padding=(0, 1))
        conv2d1 = Inception_Block_2d_SEB(in_channels, 32, 2, 0, num_kernels)
        layers += [conv2d1, nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True), nn.GELU()]

        conv2d2 = nn.Conv2d(32, 32, kernel_size=(1, 6), stride=1, padding=(0, 1))
        layers += [conv2d2, nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True), nn.GELU()]

        layers += [nn.AvgPool2d((1, 3)), nn.Dropout(0.1)]

        conv2d3 = nn.Conv2d(32, out_channel, kernel_size=(1, 6), stride=1, padding=(0, 1))
        layers += [conv2d3, nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.GELU()]

        conv2d4 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, 6), stride=1, padding=(0, 1))
        layers += [conv2d4, nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.GELU()]

        layers += [nn.AvgPool2d((1, 3)), nn.Dropout(0.1)]

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


if __name__ == '__main__':
    model = RFNet4(in_channel=4, num_kernels_1d=10, num_kernels_2d=10)
    import torch

    print(model)

    x = torch.randn(2, 4, 2, 1800)
    print(model(x).shape)
