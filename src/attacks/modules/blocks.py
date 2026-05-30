import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride)

    def forward(self, x):
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.conv(net)

        return net


class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """

    def __init__(self, kernel_size, stride_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride_size
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.max_pool(net)

        return net


class MyMaxPool2dPadSame(nn.Module):
    """
    Extend nn.MaxPool2d to support SAME padding
    """

    def __init__(self, kernel_size, stride_size):
        super(MyMaxPool2dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride_size
        self.max_pool = torch.nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        net = x

        # Compute pad shape for height
        in_height = net.shape[-2]
        out_height = (in_height + self.stride - 1) // self.stride
        pad_h = max(0, (out_height - 1) * self.stride + self.kernel_size - in_height)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # Compute pad shape for width
        in_width = net.shape[-1]
        out_width = (in_width + self.stride - 1) // self.stride
        pad_w = max(0, (out_width - 1) * self.stride + self.kernel_size - in_width)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Apply padding
        net = F.pad(net, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)

        # Apply max pooling
        net = self.max_pool(net)

        return net


class Inception_Block_1d_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_1d_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv1d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Block_2d_V1(nn.Module):
    """
    For RFNet2
    """

    def __init__(self, in_channels, out_channels, w_kernal, w_padding, num_kernels=6, init_weight=True):
        super(Inception_Block_2d_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=(w_kernal, 2 * i + 1), padding=(w_padding, i)))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class CBAMBlock(nn.Module):
    def __init__(self, in_planes, planes, kernel, padding, stride=1):
        super(CBAMBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.cbam(out)
        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiInputFusion(nn.Module):
    def __init__(self, out_channel: int = 32, reduction: int = 16, fusion_input_dim: int = 4,
                 feature_length_ratio: int = 5,
                 df: bool = True, tiktok: bool = True, tam: bool = True, fusion: bool = True):
        super(MultiInputFusion, self).__init__()

        """
        feature_length_ration means tam and fusion only use seq_length/feature_length_ratio length.
        """

        assert df or tiktok or tam or fusion, "At least one feature type should be enabled"
        self.df = df
        self.tiktok = tiktok
        self.tam = tam
        self.fusion = fusion
        self.feature_length_ratio = feature_length_ratio

        self.num_input = int(df) + int(tiktok) + int(tam) + int(fusion)
        self.fusion_input_dim = fusion_input_dim
        self.df_conv = nn.Conv2d(1, out_channel, kernel_size=(1, 7), stride=feature_length_ratio, padding=(0, 3))
        self.tiktok_conv = nn.Conv2d(1, out_channel, kernel_size=(1, 7), stride=feature_length_ratio, padding=(0, 3))
        self.tam_conv = nn.Conv2d(1, out_channel, kernel_size=(2, 3), stride=1, padding=(0, 1))
        self.fusion_conv = nn.Conv2d(fusion_input_dim, out_channel, kernel_size=(2, 3), stride=1, padding=(0, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channel, out_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel // reduction, out_channel * self.num_input, bias=False),
            nn.Sigmoid()
        )

    def forward(self, xs):
        # first is tam, second is fusion
        out = []

        length = xs.size(-1) // self.feature_length_ratio

        if self.df:
            x_df = xs[:, :1, :].reshape(xs.size(0), 1, 1, -1)
            out.append(self.df_conv(x_df))
        if self.tiktok:
            x_tiktok = xs[:, 1:2, :].reshape(xs.size(0), 1, 1, -1)
            out.append(self.tiktok_conv(x_tiktok))
        if self.tam:
            x_tam = xs[:, 2:4, :length].reshape(xs.size(0), 1, 2, -1)
            out.append(self.tam_conv(x_tam))
        if self.fusion:
            x_fusion = xs[:, 4:, :length].reshape(xs.size(0), self.fusion_input_dim, 2, -1)
            out.append(self.fusion_conv(x_fusion))

        stacked_out = torch.stack(out, dim=1)  # [b, num_input, c, h, w]
        out = torch.sum(stacked_out, dim=1)  # [b, c, h, w]

        b, c, _, _ = out.size()
        y = self.avg_pool(out).view(b, c)  # [b, c]
        y = self.fc(y).view(b, self.num_input, c, 1, 1)  # [b, num_input, c, 1, 1]

        return (stacked_out * y.expand_as(stacked_out)).sum(dim=1)


class Inception_Block_2d_SEB(nn.Module):
    def __init__(self, in_channels, out_channels, w_kernal, w_padding, num_kernels=5, reduction=16, init_weight=True):
        super(Inception_Block_2d_SEB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=(w_kernal, 2 * i + 1), padding=(w_padding, i)))
        self.kernels = nn.ModuleList(kernels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(out_channels // reduction, out_channels * num_kernels, bias=False),
            nn.Sigmoid()
        )

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))

        res = torch.stack(res_list, dim=1)  # [b, num_kernels, c, h, w]
        res_sum = res.sum(dim=1)  # [b, c, h, w]

        b, c, _, _ = res_sum.size()
        y = self.avg_pool(res_sum).view(b, c)  # [b, c]
        y = self.fc(y).view(b, self.num_kernels, c, 1, 1)

        return (res * y.expand_as(res)).sum(dim=1)


class Inception_Block_1d_SEB(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=5, reduction=16, init_weight=True):
        super(Inception_Block_1d_SEB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv1d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // reduction, out_channels * num_kernels, bias=False),
            nn.Sigmoid()
        )

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))

        res = torch.stack(res_list, dim=1)  # [b, num_kernels, c, L]
        res_sum = res.sum(dim=1)  # [b, c, L]

        b, c, _ = res_sum.size()
        y = self.avg_pool(res_sum).view(b, c)  # [b, c]
        y = self.fc(y).view(b, self.num_kernels, c, 1)

        return (res * y.expand_as(res)).sum(dim=1)
