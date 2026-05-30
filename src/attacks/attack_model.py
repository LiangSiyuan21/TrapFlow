import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torchsummaryX import summary
import torch.nn.init as init
'''Conditional WGAN with gradient penalty'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import torch.nn.functional as F



import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

class InsertPolicyNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2):
        super(InsertPolicyNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # 输出单个位置的插入长度预测值

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        outputs, _ = self.lstm(x)
        last_output = outputs[:, -1, :]  # 使用最后一个时间步的输出
        logits = self.fc(last_output)    # [batch_size, 1]
        return logits  # 返回未经过激活的连续值





# class InsertPolicyNetwork(nn.Module):
#     def __init__(self, input_dim=2, hidden_dim=128, num_layers=2, kernel_size=3, distance='levenshtein'):
#         super(InsertPolicyNetwork, self).__init__()
#         # simple        
#         # self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
#         # self.pool = nn.MaxPool1d(kernel_size=1, stride=1)

#         # complex
#         self.initial_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=(kernel_size - 1) // 2)
#         self.layers = nn.ModuleList()
#         for _ in range(num_layers - 1):
#             self.layers.append(nn.Sequential(
#                 nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=(kernel_size - 1) // 2),
#                 nn.ReLU()
#             ))
        
#         self.position_pred = nn.Linear(hidden_dim, 1)
#         self.count_pred = nn.Linear(hidden_dim, 1)
        
#         self.distance = distance 

#         self._initialize_weights()

#     # simple
#     # def _initialize_weights(self):
#     #     nn.init.xavier_uniform_(self.conv1.weight)
#     #     nn.init.zeros_(self.conv1.bias)
#     #     nn.init.xavier_uniform_(self.position_pred.weight)
#     #     nn.init.zeros_(self.position_pred.bias)
#     #     nn.init.xavier_uniform_(self.count_pred.weight)
#     #     nn.init.zeros_(self.count_pred.bias)

#     # complex
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)


#     def forward(self, x):
#         # x: [batch_size, seq_len, input_dim]
        
#         # Compute the original mask
#         non_zero_mask = torch.abs(x[:, :, 1]) > 0  # [batch_size, seq_len]
#         padding_mask = torch.cumprod((x[:, :, 1] != 0).int(), dim=1).bool()
#         mask = non_zero_mask & padding_mask  # [batch_size, seq_len]
        
#         # simple
#         # # Permute for convolution
#         # x = x.permute(0, 2, 1)  # [batch_size, input_dim, seq_len]
#         # # Convolution and activation
#         # x = F.relu(self.conv1(x))  # [batch_size, hidden_dim, seq_len]
#         # # No pooling or pooling with stride 1
#         # # x = self.pool(x)  # [batch_size, hidden_dim, seq_len]
#         # # Transpose back to [batch_size, seq_len, hidden_dim]
#         # x = x.permute(0, 2, 1)  # [batch_size, seq_len, hidden_dim]

#         x = x.permute(0, 2, 1)  # [batch_size, input_dim, seq_len]
#         x = F.relu(self.initial_conv(x))  # Initial convolution
        
#         # Residual connections
#         for layer in self.layers:
#             residual = x
#             x = layer(x)
#             x = x + residual  # Add residual connection
        
#         x = x.permute(0, 2, 1)  # [batch_size, seq_len, hidden_dim]
        
        
#         # Compute logits
#         position_logits = self.position_pred(x).squeeze(-1)  # [batch_size, seq_len]
#         count_logits = self.count_pred(x).squeeze(-1)        # [batch_size, seq_len]
        
#         # Apply mask
#         inf_mask = torch.full_like(position_logits, -1e6)
#         position_logits = torch.where(mask, position_logits, inf_mask)
#         count_logits = torch.where(mask, count_logits, inf_mask)
        
#         return position_logits, count_logits



# # class InsertPolicyNetwork(nn.Module):
# #     def __init__(self, input_dim=2, hidden_dim=128, distance='levenshtein'):
# #         super(InsertPolicyNetwork, self).__init__()

# #         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
# #         self.position_pred = nn.Linear(hidden_dim, 1)
# #         self.count_pred = nn.Linear(hidden_dim, 1)
# #         self.distance = distance 

# #         self._initialize_weights()

# #     def _initialize_weights(self):
# #         # 初始化 LSTM 权重
# #         for name, param in self.lstm.named_parameters():
# #             if 'weight' in name:
# #                 init.xavier_uniform_(param)  # 使用 Xavier 初始化 LSTM 权重
# #             elif 'bias' in name:
# #                 nn.init.zeros_(param)  # 将偏置初始化为 0

# #         # 初始化全连接层 (Linear) 的权重
# #         init.xavier_uniform_(self.position_pred.weight)  # Xavier 初始化 Linear 层权重
# #         init.zeros_(self.position_pred.bias)  # 将 bias 初始化为 0

# #         init.xavier_uniform_(self.count_pred.weight)  # Xavier 初始化 Linear 层权重
# #         init.zeros_(self.count_pred.bias)  # 将 bias 初始化为 0

# #     def forward(self, x):
# #         # x: [batch_size, seq_len, input_dim]
# #         # 计算掩码，只关注非零流量位置，但同时识别填充位置
# #         non_zero_mask = torch.abs(x[:, :, 1]) > 0  # 非零流量位置掩码
# #         # 使用cumprod找到第一个0后的所有位置
# #         padding_mask = torch.cumprod((x[:, :, 1] != 0).int(), dim=1).bool()

# #         # 确保掩码同时满足非零和非填充条件
# #         mask = non_zero_mask & padding_mask

# #         # LSTM 处理
# #         output, _ = self.transformer(x)
# #         # output, _ = self.lstm(x)

# #         # 计算位置和计数logits
# #         position_logits = self.position_pred(output).squeeze(-1)  # [batch_size, seq_len]
# #         count_logits = self.count_pred(output).squeeze(-1)        # [batch_size, seq_len]

# #         # 应用掩码：将掩码之外的位置的logits设置为负大数
# #         inf_mask = torch.full_like(position_logits, -1e6)  # 创建一个与logits形状相同，值为-1e6的tensor
# #         position_logits = torch.where(mask, position_logits, inf_mask)  # 未选中的设置为-1e6
# #         count_logits = torch.where(mask, count_logits, inf_mask)  # 未选中的设置为-1e6

# #         return position_logits, count_logits


class Generator(nn.Module):
    def __init__(self, seq_size, class_dim, latent_dim, scaler_min, scaler_max, is_gpu=False):
        super(Generator, self).__init__()
        self.seq_size = seq_size
        self.class_dim = class_dim
        self.latent_dim = latent_dim
        self.LongTensor = torch.cuda.LongTensor if is_gpu else torch.LongTensor
        self.scaler_min = scaler_min
        self.scaler_max = scaler_max

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim + self.class_dim, 512, normalize=False),
            *block(512, 1024),
            *block(1024, 2048),
            nn.Linear(2048, self.seq_size),
            nn.Sigmoid()
        )

    def forward(self, z, c):
        input = torch.cat([z, c], 1)
        trace = self.model(input)

        # 1) mask the tail of each trace according to the first element which is the learned burst seq length
        # https://discuss.pytorch.org/t/set-value-of-torch-tensor-up-to-some-index/102097
        burst_length = trace[:, 0] * (self.scaler_max - self.scaler_min) + self.scaler_min
        mask = torch.zeros_like(trace)
        mask[(torch.arange(trace.shape[0]), burst_length.type(self.LongTensor) + 1)] = 1
        mask = 1 - mask.cumsum(dim=-1)

        trace = trace * mask
        return trace


class Discriminator(nn.Module):
    def __init__(self, seq_size, class_dim):
        super(Discriminator, self).__init__()
        self.seq_size = seq_size
        self.class_dim = class_dim
        self.model = nn.Sequential(
            nn.Linear(self.seq_size + self.class_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, trace, c):
        input = torch.cat([trace, c], 1)
        validity = self.model(input)
        return validity


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


class DF(nn.Module):
    def __init__(self, length, num_classes=100):
        super(DF, self).__init__()
        self.length = length
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(
            # nn.Conv1d(1, 32, kernel_size=9, stride=1, padding=4),
            MyConv1dPadSame(1, 32, 8, 1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            # nn.Conv1d(32, 32, kernel_size=9, stride=1, padding=4),
            MyConv1dPadSame(32, 32, 8, 1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            # nn.MaxPool1d(kernel_size=4, stride=4),
            MyMaxPool1dPadSame(8, 1),
            nn.Dropout(0.1)
        )
        self.layer2 = nn.Sequential(
            # nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4),
            MyConv1dPadSame(32, 64, 8, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Conv1d(64, 64, kernel_size=9, stride=1, padding=4),
            MyConv1dPadSame(64, 64, 8, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=4, stride=4),
            MyMaxPool1dPadSame(8, 1),
            nn.Dropout(0.1)
        )
        self.layer3 = nn.Sequential(
            # nn.Conv1d(64, 128, kernel_size=9, stride=1, padding=4),
            MyConv1dPadSame(64, 128, 8, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Conv1d(128, 128, kernel_size=9, stride=1, padding=4),
            MyConv1dPadSame(128, 128, 8, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=4, stride=4),
            MyMaxPool1dPadSame(8, 1),
            nn.Dropout(0.1)
        )
        self.layer4 = nn.Sequential(
            MyConv1dPadSame(128, 256, 8, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            MyMaxPool1dPadSame(8, 1),
            nn.Dropout(0.1)
        )
        self.layer5 = nn.Sequential(
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
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer5(out)
        out = self.fc(out)
        return out

    def linear_input(self):
        res = self.length
        for i in range(4):
            res = int(np.ceil(res / 8))
        return res


if __name__ == '__main__':
    generator = Generator(1400, 100, 500, 0, 100, is_gpu=False)
    summary(generator, torch.zeros((32, 500)), c=torch.zeros(32, 100))