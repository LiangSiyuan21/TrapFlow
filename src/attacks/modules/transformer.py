"""
File: models.py
Author: lok
Edited By: jzx-bupt
"""

import copy

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from attacks.modules.df import conv_block8


class Transformer(nn.Module):

    def __init__(self, embed_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()

        self.encoder = TransformerEncoder(TransformerEncoderLayer(embed_dim, nhead, dim_feedforward, dropout),
                                          num_encoder_layers)
        self.decoder = TransformerDecoder(TransformerDecoderLayer(embed_dim, nhead, dim_feedforward, dropout),
                                          num_decoder_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed):
        bs = src.shape[0]
        memory = self.encoder(src, pos_embed)
        query_embed = query_embed.repeat(bs, 1, 1)
        tgt = torch.zeros_like(query_embed)
        output = self.decoder(tgt, memory, pos_embed, query_embed)

        return output


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos, query_pos):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, pos, query_pos)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, src, pos):
        src2 = self.self_attn(query=src + pos, key=src + pos, value=src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, embed_dim, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, tgt, memory, pos, query_pos):
        tgt2 = self.self_attn(query=tgt + query_pos, key=tgt + query_pos, value=tgt)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=tgt + query_pos, key=memory + pos, value=memory)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class BAPM_CNN(nn.Module):

    def __init__(self, filters, kernels, pools, dropout):
        super(BAPM_CNN, self).__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv1d(1, filters[0], kernel_size=kernels[0]),
            nn.ReLU(),
            nn.BatchNorm1d(filters[0]),
            nn.MaxPool1d(kernel_size=pools[0], padding=2),
            nn.Dropout(dropout),
            nn.Conv1d(filters[0], filters[1], kernel_size=kernels[1]),
            nn.ReLU(),
            nn.BatchNorm1d(filters[1]),
            nn.MaxPool1d(kernel_size=pools[1], padding=2),
            nn.Dropout(dropout),
            nn.Conv1d(filters[1], filters[2], kernel_size=kernels[2]),
            nn.ReLU(),
            nn.BatchNorm1d(filters[2]),
            nn.MaxPool1d(kernel_size=pools[2], padding=2),
            nn.Dropout(dropout),
        )

    def forward(self, input):
        if len(input.shape) == 2:
            x = input.unsqueeze(1)
        else:
            x = input
        return self.cnn_layer(x).transpose(1, 2)


class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self._norm_layer = torch.nn.BatchNorm1d
        self.stride = 1
        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, 3, self.stride, padding=1),
            self._norm_layer(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv1d(out_channels, out_channels, 3, padding=1),
            self._norm_layer(out_channels),
            torch.nn.ReLU(),
        )

        if in_channels != out_channels:
            self.res_layer = torch.nn.Conv1d(in_channels, out_channels, 1, self.stride)
        else:
            self.res_layer = None

    def forward(self, x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x) + residual


class TMWF_DFNet(nn.Module):

    def __init__(self, length, num_classes, in_channels,
                 embed_dim=64, nhead=8,
                 num_encoder_layers=0,
                 num_decoder_layers=2,
                 dropout=0.1, num_queries=1):
        super(TMWF_DFNet, self).__init__()
        dim_feedforward = embed_dim * 4
        self.cnn_layer = nn.Sequential(conv_block8(in_channels, 32, 8, 1, 0),
                                       conv_block8(32, embed_dim, 8, 1, 1))
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.trm = Transformer(embed_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.pos_embed = nn.Embedding(length // int(math.pow(8, len(self.cnn_layer))), embed_dim).weight
        self.query_embed = nn.Embedding(num_queries, embed_dim).weight
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input):
        x = self.cnn_layer(input)
        x = x.transpose(1, 2)
        feat = self.proj(x)
        o = self.trm(feat, self.query_embed.unsqueeze(0), self.pos_embed.unsqueeze(0))
        logits = self.fc(o)

        return logits[:, 0, :]


if __name__ == '__main__':
    model = TMWF_DFNet(length=1024, num_classes=101, in_channels=1)
    # model = TMWF_noDF(embed_dim=128, nhead=8, dim_feedforward=512, num_encoder_layers=2,
    #                   num_decoder_layers=2, max_len=1024 // (8 * 8 * 8), num_queries=2,
    #                   cls=100, dropout=0.1)
    # # summary(model, (1, 1024))

    x = torch.randn(2, 1, 1024)
    y = model(x)
    print(y.shape)
