import torch
import torch.nn as nn
import torch.nn.functional as F

import custom


class Encoder(nn.Module):
    def __init__(self, in_channels, heads, dropout=None):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.produce_qkv = nn.Conv1d(in_channels, 3*in_channels, 1)
        self.attention = custom.Attention(in_channels, heads, dropout)
        self.linear = nn.Linear(in_channels)
        
    def forward(self, inputs):
        qkv = self.produce_qkv(inputs)
        queries, keys, values = qkv.split(self.in_channels, 1)
        attention = self.attention(queries, keys, values)
        outputs = F.layer_norm(attention + inputs)

        outputs = F.layer_norm(self.linear(outputs) + outputs)
        return outputs 


class Decoder(nn.Module):
    def __init__(self, in_channels, heads, dropout=None):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.produce_qkv = nn.Conv1d(in_channels, 3*in_channels, 1)
        self.produce_qk = nn.Conv1d(in_channels, 2*in_channels, 1)
        self.masked_attention = custom.Attention(in_channels, heads, dropout)
        self.attention = custom.Attention(in_channels, heads, dropout)
        self.linear = nn.Linear(in_channels)

    def forward(self, inputs, outputs, mask):
        qkv = self.produce_qkv(outputs)
        queries, keys, values = qkv.split(self.in_channels, 1)
        attention = self.masked_attention(queries, keys, values, mask)
        outputs = F.layer_norm(attention + outputs)
        
        qk = self.produce_qk(inputs)
        queries, keys = qk.split(self.in_channels, 1)
        attention = self.attention(queries, keys, outputs)
        outputs = F.layer_norm(attention + outputs)

        outputs = F.layer_norm(self.linear(outputs) + outputs)
        return outputs

