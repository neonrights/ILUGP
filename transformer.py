import torch
import torch.nn as nn
import torch.nn.functional as F

# accepts input in [ batch x channels x shape ] format
class Transformer(nn.Module):
    def __init__(self, in_channels, heads, dropout=None, masked=False):
        assert in_channels % heads == 0

        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.masked = masked
        self.dropout = 0.5
        self.produce_qkv = nn.Conv1d(in_channels, 3*in_channels, 1)

    def forward(self, input_):
        qkv = self.produce_qkv(input_)
        queries, keys, values = torch.split(qkv, self.in_channels, dim=1) 
        attention = torch.bmm(keys.permute(0,2,1), queries) / self.in_channels**0.5
        if self.masked:
            pass # mask future input
        
        attention = F.softmax(attention, dim=1)
        if dropout is not None:
            attention = F.dropout(attention, self.dropout)

        output = torch.bmm(values, attention)
        return input_ + output

# adds positional encodings
class PositionalEncoding(nn.Module):
    def forward(self, input_):
        _, channels, length = input_.shape
        numerator = torch.arange(length, dtype=torch.float)
        denominator = 1e-4 ** (2 * torch.arange(channels, dtype=torch.float) / channels)
        positional_encodings = torch.sin(torch.ger(denominator, numerator))
        return input_ + positional_encodings
 
