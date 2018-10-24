import torch
import torch.nn as nn
import torch.nn.functional as F

# accepts input in [ batch x channels x shape ] format
class Attention(nn.Module):
    def __init__(self, in_channels, heads, dropout=None):
        assert in_channels % heads == 0

        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.dropout = dropout

    def forward(self, queries, keys, values, mask=None):
        attention = torch.bmm(queries, keys.permute(0,2,1)) / self.in_channels**0.5
        if mask is not None:
            attention = attention.masked_fill(mask, -1e9)
        
        attention = F.softmax(attention, dim=-1)
        if self.dropout is not None:
            attention = F.dropout(attention, self.dropout)

        output = torch.bmm(attention, values)
        return output

# adds positional encodings
class PositionalEncoding(nn.Module):
    def forward(self, input_):
        _, channels, length = input_.shape
        numerator = torch.arange(length, dtype=torch.float)
        denominator = 1e-4 ** (2 * torch.arange(channels, dtype=torch.float) / channels)
        positional_encodings = torch.sin(torch.ger(denominator, numerator))
        return input_ + positional_encodings
 

if __name__ == '__main__':
    print("please implement unittests")


