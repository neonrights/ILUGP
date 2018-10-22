import torch
import torch.nn as nn
import torch.nn.functional as F

# accepts input in [ batch x channels x shape ] format
class Transformer(nn.Module):
    def __init__(self, in_channels, masked=False):
        super().__init__()
        # initialize necessary weights
        self.produce_qkv = nn.Conv1d(in_channels, 3 * in_channels, 1)
        self.in_channels = in_channels
        self.masked = masked

    def forward(self, input_):
        qkv = self.produce_qkv(input_)
        queries, keys, values = torch.split(qkv, self.in_channels, dim=1) 
        output = F.softmax(torch.matmul(queries, torch.t(keys)) / torch.sqrt(self.in_channels))
        if self.masked:
            pass # mask output

        output = torch.matmul(output, torch.t(values))
        return input_ + output

# adds positional encodings
class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_):
        _, channels, length = input_.shape
        numerator = torch.arange(length, dtype=torch.float)
        denominator = 1e-4 ** (2 * torch.arange(channels, dtype=torch.float) / channels)
        positional_encodings = torch.sin(torch.ger(denominator, numerator))
        return input_ + positional_encodings
    
