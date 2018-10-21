import torch
import numpy as np

from torch import nn
from torch.autograd import Function

class Transformer(nn.Module):
    def __init__(self, in_channels, masked=False):
        super().__init__()
        # initialize necessary weights
        self.produce_qkv = nn.Conv1d(in_channels, 3 * in_channels, 1)

    def forward(self, input_):
        # use weights in forward pass
        return input_


class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_):
        _, positions, dimensions = input_.shape
        numerator = torch.arange(positions, dtype=torch.float)
        denominator = 1e-4 ** (2 * torch.arange(dimensions, dtype=torch.float) / dimensions)
        positional_encodings = torch.sin(torch.ger(numerator, denominator))
        return input_ + positional_encodings
    
