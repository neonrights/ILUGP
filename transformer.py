import torch
import torch.nn as nn
import torch.nn.functional as F

import custom


class EncoderLayer(nn.Module):
    def __init__(self, in_channels, heads, dropout=None):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.produce_qkv = nn.Linear(in_channels, 3*in_channels)
        self.attention = custom.Attention(in_channels, heads, dropout)
        self.linear = nn.Linear(in_channels, in_channels)
        
    def forward(self, inputs):
        qkv = self.produce_qkv(inputs)
        queries, keys, values = qkv.split(self.in_channels, -1)
        attention = self.attention(queries, keys, values)
        outputs = F.layer_norm(attention + inputs, (self.in_channels,))

        outputs = F.layer_norm(self.linear(outputs) + outputs, (self.in_channels,))
        return outputs 


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, heads, dropout=None):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.produce_qkv = nn.Linear(in_channels, 3*in_channels)
        self.produce_kv = nn.Linear(in_channels, 2*in_channels)
        self.masked_attention = custom.Attention(in_channels, heads, dropout)
        self.attention = custom.Attention(in_channels, heads, dropout)
        self.linear = nn.Linear(in_channels, in_channels)

    def forward(self, inputs, outputs):
        qkv = self.produce_qkv(outputs)
        queries, keys, values = qkv.split(self.in_channels, -1)
        
        n = inputs.shape[1]
        mask = torch.tril(torch.ones((n, n), dtype=torch.uint8))
        attention = self.masked_attention(queries, keys, values, mask)
        outputs = F.layer_norm(attention + outputs, (self.in_channels,))
        
        kv = self.produce_kv(inputs)
        keys, values = kv.split(self.in_channels, -1)
        attention = self.attention(outputs, keys, values)
        outputs = F.layer_norm(attention + outputs, (self.in_channels,))

        outputs = F.layer_norm(self.linear(outputs) + outputs, (self.in_channels,))
        return outputs


if __name__ == '__main__':
    print("Running unittests")
    test_in = torch.rand([3,4,5])
    encoder = Encoder(5, 1)
    test_out = encoder(test_in)
    assert test_out.shape == (3, 4, 5)
    print("encoder passed")
    
    decoder = Decoder(5, 1)
    test_mask = torch.tril(torch.ones((4, 4), dtype=torch.uint8))
    test_out = decoder(test_in, test_in)
    assert test_out.shape == (3, 4, 5)
    print("decoder passed")


