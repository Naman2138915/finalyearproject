import torch
from torch import nn
from torch.nn import functional as Func
import math

class SelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, inbias=True, outbias=True):
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=inbias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=outbias)
        self.num_heads = num_heads
        self.dim_head = embed_dim // num_heads
    def forward(self, x, mask=False):
        input_shape = x.shape 
        batch_size, sequence_length, embed_dim = input_shape 
        interim_shape = (batch_size, sequence_length, self.num_heads, self.dim_head) 
        query, keys, values = self.in_proj(x).chunk(3, dim=-1)
        query = query.view(interim_shape).transpose(1, 2)
        keys = keys.view(interim_shape).transpose(1, 2)
        values = values.view(interim_shape).transpose(1, 2)
        weight = query @ keys.transpose(-1, -2)
        if mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        weight /= math.sqrt(self.dim_head)
        weight = Func.softmax(weight, dim=-1)
        output = weight @ values
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)
        output = self.out_proj(output)
        return output

class CrossAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, cross_dim, inbias=True, outbias=True):
        super().__init__()
        self.query_proj   = nn.Linear(embed_dim, embed_dim, bias=inbias)
        self.keys_proj   = nn.Linear(cross_dim, embed_dim, bias=inbias)
        self.values_proj   = nn.Linear(cross_dim, embed_dim, bias=inbias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=outbias)
        self.num_heads = num_heads
        self.dim_head = embed_dim // num_heads
    def forward(self, x, y):
        input_shape = x.shape
        batch_size, sequence_length, embed_dim = input_shape
        interim_shape = (batch_size, -1, self.num_heads, self.dim_head)
        query = self.query_proj(x)
        keys = self.keys_proj(y)
        values = self.values_proj(y)
        query = query.view(interim_shape).transpose(1, 2)
        keys = keys.view(interim_shape).transpose(1, 2)
        values = values.view(interim_shape).transpose(1, 2)
        weight = query @ keys.transpose(-1, -2)
        weight /= math.sqrt(self.dim_head)
        weight = Func.softmax(weight, dim=-1)
        output = weight @ values
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)
        return output
