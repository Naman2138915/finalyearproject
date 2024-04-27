import torch
from torch import nn
from torch.nn import functional as F
from attentions import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, num_vocab: int, num_embed: int, num_token: int):
        super().__init__()    
        self.tokenembed = nn.Embedding(num_vocab, num_embed)
        self.positionembed = nn.Parameter(torch.zeros((num_token, num_embed)))
    def forward(self, tokens):
        x = self.tokenembed(tokens)
        x += self.positionembed      
        return x

class CLIPLayer(nn.Module):
    def __init__(self, num_head: int, num_embed: int):
        super().__init__()
        self.layernormalization1 = nn.LayerNorm(num_embed)
        self.attention = SelfAttention(num_head, num_embed)
        self.layernormalization2 = nn.LayerNorm(num_embed)
        self.linear1 = nn.Linear(num_embed, 4 * num_embed)
        self.linear2 = nn.Linear(4 * num_embed, num_embed)
    def forward(self, x):
        residue = x
        x = self.layernormalization1(x)
        x = self.attention(x, mask=True)
        x += residue
        residue = x
        x = self.layernormalization2(x)
        x = self.linear1(x)
        x = x * torch.sigmoid(1.702 * x)
        x = self.linear2(x)
        x += residue
        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])
        self.layernormalization = nn.LayerNorm(768)
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        state = self.embedding(tokens)
        for layer in self.layers: 
            state = layer(state)
        output = self.layernormalization(state)
        return output
