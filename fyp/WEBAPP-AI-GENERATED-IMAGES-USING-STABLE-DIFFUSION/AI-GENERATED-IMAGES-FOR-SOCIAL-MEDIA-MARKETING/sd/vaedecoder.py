import torch
from torch import nn
from torch.nn import functional as Func
from attentions import SelfAttention

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnormalization = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    def forward(self, x):
        residue = x 
        x = self.groupnormalization(x)
        batch, channel, height, width = x.shape
        x = x.view((batch, channel, height * width))
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view((batch, channel, height , width))
        x += residue
        return x 

class ResidualBlock(nn.Module):
    def __init__(self, incomingchanel, outgoingchanel):
        super().__init__()
        self.groupnormalization1 = nn.GroupNorm(32, incomingchanel)
        self.convolution1 = nn.Conv2d(incomingchanel, outgoingchanel, kernel_size=3, padding=1)
        self.groupnormalization2 = nn.GroupNorm(32, outgoingchanel)
        self.convolution2 = nn.Conv2d(outgoingchanel, outgoingchanel, kernel_size=3, padding=1)
        if incomingchanel == outgoingchanel:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(incomingchanel, outgoingchanel, kernel_size=1, padding=0)   
    def forward(self, x):
        residue = x
        x = self.groupnormalization1(x)
        x = Func.silu(x)
        x = self.convolution1(x)
        x = self.groupnormalization2(x)
        x = Func.silu(x)
        x = self.convolution2(x)
        return x + self.residual_layer(residue)

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512), 
            AttentionBlock(512), 
            ResidualBlock(512, 512), 
            ResidualBlock(512, 512), 
            ResidualBlock(512, 512), 
            ResidualBlock(512, 512), 
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            ResidualBlock(512, 512), 
            ResidualBlock(512, 512), 
            ResidualBlock(512, 512), 
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            ResidualBlock(512, 256), 
            ResidualBlock(256, 256), 
            ResidualBlock(256, 256), 
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            ResidualBlock(256, 128), 
            ResidualBlock(128, 128), 
            ResidualBlock(128, 128), 
            nn.GroupNorm(32, 128), 
            nn.SiLU(), 
            nn.Conv2d(128, 3, kernel_size=3, padding=1), 
        )
    def forward(self, x):
        x /= 0.18215
        for module in self:
            x = module(x)
        return x
