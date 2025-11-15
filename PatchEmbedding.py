import torch
import torch.nn as nn
from Config import Config
class PatchEmbedding(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.patch_size = config.patch_size
        self.stride = config.stride
        self.n_channels = config.n_channel
        self.embedding_dimensions = config.embedding_dimension
        self.patch_embed = nn.Conv2d(self.n_channels,self.embedding_dimensions,self.patch_size,stride=self.stride)

    def forward(self,x):
        x = self.patch_embed(x)    #B C H W ->B E H/P W/P
        x = x.flatten(2)           #B E H/P W/P   -> B E (H/P * W/P)
        x = x.transpose(2,1)        #B E (H/P * W/P) -> B (H/P * W/P) E
        return x