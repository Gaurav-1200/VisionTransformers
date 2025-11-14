import torch
import torch.nn as nn
from Config import Config

class MLP(nn.Module):
    def __init__(self,config:Config):
        super().__init__()
        self.embedding_dimension = config.embedding_dimension
        self.layerNorm(self.embedding_dimension)
        self.mlp_head = nn.Linear(self.embedding_dimension)

    def forward(self,x):
        x = self.layerNorm(x)
        x = self.mlp_head(x)

        return x
