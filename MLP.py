import torch
import torch.nn as nn
from Config import Config

class MLPHead(nn.Module):
    def __init__(self,config:Config):
        super().__init__()
        self.embedding_dimension = config.embedding_dimension
        self.num_classes = config.num_classes
        self.layerNorm = nn.LayerNorm(self.embedding_dimension)
        self.mlp_head = nn.Linear(self.embedding_dimension,self.num_classes)

    def forward(self,x):
        x = self.layerNorm(x)
        x = self.mlp_head(x)

        return x
