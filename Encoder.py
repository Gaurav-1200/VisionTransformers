import torch
import torch.nn as nn
from Config import Config

class Encoder(nn.Module):
    def __init__(self,config:Config):
        super().__init__()
        self.embedding_dimension = config.embedding_dimension
        self.attention_head = config.attention_heads
        self.mlp_hidden = config.mlp_hidden
        self.layerNorm1 = nn.LayerNorm(self.embedding_dimension)
        self.multihead_attention = nn.MultiheadAttention(self.embedding_dimension, self.attention_head,batch_frst= True)
        self.layerNorm2 = nn.LayerNorm(self.embedding_dimension)
        self.MLP = nn.Sequential(
            nn.Linear(self.embedding_dimension,self.mlp_hidden),
            nn.GELU(),
            nn.Linear(self.mlp_hidden,self.embedding_dimension),
        )

    def forward(self,x):
        residual_1 = x
        x = self.layerNorm1(x)
        attn_output, attn_weights = self.multihead_attention(x,x,x)
        x = attn_output + residual_1

        residual_2 = x
        x = self.layerNorm2(x)
        x = self.MLP(x)
        x = x + residual_2

        return x,attn_weights   # for plotting





