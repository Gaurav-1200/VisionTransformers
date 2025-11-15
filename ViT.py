import torch 
import torch.nn as nn

from Config import Config
from Encoder import Encoder
from MLP import MLPHead
from PatchEmbedding import PatchEmbedding

class ViT(nn.Module):
    def __init__(self,config:Config):
        super().__init__()
        self.config = config
        self.patch_embedding = PatchEmbedding(self.config)
        self.cls_token = nn.Parameter(torch.randn(1,1,self.config.embedding_dimension))  #1 token, 1 channel , E
        self.position_embedding = nn.Parameter(torch.randn(1,1+self.config.num_patches, self.config.embedding_dimension)) # 1 . patch +CLS, E
        self.blocks = nn.ModuleList([Encoder(self.config) for  _ in range(self.config.transformers_blocks)])
        self.mlp_class_head = MLPHead(self.config)

    def forward(self,x):
        self.attention_wts = []
        self.intermediate_cls_tokens = []

        x = self.patch_embedding(x)
        cls_token = self.cls_token.expand(x.shape[0],-1,-1)  #x.shape[0] == Batch_size #1,1,E -> B,1,E
        x = torch.cat((cls_token,x),dim=1)
        x = x + self.position_embedding
        for block in self.blocks:
            x,attn_wt = block(x)
            self.attention_wts.append(attn_wt)
            self.intermediate_cls_tokens.append(x[:,0])
        final_cls = x[:,0] # first token ->CLS
        x = self.mlp_class_head(final_cls)
        return x, self.attention_wts,self.intermediate_cls_tokens