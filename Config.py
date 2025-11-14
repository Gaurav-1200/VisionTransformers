from dataclasses import dataclass
@dataclass
class Config:
    num_classes = 10
    batch_size = 64
    n_channel = 3
    patch_size = 4
    stride = 4
    img_size =  32
    num_patches = ((img_size - patch_size) // stride + 1) ** 2
    embedding_dimension = 128
    attention_heads = 8
    transformers_blocks = 4
    learning_rate = 0.001
    epochs = 50
    mlp_hidden = 4*embedding_dimension
    dataset_fraction = 1.0
    
