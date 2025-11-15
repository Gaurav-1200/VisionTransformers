import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
import json
import os

from Config import Config
from ViT import ViT

from Utils import getDataLoaders,load_model


def check_accuracy(model,loader,device):
    num_correct = 0
    num_samples = 0
    model.eval()


    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores,_,_ = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

def TestLoop(model_path,test_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(model_path,device,test_config)

    _, test_loader = getDataLoaders()
    check_accuracy(model,test_loader,device)

if __name__ == "__main__":
    test_config = Config()
    model_path = os.path.join(os.getcwd(),"..","rich","vit_model_epoch50_patch4_stride4_emb128.pth")
    print(os.listdir(os.path.join(os.getcwd(),"..","rich")))
    TestLoop(model_path,test_config)
    
