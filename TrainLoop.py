import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
import json

from Config import Config
from ViT import ViT

config = Config()

def get_transformations(isAugmenting=False):
    if isAugmenting:
        transform_train = transforms.Compose([   #TAKEN FROM INTERNET

        # 1. Random Crop: Crops the image randomly at a size of 32 with 4 pixels of padding on all borders [1, 4].
        transforms.RandomCrop(size=32, padding=4),

        # 2. Random Horizontal Flip: Flips the image horizontally [2].
        transforms.RandomHorizontalFlip(),

        # 3. Color Jitter: Randomly changes brightness, contrast, saturation, and hue (all set to 0.2) [2].
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
        # no augmentation, just standard prep
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])  
        return transform_train,transform_val
    else:
        transformation_operation = transforms.Compose([transforms.ToTensor()])
    return transformation_operation,transformation_operation

def getDataLoaders(batch_size,dataFraction=1.0,isAugmenting=False):
    train_transform_op, val_transform_op = get_transformations(isAugmenting)
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=train_transform_op, download=True)
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=val_transform_op, download=True)

    if dataFraction == 1.0:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,shuffle=True)
    
    else:
        subset_size = int(len(train_dataset) * dataFraction)
        indices = np.random.choice(len(train_dataset), subset_size, replace=False)
        train_subset = Subset(train_dataset, indices)
        train_loader = DataLoader(dataset=train_subset, batch_size=batch_size,shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,shuffle=True)

    return train_loader,val_loader

def saveTrainingHistory(train_accuracies,train_losses,config):
    training_data = {
    'train_losses': train_losses,
    'train_accuracies': train_accuracies
    }

    data_save_path = f'training_history_epoch{config.epochs}_patch{config.patch_size}_stride{config.stride}_emb{config.embedding_dimension}.json'
    with open(data_save_path, 'w') as f:
        json.dump(training_data, f)

    print(f"Training history saved to {data_save_path}")

def saveTrainedModel(model,config):
    model_save_path = f'vit_model_epoch{config.epochs}_patch{config.patch_size}_stride{config.patch_size}_emb{config.embedding_dimension}.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

def train(config):

    train_loader,_ = getDataLoaders(config.batch_size,config.dataset_fraction,config.isAugmenting)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device Used :", device)


    model = ViT().to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracies = []

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        correct_epoch = 0
        total_epoch =0 
        print(f"Epoch : {epoch}/{config.epochs}")
        correct = 0
        for batch_idx,(images,labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs,attn_wts,intermediate_cls = model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct =(preds == labels).sum().item()
            accuraccy =correct*100.00/len(labels)
            correct_epoch += correct
            total_epoch += correct_epoch

            if(batch_idx %100==0):
                print(f"Batch {batch_idx}, Loss= {loss:.3f} , accuracy:{accuracy:.3f}")

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100.0 * correct_epoch/total_epoch

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f"Epoch = {epoch+1}, Total Loss = {epoch_loss:.3f} Acc ={epoch_acc:.3f}")

        saveTrainingHistory(train_accuracies,train_losses,config)
        saveTrainedModel(model,config)



