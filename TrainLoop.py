import torch
import torchvision
import torch.nn as nn

from Config import Config
from ViT import ViT

from Utils import getDataLoaders,saveTrainedModel,saveTrainingHistory

config = Config()


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



