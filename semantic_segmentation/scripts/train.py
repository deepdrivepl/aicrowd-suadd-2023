import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchmetrics import Accuracy
import numpy as np

from augment import get_transform
from dataset import Dataset
from unet_bn import U2NET_lite


def train(model, train_ds, val_ds, optimizer, epochs_no=100, patience=5):
    cel_loss = nn.CrossEntropyLoss(ignore_index=255)
    history = {"train_loss": [], "val_loss": []}
    cooldown = 0
    batch_size = 16
    steps_train = len(train_ds)/batch_size
    steps_val = len(val_ds)/batch_size
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)
    accuracy = Accuracy(task="multiclass", num_classes=16,
                       ignore_index=255).to(device)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=True, num_workers=8)

    for epoch in tqdm(range(epochs_no)):
        model.train()
        epoch_train_loss = 0
        epoch_train_acc = 0
        epoch_val_loss = 0
        epoch_val_acc = 0
        for img, target in train_loader:
            img, target = img.to(device), target.to(device)
            pred = model(img)
            loss = cel_loss(pred, torch.squeeze(target.long()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.cpu().detach().numpy()
            epoch_train_acc += accuracy(torch.argmax(pred, dim=1),
                                        torch.squeeze(target)
                                       ).cpu().detach().numpy()
        with torch.no_grad():
            model.eval()
            for img, target in val_loader:
                img, target = img.to(device), target.to(device)
                pred = model(img)
                epoch_val_loss += cel_loss(pred,
                                           torch.squeeze(target.long())
                                          ).cpu().detach().numpy()
                epoch_val_acc += accuracy(torch.argmax(pred, dim=1),
                                          torch.squeeze(target)
                                         ).cpu().detach().numpy()

        epoch_train_loss /= steps_train
        epoch_train_acc /= steps_train
        epoch_val_loss /= steps_val
        epoch_val_acc /= steps_val
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

        print("EPOCH: {}/{}".format(epoch + 1, epochs_no))
        print("Train loss: {:.6f}, Validation loss: {:.4f}".format(
              epoch_train_loss, epoch_val_loss))
        print("Train accuracy: {:.6f}, Validation accuracy: {:.6f}".format(
              epoch_train_acc, epoch_val_acc))

        cur_loss = history["val_loss"][epoch]
        if epoch != 0 and cur_loss >= history["val_loss"][epoch-1]:
            cooldown += 1
            if cooldown == patience:
                break
        else:
            cooldown = 0
            torch.save(model.state_dict(), 'model_weights.pth')
    model.load_state_dict(torch.load('model_weights.pth'))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafolder', type=str, default="")
    args = parser.parse_args()
    model = U2NET_lite()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    train_ds = Dataset(args.datafolder, get_transform())
    val_ds = Dataset(args.datafolder, get_transform(), train=False)

    model = train(model, train_ds, val_ds, optimizer)
