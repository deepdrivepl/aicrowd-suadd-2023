import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from augment import get_transform
from dataset import Dataset
from unet_bn import U2NET_lite


def train(model, train_ds, val_ds, optimizer, epochs_no=50, patience=5):
    cel_loss = nn.CrossEntropyLoss()
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
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=True, num_workers=8)

    for epoch in tqdm(range(epochs_no)):
        model.train()
        epoch_train_loss = 0
        epoch_val_loss = 0
        for img, target in train_loader:
            img, target = img.to(device), target.to(device)
            pred = torch.squeeze(model(img)[0])
            print(pred.shape)
            print(target.shape)
            print(target.dtype)
            loss = cel_loss(pred, torch.squeeze(target))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss

        with torch.no_grad():
            model.eval()
            for img, target in val_loader:
                img, target = img.to(device), target.to(device)
                pred = torch.squeeze(model(img)[0])
                epoch_val_loss += cel_loss(pred, torch.squeeze(target))

        epoch_train_loss /= steps_train
        epoch_val_loss /= steps_val
        history["train_loss"].append(epoch_train_loss.cpu().detach().numpy())
        history["val_loss"].append(epoch_val_loss.cpu().detach().numpy())

        print("EPOCH: {}/{}".format(epoch + 1, epochs_no))
        print("Train loss: {:.6f}, Validation loss: {:.4f}".format(
              epoch_train_loss, epoch_val_loss))

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

    train_ds = Dataset(args.datafolder, get_transform(True))
    val_ds = Dataset(args.datafolder, get_transform(), train=False)

    model = train(model, train_ds, val_ds, optimizer)
