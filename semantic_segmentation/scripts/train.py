# import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from unet_bn import U2NET_lite

from augment import get_transform
from dataset import Dataset


class UNET_LT(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.Net = U2NET_lite()
        self.Net.cuda(1)
        self.loss = nn.CrossEntropyLoss(ignore_index=255)
        self.lr = 3e-4
        self.accuracy = Accuracy(
            task="multiclass", num_classes=16, ignore_index=255
        ).cuda(1)

    def forward(self, x):
        pred = self.Net(x)
        return pred

    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        img, target = train_batch
        img = img.cuda(1)
        target = target.cuda(1)
        pred = self.Net(img)
        loss = self.loss(pred, torch.squeeze(target.long()))
        acc = self.accuracy(torch.argmax(pred, dim=1), torch.squeeze(target))
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        if batch_idx == 0:
            tensorboard = self.logger.experiment
            tensorboard.add_figure("predictions", show_results(img, pred, target))
        return loss

    def validation_step(self, val_batch, batch_idx):
        img, target = val_batch
        img = img.cuda(1)
        target = target.cuda(1)
        pred = self.Net(img)
        loss = self.loss(pred, torch.squeeze(target.long()))
        acc = self.accuracy(torch.argmax(pred, dim=1), torch.squeeze(target))
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        loss.backward()


def show_results(imgs, preds, gts):
    fig, ax = plt.subplots(frameon=False)
    fig.set_size_inches(6, 36)
    cmap = np.array(
        [
            (16, 64, 16),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 255, 255),
            (128, 64, 16),
            (128, 16, 64),
            (64, 16, 128),
            (16, 64, 128),
            (32, 64, 16),
            (32, 16, 64),
            (64, 16, 32),
            (16, 64, 32),
        ],
        dtype=np.uint8,
    )

    colormap255 = np.zeros((256, 3), dtype=np.uint8)
    colormap255[:16] = cmap

    images = []
    preds = torch.argmax(preds, dim=1).cpu().detach().numpy()
    gts = gts.cpu().detach().numpy()
    imgs = imgs * 0.5 + 0.5
    imgs = imgs.cpu().detach().permute(0, 2, 3, 1).numpy()
    preds = colormap255[preds]
    gts = colormap255[gts]
    for im, gt, pred in zip(imgs, gts, preds):
        images.append(np.concatenate((im, gt, pred), axis=1))
    res = np.concatenate(images, axis=0)
    plt.axis("off")
    plt.tight_layout()
    ax.imshow(res)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafolder", type=str, default="")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    train_ds = Dataset(args.datafolder, get_transform(train=True))
    val_ds = Dataset(args.datafolder, get_transform(), train=False)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=True, num_workers=8
    )

    model = UNET_LT()
    logger = TensorBoardLogger("tb_logs", name="u2net")
    callback = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    trainer = pl.Trainer(max_epochs=100, logger=logger, callbacks=[callback])
    trainer.fit(model, train_loader, val_loader)
