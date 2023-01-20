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

from models import *
from augment import get_transform
from dataset import Dataset


class SuadSemseg(pl.LightningModule):
    def __init__(self, net, lr, num_classes, **kwargs):
        super().__init__()
        self.net = net
        self.loss = nn.CrossEntropyLoss(ignore_index=255)
        self.lr = lr
        self.accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, ignore_index=255
        )
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("U2NetModel")
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--num_classes", type=int, default=16)
        return parent_parser

    def forward(self, x):
        pred = self.net(x)
        return pred

    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), lr=self.lr)
        return optimizer

    def _step(self, train_batch, batch_idx, split):
        img, target = train_batch
        pred = self.net(img)
        loss = self.loss(pred, torch.squeeze(target.long()))
        acc = self.accuracy(torch.argmax(pred, dim=1), torch.squeeze(target))
        self.log(
            split + "/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            split + "/acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        if batch_idx == 0:
            self.logger.experiment.add_figure(
                split + "/predictions",
                show_results(img, pred, target),
                global_step=self.current_epoch,
            )
        return loss

    def training_step(self, train_batch, batch_idx):
        return self._step(train_batch, batch_idx, split="train")

    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            return self._step(val_batch, batch_idx, split="val")


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
    architectures = dict(u2net=U2NET_lite)
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafolder", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--net", type=str, default="U2NET_lite")
    parser = SuadSemseg.add_model_specific_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)

    net = getattr(architectures, args.net)()

    train_ds = Dataset(args.datafolder, get_transform(train=True))
    val_ds = Dataset(args.datafolder, get_transform(), train=False)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    dict_args = dict(dict_args)
    dict_args["net"] = net
    model = SuadSemseg(**dict_args)
    logger = TensorBoardLogger("tb_logs", name="u2net")
    callback = EarlyStopping(monitor="val/loss", mode="min", patience=5)
    trainer = pl.Trainer(
        max_epochs=100,
        logger=logger,
        callbacks=[callback],
        accelerator="gpu",
        devices=[1],
    )
    trainer.fit(model, train_loader, val_loader)
