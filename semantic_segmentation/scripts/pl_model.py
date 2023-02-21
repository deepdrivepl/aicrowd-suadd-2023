import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy

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
