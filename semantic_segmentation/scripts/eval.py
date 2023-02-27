import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import torch

from augment import get_transform
from dataset import Dataset
import models
from pl_model import SuadSemseg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafolder", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    dict_args = vars(args)
    val_ds = Dataset(args.datafolder, get_transform(size=(512, 512), train=False))  # fixed by KM
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    dict_args = dict(dict_args)
    logger = TensorBoardLogger("tb_logs", name="xunet", default_hp_metric=False)
    callback = EarlyStopping(monitor="val/loss", mode="min", patience=10)
    trainer = pl.Trainer(
        max_epochs=200,
        logger=logger,
        callbacks=[callback],
        accelerator="gpu",
        devices=[1],
        accumulate_grad_batches=32,
    )
    model = SuadSemseg.load_from_checkpoint("tb_logs/xunet/version_16/checkpoints/epoch=199-step=18000.ckpt")
    trainer.test(model, val_loader)
