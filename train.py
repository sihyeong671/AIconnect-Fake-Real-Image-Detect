from modules.utils import load_yaml, seed_everything
from modules.datamodule import DataModule
from modules.trainer import LightningModule


import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


def train(config):
  
  data_module = DataModule(config=config["DATAMODULE"])
  data_module.setup(stage="train")
  
  
  model = LightningModule(config=config["TRAINER"])
  ckpt_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_f1",
    mode="max",
    dirpath=".\\ckpt",
    filename="resnet101_v1_{epoch}"
  )
  early_stop_callback = EarlyStopping(
    monitor="val_f1",
    min_delta=0.00,
    patience=5,
    verbose=False,
    mode="max")
  
  wandb_logger = WandbLogger(
    entity="bsh",
    name="resnet101_v1",
    project="aiconnect_fake_real_detect"
  )
  
  trainer = pl.Trainer(
    # accumulate_grad_batches=2,
    max_epochs=config["TRAINER"]["n_epochs"],
    accelerator="gpu",
    callbacks=[ckpt_callback, early_stop_callback],
    logger=wandb_logger
  )
  
  trainer.fit(model, data_module)
  
def test(config):
  data_module = DataModule(config=config["DATAMODULE"])
  data_module.setup(stage="test")
  
  model = LightningModule.load_from_checkpoint(".\\ckpt\\resnet101_v1_epoch=19.ckpt", config=config["TRAINER"])
  
  trainer = pl.Trainer(
    accelerator="gpu",
  )

  trainer.test(
    model,
    data_module
  )

if __name__ == "__main__":
  PROJECT_DIR = os.path.dirname(__file__)
  config_path = os.path.join(PROJECT_DIR, "config", "train_config.yaml")
  config = load_yaml(config_path)

  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, default="train")
  args = parser.parse_args()
  
  plt.rcParams["font.family"] = "MalGun Gothic"
  
  seed_everything(seed=config["seed"])

  if args.mode == "train":
    train(config)
  elif args.mode == "test":
    test(config)