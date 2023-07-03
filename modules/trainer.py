import os

from models.utils import get_model
from modules.optimizers import get_optimizer, get_scheduler
from modules.losses import get_loss
from modules.metrics import get_metric

import seaborn as sns
import pandas as pd
import numpy as np
import wandb
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl
from torchmetrics.classification import BinaryConfusionMatrix


class LightningModule(pl.LightningModule):
  def __init__(self, config: dict):
    super().__init__()
    self.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    self.model = get_model(config["model_name"])
    self.model.load_state_dict(torch.load(f".\\models\\{config['weight_file_name']}", map_location=torch.device('cuda'))['model'])
    optimizer = get_optimizer(config["optimizer"])
    self.optimizer = optimizer(params=self.parameters(), lr=config["lr"])
    scheduler = get_scheduler(config["scheduler"])
    self.scheduler = scheduler(
      self.optimizer,
      mode="max",
      factor=0.5,
    )
    self.loss_fn = get_loss(config["loss_fn"]).to(self.DEVICE)
    self.metrics = {metric_name: get_metric(metric_name).to(self.DEVICE) for metric_name in config["metric"]}
    self.confusion_matrix = BinaryConfusionMatrix()
    self.test_step_outputs = []
    
  def forward(self, x):
    x = self.model(x)
    return x
  
  def configure_optimizers(self):
    return {
      "optimizer": self.optimizer,
      "lr_scheduler": {
        "scheduler": self.scheduler,
        "monitor" : "val_f1"
      }
    }
    
  def training_step(self, batch, batch_idx):
    metrics = self._common_step(batch, batch_idx, mode="train")
    self.log_dict(metrics)
    
    return metrics["train_loss"]

  def on_train_epoch_end(self):
    for metric_fn in self.metrics.values():
      metric_fn.reset()
  
  def validation_step(self, batch, batch_idx):
    metrics = self._common_step(batch, batch_idx, mode="val")

  def on_validation_epoch_end(self):
    
    # confusion matrix 저장
    cmf = self.confusion_matrix.compute().cpu().numpy()

    f, ax = plt.subplots(figsize=(10, 10))
    df_cmf = pd.DataFrame(
      cmf,
      index=["True", "False"],
      columns=["Positive", "Negative"]
    )
    sns.heatmap(df_cmf, annot=True, ax=ax)

    metrics = {f"val_{metric_name}": metric_fn.compute().cpu() for metric_name, metric_fn in self.metrics.items()}
    self.logger.experiment.log({
      "confusion_matrix": wandb.Image(f)
    })
    self.log_dict(metrics)
    
    # reset function
    for metric_fn in self.metrics.values():
      metric_fn.reset()
    self.confusion_matrix.reset()
    
    # memory clear
    plt.close()
  
  def _common_step(self, batch, batch_idx, mode="train"):
    imgs, labels = batch
    # type casting to float
    labels = labels.type(torch.cuda.FloatTensor)
    preds = self(imgs)
    preds = preds.view(preds.size(0))
    loss = self.loss_fn(preds, labels)
    
    metrics = {f"{mode}_{metric_name}": metric_fn(preds, labels) for metric_name, metric_fn in self.metrics.items()}
    metrics[f"{mode}_loss"] = loss
    
    if mode == "train":
      pass
    elif mode == "val":
      self.confusion_matrix.update(preds, labels)
      
    return metrics
  def test_step(self, batch, batch_idx):
    imgs = batch
    probs = self(imgs)
    preds = probs.squeeze(-1)
    preds = preds.detach().cpu().numpy()
    preds = np.where(preds > 0.5, 1, 0)
    self.test_step_outputs += preds.tolist()
  
  def on_test_epoch_end(self):
    os.makedirs(".\\csv", exist_ok=True)
    submit = pd.read_csv(".\\test\\sample_submission.csv")
    submit["answer"] = self.test_step_outputs
    submit.to_csv(f".\\csv\\resnet_nodown_v0.csv", index=False)
    
  