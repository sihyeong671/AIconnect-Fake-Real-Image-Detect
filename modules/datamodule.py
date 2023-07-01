import os
import random
from glob import glob

import numpy as np
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl


class CustomDataset(Dataset):
  def __init__(self, img_paths, labels=None, transforms=None):
    self.img_paths = img_paths
    self.labels = labels
    self.transforms = transforms
    
  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self, index):
    path = self.img_paths[index]
    # img = cv2.imread(path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.open(path).convert("RGB")
    img = np.asarray(img)
    
    if self.transforms is not None:
      img = self.transforms(image=img)["image"]
    
    if self.labels is not None:
      label = self.labels[index]
      return img, label
    else:
      return img
    
    
class DataModule(pl.LightningDataModule):
  def __init__(self, config: dict):
    super().__init__()
    self.train_img_path = config["train_img_path"]
    self.test_img_path = config["test_img_path"]
    
    self.img_size = config["img_size"]
    self.val_size = config["val_size"]
    
    self.batch_size = config["batch_size"]
    self.num_workers = config["num_workers"]
    self.seed = config["seed"]
  
  def setup(self, stage=None):
    train_transform = A.Compose([
        A.Resize(self.img_size, self.img_size),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.OneOf([
          A.ChannelShuffle(p=0.2),
          A.ToGray(p=0.2),
          A.Blur(p=0.2)
        ]),
        A.Normalize(),
        ToTensorV2()
      ])
    
    test_transform = A.Compose([
      A.Resize(self.img_size, self.img_size),
      A.Normalize(),
      ToTensorV2()
    ])
    
    if stage == "train":
      # 터미널 기준 경로 or 절대경로
      real_img_paths = glob(f".\\{self.train_img_path}\\real_images\\*")
      fake_img_paths = glob(f".\\{self.train_img_path}\\fake_images\\*")
      labels = [0] * len(real_img_paths) + [1] * len(fake_img_paths)
      
      train_img_paths, val_img_paths, train_labels, val_labels = train_test_split(real_img_paths+fake_img_paths, labels, test_size=self.val_size, random_state=self.seed)
      
      self.train_dataset = CustomDataset(train_img_paths, train_labels, train_transform)
      self.val_dataset = CustomDataset(val_img_paths, val_labels, test_transform)

    elif stage == "test":
      test_img_paths = glob(f".\\{self.test_img_path}\\images\\*")
      self.test_dataset = CustomDataset(test_img_paths, transforms=test_transform)

  def train_dataloader(self):
    return DataLoader(
      dataset=self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=self.num_workers
    )
  
  def val_dataloader(self):
    return DataLoader(
      dataset=self.val_dataset,
      batch_size=2*self.batch_size,
      shuffle=False,
      num_workers=self.num_workers
    )
  
  def test_dataloader(self):
    return DataLoader(
      dataset=self.test_dataset,
      batch_size= 2*self.batch_size,
      shuffle=False,
      num_workers=self.num_workers
    )
  
  def predict_dataloader(self):
    return super().predict_dataloader()
  