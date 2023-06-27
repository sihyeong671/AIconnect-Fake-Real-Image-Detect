import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, VanForImageClassification


class VAN_base(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = VanForImageClassification.from_pretrained("Visual-Attention-Network/van-base")
    self.model.classifier = nn.Sequential(
      nn.Linear(512, 128),
      nn.BatchNorm1d(128),
      nn.ReLU(),
      nn.Dropout1d(0.3),
      nn.Linear(128, 1)
    )
  def forward(self, x):
    x = self.model(x)
    return F.sigmoid(x.logits)