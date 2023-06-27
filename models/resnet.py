import timm
import torch.nn as nn
import torch.nn.functional as F


class ResNet50(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = timm.create_model(model_name="resnet50", pretrained=True)
    self.model.fc = nn.Sequential(
      nn.Linear(2048, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(),
      nn.Linear(512, 1)
    )
    
  def forward(self, x):
    x = self.model(x)
    return F.sigmoid(x)
  

class ResNet101(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = timm.create_model(model_name="resnet101", pretrained=True)
    self.model.fc = nn.Sequential(
      nn.Linear(2048, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(),
      nn.Linear(512, 1)
    )
    
  def forward(self, x):
    x = self.model(x)
    return F.sigmoid(x)