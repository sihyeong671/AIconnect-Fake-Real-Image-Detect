from torch.nn import functional as F
import torch
import torch.nn as nn

def get_loss(loss_name: str):
    
    if loss_name == 'ce':
      return nn.CrossEntropyLoss()

    elif loss_name == 'bce':
      return nn.BCELoss()

    else:
      raise ValueError(f"{loss_name} is not found")
    
