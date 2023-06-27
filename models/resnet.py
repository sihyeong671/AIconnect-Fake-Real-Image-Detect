import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet50(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = timm.create_model(model_name="resnet50", pretrained=True)
    self.model.fc = nn.Sequential(
      nn.Linear(2054, 512),
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
    

class ResNet101_GRAM(nn.Module):
  def __init__(self):
    super().__init__()
    self.gram_list = []
    model = timm.create_model(model_name="resnet101", pretrained=True)
    self.conv1 = model.conv1
    self.bn1 = model.bn1
    self.act1 = model.act1
    self.maxpool = model.maxpool
    self.layer1 = model.layer1
    self.layer2 = model.layer2
    self.layer3 = model.layer3
    self.layer4 = model.layer4
    self.gp = model.global_pool
    
    self.gram = GramMatrix()
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.classifier = nn.Sequential(
      nn.Linear(2240, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(),
      nn.Dropout1d(0.5),
      nn.Linear(512, 1),
      nn.Sigmoid()
    )
    
    self.gram_block_0_0 = nn.Sequential(
      nn.Conv2d(3, 32, 3, 1, 1, bias=False),
      nn.BatchNorm2d(32), 
      nn.ReLU(inplace=True),
    )
    self.gram_block_0_1 = nn.Sequential(
      nn.Conv2d(1, 16, 3, 2, 1, bias=False),
      nn.BatchNorm2d(16), 
      nn.ReLU(),
      nn.Conv2d(16, 32, 3, 2, 1, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU()
    )
    
    self.gram_block_1_0 = nn.Sequential(
      nn.Conv2d(64, 32, 3, 1, 1, bias=False),
      nn.BatchNorm2d(32), 
      nn.ReLU(inplace=True),
    )
    self.gram_block_1_1 = nn.Sequential(
      nn.Conv2d(1, 16, 3, 2, 1, bias=False),
      nn.BatchNorm2d(16), 
      nn.ReLU(),
      nn.Conv2d(16, 32, 3, 2, 1, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU()
    )
    
    self.gram_block_2_0 = nn.Sequential(
      nn.Conv2d(64, 32, 3, 1, 1, bias=False),
      nn.BatchNorm2d(32), 
      nn.ReLU(inplace=True),
    )
    self.gram_block_2_1 = nn.Sequential(
      nn.Conv2d(1, 16, 3, 2, 1, bias=False),
      nn.BatchNorm2d(16), 
      nn.ReLU(),
      nn.Conv2d(16, 32, 3, 2, 1, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU()
    )
    
    self.gram_block_3_0 = nn.Sequential(
      nn.Conv2d(256, 32, 3, 1, 1, bias=False),
      nn.BatchNorm2d(32), 
      nn.ReLU(inplace=True),
    )
    self.gram_block_3_1 = nn.Sequential(
      nn.Conv2d(1, 16, 3, 2, 1, bias=False),
      nn.BatchNorm2d(16), 
      nn.ReLU(),
      nn.Conv2d(16, 32, 3, 2, 1, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU()
    )
    
    self.gram_block_4_0 = nn.Sequential(
      nn.Conv2d(512, 32, 3, 1, 1, bias=False),
      nn.BatchNorm2d(32), 
      nn.ReLU(inplace=True),
    )
    self.gram_block_4_1 = nn.Sequential(
      nn.Conv2d(1, 16, 3, 2, 1, bias=False),
      nn.BatchNorm2d(16), 
      nn.ReLU(),
      nn.Conv2d(16, 32, 3, 2, 1, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU()
    )
    
    self.gram_block_5_0 = nn.Sequential(
      nn.Conv2d(1024, 32, 3, 1, 1, bias=False),
      nn.BatchNorm2d(32), 
      nn.ReLU(inplace=True),
    )
    self.gram_block_5_1 = nn.Sequential(
      nn.Conv2d(1, 16, 3, 2, 1, bias=False),
      nn.BatchNorm2d(16), 
      nn.ReLU(),
      nn.Conv2d(16, 32, 3, 2, 1, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU()
    )
    
    
    # for m in self.modules():
    #   if isinstance(m, nn.Conv2d):
    #     nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    #   elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
    #     nn.init.constant_(m.weight, 1)
    #     nn.init.constant_(m.bias, 0)
    
  def forward(self, x):
    x0 = x
    x = self.conv1(x0)
    x = self.bn1(x)
    x1 = self.act1(x)
    x2 = self.maxpool(x1)
    
    x3 = self.layer1(x2)
    x4 = self.layer2(x3)
    x5 = self.layer3(x4)
    x = self.layer4(x5)
    x = self.gp(x)
    
    g0 = self.gram_block_0_0(x0)
    g0 = self.gram(g0)
    g0 = self.gram_block_0_1(g0)
    g0 = self.avgpool(g0)
    g0 = g0.view(g0.size(0), -1)
    
    g1 = self.gram_block_1_0(x1)
    g1 = self.gram(g1)
    g1 = self.gram_block_1_1(g1)
    g1 = self.avgpool(g1)
    g1 = g1.view(g1.size(0), -1)
    
    g2 = self.gram_block_2_0(x2)
    g2 = self.gram(g2)
    g2 = self.gram_block_2_1(g2)
    g2 = self.avgpool(g2)
    g2 = g2.view(g2.size(0), -1)
    
    g3 = self.gram_block_3_0(x3)
    g3 = self.gram(g3)
    g3 = self.gram_block_3_1(g3)
    g3 = self.avgpool(g3)
    g3 = g3.view(g3.size(0), -1)
    
    g4 = self.gram_block_4_0(x4)
    g4 = self.gram(g4)
    g4 = self.gram_block_4_1(g4)
    g4 = self.avgpool(g4)
    g4 = g4.view(g4.size(0), -1)
    
    g5 = self.gram_block_5_0(x5)
    g5 = self.gram(g5)
    g5 = self.gram_block_5_1(g5)
    g5 = self.avgpool(g5)
    g5 = g5.view(g5.size(0), -1)
    
    x = torch.cat((x, g0, g1, g2, g3, g4, g5), 1)
    x = self.classifier(x)
    
    return x
  

class GramMatrix(nn.Module):
  def __init__(self):
    super().__init__()
  
  def forward(self, x):
    b, c, h, w = x.size()
    
    features = x.view(b, c, h * w)
    a = features.transpose(1, 2)
    G = torch.bmm(features, a)
    G = G.unsqueeze(1)
    return G.div(c * h * w)
    
    
