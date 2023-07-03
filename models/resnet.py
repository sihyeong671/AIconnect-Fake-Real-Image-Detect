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
    self.model = timm.create_model(model_name="resnet101", pretrained=True)
    self.model.fc = nn.Sequential(
      nn.Linear(2048, 1)
    )
  
  def forward(self, x):
    x = self.model(x)
    return F.sigmoid(x)
    

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
    

class ChannelLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(ChannelLinear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        out_shape = [x.shape[0], x.shape[2], x.shape[3], self.out_features]
        x = x.permute(0,2,3,1).reshape(-1,self.in_features)
        x = x.matmul(self.weight.t())
        if self.bias is not None:
            x = x + self.bias[None,:]
        x = x.view(out_shape).permute(0,3,1,2)
        return x

    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetNoDown(nn.Module):

    def __init__(self, block, layers, num_classes=1, stride0=2):
        super().__init__()
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=stride0, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride0, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_features = 512 * block.expansion
        self.fc = ChannelLinear(self.num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
  
    def feature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = self.fc(x)

        return F.sigmoid(x)