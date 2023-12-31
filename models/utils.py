from models.resnet import ResNet50, ResNet101, ResNetNoDown, Bottleneck
from models.van import VAN_base

def get_model(model_name: str):
  if model_name == "resnet50":
    return ResNet50()
  elif model_name == "resnet101":
    return ResNet101()
  elif model_name == "resnet_nodown":
    return ResNetNoDown(Bottleneck, [3, 4, 6, 3], num_classes=1, stride0=1)
  elif model_name == "van":
    return VAN_base()
  else:
    ValueError(f"{model_name} is not found")

if __name__ == "__main__":
  pass