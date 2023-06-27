from models.resnet import ResNet50, ResNet101

def get_model(model_name: str):
  if model_name == "resnet50":
    return ResNet50()
  elif model_name == "resnet101":
    return ResNet101()
  else:
    ValueError(f"{model_name} is not found")

if __name__ == "__main__":
  pass