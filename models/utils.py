from models.resnet import ResNet50

def get_model(model_name: str):
  if model_name == "resnet50":
    return ResNet50()
  else:
    ValueError(f"{model_name} is not found")

if __name__ == "__main__":
  pass