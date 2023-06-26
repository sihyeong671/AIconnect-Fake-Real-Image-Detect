from torchmetrics.classification import BinaryAccuracy, BinaryF1Score,  BinaryConfusionMatrix


def get_metric(metric_name: str):
  if metric_name == 'acc':
    return BinaryAccuracy()
  
  elif metric_name == 'f1':
    return BinaryF1Score()