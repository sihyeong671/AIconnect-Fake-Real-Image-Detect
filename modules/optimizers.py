import torch.optim as optim

def get_optimizer(optimizer_name: str):
    if optimizer_name == 'sgd':
        return optim.SGD
    elif optimizer_name == 'adam':
        return optim.Adam
    elif optimizer_name == 'adamw':
        return optim.AdamW
    else:
        raise ValueError('Not a valid optimizer')
    

def get_scheduler(scheduler_name: str):
    if scheduler_name == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau