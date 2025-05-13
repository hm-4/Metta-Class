import torch

def accuracy_fn(true, pred):
    correct = torch.eq(true, pred).sum().item()
    acc = (correct / len(pred)) * 100
    return acc