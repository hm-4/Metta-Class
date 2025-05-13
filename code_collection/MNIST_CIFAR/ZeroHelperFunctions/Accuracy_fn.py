import torch

def accuracy_fn(true, pred):
    # print("what torch.eq:\n", torch.eq(true, pred))
    # print("len(pred) :", len(pred))
    correct = torch.eq(true, pred).sum().item()
    acc = (correct / len(pred)) * 100
    return acc

def repository_accuracy_fn(true, pred):
    comparisions = true == pred
    row_matches = comparisions.all(dim=1)
    correct = row_matches.sum().item()
    acc = (correct/len(pred)) * 100
    return acc
