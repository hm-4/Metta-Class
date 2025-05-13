import sys
import os
import gc
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))


import torch

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor


from ZeroHelperFunctions.PrintingFormat import torchvision_datasets_printing_format
from ZeroHelperFunctions.CustomDataset import CustomDataset


def preprocessed_cifar10():
    # Load training data
    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    # Load testing data
    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    train_data.data = torch.tensor(train_data.data)
    train_data.targets = torch.tensor(train_data.targets)
    
    test_data.data = torch.tensor(test_data.data)
    test_data.targets = torch.tensor(test_data.targets)
    # print("\ncomparision: ")
    # print(type(train_data.data))
    # print(type(train_data.targets))
    
    # print(train_data.data.dtype)
    # print(train_data.targets.dtype)
    # print("comparision ends\n") # .permute(0, 3, 1, 2)
    # print(train_data.data[0])
    train_data = CustomDataset((train_data.data.permute(0, 3, 1, 2))/255, train_data.targets)
    test_data = CustomDataset((test_data.data.permute(0, 3, 1, 2))/255, test_data.targets)
    
    
    torchvision_datasets_printing_format(train_data, "train_data")
    # torchvision_datasets_printing_format(test_data, "test_data")
    # print(train_data.data[0])
    return train_data, test_data





def class_wise_preprocessed_cifar10():
    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    # Load testing data
    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    train_data.data = torch.tensor(train_data.data)
    train_data.targets = torch.tensor(train_data.targets)
    
    test_data.data = torch.tensor(test_data.data)
    test_data.targets = torch.tensor(test_data.targets)
    
    train_datas = []
    test_datas = []
    test_data_set_for_repository = CustomDataset(test_data.data.permute(0, 3, 1, 2)/255, test_data.targets)
    
    for cl in range(10):
        cl_train_indices = [i for i, label in enumerate(train_data.targets) if label == cl]
        cl_train_data = torch.stack([train_data.data[i] for i in cl_train_indices])
        cl_train_targets = torch.ones(len(cl_train_data), dtype=torch.int64)
        # print(cl_train_data.shape)
        cl_train_data = cl_train_data.squeeze(dim=1)
        # print(cl_train_data.shape)
        
        cl_test_indices = [i for i, label in enumerate(test_data.targets) if label == cl]
        cl_test_data = torch.stack([test_data.data[i] for i in cl_test_indices])
        cl_test_targets = torch.ones(len(cl_test_data), dtype=torch.int64)
        cl_test_data = cl_test_data.squeeze(dim=1)
        
        train_datas.append(CustomDataset(cl_train_data.permute(0, 3, 1, 2)/255, cl_train_targets))
        test_datas.append(CustomDataset(cl_test_data.permute(0, 3, 1, 2)/255, cl_test_targets))
    
    torchvision_datasets_printing_format(train_datas[0], "train_datas[i]")

    return train_datas, test_datas, test_data_set_for_repository
    


