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
print("inside getting data", flush=True)
    
def preprocessed_mnist():
    print("inside getting data3", flush=True)
    train_data = datasets.MNIST(
        root="data", # where to download data to?
        train=True, # get training data
        download=True, # download data if it doesn't exist on disk
        transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
        target_transform=None # you can transform labels as well
    )

    # Setup testing data
    test_data = datasets.MNIST(
        root="data",
        train=False, # get test data
        download=True,
        transform=ToTensor()
    )
    # print("\ncomparision: ")
    # print(type(train_data.data))
    # print(type(train_data.targets))
    
    # print(train_data.data.dtype)
    # print(train_data.targets.dtype)
    # print("comparision ends\n")
    print("inside getting data2", flush=True)
    
    train_data = CustomDataset(train_data.data/255, train_data.targets)
    test_data = CustomDataset(test_data.data/255, test_data.targets)
    torchvision_datasets_printing_format(train_data, "train_data")
    return train_data, test_data


###############################################################################
def preprocessed_fashion_mnist():
    # Setup training data
    train_data = datasets.FashionMNIST(
        root="data",  # where to download data to
        train=True,  # get training data
        download=True,  # download data if it doesn't exist on disk
        transform=ToTensor(),  # images come as PIL format, we want to turn into Torch tensors
        target_transform=None  # you can transform labels as well
    )

    # Setup testing data
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,  # get test data
        download=True,
        transform=ToTensor()
    )
    # print("\ncomparision: ")
    # print(type(train_data.data))
    # print(type(train_data.targets))
    
    # print(train_data.data.dtype)
    # print(train_data.targets.dtype)
    # print("comparision ends\n")
    
    
    train_data = CustomDataset(train_data.data/255, train_data.targets)
    test_data = CustomDataset(test_data.data/255, test_data.targets)
    torchvision_datasets_printing_format(train_data, "train_data")
    return train_data, test_data


###############################################################################
def preprocessed_k_mnist():
    # Setup training data
    train_data = datasets.KMNIST(
        root="data",  # where to download data to
        train=True,  # get training data
        download=True,  # download data if it doesn't exist on disk
        transform=ToTensor(),  # images come as PIL format, we want to turn into Torch tensors
        target_transform=None  # you can transform labels as well
    )

    # Setup testing data
    test_data = datasets.KMNIST(
        root="data",
        train=False,  # get test data
        download=True,
        transform=ToTensor()
    )
    # print("\ncomparision: ")
    # print(type(train_data.data))
    # print(type(train_data.targets))
    
    # print(train_data.data.dtype)
    # print(train_data.targets.dtype)
    # print("comparision ends\n")
    
    
    train_data = CustomDataset(train_data.data/255, train_data.targets)
    test_data = CustomDataset(test_data.data/255, test_data.targets)
    torchvision_datasets_printing_format(train_data, "train_data")
    return train_data, test_data



###################################################
###################################################


def class_wise_preprocessed_mnist():
    train_data = datasets.MNIST(
        root="data", # where to download data to?
        train=True, # get training data
        download=True, # download data if it doesn't exist on disk
        transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
        target_transform=None # you can transform labels as well
    )

    # Setup testing data
    test_data = datasets.MNIST(
        root="data",
        train=False, # get test data
        download=True,
        transform=ToTensor()
    )
    
    train_datas = []
    test_datas = []
    test_data_set_for_repository = CustomDataset(test_data.data/255, test_data.targets)
    
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
        
        train_datas.append(CustomDataset(cl_train_data/255, cl_train_targets))
        test_datas.append(CustomDataset(cl_test_data/255, cl_test_targets))
    
    torchvision_datasets_printing_format(train_datas[0], "train_datas[i]")

    return train_datas, test_datas, test_data_set_for_repository
    
def preprocessed_mnist_combined_classes():
    print("inside getting data3", flush=True)
    train_data = datasets.MNIST(
        root="data", # where to download data to?
        train=True, # get training data
        download=True, # download data if it doesn't exist on disk
        transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
        target_transform=None # you can transform labels as well
    )

    # Setup testing data
    test_data = datasets.MNIST(
        root="data",
        train=False, # get test data
        download=True,
        transform=ToTensor()
    )
    # print("\ncomparision: ")
    # print(type(train_data.data))
    # print(type(train_data.targets))
    
    # print(train_data.data.dtype)
    # print(train_data.targets.dtype)
    # print("comparision ends\n")
    print("inside getting data2", flush=True)
    combined_train_targets = torch.ones_like(train_data.targets)
    combined_test_targets = torch.ones_like(test_data.targets)
    train_data = CustomDataset(train_data.data/255.0, combined_train_targets)
    test_data = CustomDataset(test_data.data/255.0, combined_test_targets)
    torchvision_datasets_printing_format(train_data, "train_data")
    return train_data, test_data






