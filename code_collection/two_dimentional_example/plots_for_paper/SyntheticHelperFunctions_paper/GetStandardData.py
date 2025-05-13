import sys
import os
import gc
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from ZeroHelperFunctions_paper.CustomDataset import CustomDataset
from ZeroHelperFunctions_paper.PrintingFormat import printing_format


def preprocessed_synthetic_data(n_train_samples,
                                n_test_samples,
                                n_features,
                                n_classes,
                                random_state=42):
    X, y = make_blobs(n_samples=n_train_samples+n_test_samples,
                    n_features=n_features,
                    centers=n_classes,
                    random_state=random_state)
    X_train, y_train = X[:n_train_samples], y[:n_train_samples]
    X_test, y_test = X[n_train_samples:], y[n_train_samples:]
    
    # X_test, y_test = make_blobs(n_samples=n_test_samples,
    #                 n_features=n_features,
    #                 centers=n_classes,
    #                 random_state=random_state)
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.int64)
    # Plotting with different shapes for each class
    # Plotting with the same color for all points but different shapes for classes
    
    x_test_for_plotting = X_test
    y_test_for_plotting = y_test
    markers = ['P', '^', 's', 'D', 'X', '*']  # Add more shapes if needed
    for cls in range(n_classes):
        plt.scatter(
            X_test[y_test == cls, 0],  # X-coordinate for class `cls`
            X_test[y_test == cls, 1],  # Y-coordinate for class `cls`
            color='none',  # Uniform color for all points
            label=f'Class {cls}', 
            marker=markers[cls % len(markers)], 
            edgecolor='black',  # Optional: Add edge color for better visibility
            alpha=0.7,  # Optional: Adjust transparency for better visualization
            linewidths=1,
            # s=20,
        )
    # plt.xticks([])
    # plt.yticks([])
    plt.legend()
    plt.show()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    train_data = CustomDataset(X_train, y_train)
    test_data = CustomDataset(X_test, y_test)
    printing_format(train_data, "train_data")
    
    return train_data, test_data, x_test_for_plotting, y_test_for_plotting
    

def class_wise_preprocessed_synthetic_data(n_train_samples,
                                        n_test_samples,
                                        n_features,
                                        n_classes,
                                        random_state=42):
    X, y = make_blobs(n_samples=n_train_samples+n_test_samples,
                    n_features=n_features,
                    centers=n_classes,
                    random_state=random_state)
    X_train, y_train = X[:n_train_samples], y[:n_train_samples]
    X_test, y_test = X[n_train_samples:], y[n_train_samples:]
    
    # X_test, y_test = make_blobs(n_samples=n_test_samples,
    #                 n_features=n_features,
    #                 centers=n_classes,
    #                 random_state=random_state)
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.int64)
    plt.scatter(x=X_train[:, 0], y=X_train[:, 1], c=y_train, cmap=plt.cm.RdYlBu)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    test_data_set_for_repository = CustomDataset(X_test, y_test)
    train_data = CustomDataset(X_train, y_train)
    test_data = CustomDataset(X_test, y_test)
    train_datas = []
    test_datas = []
    for cl in range(n_classes):
        cl_train_indices = [i for i, label in enumerate(train_data.targets) if label == cl]
        cl_train_data = torch.stack([train_data.data[i] for i in cl_train_indices])
        cl_train_targets = torch.ones(len(cl_train_data), dtype=torch.int64)
        
        cl_test_indices = [i for i, label in enumerate(test_data.targets) if label == cl]
        cl_test_data = torch.stack([test_data.data[i] for i in cl_test_indices])
        cl_test_targets = torch.ones(len(cl_test_data), dtype=torch.int64)
        cl_test_data = cl_test_data.squeeze(dim=1)
        
        train_datas.append(CustomDataset(cl_train_data, cl_train_targets))
        test_datas.append(CustomDataset(cl_test_data, cl_test_targets))
    printing_format(train_datas[0], "train_datas[0]")
    return train_datas, test_datas, test_data_set_for_repository
    