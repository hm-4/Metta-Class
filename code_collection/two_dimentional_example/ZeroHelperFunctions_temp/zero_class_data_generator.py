import numpy as np

# Import PyTorch
import torch
from torch import nn

# Import torchvision 
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, ConcatDataset
# Import matplotlib for visualization
import matplotlib.pyplot as plt

        # self._zero_data_for_train = ZeroClassDataset(num_samples=n_train_zeros,
        #                                             tensor_shape=self.tensor_shape,
        #                                             label=label_for_zero,
        #                                             data=self.train_data.data)
        
class ZeroClassDataset(Dataset):
    def __init__(self, data, num_samples: int, data_shape: tuple, label: int = 3):
        """
        images: must be normalized to [0, 1]
        image_shape : (num_channels, width, height), ex: (1, 32, 32), (3, 32, 32)
        """
        
        self.num_samples = num_samples
        self.data_shape = data_shape
        self.label = label
        self.data, self.targets = self.generate_data(data=data)

    def generate_data(self, data, percentage_padding_around = 10):
        max_vector, min_vector = self.maxmin(data, percentage_padding_around)
        generator = torch.distributions.uniform.Uniform(min_vector, max_vector)
        data = generator.sample((self.num_samples,))
        # print(data)
        # print(data.shape)
        # print(type(data))
        # print(data[0].dtype)
        targets = torch.zeros((self.num_samples,), dtype=torch.int64) + self.label
        return data, targets
    
    
    def maxmin(self, data, percentage_padding_around):
        max_vector, min_vector = torch.max(data, dim=0)[0], torch.min(data, dim=0)[0]
        for i in range(len(min_vector)):
            width_of_dim_i = max_vector[i] - min_vector[i]
            padding_width = width_of_dim_i * percentage_padding_around / 100
            min_vector[i] -= padding_width
            max_vector[i] += padding_width
        return max_vector, min_vector
    
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]