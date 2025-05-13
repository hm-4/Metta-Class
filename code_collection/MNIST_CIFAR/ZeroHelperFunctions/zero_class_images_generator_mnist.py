import numpy as np
from ZeroHelperFunctions.show_image import show_color_images_in_grid

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

class ZeroClassDataset(Dataset):
    def __init__(self, num_samples: int, image_shape: tuple, label: int = 0):
        """
        images: must be normalized to [0, 1]
        image_shape : (num_channels, width, height), ex: (1, 32, 32), (3, 32, 32)
        
        """
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.label = label
        self.data, self.targets = self.generate_data()

    def generate_data(self):
        # print(images[0])
        max_image = np.ones(self.image_shape, dtype=np.float32)
        min_image = np.zeros(self.image_shape, dtype=np.float32)
        data = np.zeros((self.num_samples, *self.image_shape), dtype=np.float32)
        for c in range(self.image_shape[0]):
            for i in range(self.image_shape[1]):
                for j in range(self.image_shape[2]):
                    # print("at this ", c, i, j)
                    # print(max_image[c, i, j], min_image[c, i, j])
                    # data[:, c, i, j] = np.random.randint(min_image[c, i, j], max_image[c, i, j]+1, self.num_samples)
                    data[:, c, i, j] = np.random.uniform(min_image[c, i, j], max_image[c, i, j], self.num_samples)
        # data = data/255
        data = np.squeeze(data)
        targets = np.zeros(self.num_samples, dtype=np.int64) + self.label
        return torch.tensor(data, dtype=torch.float32), torch.tensor(targets)
    
    
    def maxmin(self, images, percentage_padding_around):
        max_image = np.ones(self.image_shape, dtype=np.float32)
        min_image = np.zeros(self.image_shape, dtype=np.float32)
        return max_image, min_image
        # print(max_image.shape, min_image.shape)
        # plt.figure(figsize=(6, 3))
        # image1 = max_image.squeeze()
        # image2 = min_image.squeeze()
        # plt.subplot(1, 2, 1)
        # plt.imshow(image1, cmap='gray')
        # plt.title(f'Sample {1}')

        # plt.subplot(1, 2, 2)
        # plt.imshow(image2, cmap='gray')
        # plt.title(f'Sample {2}')
        # plt.show()
        
        # if len(images.shape) == 3:
        #     # print("unsqueeze: ")
        #     # print(images.shape)
        #     images = images.unsqueeze(1)
        #     # print(images.shape)
        # # print(images.shape)
        # for c in range(self.image_shape[0]):
        #     # print("current c ", c)
        #     for i in range(self.image_shape[1]):
        #         for j in range(self.image_shape[2]):
        #             # print(i, j)
        #             max_image[c, i, j] = images[:, c, i, j].max().item()
        #             min_image[c, i, j] = images[:, c, i, j].min().item()
        # # print(max_image, min_image)
        # # print(self.image_shape, self.image_shape[0])
        # if self.image_shape[0] == 1:      
        #     plt.figure(figsize=(6, 3))
        #     image1 = max_image.squeeze()
        #     image2 = min_image.squeeze()
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(image1, cmap='gray', vmin=0, vmax=1)
        #     plt.title(f'Sample {1}')

        #     plt.subplot(1, 2, 2)
        #     plt.imshow(image2, cmap='gray', vmin=0, vmax=1)
        #     plt.title(f'Sample {2}')
        #     plt.show()
        #     # print(max_image)
        # else:
        #     show_color_images_in_grid(np.stack((max_image, min_image), axis=0), 1, 2)
            
        
        # ### padding
        # for c in range(self.image_shape[0]):
        #     for i in range(self.image_shape[1]):
        #         for j in range(self.image_shape[2]):
        #             # print(max_image[c, i, j], min_image[c, i, j])
        #             width = max_image[c, i, j] - min_image[c, i, j] + 1
        #             padding_width = width * percentage_padding_around / 100
        #             # print("paddingwidth: ", padding_width)
        #             # print(min_image[c, i, j])
        #             # print("result: ", min_image[c, i, j] - padding_width)
        #             min_image[c, i, j] = min_image[c, i, j] - padding_width
        #             # print("print min image:", min_image[c, i, j])
        #             max_image[c, i, j] = padding_width + max_image[c, i, j]
        #             # print("print max image: ", max_image[c, i, j])
        # print(max_image)
        # return max_image, min_image
    
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]