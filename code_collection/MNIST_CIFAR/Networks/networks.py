import torch
from torch import nn

class FullyConnectedNet(nn.Module):
    def __init__(self, 
                input_dim: int,
                layer1_dim: int,
                layer2_dim: int,
                layer3_dim: int,
                num_classes: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_dim, out_features=layer1_dim),
            nn.ReLU(),
            nn.Linear(in_features=layer1_dim, out_features=layer2_dim),
            nn.ReLU(),
            nn.Linear(layer2_dim, layer3_dim),
            nn.ReLU(),
            nn.Linear(layer3_dim, num_classes) # +1 for zero.
        )
    
    def forward(self, x:torch.Tensor):
        return(self.layer_stack(x))


class FullyConnectedNet_powerful(nn.Module):
    def __init__(self, 
                input_dim: int,
                layer1_dim: int,
                layer2_dim: int,
                layer3_dim: int,
                layer4_dim: int,
                layer5_dim: int,
                num_classes: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_dim, out_features=layer1_dim),
            nn.ReLU(),
            nn.Linear(in_features=layer1_dim, out_features=layer2_dim),
            nn.ReLU(),
            nn.Linear(layer2_dim, layer3_dim),
            nn.ReLU(),
            nn.Linear(layer3_dim, layer4_dim),
            nn.ReLU(),
            nn.Linear(layer4_dim, layer5_dim),
            nn.ReLU(),
            nn.Linear(layer5_dim, num_classes) # +1 for zero.
        )
    
    def forward(self, x:torch.Tensor):
        return(self.layer_stack(x))




import torch
import torch.nn as nn

# class AllClassVGGNet(nn.Module):
#     def __init__(self, out_nodes):
#         super().__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),#16
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),#8

#             nn.Conv2d(256, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),#4
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),#2

#             nn.Conv2d(256, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)#1
#         )

#         self.fc_layers = nn.Sequential(
#             nn.Linear(128 * 1 * 1, 1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(1024, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(512, out_nodes)
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc_layers(x)
#         return x

class AllClassVGGNet(nn.Module):
    def __init__(self, out_nodes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),#16
            
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),#8

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),#4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),#2

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)#1
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 1 * 1, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, out_nodes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
    
class ClassWiseVGGNet(nn.Module):
    def __init__(self, out_nodes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),#16
            
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),#8

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),#4
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),#2

            nn.Conv2d(64, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)#1
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 1 * 1, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(1024, 256),
            # nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, out_nodes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x