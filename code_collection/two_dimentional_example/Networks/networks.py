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