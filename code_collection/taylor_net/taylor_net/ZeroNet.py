import torch
from torch import nn

class ZeroNet(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 layer1_dim: int,
                 layer2_dim: int,
                 layer3_dim: int,
                 num_classes: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=layer1_dim),
            nn.ReLU(),
            nn.Linear(in_features=layer1_dim, out_features=layer2_dim),
            nn.ReLU(),
            nn.Linear(layer2_dim, layer3_dim),
            nn.ReLU(),
            nn.Linear(layer3_dim, num_classes+1) # +1 for zero.
        )
    
    def forward(self, x:torch.Tensor):
        return(self.layer_stack(x))


class SimpleNet(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 layer1_dim: int,
                 layer2_dim: int,
                 layer3_dim: int,
                 num_classes: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
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


def generate_zero_data(min_vector, max_vector, percentage_padding=100):
    for i in range(len(min_vector)):
        width_of_dim_i = max_vector[i] - min_vector[i]
        padding_width = width_of_dim_i * percentage_padding / 100
        min_vector[i] -= padding_width
        max_vector[i] += padding_width
    # print(min_vector, max_vector)
    return torch.distributions.uniform.Uniform(min_vector, max_vector)
    