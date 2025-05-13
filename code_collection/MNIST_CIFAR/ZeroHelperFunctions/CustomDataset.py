from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        """get data and standardize it.
        Args:
            data (torch.Tensor): image_data in floats() must be [0, 1]
            targets (torch.Tensor): labels in float()
        """
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        data, target = self.data[idx], self.targets[idx]
        return data, target