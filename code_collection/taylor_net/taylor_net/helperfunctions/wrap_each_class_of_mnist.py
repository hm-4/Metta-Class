from torch.utils.data import Dataset

class SeparatingOneClass(Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        return image, target
