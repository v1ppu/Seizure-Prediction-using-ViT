import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F

class EEGDataset(Dataset):
    def __init__(self, images, labels, resize_to=(224, 224)):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

        if(self.images.ndim == 3):
            self.images = self.images.unsqueeze(1)

        self.images = F.interpolate(self.images, size=resize_to, mode='bilinear', align_corners=False)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]