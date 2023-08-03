import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.io import read_image, ImageReadMode

import numpy as np
import random
import yaml
from PIL import Image

class FitzpatrickDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.classes = self.get_num_classes()
        self.class_to_idx = self._get_class_to_idx()

    def __len__(self):
        return len(self.df)
    
    def get_num_classes(self):
        return self.df['label_idx'].unique()
    
    def _get_class_to_idx(self):
        return {'benign': 0, 'malignant': 1, 'non-neoplastic': 2}
    
    def __getitem__(self, idx):
        image = read_image(self.df.iloc[idx]['Path'], mode=ImageReadMode.RGB)
        image = T.ToPILImage()(image)
        label = self.df.iloc[idx]['label_idx']
        sens_attribute = self.df.iloc[idx]['skin_type']

        if self.transform:
            image = self.transform(image)
        
        return image, label, sens_attribute