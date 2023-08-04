import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.io import read_image, ImageReadMode

import numpy as np
import random
import yaml
from PIL import Image

class HAM10000Dataset(Dataset):
    def __init__(self, df, sens_attribute, transform=None):

        assert sens_attribute is not None

        self.df = df
        self.transform = transform
        self.sens_attribute = sens_attribute
        self.classes = self.get_num_classes()
        self.class_to_idx = self._get_class_to_idx()

    def __len__(self):
        return len(self.df)
    
    def get_num_classes(self):
        return self.df['dx_index'].unique()

    def _get_original_labels(self):
        return {'akiec':"Bowen's disease",
                'bcc':'basal cell carcinoma',
                'bkl':'benign keratosis-like lesions',
                'df':'dermatofibroma',
                'nv':'melanocytic nevi',
                'mel':'melanoma',
                'vasc':'vascular lesions'}
    
    def _get_class_to_idx(self):
        return {'akiec':0,
                'bcc':1,
                'bkl':2,
                'df':3,
                'nv':4,
                'mel':5,
                'vasc':6}
    
    def __getitem__(self, idx):
        image = read_image(self.df.iloc[idx]['Path'], mode=ImageReadMode.RGB)
        image = T.ToPILImage()(image)
        label = torch.tensor(self.df.iloc[idx]['dx_index'])

        if(self.sens_attribute == 'gender'):
            sens_attribute = self.df.iloc[idx]['sex']
        elif(self.sens_attribute == 'age'):
            sens_attribute = self.df.iloc[idx]['Age_multi']
        else:
            raise ValueError('Invalid sensitive attribute for HAM10000 dataset')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, sens_attribute