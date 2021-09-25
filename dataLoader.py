from PIL import Image
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from scipy import stats as ss
from config import config


# Dataset Class for MLP and CNN
class WindDataset(Dataset):
    def __init__(self, labelDataIndexes, dataset, transform = None):
        super(WindDataset, self).__init__()
        self.input_ =  labelDataIndexes           
        self.imageData, self.windData = dataset
        self.transform = transform   

    def __getitem__(self, idx):
        # Changed
        index = self.input_[idx]
        self.label = self.windData[index]
        self.image  = self.imageData[index]
        
        if self.transform:            
          self.image = self.transform(self.image)                                   

        self.image = torch.from_numpy(self.image.copy())                
        self.label = self.label
        return self.image, self.label

    def __len__(self):
        return len(self.input_)
    
# Dataset Class for CNN-LSTM
class WindDatasetSequence(Dataset):
    def __init__(self, labelDataIndexes, dataset, transform = None):
        super(WindDatasetSequence, self).__init__()
        self.input_ =  labelDataIndexes           
        self.imageSeq, self.windData = dataset        
        self.layer_dim = config['layer_dim']
        self.hidden_dim = config['hidden_dim']
        self.transform = transform        

    def __getitem__(self, idx):      
        # Changed
        idx = self.input_[idx]
        self.label = self.windData[idx + config['seq_dim']]
        
        # if self.transform:            
        #     self.image = self.transform(self.image)  
            
        self.images  = self.imageSeq[idx: idx + config['seq_dim']]                    
        self.images = torch.as_tensor(self.images)        
        self.label = self.label          
        return self.images, self.label

    def __len__(self):
      # Changed
        return len(self.input_) - config['seq_dim']