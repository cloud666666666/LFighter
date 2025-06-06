import torch
from torch.utils import data
import numpy as np


class CustomDataset(data.Dataset):
    def __init__(self, dataset, indices, source_class = None, target_class = None):
        self.dataset = dataset
        self.indices = indices
        self.source_class = source_class
        self.target_class = target_class  
        self.contains_source_class = False
            
    def __getitem__(self, index):
        x, y = self.dataset[int(self.indices[index])][0], self.dataset[int(self.indices[index])][1]
        if isinstance(y, torch.Tensor):
            y = y.item()
        elif isinstance(y, np.ndarray):
            if y.ndim > 0 and y.size > 1:
                y = int(np.argmax(y))
            else:
                y = int(y)
        else:
            y = int(y)
        if y == self.source_class:
            y = self.target_class 
        return x, y 

    def __len__(self):
        return len(self.indices)

class PoisonedDataset(data.Dataset):
    def __init__(self, dataset, source_class = None, target_class = None):
        self.dataset = dataset
        self.source_class = source_class
        self.target_class = target_class  
            
    def __getitem__(self, index):
        x, y = self.dataset[index][0], self.dataset[index][1]
        if isinstance(y, torch.Tensor):
            y = y.item()
        elif isinstance(y, np.ndarray):
            if y.ndim > 0 and y.size > 1:
                y = int(np.argmax(y))
            else:
                y = int(y)
        else:
            y = int(y)
        if y == self.source_class:
            y = self.target_class 
        return x, y 

    def __len__(self):
        return len(self.dataset)

    
class IMDBDataset:
    def __init__(self, reviews, targets):
        """
        Argument:
        reviews: a numpy array
        targets: a vector array
        
        Return xtrain and ylabel in torch tensor datatype
        """
        self.reviews = reviews
        self.target = targets
    
    def __len__(self):
        # return length of dataset
        return len(self.reviews)
    
    def __getitem__(self, index):
        # given an index (item), return review and target of that index in torch tensor
        x = torch.tensor(self.reviews[index,:], dtype = torch.long)
        y = torch.tensor(self.target[index], dtype = torch.float)
        
        return  x, y

# A method for combining datasets  
def combine_datasets(list_of_datasets):
    return data.ConcatDataset(list_of_datasets)
    