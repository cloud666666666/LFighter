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
    
# 在分割数据集后，打印全局训练集标签分布
if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms
    import os
    import numpy as np
    # 自动加载PATHMNIST训练集
    try:
        from torchvision.datasets import VisionDataset
        from torchvision.datasets.utils import download_url
        # 你主流程应该用的是torchvision.datasets.ImageFolder或类似方式
        # 这里假设数据已解压在 data/PATHMNIST/，每类一个子文件夹
        data_dir = os.path.join('data', 'PATHMNIST')
        if not os.path.exists(data_dir):
            print('请确保PATHMNIST数据已解压到 data/PATHMNIST/')
        else:
            trainset = torchvision.datasets.ImageFolder(root=data_dir, transform=transforms.ToTensor())
            labels = [label for _, label in trainset.samples]
            print('训练集标签分布:', np.bincount(np.array(labels)))
    except Exception as e:
        print('无法打印标签分布:', e)
    
    data = np.load('data/pathmnist.npz')
    y_train = data['train_labels']
    # y_train shape: (N, 1) 或 (N,)
    y_train = y_train.flatten()
    print('训练集标签分布:', np.bincount(y_train))
    