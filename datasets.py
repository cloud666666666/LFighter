import torch
from torch.utils import data
import numpy as np
import random


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
    def __init__(self, dataset, source_class=None, target_class=None, attack_config=None, current_epoch=0, peer_id=None):
        self.dataset = dataset
        self.source_class = source_class
        self.target_class = target_class
        self.attack_config = attack_config or {}
        self.current_epoch = current_epoch
        self.peer_id = peer_id
        self.attack_history = []
        self.current_round_flips = []  # 当前轮次的翻转记录
        
        # 检测历史（用于自适应攻击）
        self.detection_history = []
        self.current_detection_rate = 0.0
        
    def set_epoch(self, epoch):
        """设置当前轮次（用于时变攻击）"""
        self.current_epoch = epoch
        # 新的轮次开始，清空当前轮次的翻转记录
        self.current_round_flips = []
        
    def update_detection_rate(self, detection_rate):
        """更新检测率（用于自适应攻击）"""
        self.detection_history.append(detection_rate)
        if len(self.detection_history) > 5:  # 只保留最近5轮的历史
            self.detection_history.pop(0)
        self.current_detection_rate = np.mean(self.detection_history)
            
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
            
        original_y = y
        
        # 根据攻击配置应用不同的攻击策略
        if self.attack_config:
            y = self._apply_complex_attack(y, original_y)
        elif self.source_class is not None and y == self.source_class:
            # 简单攻击（兼容原有逻辑）
            y = self.target_class
            self._record_flip(original_y, y, 'simple')
            
        return x, y
    
    def _record_flip(self, original_label, flipped_label, attack_type):
        """记录标签翻转事件"""
        if original_label != flipped_label:
            flip_record = {
                'original': original_label,
                'flipped': flipped_label,
                'attack_type': attack_type,
                'peer_id': self.peer_id,
                'epoch': self.current_epoch
            }
            self.current_round_flips.append(flip_record)
            self.attack_history.append(flip_record)
    
    def get_current_round_flips(self):
        """获取当前轮次的翻转记录"""
        return self.current_round_flips.copy()
    
    def _apply_complex_attack(self, y, original_y):
        """应用复杂攻击策略"""
        attack_type = self.attack_config.get('type', 'simple')
        
        if attack_type == 'simple':
            new_y = self._simple_attack(y)
        elif attack_type == 'multi_target':
            new_y = self._multi_target_attack(y)
        elif attack_type == 'probabilistic':
            new_y = self._probabilistic_attack(y)
        elif attack_type == 'time_varying':
            new_y = self._time_varying_attack(y)
        elif attack_type == 'adaptive':
            new_y = self._adaptive_attack(y)
        elif attack_type == 'mixed':
            new_y = self._mixed_attack(y)
        else:
            new_y = y
        
        # 记录翻转事件
        self._record_flip(original_y, new_y, attack_type)
        
        return new_y
    
    def _simple_attack(self, y):
        """简单攻击：固定映射"""
        config = self.attack_config
        source = config.get('source_class', self.source_class)
        target = config.get('target_class', self.target_class)
        flip_rate = config.get('flip_rate', 1.0)
        
        if y == source and random.random() < flip_rate:
            return target
        return y
    
    def _multi_target_attack(self, y):
        """多源-多目标攻击"""
        mappings = self.attack_config.get('mappings', {})
        flip_probs = self.attack_config.get('flip_probabilities', {})
        
        if y in mappings:
            flip_prob = flip_probs.get(y, 0.5)
            if random.random() < flip_prob:
                return mappings[y]
        return y
    
    def _probabilistic_attack(self, y):
        """概率性攻击"""
        source_classes = self.attack_config.get('source_classes', [])
        target_classes = self.attack_config.get('target_classes', [])
        flip_rate = self.attack_config.get('flip_rate', 0.6)
        randomize = self.attack_config.get('randomize_targets', True)
        
        if y in source_classes and random.random() < flip_rate:
            if randomize:
                return random.choice(target_classes)
            else:
                # 固定映射到第一个目标类别
                return target_classes[0] if target_classes else y
        return y
    
    def _time_varying_attack(self, y):
        """时变攻击"""
        phases = self.attack_config.get('phases', [])
        
        for phase in phases:
            epoch_range = phase['epochs']
            if epoch_range[0] <= self.current_epoch < epoch_range[1]:
                mapping = phase['mapping']
                flip_rate = phase.get('flip_rate', 1.0)
                
                if y in mapping and random.random() < flip_rate:
                    return mapping[y]
                break
        return y
    
    def _adaptive_attack(self, y):
        """自适应攻击：根据检测率调整策略"""
        detection_threshold = self.attack_config.get('detection_threshold', 0.7)
        
        if self.current_detection_rate > detection_threshold:
            # 高检测率：使用隐蔽攻击
            return self._stealth_attack(y)
        else:
            # 低检测率：使用激进攻击
            return self._aggressive_attack(y)
    
    def _stealth_attack(self, y):
        """隐蔽攻击：相似类别间的低频率翻转"""
        stealth_mapping = self.attack_config.get('stealth_mapping', {})
        stealth_rate = self.attack_config.get('stealth_rate', 0.3)
        
        if y in stealth_mapping and random.random() < stealth_rate:
            return stealth_mapping[y]
        return y
    
    def _aggressive_attack(self, y):
        """激进攻击：差异大的类别间的高频率翻转"""
        aggressive_mapping = self.attack_config.get('aggressive_mapping', {})
        aggressive_rate = self.attack_config.get('aggressive_rate', 0.9)
        
        if y in aggressive_mapping and random.random() < aggressive_rate:
            return aggressive_mapping[y]
        return y
    
    def _mixed_attack(self, y):
        """混合策略攻击"""
        label_flip_prob = self.attack_config.get('label_flip_prob', 0.6)
        noise_only_prob = self.attack_config.get('noise_only_prob', 0.2)
        
        rand = random.random()
        if rand < label_flip_prob:
            # 标签翻转
            mappings = self.attack_config.get('mappings', {})
            if y in mappings:
                return mappings[y]
        elif rand < label_flip_prob + noise_only_prob:
            # 只添加噪声，标签不变（噪声在训练过程中添加）
            pass
        # 其余情况为混合策略，这里简化为随机选择标签翻转
        
        return y
    


    def get_attack_statistics(self):
        """获取攻击统计信息"""
        return {
            'total_samples': len(self.dataset),
            'attack_history_length': len(self.attack_history),
            'current_detection_rate': self.current_detection_rate,
            'detection_history': self.detection_history.copy()
        }

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
    