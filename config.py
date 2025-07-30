import torch
import random
import torch.nn as nn

#=============================== Defining global variables ========================#
DATASET_NAME = "PATHMNIST"
MODEL_NAME = "CNNPATHMNIST"  # 你需要在models.py中实现CNNMNIST模型
DD_TYPE = 'IID'
LOG_DIST_TYPE = DD_TYPE  # 新增：用于日志命名的分布类型（IID或NONIID）
ALPHA = 1
NUM_PEERS = 100     # 小规模测试：只用5个peer
FRAC_PEERS = 1.0  # 每轮采样全部peer
SEED = 7 #fixed seed
random.seed(SEED)

#select the device to work with cpu or gpu
if torch.cuda.is_available():
    DEVICE = "cuda:5"  # 指定使用GPU6
else:
    DEVICE = "cpu"
DEVICE = torch.device(DEVICE)

# 小规模快速测试配置
CRITERION = nn.CrossEntropyLoss()  # 继续使用标准CE

GLOBAL_ROUNDS = 200  # 小规模测试：只训练10轮
LOCAL_EPOCHS = 3    # 减少本地轮次到2轮
TEST_BATCH_SIZE = 1000 # 减少测试批次大小
LOCAL_BS = 64      # 减少批次大小
LOCAL_LR =  0.01   # 保持学习率
LOCAL_MOMENTUM = 0.9 #local momentum for each peer
NUM_CLASSES = 9 # PATHMNIST有9类
LABELS_DICT = {
    0: 'Adipose',
    1: 'Background',
    2: 'Debris',
    3: 'Lymphocytes',
    4: 'Mucus',
    5: 'Smooth muscle',
    6: 'Normal colon mucosa',
    7: 'Cancer-associated stroma',
    8: 'Colorectal adenocarcinoma epithelium'
}

SOURCE_CLASS = 3 
TARGET_CLASS = 5 

CLASS_PER_PEER = 9  # 每个peer拥有的类别数，建议与NUM_CLASSES一致
SAMPLES_PER_CLASS = 582  # 小规模测试：每类只用100个样本
RATE_UNBALANCE = 1

#=============================== 复杂攻击场景配置 ========================#
# 攻击场景类型选择
ATTACK_SCENARIO = 'multi_target'  # 使用多目标攻击实现移位攻击

# 简单攻击（当前默认）
SIMPLE_ATTACK = {
    'type': 'simple',
    'source_class': SOURCE_CLASS,
    'target_class': TARGET_CLASS,
    'flip_rate': 1.0
}

# 多源-多目标攻击 - 改为全局循环移位攻击
MULTI_TARGET_ATTACK = {
    'type': 'multi_target',
    'mappings': {
        0: 1,  # Adipose → Background
        1: 2,  # Background → Debris
        2: 3,  # Debris → Lymphocytes
        3: 4,  # Lymphocytes → Mucus
        4: 5,  # Mucus → Smooth muscle
        5: 6,  # Smooth muscle → Normal colon mucosa
        6: 7,  # Normal colon mucosa → Cancer-associated stroma
        7: 8,  # Cancer-associated stroma → Colorectal adenocarcinoma
        8: 0   # Colorectal adenocarcinoma → Adipose (循环移位)
    },
    'flip_probabilities': {
        0: 0.95,  # 高概率翻转，确保攻击效果明显
        1: 0.95,
        2: 0.95,
        3: 0.95,
        4: 0.95,
        5: 0.95,
        6: 0.95,
        7: 0.95,
        8: 0.95
    }
}

# 概率性攻击
PROBABILISTIC_ATTACK = {
    'type': 'probabilistic',
    'source_classes': [1, 3, 5, 7],
    'target_classes': [0, 2, 4, 6, 8],
    'flip_rate': 0.6,
    'randomize_targets': True  # 是否随机选择目标类别
}

# 时变攻击
TIME_VARYING_ATTACK = {
    'type': 'time_varying',
    'phases': [
        {'epochs': [0, 3], 'mapping': {1: 3}, 'flip_rate': 0.8},
        {'epochs': [3, 6], 'mapping': {3: 7}, 'flip_rate': 0.9},
        {'epochs': [6, 9], 'mapping': {7: 1}, 'flip_rate': 0.7},
        {'epochs': [9, 12], 'mapping': {5: 2}, 'flip_rate': 0.6}
    ]
}

# 自适应攻击
ADAPTIVE_ATTACK = {
    'type': 'adaptive',
    'stealth_mapping': {3: 4, 1: 2},      # 相似类别映射（隐蔽期）
    'aggressive_mapping': {1: 8, 3: 5, 7: 0},  # 差异大的映射（激进期）
    'stealth_rate': 0.3,
    'aggressive_rate': 0.9,
    'detection_threshold': 0.7  # 检测率超过此值时切换为隐蔽模式
}

# 混合策略攻击  
MIXED_ATTACK = {
    'type': 'mixed',
    'label_flip_prob': 0.6,  # 60%使用标签翻转
    'noise_only_prob': 0.2,  # 20%只添加噪声
    'hybrid_prob': 0.2,      # 20%混合策略
    'mappings': {1: 2, 3: 7, 5: 8},
    'noise_std': 0.1
}

# 攻击场景配置映射
ATTACK_CONFIGS = {
    'simple': SIMPLE_ATTACK,
    'multi_target': MULTI_TARGET_ATTACK, 
    'probabilistic': PROBABILISTIC_ATTACK,
    'time_varying': TIME_VARYING_ATTACK,
    'adaptive': ADAPTIVE_ATTACK,
    'mixed': MIXED_ATTACK
}