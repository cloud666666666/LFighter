import torch
import random
import torch.nn as nn

#=============================== Defining global variables ========================#
DATASET_NAME = "PATHMNIST"
MODEL_NAME = "CNNPATHMNIST"  # 你需要在models.py中实现CNNMNIST模型
DD_TYPE = 'NON_IID'
LOG_DIST_TYPE = DD_TYPE  # 新增：用于日志命名的分布类型（IID或NONIID）
ALPHA = 1
NUM_PEERS = 100     # 小规模测试：只用5个peer
FRAC_PEERS = 1.0  # 每轮采样全部peer
SEED = 7 #fixed seed
random.seed(SEED)

#select the device to work with cpu or gpu
if torch.cuda.is_available():
    DEVICE = "cuda:6"  # 指定使用GPU6
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