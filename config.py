import torch
import random
import torch.nn as nn

#=============================== Defining global variables ========================#
DATASET_NAME = "PATHMNIST"
MODEL_NAME = "CNNPATHMNIST"  # 你需要在models.py中实现CNNPATHMNIST模型
DD_TYPE = 'NON_IID'
ALPHA = 1
NUM_PEERS = 100 # "number of peers: K" 
FRAC_PEERS = 1 #'the fraction of peers: C to be selected in each round'
SEED = 7 #fixed seed
random.seed(SEED)
CRITERION = nn.CrossEntropyLoss()
GLOBAL_ROUNDS = 200 #"number of rounds of federated model training"
LOCAL_EPOCHS = 3 #"the number of local epochs: E for each peer"
TEST_BATCH_SIZE = 1000
LOCAL_BS = 64 #"local batch size: B for each peer"
LOCAL_LR =  0.01#local learning rate: lr for each peer
LOCAL_MOMENTUM = 0.9 #local momentum for each peer
NUM_CLASSES = 9 # PATHMNIST有9类
LABELS_DICT = {
    'background': 0,
    'normal': 1,
    'cancer': 2,
    'adenoma': 3,
    'inflammation': 4,
    'hyperplasia': 5,
    'be': 6,
    'dysplasia': 7,
    'other': 8
}

#select the device to work with cpu or gpu
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
DEVICE = torch.device(DEVICE)
SOURCE_CLASS = 2 # 以cancer为例，源类别（可根据实验需求调整）
TARGET_CLASS = 1 # 以normal为例，目标类别（可根据实验需求调整）

CLASS_PER_PEER = 9  # 每个peer拥有的类别数，建议与NUM_CLASSES一致
SAMPLES_PER_CLASS = 700  # 每类样本数，可根据PATHMNIST实际情况调整
RATE_UNBALANCE = 1