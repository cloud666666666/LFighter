import config
from experiment_federated import run_exp 
import os
from datetime import datetime

# 多Krum聚合防御实验
RULE = 'mkrum'
ATTACK_TYPE = 'label_flipping'
MALICIOUS_BEHAVIOR_RATE = 1

os.makedirs('log', exist_ok=True)

for atr in [0.1, 0.2, 0.3, 0.4, 0.5]:  # 统一攻击者比例
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    LOG_FILE = f"log/mkrum_{config.LOG_DIST_TYPE}_source{config.SOURCE_CLASS}_target{config.TARGET_CLASS}_atr{atr}_{timestamp}.log"
    
    run_exp(dataset_name = config.DATASET_NAME, model_name = config.MODEL_NAME, dd_type = config.DD_TYPE, num_peers = config.NUM_PEERS, 
            frac_peers = config.FRAC_PEERS, seed = config.SEED, test_batch_size = config.TEST_BATCH_SIZE,
            criterion = config.CRITERION, global_rounds = config.GLOBAL_ROUNDS, local_epochs = config.LOCAL_EPOCHS, local_bs = config.LOCAL_BS, 
            local_lr = config.LOCAL_LR, local_momentum = config.LOCAL_MOMENTUM, labels_dict = config.LABELS_DICT, device = config.DEVICE,
            attackers_ratio = atr, attack_type = ATTACK_TYPE, 
            malicious_behavior_rate = MALICIOUS_BEHAVIOR_RATE, rule = RULE,
            source_class = config.SOURCE_CLASS, target_class = config.TARGET_CLASS,
            class_per_peer = config.CLASS_PER_PEER, samples_per_class = config.SAMPLES_PER_CLASS, 
            rate_unbalance = config.RATE_UNBALANCE, alpha = config.ALPHA, resume = False, log_file=LOG_FILE) 