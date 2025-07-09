import config
from experiment_federated import run_exp 
import os
from datetime import datetime
# LFighter多视图+DBO-Net损失优化实验
RULE = 'lfighter_mv_dbo'
ATTACK_TYPE = 'label_flipping'
MALICIOUS_BEHAVIOR_RATE = 1

# 获取攻击配置
ATTACK_CONFIG = config.ATTACK_CONFIGS.get(config.ATTACK_SCENARIO, None)
os.makedirs('log', exist_ok=True)
for atr in [0.1, 0.2, 0.3, 0.4, 0.5]:  # 统一攻击者比例
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    LOG_FILE = f"log/lfighter_mv_dbo_{config.LOG_DIST_TYPE}_source{config.SOURCE_CLASS}_target{config.TARGET_CLASS}_atr{atr}_{config.ATTACK_SCENARIO}_{timestamp}.log"
    
    print(f"🔧 实验配置:")
    print(f"   攻击场景: {config.ATTACK_SCENARIO}")
    if ATTACK_CONFIG:
        print(f"   攻击配置: {ATTACK_CONFIG}")
    print(f"   攻击者比例: {atr}")
    
    run_exp(dataset_name = config.DATASET_NAME, model_name = config.MODEL_NAME, dd_type = config.DD_TYPE, num_peers = config.NUM_PEERS, 
            frac_peers = config.FRAC_PEERS, seed = config.SEED, test_batch_size = config.TEST_BATCH_SIZE,
                criterion = config.CRITERION, global_rounds = config.GLOBAL_ROUNDS, local_epochs = config.LOCAL_EPOCHS, local_bs = config.LOCAL_BS, 
                 local_lr = config.LOCAL_LR, local_momentum = config.LOCAL_MOMENTUM, labels_dict = config.LABELS_DICT, device = config.DEVICE,
                attackers_ratio = atr, attack_type = ATTACK_TYPE, 
                 malicious_behavior_rate = MALICIOUS_BEHAVIOR_RATE, rule = RULE,
                source_class = config.SOURCE_CLASS, target_class = config.TARGET_CLASS,
               class_per_peer = config.CLASS_PER_PEER, samples_per_class = config.SAMPLES_PER_CLASS, 
               rate_unbalance = config.RATE_UNBALANCE, alpha = config.ALPHA, resume = False, log_file=LOG_FILE,
               attack_config = ATTACK_CONFIG)  # 添加攻击配置参数 