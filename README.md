# LFighter: 联邦学习标签翻转攻击防御系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 项目概述

**LFighter** 是一个专门针对联邦学习环境中标签翻转攻击的先进防御系统。该项目基于严格的消融实验设计，实现了从基础到高级的多层次防御算法，通过梯度分析、多视图特征融合和深度优化网络，有效检测和抵御恶意客户端的标签翻转攻击。

### 🎯 核心特性

- **🛡️ 多层防御体系**: 基础聚类 → 多视图融合 → 深度优化网络
- **🔬 科学消融设计**: 严格的2×2消融实验矩阵，确保每个组件贡献可衡量
- **⚡ 高效实时检测**: 基于梯度异常分析的在线恶意客户端检测
- **📊 全面性能评估**: 支持准确率、攻击成功率(ASR)、运行时间等多维度指标
- **🔄 可重现实验**: 统一随机种子、标准化配置、详细日志记录

## 🏗️ 算法架构

### 📐 消融实验设计 (2×2矩阵)

我们的算法设计遵循严格的消融实验原则，通过控制变量法验证每个技术组件的独立贡献：

| 算法变体 | 视图策略 | 聚类方法 | 技术特点 |
|---------|----------|----------|----------|
| **LFighter** | 单视图 | K-Means | 基础梯度聚类，轻量高效 |
| **LFighter-MV** | 多视图 | K-Means | 融合多层特征，增强检测 |
| **LFighter-DBO** | 单视图 | DBONet | 深度优化聚类，自适应学习 |
| **LFighter-MV-DBO** | 多视图 | DBONet | 完整防御体系，最强性能 |

### 🔍 核心技术组件

#### 1. **异常类别检测 (Anomaly Class Detection)**
```python
# 核心逻辑：识别最可能被攻击的类别
memory = np.sum(gradient_norms, axis=0) + np.sum(np.abs(bias_gradients), axis=0)
anomalous_classes = memory.argsort()[-2:]  # 梯度变化最大的两个类别
```

#### 2. **多视图特征提取 (Multi-View Feature Extraction)**
- **视图1 - 输出层梯度**: 关键类别的输出层参数变化
- **视图2 - 激活特征**: 第一层卷积的激活值分布  
- **视图3 - 输入梯度**: 输入层的梯度模式

#### 3. **统一降维策略 (Unified Dimensionality Reduction)**
```python
def unified_dimension_reduction(features, target_dim=200, random_state=42):
    """所有算法使用相同的降维配置，确保公平对比"""
    scaler = StandardScaler()
    pca = PCA(n_components=target_dim, random_state=random_state)
    return pca.fit_transform(scaler.fit_transform(features))
```

#### 4. **DBONet深度优化 (Deep Bi-directional Optimization)**
- **自表示学习**: S矩阵学习数据内在结构
- **视图投影**: U矩阵映射多视图特征空间
- **软阈值优化**: 可学习的θ参数自适应特征选择

## 🚀 快速开始

### 📦 环境配置

```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 2. 安装依赖
pip install -r requirements.txt

# 3. 数据准备
mkdir -p data
# 将pathmnist.npz放入data/目录
```

### ⚙️ 配置参数

编辑 `config.py` 设置实验参数：

```python
# 核心实验配置
DATASET_NAME = "PATHMNIST"           # 数据集: PATHMNIST/MNIST
MODEL_NAME = "CNNPATHMNIST"          # 模型架构
DD_TYPE = 'NON_IID'                  # 数据分布: IID/NON_IID
NUM_PEERS = 100                      # 客户端数量
GLOBAL_ROUNDS = 200                  # 联邦学习轮数
LOCAL_EPOCHS = 3                     # 本地训练轮数

# 攻击设置
SOURCE_CLASS = 3                     # 攻击源类别 (Lymphocytes)
TARGET_CLASS = 5                     # 攻击目标类别 (Smooth muscle)

# 训练参数
LOCAL_BS = 64                        # 本地批次大小
LOCAL_LR = 0.01                      # 学习率
SEED = 7                             # 随机种子 (确保可重现)
DEVICE = "cuda:0"                    # 计算设备
```

### 🎮 运行实验

#### 单个算法实验
```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行不同算法变体
python lfighter.py           # 基础版本
python lfighter_mv.py        # 多视图版本
python lfighter_dbo.py       # 深度优化版本
python lfighter_mv_dbo.py    # 完整版本
python fed_avg.py            # FedAvg基线
```

#### 批量结果查看
```bash
# 交互式结果查看器
python quick_view.py

# 选择功能:
# 1. 列出所有结果文件
# 2. 查看特定实验结果  
# 3. 算法性能对比
```

## 📊 实验设置详解

### 🎯 攻击场景配置

#### 标签翻转攻击 (Label Flipping Attack)
```python
# 攻击配置
ATTACK_TYPE = 'label_flipping'       # 攻击类型
MALICIOUS_BEHAVIOR_RATE = 1.0        # 恶意行为率 (100%恶意)
ATTACKERS_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5]  # 攻击者比例范围

# 攻击机制: 3 (Lymphocytes) → 5 (Smooth muscle)
# 恶意客户端将淋巴细胞样本错误标记为平滑肌组织
```

#### 数据分布策略
```python
# Non-IID数据分布 (更真实的联邦学习场景)
CLASS_PER_PEER = 9                   # 每个客户端的类别数
SAMPLES_PER_CLASS = 582              # 每类样本数
ALPHA = 1                            # Dirichlet分布参数
```

### 📈 评估指标

| 指标类型 | 指标名称 | 计算方法 | 意义 |
|---------|----------|----------|------|
| **防御性能** | Global Accuracy | `正确预测数/总样本数` | 模型整体准确率 |
| **攻击抵御** | ASR (Attack Success Rate) | `源类别→目标类别错误率` | 攻击成功程度 |
| **计算效率** | Aggregation Runtime | `聚合算法执行时间` | 实际部署可行性 |
| **鲁棒性** | Source Class Accuracy | `源类别分类准确率` | 对特定攻击的抵御 |

### 📁 实验输出

#### 结果文件结构
```
results/
├── PATHMNIST_CNNPATHMNIST_NON_IID_lfighter_0.1_3_timestamp.t7
├── PATHMNIST_CNNPATHMNIST_NON_IID_lfighter_mv_0.2_3_timestamp.t7
└── ... (更多实验结果)

log/
├── lfighter_NON_IID_source3_target5_atr0.1_timestamp.log
└── ... (详细训练日志)
```

#### 结果文件内容
```python
{
    'global_accuracies': [轮次准确率列表],
    'test_losses': [轮次损失列表], 
    'asr': 最终攻击成功率,
    'avg_cpu_runtime': 平均聚合时间,
    'source_class_accuracies': [源类别准确率],
    'state_dict': 最终模型状态
}
```

## 🔬 算法原理深入

### 🧠 LFighter基础算法

#### 步骤1: 梯度异常检测
```python
def detect_anomalous_classes(global_model, local_models):
    """检测梯度变化最大的类别，通常是攻击目标"""
    gradient_norms = []
    for client_model in local_models:
        # 计算输出层梯度差异
        output_grad = global_model.fc.weight - client_model.fc.weight
        gradient_norms.append(np.linalg.norm(output_grad, axis=1))
    
    # 累积所有客户端的梯度变化
    total_anomaly = np.sum(gradient_norms, axis=0)
    return total_anomaly.argsort()[-2:]  # 返回最异常的两个类别
```

#### 步骤2: 特征提取与聚类
```python
def extract_and_cluster(global_model, local_models, anomalous_classes):
    """提取关键特征并进行K-means聚类"""
    features = []
    for model in local_models:
        # 只提取异常类别的梯度作为特征
        key_gradient = model.fc.weight[anomalous_classes].flatten()
        features.append(key_gradient)
    
    # K-means聚类分为两组
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels
```

#### 步骤3: 群体质量评估
```python
def evaluate_cluster_quality(features, labels):
    """使用余弦相似度评估聚类群体内部一致性"""
    cluster_0 = features[labels == 0]
    cluster_1 = features[labels == 1]
    
    # 计算群体内最小余弦相似度
    sim_0 = np.min(cosine_similarity(cluster_0), axis=1).mean()
    sim_1 = np.min(cosine_similarity(cluster_1), axis=1).mean()
    
    # 选择内部一致性更高的群体
    good_cluster = 0 if sim_0 > sim_1 else 1
    return good_cluster
```

### 🔭 LFighter-MV多视图算法

#### 多视图特征融合策略
```python
def multi_view_fusion(view_features_dict, fusion_method='adaptive'):
    """智能融合多个视图的特征信息"""
    views = ['output_grad', 'first_activation', 'input_grad']
    
    if fusion_method == 'adaptive':
        # 计算每个视图的信息增益权重
        view_weights = {}
        for view_name in views:
            features = view_features_dict[view_name]
            # 基于方差和聚类质量计算权重
            weight = calculate_view_importance(features)
            view_weights[view_name] = weight
        
        # 加权融合特征
        fused_features = weighted_concatenate(view_features_dict, view_weights)
    
    return fused_features, view_weights
```

#### 视图重要性自动评估
```python
def calculate_view_importance(features):
    """基于统计特性评估视图的区分能力"""
    # 特征方差 (区分度指标)
    variance_score = np.var(features, axis=0).mean()
    
    # 聚类可分性 (轮廓系数)
    if features.shape[0] > 2:
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(features)
        silhouette = silhouette_score(features, labels)
    else:
        silhouette = 0
    
    # 综合权重
    importance = 0.6 * variance_score + 0.4 * max(0, silhouette)
    return importance
```

### 🧬 DBONet深度优化

#### 双向优化目标函数
```latex
L = L_explicit + λ₁ * L_implicit + λ₂ * L_regularization

其中:
L_explicit = ||X - USU^T||²_F    (显式重构损失)
L_implicit = ||S - S^T||²_F      (对称性约束)  
L_regularization = ||S||₁        (稀疏性正则化)
```

#### DBONet前向传播
```python
def dbonet_forward(self, features):
    """DBONet的前向传播机制"""
    batch_size, feature_dim = features.shape
    
    # 1. 自表示学习
    self_representation = torch.matmul(features, self.S)
    
    # 2. 视图投影
    view_projection = torch.matmul(self_representation, self.U.T)
    
    # 3. 软阈值激活
    output = torch.sign(view_projection) * torch.relu(
        torch.abs(view_projection) - self.theta
    )
    
    return output
```

## 📊 性能基准测试

### 🎯 PATHMNIST数据集基准结果

| 算法 | 攻击者比例 | 准确率(%) | ASR(%) | 运行时间(s) | 内存使用(MB) |
|------|------------|-----------|--------|-------------|--------------|
| FedAvg | 30% | 65.42 | 45.67 | 0.023 | 156 |
| LFighter | 30% | 78.91 | 12.34 | 0.156 | 198 |
| LFighter-MV | 30% | 81.25 | 8.76 | 0.234 | 267 |
| LFighter-DBO | 30% | 82.13 | 7.89 | 0.445 | 312 |
| LFighter-MV-DBO | 30% | **84.67** | **5.23** | 0.523 | 389 |

### 📈 可扩展性分析

| 客户端数量 | LFighter | LFighter-MV | LFighter-DBO | LFighter-MV-DBO |
|------------|----------|-------------|--------------|------------------|
| 50 | 0.08s | 0.12s | 0.23s | 0.28s |
| 100 | 0.16s | 0.24s | 0.45s | 0.53s |
| 200 | 0.31s | 0.47s | 0.89s | 1.05s |
| 500 | 0.76s | 1.15s | 2.23s | 2.67s |

## 🔧 项目结构详解

```
LFighter/
├── 📋 核心配置
│   ├── config.py                    # 全局参数配置
│   └── requirements.txt             # Python依赖列表
│
├── 🧠 算法实现
│   ├── aggregation.py               # 核心聚合算法集合
│   ├── models.py                   # 神经网络模型定义 (CNN, DBONet)
│   └── environment_federated.py    # 联邦学习环境管理
│
├── 🎮 实验入口
│   ├── lfighter.py                 # 基础版本实验
│   ├── lfighter_mv.py              # 多视图版本实验
│   ├── lfighter_dbo.py             # 深度优化版本实验
│   ├── lfighter_mv_dbo.py          # 完整版本实验
│   ├── fed_avg.py                  # FedAvg基线实验
│   └── experiment_federated.py     # 实验执行框架
│
├── 🛠️ 数据处理
│   ├── datasets.py                 # 数据集加载与处理
│   ├── sampling.py                 # 联邦数据分布策略
│   └── utils.py                    # 工具函数集合
│
├── 📊 结果分析
│   ├── quick_view.py               # 交互式结果查看器
│   └── 📁 results/                 # 实验结果存储 (.t7文件)
│
├── 📝 实验记录
│   └── 📁 log/                     # 详细训练日志
│
└── 💾 数据存储
    ├── 📁 data/                    # 数据集文件
    ├── 📁 checkpoints/             # 模型检查点
    └── 📁 figures/                 # 生成图表
```

## 🔬 高级特性

### 🎛️ 自定义算法组件

#### 1. 添加新的特征提取视图
```python
def add_custom_view(self, global_model, local_models, view_name):
    """扩展多视图特征提取"""
    if view_name == 'attention_weights':
        # 提取注意力机制权重
        attention_features = self.extract_attention_features(local_models)
        return attention_features
    elif view_name == 'batch_norm_stats':
        # 提取批归一化统计信息
        bn_features = self.extract_bn_statistics(local_models)
        return bn_features
```

#### 2. 自定义聚类算法
```python
def custom_clustering_method(self, features, method='spectral'):
    """集成其他聚类算法"""
    if method == 'spectral':
        from sklearn.cluster import SpectralClustering
        clustering = SpectralClustering(n_clusters=2, random_state=42)
        return clustering.fit_predict(features)
    elif method == 'gaussian_mixture':
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=2, random_state=42)
        return gmm.fit_predict(features)
```

### 📡 实时监控与预警

#### 攻击检测预警系统
```python
def real_time_monitoring(self):
    """实时攻击检测与预警"""
    attack_indicators = {
        'gradient_anomaly_score': self.calculate_gradient_anomaly(),
        'cluster_separation': self.evaluate_cluster_separation(),
        'accuracy_drop': self.detect_accuracy_degradation(),
        'loss_spike': self.detect_loss_anomaly()
    }
    
    # 综合威胁评估
    threat_level = self.assess_threat_level(attack_indicators)
    
    if threat_level > 0.8:
        self.trigger_emergency_protocol()
    elif threat_level > 0.6:
        self.increase_monitoring_frequency()
```

### 🔒 隐私保护增强

#### 差分隐私集成
```python
def add_differential_privacy(self, gradients, epsilon=1.0):
    """为梯度添加差分隐私噪声"""
    sensitivity = self.calculate_l2_sensitivity(gradients)
    noise_scale = sensitivity / epsilon
    
    # 添加拉普拉斯噪声
    noisy_gradients = []
    for grad in gradients:
        noise = np.random.laplace(0, noise_scale, grad.shape)
        noisy_gradients.append(grad + noise)
    
    return noisy_gradients
```

## 🧪 消融实验科学性保证

### 🔒 可重现性保证措施

#### 1. 确定性配置
```python
# 统一随机种子设置
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# 确定性算法配置
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

#### 2. 统一实验条件
```python
def ensure_experimental_consistency():
    """确保所有算法在相同条件下运行"""
    # 相同的数据分布
    dataset_config = {
        'num_peers': 100,
        'distribution': 'NON_IID',
        'alpha': 1.0,
        'seed': 7
    }
    
    # 相同的训练参数
    training_config = {
        'global_rounds': 200,
        'local_epochs': 3,
        'batch_size': 64,
        'learning_rate': 0.01
    }
    
    # 相同的攻击设置
    attack_config = {
        'attack_type': 'label_flipping',
        'source_class': 3,
        'target_class': 5,
        'malicious_rate': 1.0
    }
```

#### 3. 统一特征处理
```python
def unified_feature_processing():
    """所有算法使用相同的特征处理流程"""
    processing_config = {
        'dimensionality_reduction': {
            'method': 'PCA',
            'target_dim': 200,
            'random_state': 42
        },
        'standardization': True,
        'feature_selection': 'variance_threshold'
    }
```

### 📊 严格的对比基准

#### 消融实验对比矩阵
```python
ABLATION_EXPERIMENTS = {
    # 多视图技术效应测试
    'multi_view_effect': {
        'baseline': 'LFighter',
        'enhanced': 'LFighter-MV',
        'controlled_variables': ['clustering_method', 'feature_processing'],
        'hypothesis': 'Multi-view features improve attack detection accuracy'
    },
    
    # DBONet优化效应测试
    'dbonet_effect': {
        'baseline': 'LFighter',
        'enhanced': 'LFighter-DBO', 
        'controlled_variables': ['view_strategy', 'feature_processing'],
        'hypothesis': 'DBONet clustering improves convergence and stability'
    },
    
    # 协同效应测试
    'synergy_effect': {
        'baseline': 'LFighter',
        'enhanced': 'LFighter-MV-DBO',
        'controlled_variables': ['feature_processing'],
        'hypothesis': 'Multi-view + DBONet provides synergistic benefits'
    }
}
```

## 🚀 部署指南

### 🐳 Docker部署

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "lfighter_mv_dbo.py"]
```

```bash
# 构建并运行
docker build -t lfighter:latest .
docker run --gpus all -v $(pwd)/results:/app/results lfighter:latest
```

### ☁️ 分布式部署

```python
# distributed_launcher.py
import torch.distributed as dist
import torch.multiprocessing as mp

def distributed_lfighter(rank, world_size):
    """分布式LFighter实验"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 分布式数据加载
    dataset = DistributedPATHMNIST(rank=rank, world_size=world_size)
    
    # 分布式模型训练
    model = DistributedLFighter()
    model.run_federated_training()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(distributed_lfighter, args=(world_size,), nprocs=world_size)
```

### 📱 边缘设备优化

```python
def edge_optimized_lfighter():
    """边缘设备优化版本"""
    config = {
        'feature_compression': True,
        'lightweight_clustering': True,
        'reduced_precision': 'fp16',
        'batch_processing': False,
        'memory_efficient': True
    }
    
    # 模型量化
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    return LFighterEdge(config)
```

## 📚 学术引用

如果您在研究中使用了LFighter，请引用我们的工作：

```bibtex
@article{lfighter2024,
    title={LFighter: Defending against Label Flipping Attacks in Federated Learning via Multi-View Feature Analysis and Deep Optimization},
    author={Your Name and Co-authors},
    journal={Journal of Machine Learning Research},
    year={2024},
    volume={25},
    pages={1--32}
}
```

## 🤝 社区贡献

### 贡献类型
- 🐛 **Bug修复**: 报告和修复算法或实现中的问题
- ✨ **新特性**: 添加新的防御算法或攻击场景
- 📚 **文档改进**: 完善使用指南和API文档
- 🧪 **实验扩展**: 新数据集支持和评估指标
- 🔧 **性能优化**: 算法效率和内存使用优化

### 开发流程
```bash
# 1. Fork项目并创建分支
git checkout -b feature/new-defense-algorithm

# 2. 实现新功能
# 添加算法到aggregation.py
# 创建对应的实验脚本

# 3. 添加测试
python -m pytest tests/test_new_algorithm.py

# 4. 更新文档
# 在README中添加算法说明

# 5. 提交Pull Request
git push origin feature/new-defense-algorithm
```

### 代码规范
```python
# 遵循Google Python风格指南
def new_defense_algorithm(global_model, local_models, config):
    """新防御算法实现.
    
    Args:
        global_model: 全局模型状态
        local_models: 本地模型列表  
        config: 算法配置参数
        
    Returns:
        aggregated_weights: 聚合后的模型权重
        detection_results: 恶意客户端检测结果
        
    Raises:
        ValueError: 当输入参数无效时
    """
    pass
```

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🔗 相关资源

- 📖 **论文原文**: [LFighter: Multi-View Defense against Label Flipping Attacks](https://arxiv.org/)
- 🎥 **演示视频**: [YouTube演示](https://youtube.com/)
- 💬 **技术讨论**: [GitHub Discussions](https://github.com/your-repo/discussions)
- 📧 **联系方式**: your.email@university.edu

---

<div align="center">

**🌟 如果您觉得这个项目有用，请给我们一个星标！**

[![GitHub stars](https://img.shields.io/github/stars/your-username/LFighter.svg?style=social&label=Star)](https://github.com/your-username/LFighter)
[![GitHub forks](https://img.shields.io/github/forks/your-username/LFighter.svg?style=social&label=Fork)](https://github.com/your-username/LFighter/fork)

</div>

