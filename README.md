# LFighter: 针对联邦学习标签翻转攻击的鲁棒聚合算法

## 📋 项目简介

LFighter 是一个专门针对联邦学习环境中标签翻转攻击的防御系统。该项目实现了多种鲁棒聚合算法，包括基础的 LFighter 算法以及其增强版本（多视图版本和 DBO 集成版本），用于检测和抵御恶意客户端的攻击。

## 🏗️ 项目架构

### 核心算法

1. **LFighter**: 基础版本，基于梯度聚类和余弦相似度分析
2. **LFighter-MV**: 多视图版本，利用神经网络不同层的特征表示
3. **LFighter-DBO**: 集成深度双向优化网络的版本
4. **LFighter-MV-DBO**: 结合多视图和DBO的最强版本

### 支持的基准算法

- **FedAvg**: 联邦平均算法（基准）
- **FoolsGold**: 基于梯度相似度的防御算法
- **Tolpegin**: 另一种鲁棒聚合算法
- **FLAME**, **Krum**, **Median**, **Trimmed Mean**: 其他经典防御算法

## 🎯 主要功能

### 攻击检测
- **标签翻转攻击检测**: 识别将源类别标签恶意翻转为目标类别的攻击者
- **梯度异常检测**: 通过分析模型参数更新的异常模式检测攻击
- **聚类分析**: 使用 K-means 聚类将参与者分为正常和恶意群体

### 数据集支持
- **PATHMNIST**: 病理图像分类数据集（9类）
- **MNIST**: 手写数字识别数据集（10类）
- **IMDB**: 情感分析数据集（文本分类）

### 模型架构
- **CNNPATHMNIST**: 专为 PATHMNIST 设计的 CNN 模型
- **CNNMNIST**: MNIST 数据集的 CNN 模型
- **BiLSTM**: 用于文本分类的双向 LSTM
- **DBONet**: 深度双向优化网络
- **ResNet18**: 深度残差网络

## 🚀 快速开始

### 环境要求

```bash
Python 3.8+
CUDA 12.x (可选，用于 GPU 加速)
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据准备

确保在 `data/` 目录下有以下数据：
- `pathmnist.npz`: PATHMNIST 数据集
- `MNIST/`: MNIST 数据集目录

### 配置参数

编辑 `config.py` 文件来设置实验参数：

```python
DATASET_NAME = "PATHMNIST"       # 数据集名称
MODEL_NAME = "CNNPATHMNIST"      # 模型架构
NUM_PEERS = 100                  # 客户端数量
GLOBAL_ROUNDS = 200              # 全局训练轮数
LOCAL_EPOCHS = 3                 # 本地训练轮数
SOURCE_CLASS = 3                 # 攻击源类别
TARGET_CLASS = 5                 # 攻击目标类别
```

### 运行实验

#### 1. 基础 LFighter 算法
```bash
python lfighter.py
```

#### 2. 多视图版本
```bash
python lfighter_mv.py
```

#### 3. DBO集成版本
```bash
python lfighter_dbo.py
```

#### 4. 完整版本（MV+DBO）
```bash
python lfighter_mv_dbo.py
```

#### 5. 基准算法（FedAvg）
```bash
python fed_avg.py
```

## 📊 实验设置

### 默认参数
- **参与者数量**: 100
- **攻击者比例**: 10%-50%
- **数据分布**: Non-IID（α=1）
- **本地批次大小**: 64
- **学习率**: 0.01
- **设备**: CUDA GPU（如可用）

### 攻击场景
- **攻击类型**: 标签翻转攻击
- **恶意行为率**: 100%（攻击者完全恶意）
- **源类别→目标类别**: 3→5（可在config.py中修改）

## 📈 结果分析

### 日志文件
实验结果保存在 `log/` 目录下，文件命名格式：
```
{algorithm}_{distribution}_source{source}_target{target}_atr{attack_ratio}_{timestamp}.log
```

### 结果分析工具
使用 `result_analyzer.py` 来分析实验结果：
```bash
python result_analyzer.py
```

### 可视化
- `quick_view.py`: 快速查看实验结果
- `Experiments_MNIST.ipynb`: Jupyter notebook 交互式分析

## 🔧 核心算法原理

### LFighter 基础算法
1. **梯度差异计算**: 计算全局模型与本地模型的参数差异
2. **关键类别识别**: 基于梯度变化识别最可能被攻击的类别
3. **聚类分析**: 使用K-means将参与者分为两个群体
4. **群体评估**: 通过余弦相似度评估群体内部一致性
5. **权重分配**: 为可信群体分配权重，过滤恶意更新

### LFighter-MV 多视图算法
- **特征提取**: 从神经网络的多个层提取特征表示
- **视图融合**: 智能融合不同层的特征信息
- **增强检测**: 通过多视角提高攻击检测准确性

### LFighter-DBO 算法
- **深度优化**: 集成深度双向优化网络
- **自适应聚合**: 基于网络状态动态调整聚合策略
- **高效计算**: 优化的计算流程，适合大规模部署

## 📁 项目结构

```
LFighter/
├── config.py                 # 配置文件
├── environment_federated.py  # 联邦学习环境
├── experiment_federated.py   # 实验执行框架
├── models.py                 # 神经网络模型定义
├── aggregation.py            # 聚合算法实现
├── datasets.py              # 数据集处理
├── sampling.py              # 数据采样策略
├── utils.py                 # 工具函数
├── lfighter.py              # LFighter算法入口
├── lfighter_mv.py           # 多视图版本
├── lfighter_dbo.py          # DBO版本
├── lfighter_mv_dbo.py       # 完整版本
├── fed_avg.py               # FedAvg基准
├── result_analyzer.py       # 结果分析工具
├── quick_view.py            # 快速查看工具
├── requirements.txt         # 依赖包列表
├── data/                    # 数据集目录
├── log/                     # 实验日志
├── results/                 # 实验结果
├── checkpoints/             # 模型检查点
└── figures/                 # 生成的图表
```

## 🛡️ 安全特性

- **隐私保护**: 仅交换模型参数，不暴露原始数据
- **鲁棒性**: 对各种攻击强度具有良好的抵御能力
- **可扩展性**: 支持大规模联邦学习场景
- **实时检测**: 在线检测和防御恶意行为


## 🤝 贡献指南

欢迎提交问题报告、功能请求或代码贡献：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

