# LFighter: 联邦学习中的标签翻转攻击防御

本项目实现了LFighter算法及其增强版本，用于检测联邦学习中的恶意客户端。

## 🎯 可视化功能

### LFighter-Autoencoder 可视化

LFighter-Autoencoder算法提供6种可视化方法：

1. **Autoencoder训练过程** (`ae_training_round_X.png`)
   - 训练损失随epoch变化的曲线
   - 监控模型收敛情况

2. **特征空间对比** (`feature_space_round_X.png`)
   - 2×2子图：原始vs潜在特征空间
   - 客户端类型vs聚类结果对比
   - t-SNE降维可视化

3. **重构误差分析** (`reconstruction_errors_round_X.png`)
   - 重构误差分布直方图
   - 客户端重构误差条形图

4. **客户端得分分析** (`client_scores_round_X.png`)
   - 聚类得分、重构得分、最终得分
   - 多维度得分对比散点图

5. **聚类质量对比** (`cluster_quality_round_X.png`)
   - 两个聚类的相异性得分对比

6. **总结报告** (`summary_report_round_X.txt`)
   - 详细的算法参数和性能指标

### LFighter (原版) 可视化

原版LFighter算法现已内置可视化功能：

1. **特征空间可视化** (`lfighter_feature_space_round_X.png`)
   - 1×2子图：客户端类型 vs 聚类结果
   - 仅显示原始特征空间（无潜在空间）

2. **聚类质量对比** (`lfighter_cluster_quality_round_X.png`)
   - 与LFighter-AE相同的聚类质量可视化

3. **客户端得分分析** (`lfighter_client_scores_round_X.png`)
   - 显示每个客户端的聚类得分
   - 包含得分分布直方图

4. **总结报告** (`lfighter_summary_report_round_X.txt`)
   - 详细的算法参数和性能指标

## 📊 使用方法

### 1. LFighter-Autoencoder 可视化

```python
from aggregation import LFighterAutoencoder

# 创建带可视化的LFighter-AE聚合器
aggregator = LFighterAutoencoder(
    num_classes=10,                  # 分类数量
    enable_visualization=True,       # 启用可视化
    save_path="./figures/",         # 保存路径
    visualization_frequency=1,       # 每轮都保存
    max_visualizations=0,           # 不限制文件数量
    save_final_only=False           # 不只保存最后一轮
)

# 在聚合过程中自动生成可视化
global_weights = aggregator.aggregate(global_model, local_models, ptypes)
```

### 2. 原版LFighter 可视化

```python
from aggregation import LFD

# 创建带可视化的LFighter聚合器（原版）
aggregator = LFD(
    num_classes=10,                  # 分类数量
    enable_visualization=True,       # 启用可视化
    save_path="./figures/",         # 保存路径
    visualization_frequency=1,       # 每轮都保存
    max_visualizations=0,           # 不限制文件数量
    save_final_only=False           # 不只保存最后一轮
)

# 在聚合过程中自动生成可视化
global_weights = aggregator.aggregate(global_model, local_models, ptypes)
```

### 3. 对比两种算法

```python
from aggregation import LFD, LFighterAutoencoder

# 创建两个聚合器进行对比
lfighter_original = LFD(num_classes=10, enable_visualization=True)
lfighter_ae = LFighterAutoencoder(num_classes=10, enable_visualization=True)
    
# 使用相同数据测试
weights_original = lfighter_original.aggregate(global_model, local_models, ptypes)
weights_ae = lfighter_ae.aggregate(global_model, local_models, ptypes)

# 对比生成的可视化文件：
# - lfighter_*_round_1.png (原版)
# - ae_*, feature_space_*, client_scores_*, etc. (AE版本)
```

## 🛠️ 可视化控制参数

所有可视化功能支持以下控制参数：

- `enable_visualization=True/False`: 启用/禁用可视化
- `save_path="./figures/"`: 图片和报告保存路径
- `visualization_frequency=1`: 可视化频率（每N轮保存一次）
- `max_visualizations=0`: 最大保存文件数量（0=不限制）
- `save_final_only=False`: 是否只保存最后一轮的可视化

### 示例配置

```python
# 每3轮保存一次，最多保留10组文件
aggregator = LFD(
    num_classes=10,
    enable_visualization=True,
    visualization_frequency=3,
    max_visualizations=10
)

# 只保存最后一轮的结果
aggregator = LFD(
    num_classes=10,
    enable_visualization=True,
    save_final_only=True
)
aggregator.set_total_rounds(50)  # 设置总轮数
```

## 📈 对比分析

使用可视化功能可以直接对比LFighter-AE和原版LFighter的性能差异：

1. **特征表示能力**: 对比原始特征空间vs潜在特征空间
2. **聚类质量**: 比较两种算法的聚类相异性得分
3. **检测准确率**: 通过总结报告对比选择的好客户端数量
4. **算法稳定性**: 观察训练过程和得分分布的差异

## 📋 技术细节

### 可视化技术栈
- **t-SNE降维**: 高维特征的2D投影可视化
- **matplotlib/seaborn**: 图表生成和美化
- **智能perplexity**: 自动调整处理小样本情况
- **300 DPI输出**: 高质量PNG图片
- **统一色彩方案**: 红色(恶意) vs 蓝色(良性)

### 性能优化
- **条件生成**: 只在指定轮次生成可视化
- **自动清理**: 限制文件数量，自动删除旧文件
- **异常处理**: 处理空聚类、维度不足等边界情况
- **内存管理**: 及时释放图像资源

## 🔧 故障排除

### 常见问题

1. **t-SNE失败**: 自动回退到PCA降维
2. **空聚类**: 特殊处理，赋予最差质量分数
3. **样本数不足**: 智能调整perplexity参数
4. **内存不足**: 使用Agg后端，支持无GUI环境

### 依赖要求

确保安装以下依赖包：
```bash
pip install matplotlib seaborn scikit-learn
```

## 🎨 可视化文件说明

### 文件命名规范
- **LFighter-AE**: `ae_training_`, `feature_space_`, `client_scores_`, `cluster_quality_`, `reconstruction_errors_`, `summary_report_`
- **原版LFighter**: `lfighter_feature_space_`, `lfighter_cluster_quality_`, `lfighter_client_scores_`, `lfighter_summary_report_`

### 颜色编码
- **红色**: 恶意客户端
- **蓝色**: 良性客户端  
- **绿色**: 好的聚类
- **橙色/绿色**: 不同聚类结果

