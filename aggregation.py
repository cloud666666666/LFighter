import copy
import enum
from sklearn.decomposition import PCA
import torch
import numpy as np
import math
from scipy import stats
from functools import reduce
import time
import sklearn.metrics.pairwise as smp
import hdbscan
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.cluster import KMeans
from utils import *
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
# 添加可视化相关导入
import matplotlib
matplotlib.use('Agg')  # 设置后端，支持无GUI环境
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
from matplotlib.backends.backend_pdf import PdfPages

# 设置matplotlib和seaborn样式
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def get_pca(data, threshold = 0.99):
    normalized_data = StandardScaler().fit_transform(data)
    pca = PCA()
    reduced_data = pca.fit_transform(normalized_data)
    # Determine explained variance using explained_variance_ration_ attribute
    exp_var = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var)
    select_pcas = np.where(cum_sum_eigenvalues <=threshold)[0]
    # print('Number of components with variance <= {:0.0f}%: {}'.format(threshold*100, len(select_pcas)))
    reduced_data = reduced_data[:, select_pcas]
    return reduced_data

eps = np.finfo(float).eps

class LFD():
    def __init__(self, num_classes, enable_visualization=True, save_path="./figures/", 
                 visualization_frequency=1, max_visualizations=0, save_final_only=False,
                 save_as_pdf=True, keep_individual_files=False, attack_ratio=None):
        """
        Args:
            save_as_pdf: 是否保存为PDF格式（默认True）
            keep_individual_files: 是否保留单独的PNG文件（默认False）
            attack_ratio: 攻击者比率，用于PDF文件名
        """
        self.num_classes = num_classes
        self.memory = np.zeros(num_classes)
        
        # 可视化相关参数
        self.enable_visualization = enable_visualization
        self.save_path = save_path
        self.visualization_frequency = visualization_frequency
        self.max_visualizations = max_visualizations
        self.save_final_only = save_final_only
        self.save_as_pdf = save_as_pdf
        self.keep_individual_files = keep_individual_files
        self.attack_ratio = attack_ratio
        
        # 轮数计数器
        self.round_counter = 0
        self.total_rounds = None
        self.visualization_count = 0
        
        # PDF文件管理
        self.pdf_pages = None
        self.pdf_filename = None
        
        # 确保保存目录存在
        if self.enable_visualization:
            os.makedirs(self.save_path, exist_ok=True)
            
            # 创建PDF文件（如果启用PDF保存）
            if self.save_as_pdf:
                self._initialize_pdf()
    
    def _initialize_pdf(self):
        """初始化PDF文件"""
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            attack_str = f"_atr{self.attack_ratio:.1f}" if self.attack_ratio is not None else ""
            self.pdf_filename = f'{self.save_path}/lfighter_complete_report{attack_str}.pdf'
            self.pdf_pages = PdfPages(self.pdf_filename)
            print(f"[LFighter] 🔗 PDF报告初始化成功")
            print(f"[LFighter] 📄 实时查看路径: {self.pdf_filename}")
            print(f"[LFighter] 💡 提示: 每个epoch后PDF会自动更新，可随时查看")
        except Exception as e:
            print(f"[LFighter] PDF初始化失败: {e}")
            self.pdf_pages = None
    
    def finalize_pdf(self):
        """关闭PDF文件"""
        if self.pdf_pages is not None:
            try:
                self.pdf_pages.close()
                print(f"[LFighter] PDF报告已保存: {self.pdf_filename}")
            except Exception as e:
                print(f"[LFighter] PDF关闭失败: {e}")
            finally:
                self.pdf_pages = None
    
    def set_total_rounds(self, total_rounds):
        """设置总训练轮数，用于save_final_only模式"""
        self.total_rounds = total_rounds
        if self.enable_visualization:
            print(f"[LFighter] Set total rounds to {total_rounds} for visualization control")
    
    def should_visualize_this_round(self):
        """判断当前轮次是否应该保存可视化"""
        if not self.enable_visualization:
            return False
            
        # 如果设置为只保存最后一轮
        if self.save_final_only:
            return self.total_rounds is not None and self.round_counter == self.total_rounds
        
        # 动态调整可视化频率：前20轮每轮可视化，后续每10轮一次
        if self.round_counter <= 20:
            current_frequency = 1
        else:
            current_frequency = 10
        
        # 按动态频率保存
        return self.round_counter % current_frequency == 0
    
    def cleanup_old_visualizations(self):
        """清理旧的可视化文件"""
        if not self.enable_visualization:
            return
        
        if self.max_visualizations > 0 and self.visualization_count > self.max_visualizations:
            # 清理PNG文件
            png_files = [f for f in os.listdir(self.save_path) if f.startswith('lfighter_') and f.endswith('.png')]
            png_files.sort(key=lambda x: os.path.getctime(os.path.join(self.save_path, x)))
            
            while len(png_files) > self.max_visualizations:
                old_file = os.path.join(self.save_path, png_files.pop(0))
                if os.path.exists(old_file):
                    os.remove(old_file)
            
            # 清理文本文件
            txt_files = [f for f in os.listdir(self.save_path) if f.startswith('lfighter_') and f.endswith('.txt')]
            txt_files.sort(key=lambda x: os.path.getctime(os.path.join(self.save_path, x)))
            
            while len(txt_files) > self.max_visualizations:
                old_file = os.path.join(self.save_path, txt_files.pop(0))
                if os.path.exists(old_file):
                    os.remove(old_file)
    
    def create_pdf_report(self, round_num, features, labels, ptypes, scores, cs0, cs1, good_cl, metrics):
        """向已有PDF文件添加当前轮次的可视化页面"""
        if not self.should_visualize_this_round() or not self.save_as_pdf or self.pdf_pages is None:
            return
        
        try:
            # 第一页：特征空间可视化
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # t-SNE降维
            if features.shape[1] > 2:
                try:
                    perplexity = min(30, len(features)-1)
                    if perplexity < 1:
                        perplexity = 1
                    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                    features_2d = tsne.fit_transform(features)
                except:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2, random_state=42)
                    features_2d = pca.fit_transform(features)
            else:
                features_2d = features
            
            # 按客户端类型着色
            colors = []
            malicious_count = 0
            for ptype in ptypes:
                ptype_str = str(ptype).lower()
                if ('malicious' in ptype_str or 'attack' in ptype_str or 
                    'bad' in ptype_str or 'adversarial' in ptype_str):
                    colors.append('red')
                    malicious_count += 1
                else:
                    colors.append('blue')
            
            axes[0].scatter(features_2d[:, 0], features_2d[:, 1], c=colors, alpha=0.7, s=100)
            axes[0].set_title('LFighter: Feature Space (by Client Type)', fontsize=14)
            axes[0].set_xlabel('Dimension 1')
            axes[0].set_ylabel('Dimension 2')
            
            red_patch = plt.matplotlib.patches.Patch(color='red', label='Malicious')
            blue_patch = plt.matplotlib.patches.Patch(color='blue', label='Benign')
            axes[0].legend(handles=[red_patch, blue_patch])
            
            # 按聚类结果着色
            cluster_colors = ['orange', 'green']
            for i in range(len(features_2d)):
                axes[1].scatter(features_2d[i, 0], features_2d[i, 1], 
                               c=cluster_colors[labels[i]], alpha=0.7, s=100)
            axes[1].set_title('LFighter: Feature Space (by Cluster)', fontsize=14)
            axes[1].set_xlabel('Dimension 1')
            axes[1].set_ylabel('Dimension 2')
            
            orange_patch = plt.matplotlib.patches.Patch(color='orange', label='Cluster 0')
            green_patch = plt.matplotlib.patches.Patch(color='green', label='Cluster 1')
            axes[1].legend(handles=[orange_patch, green_patch])
            
            plt.suptitle(f'LFighter Algorithm - Round {round_num} - Feature Space Analysis', fontsize=16)
            plt.tight_layout()
            self.pdf_pages.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # 第二页：聚类质量和客户端得分
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 聚类质量
            clusters = ['Cluster 0', 'Cluster 1']
            dissimilarities = [cs0, cs1]
            cluster_quality_colors = ['green' if i == good_cl else 'red' for i in range(2)]
            
            bars = axes[0, 0].bar(clusters, dissimilarities, color=cluster_quality_colors, alpha=0.7)
            axes[0, 0].set_ylabel('Dissimilarity Score')
            axes[0, 0].set_title('Cluster Quality Comparison')
            axes[0, 0].grid(True, alpha=0.3)
            
            for i, (bar, score) in enumerate(zip(bars, dissimilarities)):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              f'{score:.4f}', ha='center', va='bottom')
            
            axes[0, 0].text(good_cl, dissimilarities[good_cl] + 0.05, '✓ Good Cluster', 
                           ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # 客户端得分条形图
            client_indices = range(len(scores))
            bars = axes[0, 1].bar(client_indices, scores, color=colors, alpha=0.7)
            axes[0, 1].set_title('Client Scores')
            axes[0, 1].set_xlabel('Client Index')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_ylim(0, 1.1)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 得分分布直方图
            axes[1, 0].hist(scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 0].axvline(np.mean(scores), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(scores):.4f}')
            axes[1, 0].axvline(np.median(scores), color='orange', linestyle='--', 
                              label=f'Median: {np.median(scores):.4f}')
            axes[1, 0].set_xlabel('Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Score Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 性能指标文本
            axes[1, 1].axis('off')
            attack_ratio_str = f"Attack Ratio: {self.attack_ratio:.1f}" if self.attack_ratio is not None else "Attack Ratio: N/A"
            metrics_text = f"""
Performance Metrics - Round {round_num}

{attack_ratio_str}
Total Clients: {metrics.get('total_clients', 'N/A')}
Good Clients Selected: {metrics.get('good_clients', 'N/A')}
Selection Accuracy: {metrics.get('accuracy', 'N/A'):.2%}

Cluster Analysis:
• Cluster 0 Dissimilarity: {metrics.get('cs0', 'N/A'):.4f}
• Cluster 1 Dissimilarity: {metrics.get('cs1', 'N/A'):.4f}
• Good Cluster: {metrics.get('good_cluster', 'N/A')}

Feature Processing:
• Reduction Method: {metrics.get('reduction_method', 'N/A')}
• Detected Anomalous Classes: {metrics.get('anomalous_classes', 'N/A')}

Algorithm: Original LFighter (K-means clustering)
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
            axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            plt.suptitle(f'LFighter Algorithm - Round {round_num} - Performance Analysis', fontsize=16)
            plt.tight_layout()
            self.pdf_pages.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # 强制刷新PDF文件，确保实时可见
            try:
                # matplotlib PdfPages 的正确flush方法
                if hasattr(self.pdf_pages, '_file') and hasattr(self.pdf_pages._file, '_file'):
                    self.pdf_pages._file._file.flush()
                    import os
                    os.fsync(self.pdf_pages._file._file.fileno())
                elif hasattr(self.pdf_pages, 'flush'):
                    self.pdf_pages.flush()
            except:
                pass  # 忽略flush错误，不影响正常功能
            
            print(f"[LFighter] Round {round_num} 添加到PDF报告 - 实时可查看: {self.pdf_filename}")
            
        except Exception as e:
            print(f"[LFighter] PDF页面添加失败 Round {round_num}: {e}")
    
    def visualize_feature_space(self, features, labels, ptypes, round_num):
        """可视化原始特征空间"""
        if not self.should_visualize_this_round():
            return
        
        # 调试信息：打印客户端类型
        if self.enable_visualization and round_num == 1:  # 只在第一轮打印
            print(f"[LFighter Debug] Client types: {ptypes[:10]}...")  # 打印前10个客户端类型
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # t-SNE降维可视化
        if features.shape[1] > 2:
            try:
                perplexity = min(30, len(features)-1)
                if perplexity < 1:
                    perplexity = 1
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                features_2d = tsne.fit_transform(features)
            except:
                # 如果t-SNE失败，使用PCA
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2, random_state=42)
                features_2d = pca.fit_transform(features)
        else:
            features_2d = features
        
        # 1. 按客户端类型着色 - 增强检测逻辑
        colors = []
        malicious_count = 0
        for ptype in ptypes:
            ptype_str = str(ptype).lower()
            if ('malicious' in ptype_str or 'attack' in ptype_str or 
                'bad' in ptype_str or 'adversarial' in ptype_str):
                colors.append('red')
                malicious_count += 1
            else:
                colors.append('blue')
        
        # 调试信息：打印检测结果
        if self.enable_visualization and round_num == 1:
            print(f"[LFighter Debug] Detected {malicious_count}/{len(ptypes)} malicious clients")
        
        axes[0].scatter(features_2d[:, 0], features_2d[:, 1], c=colors, alpha=0.7, s=100)
        axes[0].set_title('LFighter: Feature Space (by Client Type)')
        axes[0].set_xlabel('Dimension 1')
        axes[0].set_ylabel('Dimension 2')
        
        # 添加图例
        red_patch = plt.matplotlib.patches.Patch(color='red', label='Malicious')
        blue_patch = plt.matplotlib.patches.Patch(color='blue', label='Benign')
        axes[0].legend(handles=[red_patch, blue_patch])
        
        # 2. 按聚类结果着色
        cluster_colors = ['orange', 'green']
        for i in range(len(features_2d)):
            axes[1].scatter(features_2d[i, 0], features_2d[i, 1], 
                           c=cluster_colors[labels[i]], alpha=0.7, s=100)
        axes[1].set_title('LFighter: Feature Space (by Cluster)')
        axes[1].set_xlabel('Dimension 1')
        axes[1].set_ylabel('Dimension 2')
        
        # 添加聚类图例
        orange_patch = plt.matplotlib.patches.Patch(color='orange', label='Cluster 0')
        green_patch = plt.matplotlib.patches.Patch(color='green', label='Cluster 1')
        axes[1].legend(handles=[orange_patch, green_patch])
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/lfighter_feature_space_round_{round_num}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_cluster_quality(self, cs0, cs1, good_cl, round_num):
        """可视化聚类质量"""
        if not self.should_visualize_this_round():
            return
            
        # 只在PDF报告中添加聚类质量可视化，不生成单独的图片文件
        if self.save_as_pdf and self.pdf_pages is not None:
            plt.figure(figsize=(10, 6))
            clusters = ['Cluster 0', 'Cluster 1']
            dissimilarities = [cs0, cs1]
            colors = ['green' if i == good_cl else 'red' for i in range(2)]
            
            plt.bar(clusters, dissimilarities, color=colors, alpha=0.7)
            plt.ylabel('Dissimilarity Score')
            plt.title('Cluster Quality Comparison')
            plt.grid(True, alpha=0.3)
            
            # 添加图例
            green_patch = plt.matplotlib.patches.Patch(color='green', label='Good Cluster')
            red_patch = plt.matplotlib.patches.Patch(color='red', label='Bad Cluster')
            plt.legend(handles=[green_patch, red_patch])
            
            try:
                self.pdf_pages.savefig(plt.gcf(), bbox_inches='tight')
            except Exception as e:
                print(f"[LFighter-AE] 聚类质量可视化添加到PDF失败: {e}")
            plt.close()
    
    def visualize_client_scores(self, scores, ptypes, round_num):
        """可视化客户端得分"""
        if not self.should_visualize_this_round():
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        client_indices = range(len(scores))
        
        # 增强的客户端类型检测逻辑
        colors = []
        malicious_count = 0
        for ptype in ptypes:
            ptype_str = str(ptype).lower()
            if ('malicious' in ptype_str or 'attack' in ptype_str or 
                'bad' in ptype_str or 'adversarial' in ptype_str):
                colors.append('red')
                malicious_count += 1
            else:
                colors.append('blue')
        
        # 调试信息：打印检测结果
        if self.enable_visualization and round_num == 1:
            print(f"[LFighter Client Scores] Detected {malicious_count}/{len(ptypes)} malicious clients")
        
        # 1. 客户端得分条形图
        bars = axes[0].bar(client_indices, scores, color=colors, alpha=0.7)
        axes[0].set_title('LFighter: Client Scores')
        axes[0].set_xlabel('Client Index')
        axes[0, 1].set_ylim(0, 1.1)
        axes[0].grid(True, alpha=0.3)
        
        # 添加图例
        red_patch = plt.matplotlib.patches.Patch(color='red', label='Malicious')
        blue_patch = plt.matplotlib.patches.Patch(color='blue', label='Benign')
        axes[0].legend(handles=[red_patch, blue_patch])
        
        # 2. 得分分布直方图
        axes[1].hist(scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1].axvline(np.mean(scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(scores):.4f}')
        axes[1].axvline(np.median(scores), color='orange', linestyle='--', 
                       label=f'Median: {np.median(scores):.4f}')
        axes[1].set_xlabel('Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('LFighter: Score Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/lfighter_client_scores_round_{round_num}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_report(self, round_num, metrics):
        """创建总结报告"""
        if not self.should_visualize_this_round():
            return
        
        # 只在PDF报告中添加总结报告，不生成单独的文本文件
        if self.save_as_pdf and self.pdf_pages is not None:
            # 创建文本报告
            report_content = f"""
LFighter-Autoencoder Algorithm Summary Report
==========================================
Round: {round_num}
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}

Algorithm Parameters:
- Hidden Dimension: {self.ae_hidden_dim}
- Latent Dimension: {self.ae_latent_dim}
- Training Epochs: {self.ae_epochs}
- Reconstruction Weight: {self.reconstruction_weight}

Performance Metrics:
- Total Clients: {metrics.get('total_clients', 'N/A')}
- Good Clients Selected: {metrics.get('good_clients', 'N/A')}
- Selection Accuracy: {metrics.get('accuracy', 'N/A'):.2%}
- Cluster 0 Dissimilarity: {metrics.get('cs0', 'N/A'):.4f}
- Cluster 1 Dissimilarity: {metrics.get('cs1', 'N/A'):.4f}
- Good Cluster: {metrics.get('good_cluster', 'N/A')}
- Mean Reconstruction Error: {metrics.get('mean_recon_error', 'N/A'):.4f}
- Std Reconstruction Error: {metrics.get('std_recon_error', 'N/A'):.4f}

Feature Processing:
- Feature Strategy: {metrics.get('feature_strategy', '使用全部输出层梯度')}
- Original Feature Dimension: {metrics.get('feature_dim', 'N/A')}
- Dimension Reduction Method: {metrics.get('reduction_method', 'N/A')}
- Final Loss: {metrics.get('final_loss', 'N/A'):.6f}

Attack Analysis:
- Attack Type: {metrics.get('attack_type', 'N/A')}
- Attack Scope: {metrics.get('attack_scope', 'N/A')}
- Attack Description: {metrics.get('attack_description', 'N/A')}
- Confidence: {metrics.get('attack_confidence', 'N/A'):.2f}
"""
            
            # 创建一个文本图像，添加到PDF
            fig = plt.figure(figsize=(12, 10))
            plt.text(0.1, 0.5, report_content, fontsize=10, family='monospace')
            plt.axis('off')
            
            try:
                self.pdf_pages.savefig(fig, bbox_inches='tight')
            except Exception as e:
                print(f"[LFighter-AE] 总结报告添加到PDF失败: {e}")
            plt.close()
    
    def clusters_dissimilarity(self, clusters):
        """计算聚类间相异性，处理空聚类情况"""
        n0 = len(clusters[0])
        n1 = len(clusters[1])
        m = n0 + n1 
        
        # 处理空聚类情况
        if n0 == 0:
            return 1.0, 0.0  # 空聚类0，聚类1更好
        if n1 == 0:
            return 0.0, 1.0  # 空聚类1，聚类0更好
        if n0 == 1:
            ds0 = 1.0  # 单样本聚类质量设为最差
        else:
            cs0 = smp.cosine_similarity(clusters[0]) - np.eye(n0)
            mincs0 = np.min(cs0, axis=1)
            ds0 = n0/m * (1 - np.mean(mincs0))
        
        if n1 == 1:
            ds1 = 1.0  # 单样本聚类质量设为最差
        else:
            cs1 = smp.cosine_similarity(clusters[1]) - np.eye(n1)
            mincs1 = np.min(cs1, axis=1)
            ds1 = n1/m * (1 - np.mean(mincs1))
        
        return ds0, ds1

    def aggregate(self, global_model, local_models, ptypes):
        local_weights = [copy.deepcopy(model).state_dict() for model in local_models]
        m = len(local_models)
        
        # 可视化模式下增加轮数计数
        if self.enable_visualization:
            self.round_counter += 1
            # 提示当前可视化频率
            current_freq = 1 if self.round_counter <= 20 else 10
            next_viz_round = self.round_counter if self.round_counter % current_freq == 0 else ((self.round_counter // current_freq) + 1) * current_freq
            print(f"[LFighter] Round {self.round_counter} - Visualization freq: every {current_freq} rounds (next: round {next_viz_round})")
        
        for i in range(m):
            local_models[i] = list(local_models[i].parameters())
        global_model = list(global_model.parameters())
        dw = [None for i in range(m)]
        db = [None for i in range(m)]
        for i in range(m):
            dw[i]= global_model[-2].cpu().data.numpy() - \
                local_models[i][-2].cpu().data.numpy() 
            db[i]= global_model[-1].cpu().data.numpy() - \
                local_models[i][-1].cpu().data.numpy()
        dw = np.asarray(dw)
        db = np.asarray(db)

        "If one class or two classes classification model"
        if len(db[0]) <= 2:
            data = []
            for i in range(m):
                data.append(dw[i].reshape(-1))
        
            kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
            labels = kmeans.labels_

            clusters = {0:[], 1:[]}
            for i, l in enumerate(labels):
                clusters[l].append(data[i])

            good_cl = 0
            cs0, cs1 = self.clusters_dissimilarity(clusters)
            if cs0 < cs1:
                good_cl = 1

            scores = np.ones([m])
            for i, l in enumerate(labels):
                if l != good_cl:
                    scores[i] = 0
            
            # 可视化结果
            if self.enable_visualization:
                print(f'[LFighter] Cluster quality: cs0={cs0:.4f}, cs1={cs1:.4f}, good_cluster={good_cl}')
                print(f'[LFighter] Selected good clients: {np.sum(scores)}/{m}')
                
                if self.should_visualize_this_round():
                    # 创建总结报告
                    metrics = {
                        'total_clients': m,
                        'good_clients': int(np.sum(scores)),
                        'accuracy': int(np.sum(scores)) / m,
                        'cs0': cs0,
                        'cs1': cs1,
                        'good_cluster': good_cl,
                        'reduction_method': 'Binary classification',
                        'anomalous_classes': 'N/A (Binary)'
                    }
                    
                    # 生成PDF报告（如果启用）
                    self.create_pdf_report(self.round_counter, np.array(data), labels, ptypes, scores, cs0, cs1, good_cl, metrics)
                    
                    # 生成单独的PNG文件（如果需要）
                    if self.keep_individual_files:
                        self.visualize_feature_space(np.array(data), labels, ptypes, self.round_counter)
                        self.visualize_cluster_quality(cs0, cs1, good_cl, self.round_counter)
                        self.visualize_client_scores(scores, ptypes, self.round_counter)
                        self.create_summary_report(self.round_counter, metrics)
                    
                    # 清理旧的可视化文件
                    self.cleanup_old_visualizations()
                
            global_weights = average_weights(local_weights, scores)
            return global_weights

        "For multiclassification models"
        norms = np.linalg.norm(dw, axis = -1) 
        self.memory = np.sum(norms, axis = 0)
        self.memory +=np.sum(abs(db), axis = 0)
        max_two_freq_classes = self.memory.argsort()[-2:]
        print('Potential source and target classes:', max_two_freq_classes)
        data = []
        for i in range(m):
            data.append(dw[i][max_two_freq_classes].reshape(-1))

        # === 统一降维处理：确保消融实验公平性 ===
        def unified_dimension_reduction(features, target_dim=200, method='auto', standardize=True):
            """统一降维函数：与多视图算法保持一致"""
            features_array = np.array(features)
            n_samples, original_dim = features_array.shape
            
            # 1. 标准化处理
            if standardize:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                features_array = scaler.fit_transform(features_array)
                std_info = "standardized"
            else:
                std_info = "raw"
            
            # 2. 智能降维策略
            max_pca_components = min(n_samples, original_dim)
            effective_target_dim = min(target_dim, max_pca_components)
            
            if original_dim <= effective_target_dim:
                return features_array, f"keep_all_{original_dim}_{std_info}"
            elif original_dim <= effective_target_dim * 2:
                return features_array[:, :effective_target_dim], f"truncate_{effective_target_dim}_{std_info}"
            else:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=effective_target_dim, random_state=42)
                reduced_features = pca.fit_transform(features_array)
                explained_ratio = np.sum(pca.explained_variance_ratio_)
                return reduced_features, f"pca_{effective_target_dim}_var{explained_ratio:.3f}_{std_info}"
        
        # 应用统一降维（与多视图算法保持一致）
        data_reduced, reduction_method = unified_dimension_reduction(data, target_dim=200)
        print(f'[LFighter] Applied unified dimension reduction: {reduction_method}')
        
        kmeans = KMeans(n_clusters=2, random_state=0).fit(data_reduced)
        labels = kmeans.labels_

        clusters = {0:[], 1:[]}
        for i, l in enumerate(labels):
          clusters[l].append(data_reduced[i])

        good_cl = 0
        cs0, cs1 = self.clusters_dissimilarity(clusters)
        if cs0 < cs1:
            good_cl = 1

        scores = np.ones([m])
        for i, l in enumerate(labels):
            if l != good_cl:
                scores[i] = 0
        
        # 可视化结果
        if self.enable_visualization:
            print(f'[LFighter] Cluster quality: cs0={cs0:.4f}, cs1={cs1:.4f}, good_cluster={good_cl}')
            print(f'[LFighter] Selected good clients: {np.sum(scores)}/{m}')
            
            if self.should_visualize_this_round():
                # 创建总结报告
                metrics = {
                    'total_clients': m,
                    'good_clients': int(np.sum(scores)),
                    'accuracy': int(np.sum(scores)) / m,
                    'cs0': cs0,
                    'cs1': cs1,
                    'good_cluster': good_cl,
                    'reduction_method': reduction_method,
                    'anomalous_classes': str(max_two_freq_classes)
                }
                
                # 生成PDF报告（如果启用）
                self.create_pdf_report(self.round_counter, data_reduced, labels, ptypes, scores, cs0, cs1, good_cl, metrics)
                
                # 生成单独的PNG文件（如果需要）
                if self.keep_individual_files:
                    self.visualize_feature_space(data_reduced, labels, ptypes, self.round_counter)
                    self.visualize_cluster_quality(cs0, cs1, good_cl, self.round_counter)
                    self.visualize_client_scores(scores, ptypes, self.round_counter)
                    self.create_summary_report(self.round_counter, metrics)
                
                # 清理旧的可视化文件
                self.cleanup_old_visualizations()
            
        global_weights = average_weights(local_weights, scores)
        return global_weights


################################################
# Takes in grad
# Compute similarity
# Get weightings
def foolsgold(grads):
    n_clients = grads.shape[0]
    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1)
    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    return wv


class FoolsGold:
    def __init__(self, num_peers):
        self.memory = None
        self.wv_history = []
        self.num_peers = num_peers
       
    def score_gradients(self, local_grads, selectec_peers):
        m = len(local_grads)
        grad_len = np.array(local_grads[0][-2].cpu().data.numpy().shape).prod()
        if self.memory is None:
            self.memory = np.zeros((self.num_peers, grad_len))

        grads = np.zeros((m, grad_len))
        for i in range(m):
            grads[i] = np.reshape(local_grads[i][-2].cpu().data.numpy(), (grad_len))
        self.memory[selectec_peers]+= grads
        wv = foolsgold(self.memory)  # Use FG
        self.wv_history.append(wv)
        return wv[selectec_peers]


#######################################################################################



class LFighterDBO:
    def __init__(self):
        pass
    
    def aggregate(self, simulation_model, local_weights, local_features=None):
        import config
        device = simulation_model.device if hasattr(simulation_model, 'device') else config.DEVICE
        
        import torch
        from models import DBONet
        from sklearn.neighbors import kneighbors_graph
        
        # 直接在这里定义归一化函数，避免循环导入
        def normalization(data):
            """数据归一化函数"""
            maxVal = torch.max(data)
            minVal = torch.min(data)
            data = (data - minVal) / (maxVal - minVal + 1e-10)  # 避免除零
            return data

        def standardization(data):
            """数据标准化函数"""
            rowSum = torch.sqrt(torch.sum(data**2, 1))
            repMat = rowSum.repeat((data.shape[1], 1)) + 1e-10
            data = torch.div(data, repMat.t())
            return data
        
        m = len(local_weights)
        print(f"[LFighter-DBO] Processing {m} clients with lightweight DBONet")
        
        # 只提取输出层梯度特征
        local_models = []
        for local_weight in local_weights:
            model = copy.deepcopy(simulation_model)
            model.load_state_dict(local_weight)
            local_models.append(list(model.parameters()))
        
        global_model = list(simulation_model.parameters())
        
        # 使用与LFighter一致的关键类别检测和提取
        # 首先检测最异常的两个类别（与LFighter逻辑完全一致）
        memory = np.zeros(global_model[-1].shape[0])  # 输出层偏置的维度
        
        for i in range(m):
            # 计算权重和偏置的差异
            dw = global_model[-2].cpu().data.numpy() - local_models[i][-2].cpu().data.numpy()
            db = global_model[-1].cpu().data.numpy() - local_models[i][-1].cpu().data.numpy()
            
            # 累积异常程度（与原版LFighter的逻辑一致）
            norms = np.linalg.norm(dw, axis=-1)
            memory += norms + np.abs(db)
        
        # 找到最异常的两个类别
        max_two_freq_classes = memory.argsort()[-2:]
        print(f'[LFighter-DBO] Detected anomalous classes (same as LFighter): {max_two_freq_classes}')
        
        # 只提取这两个类别的输出层梯度（与LFighter完全一致）
        gradients = []
        for i in range(m):
            dw = global_model[-2].cpu().data.numpy() - local_models[i][-2].cpu().data.numpy()
            key_classes_grad = dw[max_two_freq_classes].reshape(-1)
            gradients.append(key_classes_grad)
        
        feature_matrix = np.array(gradients)
        print(f"[LFighter-DBO] Gradient matrix shape: {feature_matrix.shape}")
        
        # === 统一降维处理：确保消融实验公平性 ===
        def unified_dimension_reduction(features, target_dim=200, method='auto', standardize=True):
            """统一降维函数：与多视图算法保持一致"""
            features_array = np.array(features)
            n_samples, original_dim = features_array.shape
            
            # 1. 标准化处理
            if standardize:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                features_array = scaler.fit_transform(features_array)
                std_info = "standardized"
            else:
                std_info = "raw"
            
            # 2. 智能降维策略
            max_pca_components = min(n_samples, original_dim)
            effective_target_dim = min(target_dim, max_pca_components)
            
            if original_dim <= effective_target_dim:
                return features_array, f"keep_all_{original_dim}_{std_info}"
            elif original_dim <= effective_target_dim * 2:
                return features_array[:, :effective_target_dim], f"truncate_{effective_target_dim}_{std_info}"
            else:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=effective_target_dim, random_state=42)
                reduced_features = pca.fit_transform(features_array)
                explained_ratio = np.sum(pca.explained_variance_ratio_)
                return reduced_features, f"pca_{effective_target_dim}_var{explained_ratio:.3f}_{std_info}"
        
        # 应用统一降维（与多视图算法保持一致）
        feature_matrix_reduced, reduction_method = unified_dimension_reduction(feature_matrix, target_dim=200)
        print(f"[LFighter-DBO] Applied unified dimension reduction: {reduction_method}")
        
        # 轻量级DBONet配置：blocks=1-2，快速训练
        n_view = 1
        nfeats = [feature_matrix_reduced.shape[1]]  # 使用降维后的维度
        n_clusters = 2
        blocks = 2  # 统一配置
        para = 0.05  # 统一参数（与MV-DBO保持一致）
        np.random.seed(42)
        Z_init = np.random.randn(m, n_clusters) * 0.01  # 统一初始化缩放
        
        # 统一邻接矩阵配置
        n_neighbors = min(5, m-1)  # 与MV-DBO保持一致的邻居数量
        adj = kneighbors_graph(feature_matrix_reduced, n_neighbors=n_neighbors, mode='connectivity', include_self=True)
        adj_tensor = torch.tensor(adj.toarray(), dtype=torch.float32, device=device)
        
        # 创建轻量级DBONet
        dbo_model = DBONet(nfeats, n_view, n_clusters, blocks, para, Z_init, device)
        features_tensor = [torch.tensor(feature_matrix_reduced, dtype=torch.float32, device=device)]
        adjs = [adj_tensor]
        
        # 统一归一化流程（与MV-DBO一致）
        features_norm = []
        for i in range(n_view):
            # 多步归一化
            feature = features_tensor[i]
            feature = (feature - feature.mean(dim=0)) / (feature.std(dim=0) + 1e-8)  # 标准化
            feature = standardization(normalization(feature))  # 进一步归一化
            features_norm.append(feature)
        
        # 统一训练配置：与MV-DBO完全相同
        dbo_model.train()
        optimizer = torch.optim.Adam(dbo_model.parameters(), lr=5e-4, weight_decay=1e-5)  # 统一优化器配置
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)  # 统一调度器
        criterion = torch.nn.MSELoss()
        
        best_loss = float('inf')
        for epoch in range(8):  # 统一训练轮数：8个epoch
            optimizer.zero_grad()
            output_z = dbo_model(features_tensor, adjs)
            
            # 统一损失函数：显式+隐式损失（与MV-DBO一致）
            loss_dis = torch.tensor(0., device=device)
            loss_lap = torch.tensor(0., device=device)
            
            for k in range(n_view):
                # 显式损失：特征重构
                target_sim = features_norm[k] @ features_norm[k].t()
                pred_sim = output_z @ output_z.t()
                loss_dis += criterion(pred_sim, target_sim)
                
                # 隐式损失：图拉普拉斯正则化
                loss_lap += criterion(pred_sim, adjs[k])
            
            # 统一总损失权重
            total_loss = loss_dis + 0.3 * loss_lap
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(dbo_model.parameters(), max_norm=1.0)  # 统一梯度裁剪
            optimizer.step()
            scheduler.step()
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
            
            if epoch % 2 == 0:
                print(f"[LFighter-DBO] Epoch {epoch+1}: Loss={total_loss.item():.6f}")
        
        # === 使用LFD clusters_dissimilarity替换Silhouette评估 ===
        dbo_model.eval()
        with torch.no_grad():
            output_z = dbo_model(features_tensor, adjs)
        
        z_np = output_z.detach().cpu().numpy()
        
        # 简化的kmeans聚类（与LFighter-MV-DBO一致）
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(z_np)
        labels = kmeans.labels_
        
        # === 使用LFD的clusters_dissimilarity进行质量评估 ===
        clusters = {0: [], 1: []}
        for i, l in enumerate(labels):
            clusters[l].append(z_np[i])
        
        # 创建LFD实例并调用clusters_dissimilarity方法
        lfd = LFD(config.NUM_CLASSES)
        cs0, cs1 = lfd.clusters_dissimilarity(clusters)
        good_cl = 0 if cs0 < cs1 else 1
        
        print(f"[LFighter-DBO] LFD cluster quality: cs0={cs0:.4f}, cs1={cs1:.4f}, good_cluster={good_cl}")
        
        # 权重分配
        scores = np.ones(m)
        for i, l in enumerate(labels):
            if l != good_cl:
                scores[i] = 0
        
        print(f"[LFighter-DBO] Good clients: {np.sum(scores)}/{m}")
        return average_weights(local_weights, scores)


class LFighterMV:
    def __init__(self):
        pass
    
    def aggregate(self, simulation_model, local_weights, local_features, local_models, peers_types, lfd):
        """LFighter-MV: 三视图拼接 - 输出层梯度，第一层激活值，输入层梯度"""
        # 严格检查：缺少特征时直接报错
        if not local_features or len(local_features) == 0:
            raise ValueError("LFighter-MV requires local_features but got None or empty list. Please check if 'mv' rule is properly extracting features during training.")
        
        m = len(local_weights)
        
        # 视图1: 输出层梯度（权重差异）- 与原版LFighter保持一致
        output_grad_features = []
        
        # 首先计算所有类别的梯度差异来检测最异常的两个类别（与LFighter保持一致）
        global_params = list(simulation_model.parameters())
        
        # 计算所有客户端的输出层偏置差异，用于检测异常类别
        memory = np.zeros(global_params[-1].shape[0])  # 输出层偏置的维度
        
        for i in range(m):
            local_model = copy.deepcopy(simulation_model)
            local_model.load_state_dict(local_weights[i])
            local_params = list(local_model.parameters())
            
            # 计算权重和偏置的差异
            dw = global_params[-2].cpu().data.numpy() - local_params[-2].cpu().data.numpy()
            db = global_params[-1].cpu().data.numpy() - local_params[-1].cpu().data.numpy()
            
            # 累积异常程度（与原版LFighter的逻辑一致）
            norms = np.linalg.norm(dw, axis=-1)
            memory += norms + np.abs(db)
        
        # 找到最异常的两个类别（与原版LFighter一致）
        max_two_freq_classes = memory.argsort()[-2:]
        print(f'[LFighter-MV] Detected anomalous classes (same as LFighter): {max_two_freq_classes}')
        
        # 现在只提取这两个类别的输出层梯度
        for i in range(m):
            local_model = copy.deepcopy(simulation_model)
            local_model.load_state_dict(local_weights[i])
            local_params = list(local_model.parameters())
            
            # 只提取最异常两个类别的权重差异（与LFighter完全一致）
            output_grad = global_params[-2].cpu().data.numpy() - local_params[-2].cpu().data.numpy()
            key_classes_grad = output_grad[max_two_freq_classes].reshape(-1)
            output_grad_features.append(key_classes_grad)
        
        # 视图2: 第一层激活值（从local_features中提取）
        first_activation_features = []
        for i, peer_features in enumerate(local_features):
            if not peer_features or len(peer_features) == 0:
                raise ValueError(f"Empty peer_features for peer {i}. Check feature extraction during training.")
            
            # 调试：打印实际的数据结构
            print(f"[Debug] Peer {i} features structure: type={type(peer_features)}, len={len(peer_features) if hasattr(peer_features, '__len__') else 'N/A'}")
            if hasattr(peer_features, '__len__') and len(peer_features) > 0:
                print(f"[Debug] First element type: {type(peer_features[0])}")
                if hasattr(peer_features[0], '__len__'):
                    print(f"[Debug] First element length: {len(peer_features[0])}")
            
            # 处理可能的嵌套结构
            # peer_features应该是[input_flat, first_layer_activation, logits]
            # 但如果是嵌套的，可能需要额外处理
            actual_features = peer_features
            
            # 检查是否是嵌套结构（list of lists）
            if isinstance(peer_features, (list, tuple)) and len(peer_features) > 0:
                if isinstance(peer_features[0], (list, tuple)):
                    # 如果是嵌套的，取第一个元素（假设它是我们需要的特征）
                    actual_features = peer_features[0]
                    print(f"[Debug] Detected nested structure, using first element")
            
            # 现在检查actual_features是否符合预期格式
            if not isinstance(actual_features, (list, tuple)) or len(actual_features) < 2:
                raise ValueError(f"Invalid features structure for peer {i}: expected list/tuple with >=2 elements, got {type(actual_features)} with length {len(actual_features) if hasattr(actual_features, '__len__') else 'unknown'}")
            
            # 从CNNPATHMNIST的return_features中提取第一层激活值（索引1）
            first_activation = actual_features[1]  # first_layer_activation
            print(f"[Debug] Peer {i} first_activation type: {type(first_activation)}")
            
            if not hasattr(first_activation, 'detach'):
                # 如果first_activation不是tensor，尝试转换
                if isinstance(first_activation, (list, tuple)):
                    # 如果是list/tuple，可能包含多个tensor，取第一个
                    if len(first_activation) > 0 and hasattr(first_activation[0], 'detach'):
                        first_activation = first_activation[0]
                        print(f"[Debug] Converted from list to tensor for peer {i}")
                    else:
                        raise ValueError(f"Cannot convert first_activation to tensor for peer {i}: {type(first_activation)}")
                else:
                    raise ValueError(f"Expected tensor with .detach() method for peer {i}, got {type(first_activation)}")
            
            # 只取第一个样本的激活值作为代表
            first_activation_flat = first_activation[0].detach().cpu().numpy().flatten()
            first_activation_features.append(first_activation_flat)
            print(f"[Debug] Successfully extracted features for peer {i}, shape: {first_activation_flat.shape}")
        
        # 视图3: 输入层梯度（第一层卷积权重的梯度）
        input_grad_features = []
        for i in range(m):
            # 获取第一层卷积层的权重梯度（conv1权重）
            input_grad = list(simulation_model.parameters())[0].cpu().data.numpy() - list(local_models[i].parameters())[0].cpu().data.numpy()
            input_grad_features.append(input_grad.reshape(-1))
        
        # 智能维度处理：根据特征维度选择裁剪或PCA降维策略
        def smart_dimension_reduction(features, target_dim=200, method='auto', standardize=True):
            """智能维度缩减：标准化 + PCA/裁剪，考虑样本数限制"""
            features_array = np.array(features)
            n_samples, original_dim = features_array.shape
            
            # 1. 强烈建议的标准化处理（均值0，方差1）
            if standardize:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                features_array = scaler.fit_transform(features_array)
                std_info = "standardized"
            else:
                std_info = "raw"
            
            # 2. PCA最大维度不能超过min(样本数, 特征数)
            max_pca_components = min(n_samples, original_dim)
            effective_target_dim = min(target_dim, max_pca_components)
            
            if original_dim <= effective_target_dim:
                # 维度已经足够小，直接返回标准化后的特征
                return features_array, f"keep_all_{original_dim}_{std_info}"
            elif original_dim <= effective_target_dim * 2:
                # 维度适中，使用简单裁剪（速度优先）
                return features_array[:, :effective_target_dim], f"truncate_{effective_target_dim}_{std_info}"
            else:
                # 维度很大，使用PCA降维（信息保留优先）
                if method == 'pca' or method == 'auto':
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=effective_target_dim, random_state=42)
                    reduced_features = pca.fit_transform(features_array)
                    explained_ratio = np.sum(pca.explained_variance_ratio_)
                    return reduced_features, f"pca_{effective_target_dim}_var{explained_ratio:.3f}_{std_info}"
                else:
                    return features_array[:, :effective_target_dim], f"truncate_{effective_target_dim}_{std_info}"
        
        # 对三个视图分别进行智能降维
        output_reduced, output_method = smart_dimension_reduction(output_grad_features, target_dim=200)
        activation_reduced, activation_method = smart_dimension_reduction(first_activation_features, target_dim=200)
        input_reduced, input_method = smart_dimension_reduction(input_grad_features, target_dim=200)
        
        # 自动加权：基于聚类分离度计算每个视图的贡献权重
        def compute_view_weights(view_features_list, view_names):
            """基于标签翻转检测理论计算视图权重 - 输出层应占主导地位"""
            from sklearn.metrics import silhouette_score
            from sklearn.cluster import KMeans
            
            weights = []
            scores = []
            
            # 定义基于理论的基础权重（输出层最重要）
            theory_based_weights = {
                'Output_Grad': 0.9,      # 输出层梯度最重要 - 直接反映标签翻转
                'First_Activation': 0.05, # 激活值特征次要 - 反映中间表示变化  
                'Input_Grad': 0.05       # 输入层梯度最低 - 距离攻击最远
            }
            
            for i, (view_features, view_name) in enumerate(zip(view_features_list, view_names)):
                # 获取理论权重作为基准
                base_weight = theory_based_weights.get(view_name, 0.33)
                
                # 对每个视图单独做聚类评估
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(view_features)
                labels = kmeans.labels_
                
                # 检查聚类是否有效（两个簇都有样本）
                if len(set(labels)) > 1:
                    # 计算Silhouette分数
                    sil_score = silhouette_score(view_features, labels)
                    # 计算聚类分离度（簇间距离/簇内距离）
                    centers = kmeans.cluster_centers_
                    inter_dist = np.linalg.norm(centers[0] - centers[1])
                    intra_dist = np.mean([np.std(view_features[labels == k], axis=0).mean() for k in range(2)])
                    separation_ratio = inter_dist / (intra_dist + 1e-8)
                    
                    # 聚类质量分数 (0-1范围)
                    cluster_quality = max(0, sil_score) + min(1.0, np.log(1 + separation_ratio) * 0.1)
                    cluster_quality = min(1.0, cluster_quality)  # 限制在[0,1]
                else:
                    cluster_quality = 0.0
                    
                # 结合理论权重和聚类质量：理论权重70%，聚类质量30%
                # 这确保输出层始终保持较高权重，同时考虑实际数据的聚类效果
                combined_score = base_weight * 0.7 + cluster_quality * base_weight * 0.3
                scores.append(combined_score)
            
            # 权重归一化
            scores = np.array(scores)
            weights = scores / np.sum(scores)  # 归一化
            
            return weights, scores
        
        # 计算三个视图的自适应权重
        view_features_list = [output_reduced, activation_reduced, input_reduced]
        view_names = ['Output_Grad', 'First_Activation', 'Input_Grad']
        view_weights, view_scores = compute_view_weights(view_features_list, view_names)
        
        # 加权拼接降维后的三个视图
        fused_features = []
        for i in range(m):
            # 应用自适应权重
            weighted_output = output_reduced[i] * view_weights[0]
            weighted_activation = activation_reduced[i] * view_weights[1] 
            weighted_input = input_reduced[i] * view_weights[2]
            
            fused = np.concatenate([weighted_output, weighted_activation, weighted_input])
            fused_features.append(fused)
        
        fused_array = np.array(fused_features)
        
        # 一致性约束/对齐机制：融合后再整体标准化+可选PCA
        def post_fusion_alignment(fused_features, apply_pca=True, final_dim=None):
            """融合后一致性对齐：整体标准化 + 可选PCA"""
            from sklearn.preprocessing import StandardScaler
            
            # 整体标准化（确保不同视图融合后的数值一致性）
            scaler = StandardScaler()
            aligned_features = scaler.fit_transform(fused_features)
            alignment_info = "post_standardized"
            
            # 可选的整体PCA（进一步降维和去相关）
            if apply_pca and final_dim is not None:
                n_samples, total_dim = aligned_features.shape
                max_components = min(n_samples, total_dim)
                effective_final_dim = min(final_dim, max_components)
                
                if total_dim > effective_final_dim:
                    from sklearn.decomposition import PCA
                    final_pca = PCA(n_components=effective_final_dim, random_state=42)
                    aligned_features = final_pca.fit_transform(aligned_features)
                    final_explained = np.sum(final_pca.explained_variance_ratio_)
                    alignment_info += f"_pca{effective_final_dim}_var{final_explained:.3f}"
                
            return aligned_features, alignment_info
        
        # 应用融合后对齐（可选最终降维到150维，保持合理复杂度）
        final_features, alignment_method = post_fusion_alignment(fused_array, apply_pca=True, final_dim=150)
        
        # 直接K-means聚类（检验多模态聚类效果）
        kmeans = KMeans(n_clusters=2, random_state=0).fit(final_features)
        labels = kmeans.labels_
        
        # 使用LFD的聚类质量评估
        clusters = {0: [], 1: []}
        for i, l in enumerate(labels):
            clusters[l].append(final_features[i])
        
        # 计算聚类质量并生成权重
        cs0, cs1 = lfd.clusters_dissimilarity(clusters)
        good_cl = 0 if cs0 < cs1 else 1
        
        # 根据聚类质量选择好的cluster生成权重
        scores = []
        for i in range(m):
            if labels[i] == good_cl:  # 选择质量更好的cluster
                scores.append(1.0)
            else:  # 质量较差的cluster
                scores.append(0.0)  # 完全排除，与原始LFighter一致
        
        # 使用计算出的scores进行加权平均（移除循环导入）
        aggregated_weights = average_weights(local_weights, scores)
        
        # 返回聚合权重和视图权重信息
        view_weights_info = {
            'output_grad': float(view_weights[0]),
            'first_activation': float(view_weights[1]),
            'input_grad': float(view_weights[2])
        }
        
        return aggregated_weights, view_weights_info


class LFighterMVDBO:
    def __init__(self):
        pass
    
    def aggregate(self, simulation_model, local_weights, local_features):
        """LFighter-MV-DBO: 重用LFighterMV多视图特征提取 + DBONet训练 + LFD聚类质量评估"""
        import config
        device = simulation_model.device if hasattr(simulation_model, 'device') else config.DEVICE
        
        import torch
        from models import DBONet
        from sklearn.neighbors import kneighbors_graph
        
        # 检查输入
        if not local_features or len(local_features) == 0:
            raise ValueError("LFighter-MV-DBO requires local_features but got None or empty list.")
        
        m = len(local_weights)
        print(f"[LFighter-MV-DBO] Processing {m} clients with DBONet + LFD quality assessment")
        
        # === 步骤1: 重用LFighterMV的多视图特征构建逻辑 ===
        lfighter_mv = LFighterMV()
        
        # 创建一个临时的LFD实例用于聚类质量评估  
        lfd = LFD(config.NUM_CLASSES, attack_ratio=getattr(config, 'ATTACKERS_RATIO', None))
        
        # 调用LFighterMV来获取处理好的多视图特征（但不用它的聚类结果）
        # 我们需要手动调用LFighterMV的特征提取部分
        local_models = []
        for local_weight in local_weights:
            model = copy.deepcopy(simulation_model)
            model.load_state_dict(local_weight)
            local_models.append(list(model.parameters()))
        
        global_model = list(simulation_model.parameters())
        
        # 异常类别检测（与LFighterMV一致）
        memory = np.zeros(global_model[-1].shape[0])
        for i in range(m):
            dw = global_model[-2].cpu().data.numpy() - local_models[i][-2].cpu().data.numpy()
            db = global_model[-1].cpu().data.numpy() - local_models[i][-1].cpu().data.numpy()
            norms = np.linalg.norm(dw, axis=-1)
            memory += norms + np.abs(db)
        
        max_two_freq_classes = memory.argsort()[-2:]
        print(f'[LFighter-MV-DBO] Detected anomalous classes: {max_two_freq_classes}')
        
        # 视图1: 输出层梯度
        output_grad_features = []
        for i in range(m):
            output_grad = global_model[-2].cpu().data.numpy() - local_models[i][-2].cpu().data.numpy()
            key_classes_grad = output_grad[max_two_freq_classes].reshape(-1)
            output_grad_features.append(key_classes_grad)
        
        # 视图2: 第一层激活值
        first_activation_features = []
        for i, peer_features in enumerate(local_features):
            actual_features = peer_features
            if isinstance(peer_features, (list, tuple)) and len(peer_features) > 0:
                if isinstance(peer_features[0], (list, tuple)):
                    actual_features = peer_features[0]
            
            first_activation = actual_features[1]
            if not hasattr(first_activation, 'detach'):
                if isinstance(first_activation, (list, tuple)) and len(first_activation) > 0:
                    first_activation = first_activation[0]
            
            first_activation_flat = first_activation[0].detach().cpu().numpy().flatten()
            first_activation_features.append(first_activation_flat)
        
        # 视图3: 输入层梯度
        input_grad_features = []
        for i in range(m):
            input_grad = global_model[0].cpu().data.numpy() - local_models[i][0].cpu().data.numpy()
            input_grad_features.append(input_grad.reshape(-1))
        
        # === 步骤2: 统一降维处理 ===
        def unified_dimension_reduction(features, target_dim=200, standardize=True):
            features_array = np.array(features)
            n_samples, original_dim = features_array.shape
            
            if standardize:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                features_array = scaler.fit_transform(features_array)
            
            max_pca_components = min(n_samples, original_dim)
            effective_target_dim = min(target_dim, max_pca_components)
            
            if original_dim > effective_target_dim:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=effective_target_dim, random_state=42)
                features_array = pca.fit_transform(features_array)
                
            return features_array
        
        # 应用统一降维
        output_reduced = unified_dimension_reduction(output_grad_features, target_dim=200)
        activation_reduced = unified_dimension_reduction(first_activation_features, target_dim=200)
        input_reduced = unified_dimension_reduction(input_grad_features, target_dim=200)
        
        # === 步骤3: 视图权重计算（基于理论） ===
        view_weights = np.array([0.9, 0.05, 0.05])  # Output主导，与理论一致
        print(f"[LFighter-MV-DBO] View weights: Output={view_weights[0]:.3f}, Activation={view_weights[1]:.3f}, Input={view_weights[2]:.3f}")
        
        # === 步骤4: DBONet训练 ===
        # 应用权重并构建多视图特征
        view_features = [
            output_reduced * np.sqrt(view_weights[0]),
            activation_reduced * np.sqrt(view_weights[1]), 
            input_reduced * np.sqrt(view_weights[2])
        ]
        
        nfeats = [f.shape[1] for f in view_features]
        n_view = len(view_features)
        
        # 构建邻接矩阵
        adjs = []
        for v in range(n_view):
            n_neighbors = min(5, m-1)
            adj = kneighbors_graph(view_features[v], n_neighbors=n_neighbors, 
                                 mode='connectivity', include_self=True)
            adj_tensor = torch.tensor(adj.toarray(), dtype=torch.float32, device=device)
            adjs.append(adj_tensor)
        
        # DBONet配置
        n_clusters = 2
        blocks = 2
        para = 0.05
        np.random.seed(42)
        Z_init = np.random.randn(m, n_clusters) * 0.01
        
        # 创建和训练DBONet
        dbo_model = DBONet(nfeats, n_view, n_clusters, blocks, para, Z_init, device)
        features_tensor = [torch.tensor(view_features[v], dtype=torch.float32, device=device) 
                         for v in range(n_view)]
        
        # 归一化处理
        def normalization(data):
            maxVal = torch.max(data)
            minVal = torch.min(data)
            return (data - minVal) / (maxVal - minVal + 1e-10)

        def standardization(data):
            rowSum = torch.sqrt(torch.sum(data**2, 1))
            repMat = rowSum.repeat((data.shape[1], 1)) + 1e-10
            return torch.div(data, repMat.t())
        
        features_norm = []
        for i in range(n_view):
            feature = features_tensor[i]
            feature = (feature - feature.mean(dim=0)) / (feature.std(dim=0) + 1e-8)
            feature = standardization(normalization(feature))
            features_norm.append(feature)
        
        # 训练DBONet
        dbo_model.train()
        optimizer = torch.optim.Adam(dbo_model.parameters(), lr=5e-4, weight_decay=1e-5)
        criterion = torch.nn.MSELoss()
        
        for epoch in range(8):
            optimizer.zero_grad()
            output_z = dbo_model(features_tensor, adjs)
            
            # 加权损失计算
            loss_dis = torch.tensor(0., device=device)
            loss_lap = torch.tensor(0., device=device)
            
            for k in range(n_view):
                weight = float(view_weights[k])
                target_sim = features_norm[k] @ features_norm[k].t()
                pred_sim = output_z @ output_z.t()
                loss_dis += criterion(pred_sim, target_sim) * weight
                loss_lap += criterion(pred_sim, adjs[k]) * weight
            
            total_loss = loss_dis + 0.3 * loss_lap
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(dbo_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if epoch % 2 == 0:
                print(f"[LFighter-MV-DBO] Epoch {epoch+1}: Loss={total_loss.item():.6f}")
        
        # === 步骤5: 使用LFD聚类质量评估替换简单kmeans ===
        dbo_model.eval()
        with torch.no_grad():
            output_z = dbo_model(features_tensor, adjs)
        
        z_np = output_z.detach().cpu().numpy()
        
        # 简化的kmeans聚类
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(z_np)
        labels = kmeans.labels_
        
        # === 使用LFD的clusters_dissimilarity进行质量评估 ===
        clusters = {0: [], 1: []}
        for i, l in enumerate(labels):
            clusters[l].append(z_np[i])
        
        # 调用LFD的聚类质量评估方法
        cs0, cs1 = lfd.clusters_dissimilarity(clusters)
        good_cl = 0 if cs0 < cs1 else 1
        
        print(f"[LFighter-MV-DBO] LFD cluster quality: cs0={cs0:.4f}, cs1={cs1:.4f}, good_cluster={good_cl}")
        
        # 权重分配
        scores = np.ones(m)
        for i, l in enumerate(labels):
            if l != good_cl:
                scores[i] = 0
        
        print(f"[LFighter-MV-DBO] Good clients: {np.sum(scores)}/{m}")
        
        # 返回结果
        aggregated_weights = average_weights(local_weights, scores)
        view_weights_info = {
            'output_grad': float(view_weights[0]),
            'first_activation': float(view_weights[1]),
            'input_grad': float(view_weights[2])
        }
        
        return aggregated_weights, view_weights_info


class Tolpegin:
    def __init__(self):
        pass
    
    def score(self, global_model, local_models, peers_types, selected_peers):
        global_model = list(global_model.parameters())
        last_g = global_model[-2].cpu().data.numpy()
        m = len(local_models)
        grads = [None for i in range(m)]
        for i in range(m):
            grad= (last_g - \
                    list(local_models[i].parameters())[-2].cpu().data.numpy())
            grads[i] = grad
        
        grads = np.array(grads)
        num_classes = grad.shape[0]
        # print('Number of classes:', num_classes)
        dist = [ ]
        labels = [ ]
        for c in range(num_classes):
            data = grads[:, c]
            data = get_pca(copy.deepcopy(data))
            kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
            cl = kmeans.cluster_centers_
            dist.append(((cl[0] - cl[1])**2).sum())
            labels.append(kmeans.labels_)
        
        dist = np.array(dist)
        candidate_class = dist.argmax()
        print("Candidate source/target class", candidate_class)
        labels = labels[candidate_class]
        if sum(labels) < m/2:
            scores = 1 - labels
        else:
            scores = labels
        
        for i, pt in enumerate(peers_types):
            print(pt, 'scored', scores[i])
        return scores
#################################################################################################################
# Clip local updates
def clipp_model(g_w, w, gamma =  1):
    for layer in w.keys():
        w[layer] = g_w[layer] + (w[layer] - g_w[layer])*min(1, gamma)
    return w
def FLAME(global_model, local_models, noise_scalar):
    # Compute number of local models
    m = len(local_models)
    
    # Flattent local models
    g_m = np.array([torch.nn.utils.parameters_to_vector(global_model.parameters()).cpu().data.numpy()])
    f_m = np.array([torch.nn.utils.parameters_to_vector(model.parameters()).cpu().data.numpy() for model in local_models])
    grads = g_m - f_m
    # Compute model-wise cosine similarity
    cs = smp.cosine_similarity(grads)
    # Compute the minimum cluster size value
    msc = int(m*0.5) + 1 
    # Apply HDBSCAN on the computed cosine similarities
    clusterer = hdbscan.HDBSCAN(min_cluster_size=msc, min_samples=1, allow_single_cluster = True)
    clusterer.fit(cs)
    labels = clusterer.labels_
    # print('Clusters:', labels)

    if sum(labels) == -(m):
        # In case all of the local models identified as outliers, consider all of as benign
        benign_idxs = np.arange(m)
    else:
        benign_idxs = np.where(labels!=-1)[0]
        
    # Compute euclidean distances to the current global model
    euc_d = cdist(g_m, f_m)[0]
    # Identify the median of computed distances
    st = np.median(euc_d)
    # Clipp admitted updates
    W_c = []
    for i, idx in enumerate(benign_idxs):
        w_c = clipp_model(global_model.state_dict(), local_models[idx].state_dict(), gamma =  st/euc_d[idx])
        W_c.append(w_c)
    
    # Average admitted clipped updates to obtain a new global model
    g_w = average_weights(W_c, np.ones(len(W_c)))
    
    '''From the original paper: {We use standard DP parameters and set eps = 3705 for IC, 
    eps = 395 for the NIDS and eps = 4191 for the NLP scenario. 
    Accordingly, lambda = 0.001 for IC and NLP, and lambda = 0.01 for the NIDS scenario.}
    However, we found lambda = 0.001 with the CIFAR10-ResNet18 benchmark spoils the model
    and therefore we tried lower lambda values, which correspond to greater eps values.'''
    
    # Add adaptive noise to the global model
    lamb = 0.001
    sigma = lamb*st*noise_scalar
    # print('Sigma:{:0.4f}'.format(sigma))
    for key in g_w.keys():
        noise = torch.FloatTensor(g_w[key].shape).normal_(mean=0, std=(sigma**2)).to(g_w[key].device)
        g_w[key] = g_w[key] + noise
        
    return g_w 
#################################################################################################################

def median_opt(input):
    shape = input.shape
    input = input.sort()[0]
    if shape[-1] % 2 != 0:
        output = input[..., int((shape[-1] - 1) / 2)]
    else:
        output = (input[..., int(shape[-1] / 2 - 1)] + input[..., int(shape[-1] / 2)]) / 2.0
    return output

def Repeated_Median_Shard(w):
    SHARD_SIZE = 100000
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        if total_num < SHARD_SIZE:
            slopes, intercepts = repeated_median(y)
            y = intercepts + slopes * (len(w) - 1) / 2.0
        else:
            y_result = torch.FloatTensor(total_num).to(device)
            assert total_num == y.shape[0]
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y_shard = y[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                slopes_shard, intercepts_shard = repeated_median(y_shard)
                y_shard = intercepts_shard + slopes_shard * (len(w) - 1) / 2.0
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE] = y_shard
            y = y_result
        y = y.reshape(shape)
        w_med[k] = y
    return w_med


def repeated_median(y):
    num_models = y.shape[1]
    total_num = y.shape[0]
    y = y.sort()[0]
    yyj = y.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
    yyi = yyj.transpose(-1, -2)
    xx = torch.FloatTensor(range(num_models)).to(y.device)
    xxj = xx.repeat(total_num, num_models, 1)
    xxi = xxj.transpose(-1, -2) + eps

    diag = torch.Tensor([float('Inf')] * num_models).to(y.device)
    diag = torch.diag(diag).repeat(total_num, 1, 1)

    dividor = xxi - xxj + diag
    slopes = (yyi - yyj) / dividor + diag
    slopes, _ = slopes.sort()
    slopes = median_opt(slopes[:, :, :-1])
    slopes = median_opt(slopes)

    # get intercepts (intercept of median)
    yy_median = median_opt(y)
    xx_median = [(num_models - 1) / 2.0] * total_num
    xx_median = torch.Tensor(xx_median).to(y.device)
    intercepts = yy_median - slopes * xx_median

    return slopes, intercepts


# Repeated Median estimator
def Repeated_Median(w):
    cur_time = time.time()
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        slopes, intercepts = repeated_median(y)
        y = intercepts + slopes * (len(w) - 1) / 2.0

        y = y.reshape(shape)
        w_med[k] = y

    print('repeated median aggregation took {}s'.format(time.time() - cur_time))
    return w_med
        
# simple median estimator
def simple_median(w):
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        median_result = median_opt(y)
        assert total_num == len(median_result)

        weight = torch.reshape(median_result, shape)
        w_med[k] = weight
    return w_med

def trimmed_mean(w, trim_ratio):
    if trim_ratio == 0:
        return average_weights(w, [1 for i in range(len(w))])
        
    assert trim_ratio < 0.5, 'trim ratio is {}, but it should be less than 0.5'.format(trim_ratio)
    trim_num = int(trim_ratio * len(w))
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        y_sorted = y.sort()[0]
        result = y_sorted[:, trim_num:-trim_num]
        result = result.mean(dim=-1)
        assert total_num == len(result)

        weight = torch.reshape(result, shape)
        w_med[k] = weight
    return w_med


# Get average weights
def average_weights(w, marks):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * marks[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * marks[i]
        w_avg[key] = w_avg[key] *(1/sum(marks))
    return w_avg
   
def Krum(updates, f, multi = False):
    n = len(updates)
    updates = [torch.nn.utils.parameters_to_vector(update.parameters()) for update in updates]
    updates_ = torch.empty([n, len(updates[0])])
    for i in range(n):
      updates_[i] = updates[i]
    k = n - f - 2
    # collection distance, distance from points to points
    cdist = torch.cdist(updates_, updates_, p=2)
    dist, idxs = torch.topk(cdist, k , largest=False)
    dist = dist.sum(1)
    idxs = dist.argsort()
    if multi:
      return idxs[:k]
    else:
      return idxs[0]
##################################################################

class LFighterAutoencoder:
    def __init__(self, num_classes, ae_hidden_dim=128, ae_latent_dim=32, ae_epochs=50, reconstruction_weight=0.8, 
                 enable_visualization=True, save_path="./figures/", 
                 visualization_frequency=1, max_visualizations=0, save_final_only=False,
                 save_as_pdf=True, keep_individual_files=False, attack_ratio=None):
        """
        LFighterAutoencoder类初始化
        
        Args:
            num_classes: 类别数量
            ae_hidden_dim: 自编码器隐藏层维度
            ae_latent_dim: 自编码器潜在空间维度
            ae_epochs: 自编码器训练轮数
            reconstruction_weight: 重构误差权重
            enable_visualization: 是否启用可视化
            save_path: 保存路径
            visualization_frequency: 可视化频率
            max_visualizations: 最大可视化数量
            save_final_only: 是否只保存最终轮次
            save_as_pdf: 是否保存为PDF (默认True，始终启用)
            keep_individual_files: 始终禁用单独文件，忽略传入的参数
            attack_ratio: 攻击者比率，用于PDF文件名
            
        特性:
            - 使用输出层的全部梯度作为特征，而不是仅选择部分类别
            - 通过传统自编码器进行特征降维和异常检测
            - 结合聚类和重构误差进行攻击者识别
            - 仅生成PDF报告，不生成单独文件
        """
        self.num_classes = num_classes
        self.memory = np.zeros(num_classes)
        
        # Autoencoder相关参数
        self.ae_hidden_dim = ae_hidden_dim
        self.ae_latent_dim = ae_latent_dim
        self.ae_epochs = ae_epochs
        self.reconstruction_weight = reconstruction_weight
    
        # 可视化相关参数
        self.enable_visualization = enable_visualization
        self.save_path = save_path
        self.visualization_frequency = visualization_frequency
        self.max_visualizations = max_visualizations
        self.save_final_only = save_final_only
        self.save_as_pdf = True  # 始终保存为PDF，忽略传入的参数
        self.keep_individual_files = False  # 始终禁用单独文件，忽略传入的参数
        self.attack_ratio = attack_ratio
        
        # 轮数计数器
        self.round_counter = 0
        self.total_rounds = None
        self.visualization_count = 0
        
        # PDF文件管理
        self.pdf_pages = None
        self.pdf_filename = None
        
        # 确保保存目录存在
        if self.enable_visualization:
            os.makedirs(self.save_path, exist_ok=True)
            
            # 创建PDF文件（如果启用PDF保存）
            if self.save_as_pdf:
                self._initialize_pdf()
    
    def _initialize_pdf(self):
        """初始化PDF文件"""
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            attack_str = f"_atr{self.attack_ratio:.1f}" if self.attack_ratio is not None else ""
            self.pdf_filename = f'{self.save_path}/lfighter_ae_complete_report{attack_str}.pdf'
            self.pdf_pages = PdfPages(self.pdf_filename)
            print(f"[LFighter-AE] 🔗 PDF报告初始化成功")
            print(f"[LFighter-AE] 📄 实时查看路径: {self.pdf_filename}")
        except Exception as e:
            print(f"[LFighter-AE] PDF初始化失败: {e}")
            self.pdf_pages = None
    
    def finalize_pdf(self):
        """关闭PDF文件"""
        if self.pdf_pages is not None:
            try:
                self.pdf_pages.close()
                print(f"[LFighter-AE] PDF报告已保存: {self.pdf_filename}")
            except Exception as e:
                print(f"[LFighter-AE] PDF关闭失败: {e}")
            finally:
                self.pdf_pages = None
    
    def set_total_rounds(self, total_rounds):
        """设置总训练轮数，用于save_final_only模式"""
        self.total_rounds = total_rounds
        print(f"[LFighter-AE] Set total rounds to {total_rounds} for visualization control")
    
    def should_visualize_this_round(self):
        """判断当前轮次是否应该保存可视化"""
        if not self.enable_visualization:
            return False
            
        # 如果设置为只保存最后一轮
        if self.save_final_only:
            return self.total_rounds is not None and self.round_counter == self.total_rounds
        
        # 动态调整可视化频率：前20轮每轮可视化，后续每10轮一次
        if self.round_counter <= 20:
            current_frequency = 1
        else:
            current_frequency = 10
        
        # 按动态频率保存
        return self.round_counter % current_frequency == 0
    
    def cleanup_old_visualizations(self):
        """清理旧的可视化文件"""
        if not self.enable_visualization:
            return
        
        if self.max_visualizations > 0 and self.visualization_count > self.max_visualizations:
            # 清理PNG文件
            png_files = [f for f in os.listdir(self.save_path) if f.startswith('lfighter_') and f.endswith('.png')]
            png_files.sort(key=lambda x: os.path.getctime(os.path.join(self.save_path, x)))
            
            while len(png_files) > self.max_visualizations:
                old_file = os.path.join(self.save_path, png_files.pop(0))
                if os.path.exists(old_file):
                    os.remove(old_file)
            
            # 清理文本文件
            txt_files = [f for f in os.listdir(self.save_path) if f.startswith('lfighter_') and f.endswith('.txt')]
            txt_files.sort(key=lambda x: os.path.getctime(os.path.join(self.save_path, x)))
            
            while len(txt_files) > self.max_visualizations:
                old_file = os.path.join(self.save_path, txt_files.pop(0))
                if os.path.exists(old_file):
                    os.remove(old_file)
    
    def create_pdf_report(self, round_num, features, labels, ptypes, scores, cs0, cs1, good_cl, metrics):
        """向已有PDF文件添加当前轮次的可视化页面 - LFighter-AE版本"""
        if not self.should_visualize_this_round() or not self.save_as_pdf or self.pdf_pages is None:
            return
        
        try:
            # 第一页：特征空间可视化（原始vs潜在特征）
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # t-SNE降维
            if features.shape[1] > 2:
                try:
                    perplexity = min(30, len(features)-1)
                    if perplexity < 1:
                        perplexity = 1
                    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                    features_2d = tsne.fit_transform(features)
                except:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2, random_state=42)
                    features_2d = pca.fit_transform(features)
            else:
                features_2d = features
            
            # 按客户端类型着色
            colors = []
            malicious_count = 0
            for ptype in ptypes:
                ptype_str = str(ptype).lower()
                if ('malicious' in ptype_str or 'attack' in ptype_str or 
                    'bad' in ptype_str or 'adversarial' in ptype_str):
                    colors.append('red')
                    malicious_count += 1
                else:
                    colors.append('blue')
            
            axes[0].scatter(features_2d[:, 0], features_2d[:, 1], c=colors, alpha=0.7, s=100)
            axes[0].set_title('LFighter-AE: Feature Space (by Client Type)', fontsize=14)
            axes[0].set_xlabel('Dimension 1')
            axes[0].set_ylabel('Dimension 2')
            
            red_patch = plt.matplotlib.patches.Patch(color='red', label='Malicious')
            blue_patch = plt.matplotlib.patches.Patch(color='blue', label='Benign')
            axes[0].legend(handles=[red_patch, blue_patch])
            
            # 按聚类结果着色
            cluster_colors = ['orange', 'green']
            for i in range(len(features_2d)):
                axes[1].scatter(features_2d[i, 0], features_2d[i, 1], 
                               c=cluster_colors[labels[i]], alpha=0.7, s=100)
            axes[1].set_title('LFighter-AE: Feature Space (by Cluster)', fontsize=14)
            axes[1].set_xlabel('Dimension 1')
            axes[1].set_ylabel('Dimension 2')
            
            orange_patch = plt.matplotlib.patches.Patch(color='orange', label='Cluster 0')
            green_patch = plt.matplotlib.patches.Patch(color='green', label='Cluster 1')
            axes[1].legend(handles=[orange_patch, green_patch])
            
            plt.suptitle(f'LFighter-AE Algorithm - Round {round_num} - Feature Space Analysis', fontsize=16)
            plt.tight_layout()
            self.pdf_pages.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # 第二页：聚类质量和客户端得分
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 聚类质量
            clusters = ['Cluster 0', 'Cluster 1']
            dissimilarities = [cs0, cs1]
            cluster_quality_colors = ['green' if i == good_cl else 'red' for i in range(2)]
            
            bars = axes[0, 0].bar(clusters, dissimilarities, color=cluster_quality_colors, alpha=0.7)
            axes[0, 0].set_ylabel('Dissimilarity Score')
            axes[0, 0].set_title('Cluster Quality Comparison')
            axes[0, 0].grid(True, alpha=0.3)
            
            for i, (bar, score) in enumerate(zip(bars, dissimilarities)):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              f'{score:.4f}', ha='center', va='bottom')
            
            axes[0, 0].text(good_cl, dissimilarities[good_cl] + 0.05, '✓ Good Cluster', 
                           ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # 客户端得分条形图
            client_indices = range(len(scores))
            bars = axes[0, 1].bar(client_indices, scores, color=colors, alpha=0.7)
            axes[0, 1].set_title('Client Scores')
            axes[0, 1].set_xlabel('Client Index')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_ylim(0, 1.1)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 得分分布直方图
            axes[1, 0].hist(scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 0].axvline(np.mean(scores), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(scores):.4f}')
            axes[1, 0].axvline(np.median(scores), color='orange', linestyle='--', 
                              label=f'Median: {np.median(scores):.4f}')
            axes[1, 0].set_xlabel('Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Score Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 性能指标文本
            axes[1, 1].axis('off')
            attack_ratio_str = f"Attack Ratio: {self.attack_ratio:.1f}" if self.attack_ratio is not None else "Attack Ratio: N/A"
            metrics_text = f"""
Performance Metrics - Round {round_num}

{attack_ratio_str}
Total Clients: {metrics.get('total_clients', 'N/A')}
Good Clients Selected: {metrics.get('good_clients', 'N/A')}
Selection Accuracy: {metrics.get('accuracy', 'N/A'):.2%}

KMeans Clustering:
• Method: {metrics.get('clustering_method', 'N/A')}
• Number of Clusters: {metrics.get('n_clusters', 'N/A')}
• Cluster Labels: {metrics.get('cluster_labels', 'N/A')}
• Cluster Distribution: {metrics.get('cluster_counts', 'N/A')}

Cluster Analysis:
• Cluster 0 Dissimilarity: {metrics.get('cs0', 'N/A'):.4f}
• Cluster 1 Dissimilarity: {metrics.get('cs1', 'N/A'):.4f}
• Good Cluster: {metrics.get('good_cluster', 'N/A')}

Feature Processing:
• Reduction Method: {metrics.get('reduction_method', 'N/A')}
• Attack Scope: {metrics.get('attack_scope', 'N/A')}
• Attack Type: {metrics.get('attack_type', 'N/A')}

AE (Autoencoder):
• Reconstruction Weight: {metrics.get('fixed_recon_weight', 'N/A'):.2f}
• Mean Recon Error: {metrics.get('mean_recon_error', 'N/A'):.4f}

Algorithm: LFighter + AE + KMeans
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
            axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            plt.suptitle(f'LFighter-AE Algorithm - Round {round_num} - Performance Analysis', fontsize=16)
            plt.tight_layout()
            self.pdf_pages.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # 强制刷新PDF文件，确保实时可见
            try:
                # matplotlib PdfPages 的正确flush方法
                if hasattr(self.pdf_pages, '_file') and hasattr(self.pdf_pages._file, '_file'):
                    self.pdf_pages._file._file.flush()
                    import os
                    os.fsync(self.pdf_pages._file._file.fileno())
                elif hasattr(self.pdf_pages, 'flush'):
                    self.pdf_pages.flush()
            except:
                pass  # 忽略flush错误，不影响正常功能
            
            print(f"[LFighter-AE] Round {round_num} 添加到PDF报告 - 实时可查看: {self.pdf_filename}")
            
        except Exception as e:
            print(f"[LFighter-AE] PDF页面添加失败 Round {round_num}: {e}")
    
    def visualize_training_process(self, losses, round_num):
        """可视化autoencoder训练过程"""
        if not self.should_visualize_this_round():
            return
            
        # 只在PDF报告中添加训练过程图表，不生成单独的图片文件
        if self.save_as_pdf and self.pdf_pages is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(losses, 'b-', linewidth=2, label='Reconstruction Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'AE Training Process (Round {round_num})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            try:
                self.pdf_pages.savefig(plt.gcf(), bbox_inches='tight')
            except Exception as e:
                print(f"[LFighter-AE] 训练过程可视化添加到PDF失败: {e}")
            plt.close()
    
    def visualize_feature_space(self, original_features, latent_features, labels, ptypes, round_num):
        """可视化原始和潜在特征空间的对比"""
        if not self.should_visualize_this_round():
            return
        
        # 调试信息：打印客户端类型
        if self.enable_visualization and round_num == 1:  # 只在第一轮打印
            print(f"[LFighter-AE Debug] Client types: {ptypes[:10]}...")  # 打印前10个客户端类型
        
        # 只在PDF报告中添加特征空间可视化，不生成单独的图片文件
        if self.save_as_pdf and self.pdf_pages is not None:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 对两种特征空间进行t-SNE降维
            features_list = [original_features, latent_features]
            feature_names = ['Original Features', 'Latent Features']
            
            for row, (features, name) in enumerate(zip(features_list, feature_names)):
                # t-SNE降维
                if features.shape[1] > 2:
                    try:
                        perplexity = min(30, len(features)-1)
                        if perplexity < 1:
                            perplexity = 1
                        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                        features_2d = tsne.fit_transform(features)
                    except:
                        # 如果t-SNE失败，使用PCA
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=2, random_state=42)
                        features_2d = pca.fit_transform(features)
                else:
                    features_2d = features
                
                # 1. 按客户端类型着色 - 增强检测逻辑
                colors = []
                malicious_count = 0
                for ptype in ptypes:
                    ptype_str = str(ptype).lower()
                    if ('malicious' in ptype_str or 'attack' in ptype_str or 
                        'bad' in ptype_str or 'adversarial' in ptype_str):
                        colors.append('red')
                        malicious_count += 1
                    else:
                        colors.append('blue')
                
                # 调试信息：打印检测结果（只在第一轮和原始特征时打印）
                if self.enable_visualization and round_num == 1 and row == 0:
                    print(f"[LFighter-AE Debug] Detected {malicious_count}/{len(ptypes)} malicious clients")
                
                axes[row, 0].scatter(features_2d[:, 0], features_2d[:, 1], c=colors, alpha=0.7, s=100)
                axes[row, 0].set_title(f'{name} (by Client Type)')
                axes[row, 0].set_xlabel('Dimension 1')
                axes[row, 0].set_ylabel('Dimension 2')
                
                # 添加图例
                red_patch = plt.matplotlib.patches.Patch(color='red', label='Malicious')
                blue_patch = plt.matplotlib.patches.Patch(color='blue', label='Benign')
                axes[row, 0].legend(handles=[red_patch, blue_patch])
                
                # 2. 按聚类结果着色
                cluster_colors = ['orange', 'green']
                for i in range(len(features_2d)):
                    axes[row, 1].scatter(features_2d[i, 0], features_2d[i, 1], 
                                       c=cluster_colors[labels[i]], alpha=0.7, s=100)
                axes[row, 1].set_title(f'{name} (by Cluster)')
                axes[row, 1].set_xlabel('Dimension 1')
                axes[row, 1].set_ylabel('Dimension 2')
                
                # 添加聚类图例
                orange_patch = plt.matplotlib.patches.Patch(color='orange', label='Cluster 0')
                green_patch = plt.matplotlib.patches.Patch(color='green', label='Cluster 1')
                axes[row, 1].legend(handles=[orange_patch, green_patch])
            
            plt.tight_layout()
            try:
                self.pdf_pages.savefig(fig, bbox_inches='tight')
            except Exception as e:
                print(f"[LFighter-AE] 特征空间可视化添加到PDF失败: {e}")
            plt.close()
    
    def visualize_reconstruction_errors(self, reconstruction_errors, ptypes, round_num):
        """可视化重构误差分布和各客户端的重构误差"""
        if not self.should_visualize_this_round():
            return
        
        # 只在PDF报告中添加重构误差可视化，不生成单独的图片文件
        if self.save_as_pdf and self.pdf_pages is not None:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # 1. 重构误差分布直方图
            axes[0].hist(reconstruction_errors, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0].axvline(np.mean(reconstruction_errors), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(reconstruction_errors):.4f}')
            axes[0].axvline(np.median(reconstruction_errors), color='orange', linestyle='--', 
                           label=f'Median: {np.median(reconstruction_errors):.4f}')
            axes[0].set_xlabel('Reconstruction Error')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Reconstruction Error Distribution')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 2. 各客户端重构误差条形图 - 增强检测逻辑
            client_indices = range(len(reconstruction_errors))
            colors = []
            for ptype in ptypes:
                ptype_str = str(ptype).lower()
                if ('malicious' in ptype_str or 'attack' in ptype_str or 
                    'bad' in ptype_str or 'adversarial' in ptype_str):
                    colors.append('red')
                else:
                    colors.append('blue')
            
            bars = axes[1].bar(client_indices, reconstruction_errors, color=colors, alpha=0.7)
            axes[1].set_title('Client Reconstruction Errors')
            axes[1].set_xlabel('Client Index')
            axes[1].set_ylabel('Reconstruction Error')
            axes[1].grid(True, alpha=0.3)
            
            # 添加图例
            red_patch = plt.matplotlib.patches.Patch(color='red', label='Malicious')
            blue_patch = plt.matplotlib.patches.Patch(color='blue', label='Benign')
            axes[1].legend(handles=[red_patch, blue_patch])
            
            plt.tight_layout()
            try:
                self.pdf_pages.savefig(fig, bbox_inches='tight')
            except Exception as e:
                print(f"[LFighter-AE] 重构误差可视化添加到PDF失败: {e}")
            plt.close()
    
    def visualize_client_scores(self, cluster_scores, recon_scores, final_scores, ptypes, round_num):
        """可视化客户端的聚类得分、重构得分和最终得分"""
        if not self.should_visualize_this_round():
            return
        
        # 只在PDF报告中添加客户端得分可视化，不生成单独的图片文件
        if self.save_as_pdf and self.pdf_pages is not None:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            client_indices = range(len(final_scores))
            
            # 增强的客户端类型检测逻辑
            colors = []
            malicious_count = 0
            for ptype in ptypes:
                ptype_str = str(ptype).lower()
                if ('malicious' in ptype_str or 'attack' in ptype_str or 
                    'bad' in ptype_str or 'adversarial' in ptype_str):
                    colors.append('red')
                    malicious_count += 1
                else:
                    colors.append('blue')
            
            # 调试信息：打印检测结果
            if self.enable_visualization and round_num == 1:
                print(f"[LFighter-AE Client Scores] Detected {malicious_count}/{len(ptypes)} malicious clients")
            
            # 1. 聚类得分
            axes[0, 0].bar(client_indices, cluster_scores, color=colors, alpha=0.7)
            axes[0, 0].set_title('Cluster Scores')
            axes[0, 0].set_xlabel('Client Index')
            axes[0, 0].set_ylabel('Cluster Score')
            axes[0, 0].set_ylim(0, 1.1)
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 重构得分
            axes[0, 1].bar(client_indices, recon_scores, color=colors, alpha=0.7)
            axes[0, 1].set_title('Reconstruction Scores')
            axes[0, 1].set_xlabel('Client Index')
            axes[0, 1].set_ylabel('Reconstruction Score')
            axes[0, 1].set_ylim(0, 1.1)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 最终得分
            axes[1, 0].bar(client_indices, final_scores, color=colors, alpha=0.7)
            axes[1, 0].set_title('Final Scores (Weighted Combination)')
            axes[1, 0].set_xlabel('Client Index')
            axes[1, 0].set_ylabel('Final Score')
            axes[1, 0].set_ylim(0, 1.1)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 添加图例
            red_patch = plt.matplotlib.patches.Patch(color='red', label='Malicious')
            blue_patch = plt.matplotlib.patches.Patch(color='blue', label='Benign')
            axes[1, 0].legend(handles=[red_patch, blue_patch])
            
            # 4. 得分对比散点图
            axes[1, 1].scatter(cluster_scores, recon_scores, c=colors, alpha=0.7, s=100)
            axes[1, 1].set_xlabel('Cluster Score')
            axes[1, 1].set_ylabel('Reconstruction Score')
            axes[1, 1].set_title('Cluster vs Reconstruction Scores')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            try:
                self.pdf_pages.savefig(fig, bbox_inches='tight')
            except Exception as e:
                print(f"[LFighter-AE] 客户端得分可视化添加到PDF失败: {e}")
            plt.close()
    
    
    def visualize_cluster_quality(self, cs0, cs1, good_cl, round_num):
        """可视化聚类质量"""
        if not self.should_visualize_this_round():
            return
            
        # 只在PDF报告中添加聚类质量可视化，不生成单独的图片文件
        if self.save_as_pdf and self.pdf_pages is not None:
            plt.figure(figsize=(10, 6))
            clusters = ['Cluster 0', 'Cluster 1']
            dissimilarities = [cs0, cs1]
            colors = ['green' if i == good_cl else 'red' for i in range(2)]
            
            plt.bar(clusters, dissimilarities, color=colors, alpha=0.7)
            plt.ylabel('Dissimilarity Score')
            plt.title('Cluster Quality Comparison')
            plt.grid(True, alpha=0.3)
            
            # 添加图例
            green_patch = plt.matplotlib.patches.Patch(color='green', label='Good Cluster')
            red_patch = plt.matplotlib.patches.Patch(color='red', label='Bad Cluster')
            plt.legend(handles=[green_patch, red_patch])
            
            try:
                self.pdf_pages.savefig(plt.gcf(), bbox_inches='tight')
            except Exception as e:
                print(f"[LFighter-AE] 聚类质量可视化添加到PDF失败: {e}")
            plt.close()
    
            plt.close()
    
            if attack_scope == 'large_multi':
                analysis['attack_type'] = 'large_multi_target'
                analysis['confidence'] = 0.85
                analysis['pattern_description'] = f'大规模多标签攻击，涉及{len(selected_classes)}个类别'
            elif attack_scope == 'small_multi':
                high_score_classes = selected_classes[relative_scores > 0.7]
                if len(high_score_classes) >= 3:
                    analysis['attack_type'] = 'small_multi_target'
                    analysis['confidence'] = 0.8
                    analysis['pattern_description'] = f'小规模多标签攻击，{len(high_score_classes)}个高影响类别'
                else:
                    analysis['attack_type'] = 'complex_sparse'
                    analysis['confidence'] = 0.6
                    analysis['pattern_description'] = f'复杂稀疏攻击模式，涉及{len(selected_classes)}个类别'
            elif attack_scope == 'traditional':
                analysis['attack_type'] = 'simple_targeted'
                analysis['confidence'] = 0.9
                analysis['pattern_description'] = '传统单源-单目标攻击'
            else:
                analysis['attack_type'] = 'unknown_local'
                analysis['confidence'] = 0.3
                analysis['pattern_description'] = '未知局部攻击模式'
        
        print(f'[LFighter-AE] 攻击模式分析: {analysis["attack_type"]} (confidence: {analysis["confidence"]:.2f})')
        print(f'[LFighter-AE] 模式描述: {analysis["pattern_description"]}')
        print(f'[LFighter-AE] 攻击范围: {attack_scope}')
        
        return analysis
        
    def clusters_dissimilarity(self, clusters):
        """计算聚类间相异性，处理空聚类情况"""
        n0 = len(clusters[0])
        n1 = len(clusters[1])
        m = n0 + n1 
        
        # 处理空聚类情况
        if n0 == 0:
            return 1.0, 0.0  # 空聚类0，聚类1更好
        if n1 == 0:
            return 0.0, 1.0  # 空聚类1，聚类0更好
        if n0 == 1:
            ds0 = 1.0  # 单样本聚类质量设为最差
        else:
            cs0 = smp.cosine_similarity(clusters[0]) - np.eye(n0)
            mincs0 = np.min(cs0, axis=1)
            ds0 = n0/m * (1 - np.mean(mincs0))
        
        if n1 == 1:
            ds1 = 1.0  # 单样本聚类质量设为最差
        else:
            cs1 = smp.cosine_similarity(clusters[1]) - np.eye(n1)
            mincs1 = np.min(cs1, axis=1)
            ds1 = n1/m * (1 - np.mean(mincs1))
        
        return ds0, ds1
    
    def create_autoencoder(self, input_dim, device):
        """创建AE模型"""
        class AE(nn.Module):
            def __init__(self, input_dim, hidden_dim, latent_dim):
                super(AE, self).__init__()
                # 编码器
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, latent_dim)
                )
                
                # 解码器
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, input_dim)
                )
            
            def encode(self, x):
                return self.encoder(x)
            
            def decode(self, z):
                return self.decoder(z)
            
            def forward(self, x):
                z = self.encode(x)
                decoded = self.decode(z)
                return z, decoded
        
        return AE(input_dim, self.ae_hidden_dim, self.ae_latent_dim).to(device)
    
    def train_autoencoder(self, features, device):
        """训练AE并返回潜在表示和重构误差"""
        features_tensor = torch.FloatTensor(features).to(device)
        input_dim = features_tensor.shape[1]
        
        # 创建AE
        ae = self.create_autoencoder(input_dim, device)
        optimizer = optim.Adam(ae.parameters(), lr=0.001, weight_decay=1e-5)
        
        # AE损失函数 - 只有重构损失
        criterion = nn.MSELoss(reduction='mean')
        
        # 记录训练损失用于可视化
        training_losses = []
        
        # 训练AE
        ae.train()
        for epoch in range(self.ae_epochs):
            optimizer.zero_grad()
            z, decoded = ae(features_tensor)
            
            # 计算重构损失
            recon_loss = criterion(decoded, features_tensor)
            recon_loss.backward()
            optimizer.step()
            
            # 记录损失
            training_losses.append(recon_loss.item())
            
            if epoch % 10 == 0:
                print(f'[LFighter-AE] AE Epoch {epoch+1}/{self.ae_epochs}: Recon Loss={recon_loss.item():.6f}')
        
        # 可视化训练过程
        if self.should_visualize_this_round():
            self.visualize_training_process(training_losses, self.round_counter)
        
        # 获取潜在表示和重构误差
        ae.eval()
        with torch.no_grad():
            z, decoded = ae(features_tensor)
            reconstruction_errors = torch.mean((features_tensor - decoded) ** 2, dim=1)
        
        return z.cpu().numpy(), reconstruction_errors.cpu().numpy(), training_losses[-1]
    
    def aggregate(self, global_model, local_models, ptypes):
        """基于LFighter + 传统Autoencoder的聚合方法"""
        import config
        device = getattr(config, 'DEVICE', 'cpu')
        
        # 增加轮数计数
        self.round_counter += 1
        # 提示当前可视化频率
        current_freq = 1 if self.round_counter <= 20 else 10
        next_viz_round = self.round_counter if self.round_counter % current_freq == 0 else ((self.round_counter // current_freq) + 1) * current_freq
        print(f"[LFighter-AE] Round {self.round_counter} - Visualization freq: every {current_freq} rounds (next: round {next_viz_round})")
        
        local_weights = [copy.deepcopy(model).state_dict() for model in local_models]
        m = len(local_models)
        
        # 转换为参数列表（与原LFighter一致）
        for i in range(m):
            local_models[i] = list(local_models[i].parameters())
        global_model = list(global_model.parameters())
        
        # 计算梯度差异（与原LFighter一致）
        dw = [None for i in range(m)]
        db = [None for i in range(m)]
        for i in range(m):
            dw[i] = global_model[-2].cpu().data.numpy() - local_models[i][-2].cpu().data.numpy()
            db[i] = global_model[-1].cpu().data.numpy() - local_models[i][-1].cpu().data.numpy()
        dw = np.asarray(dw)
        db = np.asarray(db)

        # 处理二分类情况（与原LFighter一致）
        if len(db[0]) <= 2:
            data = []
            for i in range(m):
                data.append(dw[i].reshape(-1))
            
            # 使用autoencoder进行特征学习
            latent_features, reconstruction_errors, final_loss = self.train_autoencoder(data, device)
            
            # Use KMeans clustering (in latent space)
            from sklearn.cluster import KMeans
            
            # KMeans聚类 (n_clusters=2)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(latent_features)
            
            print(f'[LFighter-AE] KMeans聚类: 2个聚类')
            print(f'[LFighter-AE] 聚类标签分布: {np.unique(labels)}')
            print(f'[LFighter-AE] 各标签数量: {[np.sum(labels == label) for label in [0, 1]]}')
            
            # 构建聚类用于质量评估
            clusters = {0: [], 1: []}
            for i, l in enumerate(labels):
                clusters[l].append(latent_features[i])
            
            # 聚类质量评估
            good_cl = 0
            cs0, cs1 = self.clusters_dissimilarity(clusters)
            if cs0 < cs1:
                good_cl = 1

            # 结合聚类结果和重构误差计算最终得分
            cluster_scores = np.ones([m])
            for i, l in enumerate(labels):
                if l != good_cl:
                    cluster_scores[i] = 0
            
            # 重构误差归一化（误差越大，得分越低）
            normalized_recon_errors = (reconstruction_errors - reconstruction_errors.min()) / (reconstruction_errors.max() - reconstruction_errors.min() + 1e-8)
            recon_scores = 1 - normalized_recon_errors
            
            # 组合得分：聚类得分和重构得分加权平均
            final_scores = (1 - self.reconstruction_weight) * cluster_scores + self.reconstruction_weight * recon_scores
            
            print(f'[LFighter-AE] Cluster quality: cs0={cs0:.4f}, cs1={cs1:.4f}, good_cluster={good_cl}')
            print(f'[LFighter-AE] Reconstruction errors: mean={reconstruction_errors.mean():.4f}, std={reconstruction_errors.std():.4f}')
            
                        # 可视化结果
            if self.should_visualize_this_round():
                # 创建总结报告
                metrics = {
                    'total_clients': m,
                    'good_clients': int(np.sum(final_scores > 0.5)),
                    'accuracy': int(np.sum(final_scores > 0.5)) / m,
                    'cs0': cs0,
                    'cs1': cs1,
                    'good_cluster': good_cl,
                    'mean_recon_error': reconstruction_errors.mean(),
                    'std_recon_error': reconstruction_errors.std(),
                    'final_loss': final_loss,
                    'reduction_method': 'Binary classification',
                    'attack_type': 'Binary classification mode',
                    'fixed_recon_weight': self.reconstruction_weight,
                    # KMeans配置信息
                    'clustering_method': 'KMeans',
                    'n_clusters': 2,
                    'cluster_labels': str([0, 1]),
                    'cluster_counts': str([np.sum(labels == label) for label in [0, 1]])
                }
                
                # 仅生成PDF报告，不生成单独的图片和文本文件
                self.create_pdf_report(self.round_counter, latent_features, labels, ptypes, final_scores, cs0, cs1, good_cl, metrics)
            
            global_weights = average_weights(local_weights, final_scores)
            return global_weights

        # 处理多分类情况
        # 检测异常类别（改进版：支持多标签攻击）
        norms = np.linalg.norm(dw, axis=-1)
        self.memory = np.sum(norms, axis=0)
        self.memory += np.sum(abs(db), axis=0)
        
        # 通用攻击检测策略：自适应选择局部vs全局特征
        sorted_classes = self.memory.argsort()
        memory_normalized = self.memory / (np.max(self.memory) + 1e-8)
        
        # 分析攻击分布模式
        high_threshold = 0.6  # 高影响阈值
        medium_threshold = 0.3  # 中等影响阈值
        
        high_impact_classes = sorted_classes[memory_normalized[sorted_classes] > high_threshold]
        medium_impact_classes = sorted_classes[memory_normalized[sorted_classes] > medium_threshold]
        
        # 计算攻击分布的均匀性
        memory_std = np.std(memory_normalized)
        memory_cv = memory_std / (np.mean(memory_normalized) + 1e-8)  # 变异系数
        
        print(f'[LFighter-AE] Memory scores: {dict(zip(range(len(self.memory)), self.memory))}')
        print(f'[LFighter-AE] Memory CV (变异系数): {memory_cv:.4f}')
        print(f'[LFighter-AE] High impact classes (>{high_threshold}): {high_impact_classes}')
        print(f'[LFighter-AE] Medium impact classes (>{medium_threshold}): {medium_impact_classes}')
        
        # 根据攻击分布选择特征提取策略
        if memory_cv < 0.5 and len(medium_impact_classes) >= len(self.memory) * 0.6:
            # 全局攻击模式（如1移位攻击）：所有类别都受到较均匀的影响
            attack_scope = 'global'
            selected_classes = list(range(len(self.memory)))  # 使用所有类别
            print(f'[LFighter-AE] 检测到全局攻击模式，使用所有{len(selected_classes)}个类别')
        elif len(high_impact_classes) >= 4:
            # 大规模多标签攻击
            attack_scope = 'large_multi'
            selected_classes = sorted_classes[-min(8, len(high_impact_classes) + 2):]
            print(f'[LFighter-AE] 检测到大规模多标签攻击，使用{len(selected_classes)}个高影响类别')
        elif len(high_impact_classes) >= 2:
            # 小规模多标签攻击
            attack_scope = 'small_multi'
            selected_classes = sorted_classes[-min(6, len(high_impact_classes) + 1):]
            print(f'[LFighter-AE] 检测到小规模多标签攻击，使用{len(selected_classes)}个类别')
        else:
            # 传统单一攻击或无明显攻击
            attack_scope = 'traditional'
            selected_classes = sorted_classes[-2:]
            print(f'[LFighter-AE] 检测到传统攻击模式，使用{len(selected_classes)}个类别')
        
        # 分析攻击模式（使用新的参数）
        attack_analysis = self.analyze_attack_pattern(selected_classes, self.memory, attack_scope, memory_cv)
        
        # 提取特征 - 修改为始终使用所有输出层梯度
        data = []
        for i in range(m):
            # 使用完整的梯度向量，不再区分攻击模式
            global_features = np.concatenate([dw[i].reshape(-1), db[i].reshape(-1)])
            data.append(global_features)
            
        print(f'[LFighter-AE] 使用全部输出层梯度作为特征 (维度: {len(data[0])})')

        # 统一降维处理（与原LFighter一致）
        def unified_dimension_reduction(features, target_dim=200, method='auto', standardize=True):
            features_array = np.array(features)
            n_samples, original_dim = features_array.shape
            
            if standardize:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                features_array = scaler.fit_transform(features_array)
                std_info = "standardized"
            else:
                std_info = "raw"
            
            max_pca_components = min(n_samples, original_dim)
            effective_target_dim = min(target_dim, max_pca_components)
            
            if original_dim <= effective_target_dim:
                return features_array, f"keep_all_{original_dim}_{std_info}"
            elif original_dim <= effective_target_dim * 2:
                return features_array[:, :effective_target_dim], f"truncate_{effective_target_dim}_{std_info}"
            else:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=effective_target_dim, random_state=42)
                reduced_features = pca.fit_transform(features_array)
                explained_ratio = np.sum(pca.explained_variance_ratio_)
                return reduced_features, f"pca_{effective_target_dim}_var{explained_ratio:.3f}_{std_info}"
        
        # 应用降维
        data_reduced, reduction_method = unified_dimension_reduction(data, target_dim=200)
        print(f'[LFighter-AE] Applied dimension reduction: {reduction_method}')
        
        # 使用autoencoder进行深度特征学习
        latent_features, reconstruction_errors, final_loss = self.train_autoencoder(data_reduced, device)
        
        # Use KMeans clustering (in latent space)
        from sklearn.cluster import KMeans
        
        # KMeans聚类 (n_clusters=2)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(latent_features)
        
        print(f'[LFighter-AE] KMeans聚类: 2个聚类')
        print(f'[LFighter-AE] 聚类标签分布: {np.unique(labels)}')
        print(f'[LFighter-AE] 各标签数量: {[np.sum(labels == label) for label in [0, 1]]}')
        
        # 构建聚类用于质量评估
        clusters = {0: [], 1: []}
        for i, l in enumerate(labels):
            clusters[l].append(latent_features[i])
        
        # 聚类质量评估
        good_cl = 0
        cs0, cs1 = self.clusters_dissimilarity(clusters)
        if cs0 < cs1:
            good_cl = 1

        # 结合聚类结果和重构误差
        cluster_scores = np.ones([m])
        for i, l in enumerate(labels):
            if l != good_cl:
                cluster_scores[i] = 0
        
        # 重构误差得分（误差越大，越可能是攻击者）
        normalized_recon_errors = (reconstruction_errors - reconstruction_errors.min()) / (reconstruction_errors.max() - reconstruction_errors.min() + 1e-8)
        recon_scores = 1 - normalized_recon_errors
        
        # 使用固定权重（不进行动态调整）
        print(f'[LFighter-AE] 使用固定重构权重: {self.reconstruction_weight:.2f}')
        
        # 组合最终得分
        final_scores = (1 - self.reconstruction_weight) * cluster_scores + self.reconstruction_weight * recon_scores
        
        # 设置阈值：只有组合得分超过0.5的客户端被认为是好的
        binary_scores = (final_scores > 0.5).astype(float)
        
        print(f'[LFighter-AE] Cluster quality: cs0={cs0:.4f}, cs1={cs1:.4f}, good_cluster={good_cl}')
        print(f'[LFighter-AE] Reconstruction errors: mean={reconstruction_errors.mean():.4f}, std={reconstruction_errors.std():.4f}')
        print(f'[LFighter-AE] Fixed reconstruction weight: {self.reconstruction_weight:.2f}')
        print(f'[LFighter-AE] Final scores: {final_scores}')
        print(f'[LFighter-AE] Selected good clients: {np.sum(binary_scores)}/{m}')
        
        # 可视化结果
        # 可视化结果
        if self.should_visualize_this_round():
            metrics = {
                'total_clients': m,
                'good_clients': int(np.sum(binary_scores)),
                'accuracy': int(np.sum(binary_scores)) / m,
                'cs0': cs0,
                'cs1': cs1,
                'good_cluster': good_cl,
                'mean_recon_error': reconstruction_errors.mean(),
                'std_recon_error': reconstruction_errors.std(),
                'final_loss': final_loss,
                'reduction_method': reduction_method,
                'feature_strategy': '使用全部输出层梯度',
                'feature_dim': len(data[0]),
                'selected_classes': str(selected_classes),  # 仅用于攻击模式分析
                'num_selected_classes': len(selected_classes),
                'attack_scope': attack_scope,
                'memory_max': self.memory.max(),
                'memory_mean': self.memory.mean(),
                'attack_type': attack_analysis['attack_type'],
                'attack_confidence': attack_analysis['confidence'],
                'attack_description': attack_analysis['pattern_description'],
                'fixed_recon_weight': self.reconstruction_weight,
                # KMeans配置信息
                'clustering_method': 'KMeans',
                'n_clusters': 2,
                'cluster_labels': str([0, 1]),
                'cluster_counts': str([np.sum(labels == label) for label in [0, 1]])
            }
            
            # 仅生成PDF报告，不生成单独的图片和文本文件
            self.create_pdf_report(self.round_counter, latent_features, labels, ptypes, binary_scores, cs0, cs1, good_cl, metrics)
        
        global_weights = average_weights(local_weights, binary_scores)
        return global_weights

    def create_summary_report(self, round_num, metrics):
        """创建总结报告"""
        if not self.should_visualize_this_round():
            return
        
    def analyze_attack_pattern(self, selected_classes, memory_scores, attack_scope, memory_cv):
        """
        通用攻击模式分析，支持全局和局部攻击
        
        注意：虽然此方法仍然分析攻击模式，但现在特征提取总是使用全部输出层梯度，
        不再根据攻击模式选择不同的特征子集。此分析仅用于调整重构权重和提供报告信息。
        
        Args:
            selected_classes: 选定的类别索引（仅用于分析）
            memory_scores: 每个类别的内存得分
            attack_scope: 攻击范围类型
            memory_cv: 内存得分的变异系数
            
        Returns:
            包含攻击类型、置信度和描述的字典
        """
        analysis = {
            'attack_type': 'unknown',
            'confidence': 0.0,
            'pattern_description': '',
            'scope': attack_scope
        }
        
        # 根据攻击范围进行不同的分析
        if attack_scope == 'global':
            # 全局攻击分析（如1移位攻击）
            if memory_cv < 0.3:
                analysis['attack_type'] = 'global_uniform'
                analysis['confidence'] = 0.9
                analysis['pattern_description'] = f'全局均匀攻击 (CV={memory_cv:.3f})，可能是移位攻击或全局标签翻转'
            else:
                analysis['attack_type'] = 'global_mixed'
                analysis['confidence'] = 0.7
                analysis['pattern_description'] = f'全局混合攻击 (CV={memory_cv:.3f})，影响所有类别但程度不均'
        else:
            # 局部攻击分析
            max_score = memory_scores.max()
            class_scores = memory_scores[selected_classes]
            relative_scores = class_scores / max_score if max_score > 0 else class_scores
            
            if attack_scope == 'large_multi':
                analysis['attack_type'] = 'large_multi_target'
                analysis['confidence'] = 0.85
                analysis['pattern_description'] = f'大规模多标签攻击，涉及{len(selected_classes)}个类别'
            elif attack_scope == 'small_multi':
                high_score_classes = selected_classes[relative_scores > 0.7]
                if len(high_score_classes) >= 3:
                    analysis['attack_type'] = 'small_multi_target'
                    analysis['confidence'] = 0.8
                    analysis['pattern_description'] = f'小规模多标签攻击，{len(high_score_classes)}个高影响类别'
                else:
                    analysis['attack_type'] = 'complex_sparse'
                    analysis['confidence'] = 0.6
                    analysis['pattern_description'] = f'复杂稀疏攻击模式，涉及{len(selected_classes)}个类别'
            elif attack_scope == 'traditional':
                analysis['attack_type'] = 'simple_targeted'
                analysis['confidence'] = 0.9
                analysis['pattern_description'] = '传统单源-单目标攻击'
            else:
                analysis['attack_type'] = 'unknown_local'
                analysis['confidence'] = 0.3
                analysis['pattern_description'] = '未知局部攻击模式'
        
        print(f'[LFighter-AE] 攻击模式分析: {analysis["attack_type"]} (confidence: {analysis["confidence"]:.2f})')
        print(f'[LFighter-AE] 模式描述: {analysis["pattern_description"]}')
        print(f'[LFighter-AE] 攻击范围: {attack_scope}')
        
        return analysis

