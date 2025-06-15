#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验结果分析器 - 用于读取和分析.t7格式的实验结果文件
"""

import torch
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
from pathlib import Path

class ResultAnalyzer:
    def __init__(self, results_dir="./results"):
        self.results_dir = Path(results_dir)
        self.results = {}
        self.df = None
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def load_all_results(self):
        """加载所有.t7结果文件"""
        if not self.results_dir.exists():
            print(f"结果目录 {self.results_dir} 不存在！")
            return
        
        t7_files = list(self.results_dir.glob("*.t7"))
        if not t7_files:
            print(f"在 {self.results_dir} 中未找到.t7文件！")
            return
        
        print(f"找到 {len(t7_files)} 个结果文件")
        
        for file_path in t7_files:
            try:
                # 解析文件名
                filename = file_path.stem
                parts = filename.split('_')
                
                if len(parts) >= 5:
                    dataset = parts[0]
                    model = parts[1] 
                    
                    # 处理数据分布类型 (可能是NON_IID等复合名称)
                    if len(parts) >= 6 and parts[2] == 'NON' and parts[3] == 'IID':
                        dd_type = 'NON_IID'
                        rule = '_'.join(parts[4:-2])
                    else:
                        dd_type = parts[2]
                        rule = '_'.join(parts[3:-2])  # 处理可能的多部分规则名
                    
                    attackers_ratio = parts[-2]
                    local_epochs = parts[-1]
                else:
                    print(f"跳过文件名格式不正确的文件: {filename}")
                    continue
                
                # 加载结果（兼容PyTorch 2.6+）
                result = torch.load(file_path, map_location='cpu', weights_only=False)
                
                # 提取关键信息
                exp_info = {
                    'file_path': str(file_path),
                    'dataset': dataset,
                    'model': model,
                    'dd_type': dd_type,
                    'rule': rule,
                    'attackers_ratio': float(attackers_ratio),
                    'local_epochs': int(local_epochs),
                    'final_accuracy': result.get('global_accuracies', [0])[-1] if result.get('global_accuracies') else 0,
                    'max_accuracy': max(result.get('global_accuracies', [0])) if result.get('global_accuracies') else 0,
                    'final_loss': result.get('test_losses', [float('inf')])[-1] if result.get('test_losses') else float('inf'),
                    'min_loss': min(result.get('test_losses', [float('inf')])) if result.get('test_losses') else float('inf'),
                    'asr': result.get('asr', 0),
                    'avg_cpu_runtime': result.get('avg_cpu_runtime', 0),
                    'global_accuracies': result.get('global_accuracies', []),
                    'test_losses': result.get('test_losses', []),
                    'source_class_accuracies': result.get('source_class_accuracies', [])
                }
                
                key = f"{dataset}_{model}_{dd_type}_{rule}_{attackers_ratio}_{local_epochs}"
                self.results[key] = exp_info
                
                print(f"✓ 加载: {filename}")
                
            except Exception as e:
                print(f"✗ 加载失败: {filename} - {e}")
        
        print(f"\n成功加载 {len(self.results)} 个实验结果")
        self._create_dataframe()
    
    def _create_dataframe(self):
        """将结果转换为DataFrame以便分析"""
        if not self.results:
            return
        
        data = []
        for key, result in self.results.items():
            row = {
                'experiment': key,
                'dataset': result['dataset'],
                'model': result['model'],
                'dd_type': result['dd_type'],
                'rule': result['rule'],
                'attackers_ratio': result['attackers_ratio'],
                'local_epochs': result['local_epochs'],
                'final_accuracy': result['final_accuracy'],
                'max_accuracy': result['max_accuracy'],
                'final_loss': result['final_loss'],
                'min_loss': result['min_loss'],
                'asr': result['asr'],
                'cpu_runtime': result['avg_cpu_runtime']
            }
            data.append(row)
        
        self.df = pd.DataFrame(data)
        print(f"\n数据框创建完成，共 {len(self.df)} 条记录")
    
    def show_summary(self):
        """显示实验结果摘要"""
        if self.df is None:
            print("请先加载结果文件！")
            return
        
        print("\n" + "="*60)
        print("📊 实验结果摘要")
        print("="*60)
        
        print(f"总实验数量: {len(self.df)}")
        print(f"数据集: {self.df['dataset'].unique()}")
        print(f"模型: {self.df['model'].unique()}")
        print(f"聚合规则: {self.df['rule'].unique()}")
        print(f"攻击者比例: {sorted(self.df['attackers_ratio'].unique())}")
        
        print("\n📈 性能统计:")
        print(f"最高准确率: {self.df['max_accuracy'].max():.2f}%")
        print(f"平均准确率: {self.df['final_accuracy'].mean():.2f}%")
        print(f"最低损失: {self.df['min_loss'].min():.4f}")
        print(f"平均ASR: {self.df['asr'].mean():.2f}%")
        
        # 按规则分组的性能
        print("\n📋 各聚合规则性能对比:")
        rule_stats = self.df.groupby('rule').agg({
            'final_accuracy': ['mean', 'std', 'max'],
            'asr': ['mean', 'std'],
            'cpu_runtime': ['mean', 'std']
        }).round(3)
        
        print(rule_stats)
    
    def compare_rules(self, metrics=['final_accuracy', 'asr', 'cpu_runtime']):
        """比较不同聚合规则的性能"""
        if self.df is None:
            print("请先加载结果文件！")
            return
        
        rules = self.df['rule'].unique()
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # 箱线图
            sns.boxplot(data=self.df, x='rule', y=metric, ax=ax)
            ax.set_title(f'{metric} 对比')
            ax.tick_params(axis='x', rotation=45)
            
            # 添加均值点
            rule_means = self.df.groupby('rule')[metric].mean()
            for j, rule in enumerate(rules):
                ax.scatter(j, rule_means[rule], color='red', s=50, zorder=5)
        
        plt.tight_layout()
        plt.show()
    
    def plot_convergence(self, filter_dict=None):
        """绘制收敛曲线"""
        if self.df is None:
            print("请先加载结果文件！")
            return
        
        # 筛选实验
        filtered_results = self.results
        if filter_dict:
            filtered_results = {k: v for k, v in self.results.items() 
                              if all(v.get(key) == value for key, value in filter_dict.items())}
        
        if not filtered_results:
            print("没有找到匹配的实验结果！")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 准确率曲线
        for key, result in filtered_results.items():
            accuracies = result['global_accuracies']
            if accuracies:
                ax1.plot(accuracies, label=result['rule'], marker='o', markersize=4)
        
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('全局准确率 (%)')
        ax1.set_title('训练准确率收敛曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 损失曲线
        for key, result in filtered_results.items():
            losses = result['test_losses']
            if losses:
                ax2.plot(losses, label=result['rule'], marker='s', markersize=4)
        
        ax2.set_xlabel('训练轮次')
        ax2.set_ylabel('测试损失')
        ax2.set_title('测试损失收敛曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def show_best_results(self, metric='final_accuracy', top_k=5):
        """显示最佳结果"""
        if self.df is None:
            print("请先加载结果文件！")
            return
        
        ascending = metric in ['final_loss', 'min_loss', 'asr', 'cpu_runtime']
        best_results = self.df.nlargest(top_k, metric) if not ascending else self.df.nsmallest(top_k, metric)
        
        print(f"\n🏆 {metric} 排名前 {top_k} 的实验:")
        print("-" * 80)
        
        for idx, (_, row) in enumerate(best_results.iterrows(), 1):
            print(f"{idx}. {row['rule']} (攻击者比例: {row['attackers_ratio']})")
            print(f"   准确率: {row['final_accuracy']:.2f}% | ASR: {row['asr']:.2f}% | 运行时间: {row['cpu_runtime']:.3f}s")
            print(f"   数据集: {row['dataset']} | 模型: {row['model']}")
            print()
    
    def export_results(self, output_path="experiment_results.csv"):
        """导出结果到CSV文件"""
        if self.df is None:
            print("请先加载结果文件！")
            return
        
        self.df.to_csv(output_path, index=False)
        print(f"结果已导出到: {output_path}")
    
    def search_experiments(self, **kwargs):
        """搜索特定的实验结果"""
        if self.df is None:
            print("请先加载结果文件！")
            return
        
        filtered_df = self.df.copy()
        
        for key, value in kwargs.items():
            if key in filtered_df.columns:
                if isinstance(value, (list, tuple)):
                    filtered_df = filtered_df[filtered_df[key].isin(value)]
                else:
                    filtered_df = filtered_df[filtered_df[key] == value]
        
        print(f"\n🔍 搜索结果 (共 {len(filtered_df)} 条):")
        if len(filtered_df) > 0:
            print(filtered_df[['rule', 'attackers_ratio', 'final_accuracy', 'asr', 'cpu_runtime']].to_string(index=False))
        else:
            print("未找到匹配的实验结果")
        
        return filtered_df

def main():
    """主函数 - 交互式结果分析"""
    analyzer = ResultAnalyzer()
    
    print("🔬 联邦学习实验结果分析器")
    print("=" * 50)
    
    # 加载所有结果
    analyzer.load_all_results()
    
    if not analyzer.results:
        print("未找到任何结果文件，请检查results目录！")
        return
    
    # 显示摘要
    analyzer.show_summary()
    
    # 交互式菜单
    while True:
        print("\n" + "="*50)
        print("📋 选择操作:")
        print("1. 显示最佳结果")
        print("2. 比较聚合规则")
        print("3. 绘制收敛曲线")
        print("4. 搜索实验")
        print("5. 导出结果")
        print("6. 重新加载")
        print("0. 退出")
        print("="*50)
        
        choice = input("请输入选择 (0-6): ").strip()
        
        if choice == '0':
            print("👋 再见！")
            break
        elif choice == '1':
            metric = input("选择指标 (final_accuracy/asr/cpu_runtime): ").strip() or 'final_accuracy'
            analyzer.show_best_results(metric=metric)
        elif choice == '2':
            analyzer.compare_rules()
        elif choice == '3':
            # 可以添加筛选条件
            print("绘制所有实验的收敛曲线...")
            analyzer.plot_convergence()
        elif choice == '4':
            print("搜索示例: rule='lfighter_dbo', attackers_ratio=0.2")
            search_input = input("输入搜索条件 (格式: key=value, key=value): ").strip()
            if search_input:
                try:
                    search_dict = {}
                    for item in search_input.split(','):
                        key, value = item.split('=')
                        key = key.strip()
                        value = value.strip().strip("'\"")
                        # 尝试转换数值
                        try:
                            value = float(value)
                            if value.is_integer():
                                value = int(value)
                        except ValueError:
                            pass
                        search_dict[key] = value
                    analyzer.search_experiments(**search_dict)
                except Exception as e:
                    print(f"搜索格式错误: {e}")
        elif choice == '5':
            filename = input("输入文件名 (默认: experiment_results.csv): ").strip() or "experiment_results.csv"
            analyzer.export_results(filename)
        elif choice == '6':
            analyzer.load_all_results()
        else:
            print("❌ 无效选择，请重试")

if __name__ == "__main__":
    main() 