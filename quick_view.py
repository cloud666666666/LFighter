#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick viewer script for .t7 result files
Enhanced with comprehensive metrics: TE, all-acc, source-acc, ASR, COV, average time
"""

import torch
import os
import sys
import numpy as np
from pathlib import Path

def calculate_coefficient_of_variation(values):
    """计算变异系数 (COV) - Coefficient of Variation"""
    if not values or len(values) < 2:
        return 0.0
    
    values_array = np.array(values)
    mean_val = np.mean(values_array)
    std_val = np.std(values_array, ddof=1)  # 样本标准差
    
    if mean_val == 0:
        return 0.0
    
    # 变异系数 = (标准差 / 平均值) × 100%
    cov = (std_val / mean_val) * 100
    return cov

def load_single_result(file_path):
    """Load a single .t7 result file with comprehensive metrics"""
    try:
        result = torch.load(file_path, map_location='cpu', weights_only=False)
        filename = Path(file_path).stem
        
        print(f"\n📊 Experiment Result: {filename}")
        print("=" * 80)
        
        # Parse filename (improved version, handles composite names like NON_IID)
        parts = filename.split('_')
        if len(parts) >= 5:
            dataset = parts[0]
            model = parts[1]
            
            # Handle data distribution type (may be composite names like NON_IID)
            if len(parts) >= 6 and parts[2] == 'NON' and parts[3] == 'IID':
                dd_type = 'NON_IID'
                rule = '_'.join(parts[4:-2])
            else:
                dd_type = parts[2]
                rule = '_'.join(parts[3:-2])
            
            attackers_ratio = parts[-2]
            local_epochs = parts[-1]
            
            print(f"📋 Experiment Configuration:")
            print(f"    Dataset: {dataset}")
            print(f"    Model: {model}")
            print(f"    Data Distribution: {dd_type}")
            print(f"    Aggregation Rule: {rule}")
            print(f"    Attackers Ratio: {attackers_ratio}")
            print(f"    Local Epochs: {local_epochs}")
            print()
        
        # 📊 Model Saved Metrics (Direct Output)
        print(f"🎯 Model Saved Metrics:")
        print("    (Raw data from model's saved state)")
        print("-" * 50)
        
        # 1. Global Accuracies (从模型直接读取)
        if 'global_accuracies' in result and result['global_accuracies']:
            accuracies = result['global_accuracies']
            clean_accuracies = [float(acc) for acc in accuracies]
            overall_accuracy = clean_accuracies[-1]
            print(f"📈 Overall Accuracy (Final):     {overall_accuracy:.2f}%")
            print(f"📊 Global Accuracies (All):      {clean_accuracies}")
        else:
            print(f"📈 Overall Accuracy:             N/A")
            overall_accuracy = 0
        
        # 2. Source Class Accuracies (从模型直接读取)
        if 'source_class_accuracies' in result and result['source_class_accuracies']:
            source_accs = result['source_class_accuracies']
            clean_source_accs = [float(acc) for acc in source_accs]
            if clean_source_accs:
                final_source_acc = clean_source_accs[-1]
                # COV (Coefficient of Variation) - 从现有数据计算
                cov_src_acc = calculate_coefficient_of_variation(clean_source_accs)
                print(f"🎯 Source-Acc (Final):           {final_source_acc:.2f}%")
                print(f"🎯 Source-Acc (All):             {clean_source_accs}")
                print(f"🔄 COV (Calculated):             {cov_src_acc:.2f}%")
            else:
                print(f"🎯 Source-Acc:                   N/A")
                print(f"🔄 COV:                          N/A")
        else:
            print(f"🎯 Source-Acc:                   N/A")
            print(f"🔄 COV:                          N/A")
        
        # 3. Test Losses (从模型直接读取)
        if 'test_losses' in result and result['test_losses']:
            losses = result['test_losses']
            clean_losses = [float(loss) for loss in losses]
            test_error = clean_losses[-1]  # 最终损失值作为测试错误
            print(f"❌ Test Error (TE):              {test_error:.4f}")
            print(f"📉 Test Losses (All):            {clean_losses}")
        else:
            print(f"❌ Test Error (TE):              N/A")
        
        # 4. ASR (Attack Success Rate) - 从模型直接读取
        if 'asr' in result:
            asr_value = float(result['asr']) if hasattr(result['asr'], 'item') else result['asr']
            print(f"⚔️  ASR (Direct from model):     {asr_value:.2f}%")
        else:
            print(f"⚔️  ASR:                         N/A")
        
        # 5. Average CPU Runtime - 从模型直接读取
        if 'avg_cpu_runtime' in result and result['avg_cpu_runtime'] is not None:
            runtime_raw = result['avg_cpu_runtime']
            if hasattr(runtime_raw, 'item'):
                runtime = float(runtime_raw.item())
            elif isinstance(runtime_raw, (int, float)):
                runtime = float(runtime_raw)
            else:
                runtime = 0.0
            
            if runtime > 0:
                print(f"⏱️  Average Time (Direct):       {runtime:.3f}s")
            else:
                print(f"⏱️  Average Time:                0.000s (No timing data)")
        else:
            print(f"⏱️  Average Time:                N/A (Not saved in model)")
        
        # 📊 Summary of 5 Core Metrics
        print(f"\n📊 Five Core Metrics Summary:")
        print("-" * 50)
        print(f"1. Test Error (TE):               {test_error:.4f}" if 'test_error' in locals() else "1. Test Error (TE):               N/A")
        print(f"2. Overall Accuracy:              {overall_accuracy:.2f}%")
        print(f"3. Source Class Accuracy:         {final_source_acc:.2f}%" if 'final_source_acc' in locals() else "3. Source Class Accuracy:         N/A")
        print(f"4. Attack Success Rate (ASR):     {asr_value:.2f}%" if 'asr_value' in locals() else "4. Attack Success Rate (ASR):     N/A")
        print(f"5. Coefficient of Variation:      {cov_src_acc:.2f}%" if 'cov_src_acc' in locals() else "5. Coefficient of Variation:      N/A")
        print(f"6. Average Time:                  {runtime:.3f}s" if 'runtime' in locals() and runtime > 0 else "6. Average Time:                  N/A")
        
        # 📋 Note about COV calculation
        if 'cov_src_acc' in locals():
            print(f"\n📋 Note: COV is calculated from source_class_accuracies array")
            print(f"   (COV was not saved in the original model results)")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Loading failed: {file_path} - {e}")

def list_all_results(results_dir="./results"):
    """List all .t7 result files"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ Results directory {results_dir} does not exist!")
        return []
    
    t7_files = list(results_path.glob("*.t7"))
    
    if not t7_files:
        print(f"❌ No .t7 files found in {results_dir}!")
        return []
    
    print(f"\n📁 Found {len(t7_files)} result files:")
    print("-" * 60)
    
    for i, file_path in enumerate(t7_files, 1):
        filename = file_path.stem
        print(f"{i:2d}. {filename}")
    
    return t7_files

def compare_rules(results_dir="./results"):
    """Enhanced comparison with comprehensive metrics"""
    results_path = Path(results_dir)
    t7_files = list(results_path.glob("*.t7"))
    
    if not t7_files:
        print("❌ No result files found!")
        return
    
    rule_stats = {}
    
    for file_path in t7_files:
        try:
            result = torch.load(file_path, map_location='cpu', weights_only=False)
            filename = file_path.stem
            parts = filename.split('_')
            
            if len(parts) >= 5:
                # Handle data distribution type (may be composite names like NON_IID)
                if len(parts) >= 6 and parts[2] == 'NON' and parts[3] == 'IID':
                    rule = '_'.join(parts[4:-2])
                else:
                    rule = '_'.join(parts[3:-2])
                attackers_ratio = float(parts[-2])
                
                if rule not in rule_stats:
                    rule_stats[rule] = []
                
                # Extract comprehensive metrics
                metrics = {'attackers_ratio': attackers_ratio}
                
                                 # Overall Accuracy and Test Error
                if 'global_accuracies' in result and result['global_accuracies']:
                    accuracies = [float(acc) for acc in result['global_accuracies']]
                    metrics['overall_accuracy'] = accuracies[-1]  # 重命名为更准确的名称
                    metrics['avg_accuracy'] = np.mean(accuracies)
                    metrics['best_accuracy'] = max(accuracies)
                else:
                    metrics['overall_accuracy'] = 0
                    metrics['avg_accuracy'] = 0
                    metrics['best_accuracy'] = 0
                
                # Test Error (TE) - 从损失函数获取
                if 'test_losses' in result and result['test_losses']:
                    losses = [float(loss) for loss in result['test_losses']]
                    metrics['test_error'] = losses[-1]  # 最终损失值
                else:
                    metrics['test_error'] = 0
                
                                 # Source-Acc and COV
                if 'source_class_accuracies' in result and result['source_class_accuracies']:
                    source_accs = [float(acc) for acc in result['source_class_accuracies']]
                    metrics['source_acc'] = source_accs[-1] if source_accs else 0
                    metrics['avg_source_acc'] = np.mean(source_accs) if source_accs else 0
                    metrics['cov_src_acc'] = calculate_coefficient_of_variation(source_accs) if source_accs else 0
                else:
                    metrics['source_acc'] = 0
                    metrics['avg_source_acc'] = 0
                    metrics['cov_src_acc'] = 0
                
                # ASR
                metrics['asr'] = result.get('asr', 0)
                
                # Average Time - 更好地处理缺失或无效数据
                runtime_raw = result.get('avg_cpu_runtime', None)
                if runtime_raw is not None:
                    if hasattr(runtime_raw, 'item'):
                        metrics['avg_time'] = float(runtime_raw.item())
                    elif isinstance(runtime_raw, (int, float)):
                        metrics['avg_time'] = float(runtime_raw)
                    else:
                        metrics['avg_time'] = 0.0
                else:
                    metrics['avg_time'] = 0.0
                
                rule_stats[rule].append(metrics)
                
        except Exception as e:
            print(f"⚠️  Skipping file {file_path}: {e}")
            continue
    
    if not rule_stats:
        print("❌ No valid result data found!")
        return
    
    print("\n📊 Comprehensive Algorithm Performance Comparison:")
    print("=" * 130)
    print(f"{'Algorithm':15} {'TE':>8} {'Overall(%)':>11} {'Source(%)':>10} {'ASR(%)':>8} {'COV(%)':>8} {'Time(s)':>8} {'Count':>6}")
    print("-" * 130)
    
    # Sort algorithms for better display
    sorted_rules = sorted(rule_stats.items(), key=lambda x: x[0])
    
    for rule, stats in sorted_rules:
        if stats:
            # Calculate averages across all experiments for this rule
            avg_overall_acc = np.mean([s['overall_accuracy'] for s in stats])
            avg_test_error = np.mean([s['test_error'] for s in stats])
            avg_source_acc = np.mean([s['source_acc'] for s in stats])
            avg_asr = np.mean([s['asr'] for s in stats])
            avg_cov = np.mean([s['cov_src_acc'] for s in stats])
            avg_time = np.mean([s['avg_time'] for s in stats])
            count = len(stats)
            
            print(f"{rule:15} {avg_test_error:7.4f} {avg_overall_acc:10.2f} {avg_source_acc:9.2f} {avg_asr:7.2f} {avg_cov:7.2f} {avg_time:7.3f} {count:5d}")
    
    print("-" * 130)
    
    # Detailed breakdown by attacker ratio
    print(f"\n📋 Detailed Performance by Attacker Ratio:")
    print("=" * 140)
    
    # Get all unique attacker ratios
    all_ratios = set()
    for stats in rule_stats.values():
        for s in stats:
            all_ratios.add(s['attackers_ratio'])
    
    sorted_ratios = sorted(all_ratios)
    
    for ratio in sorted_ratios:
        print(f"\n🎯 Attacker Ratio: {ratio*100:.0f}%")
        print("-" * 100)
        print(f"{'Algorithm':15} {'TE':>8} {'Overall(%)':>11} {'Source(%)':>10} {'ASR(%)':>8} {'COV(%)':>8} {'Time(s)':>8}")
        print("-" * 110)
        
        for rule, stats in sorted_rules:
            # Filter stats for this ratio
            ratio_stats = [s for s in stats if abs(s['attackers_ratio'] - ratio) < 0.01]
            
            if ratio_stats:
                # Average across multiple runs (if any) for this ratio
                overall_acc = np.mean([s['overall_accuracy'] for s in ratio_stats])
                test_error = np.mean([s['test_error'] for s in ratio_stats])
                source_acc = np.mean([s['source_acc'] for s in ratio_stats])
                asr = np.mean([s['asr'] for s in ratio_stats])
                cov = np.mean([s['cov_src_acc'] for s in ratio_stats])
                time_avg = np.mean([s['avg_time'] for s in ratio_stats])
                
                print(f"{rule:15} {test_error:7.4f} {overall_acc:10.2f} {source_acc:9.2f} {asr:7.2f} {cov:7.2f} {time_avg:7.3f}")
            else:
                print(f"{rule:15} {'N/A':>8} {'N/A':>11} {'N/A':>10} {'N/A':>8} {'N/A':>8} {'N/A':>8}")

def export_csv(results_dir="./results", output_file="experiment_results.csv"):
    """Export comprehensive results to CSV file"""
    results_path = Path(results_dir)
    t7_files = list(results_path.glob("*.t7"))
    
    if not t7_files:
        print("❌ No result files found!")
        return
    
    import csv
    
    csv_rows = []
    headers = ['Algorithm', 'Dataset', 'Distribution', 'Attacker_Ratio', 'Test_Error', 'Overall_Accuracy', 'Source_Acc', 'ASR', 'COV_Src_Acc', 'Avg_Time', 'Best_Accuracy', 'Final_Loss']
    
    for file_path in t7_files:
        try:
            result = torch.load(file_path, map_location='cpu', weights_only=False)
            filename = file_path.stem
            parts = filename.split('_')
            
            if len(parts) >= 5:
                dataset = parts[0]
                
                # Handle data distribution type
                if len(parts) >= 6 and parts[2] == 'NON' and parts[3] == 'IID':
                    dd_type = 'NON_IID'
                    rule = '_'.join(parts[4:-2])
                else:
                    dd_type = parts[2]
                    rule = '_'.join(parts[3:-2])
                
                attackers_ratio = float(parts[-2])
                
                # Extract metrics
                row = [rule, dataset, dd_type, attackers_ratio]
                
                if 'global_accuracies' in result and result['global_accuracies']:
                    accuracies = [float(acc) for acc in result['global_accuracies']]
                    overall_acc = accuracies[-1]
                    
                    # Test Error from loss function
                    test_error = 0
                    if 'test_losses' in result and result['test_losses']:
                        losses = [float(loss) for loss in result['test_losses']]
                        test_error = losses[-1]
                    
                    row.extend([
                        test_error,  # Test Error (from loss function)
                        overall_acc,  # Overall Accuracy
                    ])
                    
                    # Source-Acc
                    if 'source_class_accuracies' in result and result['source_class_accuracies']:
                        source_accs = [float(acc) for acc in result['source_class_accuracies']]
                        row.append(source_accs[-1] if source_accs else 0)
                    else:
                        row.append(0)
                    
                    # ASR, COV, Time
                    cov_src_acc = 0
                    if 'source_class_accuracies' in result and result['source_class_accuracies']:
                        source_accs = [float(acc) for acc in result['source_class_accuracies']]
                        cov_src_acc = calculate_coefficient_of_variation(source_accs) if source_accs else 0
                    
                    # 处理平均时间
                    avg_time = 0.0
                    runtime_raw = result.get('avg_cpu_runtime', None)
                    if runtime_raw is not None:
                        if hasattr(runtime_raw, 'item'):
                            avg_time = float(runtime_raw.item())
                        elif isinstance(runtime_raw, (int, float)):
                            avg_time = float(runtime_raw)
                    
                    row.extend([
                        result.get('asr', 0),
                        cov_src_acc,
                        avg_time,
                        max(accuracies),
                    ])
                    
                    # Final Loss
                    if 'test_losses' in result and result['test_losses']:
                        losses = [float(loss) for loss in result['test_losses']]
                        row.append(losses[-1])
                    else:
                        row.append(0)
                    
                    csv_rows.append(row)
                    
        except Exception as e:
            print(f"⚠️  Skipping file {file_path}: {e}")
            continue
    
    # Write CSV
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(csv_rows)
        
        print(f"✅ Results exported to {output_file}")
        print(f"📊 Total {len(csv_rows)} experiments exported")
        
    except Exception as e:
        print(f"❌ Failed to export CSV: {e}")

def main():
    """Main function with enhanced options"""
    if len(sys.argv) > 1:
        # View specific file directly
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            load_single_result(file_path)
        else:
            print(f"❌ File does not exist: {file_path}")
        return
    
    print("🔍 LFighter Experiment Results Comprehensive Viewer")
    print("=" * 60)
    
    while True:
        print("\nSelect operation:")
        print("1. List all result files")
        print("2. View specific result file") 
        print("3. Compare algorithm performance (comprehensive)")
        print("4. Export results to CSV")
        print("0. Exit")
        
        choice = input("\nEnter choice (0-4): ").strip()
        
        if choice == '0':
            print("👋 Goodbye!")
            break
        elif choice == '1':
            list_all_results()
        elif choice == '2':
            files = list_all_results()
            if files:
                try:
                    idx = int(input(f"\nEnter file number (1-{len(files)}): ")) - 1
                    if 0 <= idx < len(files):
                        load_single_result(files[idx])
                    else:
                        print("❌ Invalid file number!")
                except ValueError:
                    print("❌ Please enter a valid number!")
        elif choice == '3':
            compare_rules()
        elif choice == '4':
            output_file = input("Enter CSV filename (default: experiment_results.csv): ").strip()
            if not output_file:
                output_file = "experiment_results.csv"
            export_csv(output_file=output_file)
        else:
            print("❌ Invalid choice, please try again!")

if __name__ == "__main__":
    main() 