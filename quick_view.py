#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick viewer script for .t7 result files
"""

import torch
import os
import sys
from pathlib import Path

def load_single_result(file_path):
    """Load a single .t7 result file"""
    try:
        result = torch.load(file_path, map_location='cpu', weights_only=False)
        filename = Path(file_path).stem
        
        print(f"\n📊 Experiment Result: {filename}")
        print("=" * 60)
        
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
            
            print(f"Dataset: {dataset}")
            print(f"Model: {model}")
            print(f"Data Distribution: {dd_type}")
            print(f"Aggregation Rule: {rule}")
            print(f"Attackers Ratio: {attackers_ratio}")
            print(f"Local Epochs: {local_epochs}")
            print()
        
        # Display key performance metrics (improved format)
        if 'global_accuracies' in result and result['global_accuracies']:
            accuracies = result['global_accuracies']
            # Convert numpy types to Python float
            clean_accuracies = [float(acc) for acc in accuracies]
            print(f"🎯 Final Accuracy: {clean_accuracies[-1]:.2f}%")
            print(f"📈 Best Accuracy: {max(clean_accuracies):.2f}%")
            print(f"📊 Training Rounds: {len(clean_accuracies)-1} rounds")
            # Only show first 5 and last 5 values to avoid too long output
            if len(clean_accuracies) > 10:
                print(f"📊 Accuracy Trend: {clean_accuracies[:5]} ... {clean_accuracies[-5:]}")
            else:
                print(f"📊 Accuracy Changes: {[round(acc, 2) for acc in clean_accuracies]}")
        
        if 'test_losses' in result and result['test_losses']:
            losses = result['test_losses']
            clean_losses = [float(loss) for loss in losses]
            print(f"📉 Final Loss: {clean_losses[-1]:.4f}")
            print(f"📉 Best Loss: {min(clean_losses):.4f}")
        
        if 'asr' in result:
            asr_value = float(result['asr']) if hasattr(result['asr'], 'item') else result['asr']
            print(f"⚔️  Attack Success Rate (ASR): {asr_value:.2f}%")
        
        if 'avg_cpu_runtime' in result:
            runtime = float(result['avg_cpu_runtime']) if hasattr(result['avg_cpu_runtime'], 'item') else result['avg_cpu_runtime']
            print(f"⏱️  Average Runtime: {runtime:.3f}s")
        
        if 'source_class_accuracies' in result and result['source_class_accuracies']:
            source_accs = result['source_class_accuracies']
            clean_source_accs = [float(acc) for acc in source_accs]
            print(f"🎯 Source Class Accuracy (Total rounds: {len(clean_source_accs)}):")
            if len(clean_source_accs) > 10:
                print(f"    First 5: {[round(acc, 2) for acc in clean_source_accs[:5]]}")
                print(f"    Last 5: {[round(acc, 2) for acc in clean_source_accs[-5:]]}")
            else:
                print(f"    All: {[round(acc, 2) for acc in clean_source_accs]}")
        
        print("=" * 60)
        
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
    print("-" * 50)
    
    for i, file_path in enumerate(t7_files, 1):
        filename = file_path.stem
        print(f"{i:2d}. {filename}")
    
    return t7_files

def compare_rules(results_dir="./results"):
    """Simple comparison of different rules' performance"""
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
                
                if 'global_accuracies' in result and result['global_accuracies']:
                    final_acc = result['global_accuracies'][-1]
                    asr = result.get('asr', 0)
                    runtime = result.get('avg_cpu_runtime', 0)
                    
                    rule_stats[rule].append({
                        'accuracy': final_acc,
                        'asr': asr,
                        'runtime': runtime,
                        'attackers_ratio': attackers_ratio
                    })
        except:
            continue
    
    if not rule_stats:
        print("❌ No valid result data found!")
        return
    
    print("\n📊 Rule Performance Comparison:")
    print("=" * 80)
    print(f"{'Rule':15} {'Avg Accuracy':>12} {'Avg ASR':>10} {'Avg Time':>10} {'Exp Count':>10}")
    print("-" * 80)
    
    for rule, stats in rule_stats.items():
        if stats:
            avg_acc = sum(s['accuracy'] for s in stats) / len(stats)
            avg_asr = sum(s['asr'] for s in stats) / len(stats)
            avg_runtime = sum(s['runtime'] for s in stats) / len(stats)
            
            print(f"{rule:15} {avg_acc:11.2f}% {avg_asr:9.2f}% {avg_runtime:9.3f}s {len(stats):9d}")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        # View specific file directly
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            load_single_result(file_path)
        else:
            print(f"❌ File does not exist: {file_path}")
        return
    
    print("🔍 Federated Learning Experiment Results Quick Viewer")
    print("=" * 50)
    
    while True:
        print("\nSelect operation:")
        print("1. List all result files")
        print("2. View specific result file")
        print("3. Compare rule performance")
        print("0. Exit")
        
        choice = input("\nEnter choice (0-3): ").strip()
        
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
        else:
            print("❌ Invalid choice, please try again!")

if __name__ == "__main__":
    main() 