import os
import csv
import matplotlib.pyplot as plt
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

def generate_all_plots(folder_path):
   
    csv_path = folder_path + "comparative_table.csv"

    if not os.path.exists(csv_path):
        print(f"Skipping plots, {csv_path} not found.")
        return
        
    data = {}
    
    # Read the data from the CSV file
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Status'] == 'OK':
                prec = row['Precision']
                bs = int(row['Batch Size'])
                
                try: thpt = float(row['Throughput (Patches/sec)'])
                except ValueError: thpt = np.nan
                    
                try: vram = float(row['Peak VRAM (MB)'])
                except ValueError: vram = np.nan
                    
                try: acc = float(row['Accuracy (%)'])
                except ValueError: acc = np.nan
                    
                # Initialize precision dictionary if not exists
                if prec not in data:
                    data[prec] = {'bs': [], 'thpt': [], 'vram': [], 'acc': []}
                    
                # Store the parsed values
                data[prec]['bs'].append(bs)
                data[prec]['thpt'].append(thpt)
                data[prec]['vram'].append(vram)
                data[prec]['acc'].append(acc)
                
    if not data:
        print("No valid data to plot.")
        return

    # Define markers and extract all unique batch sizes
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    all_bs = set()
    for prec in data:
        all_bs.update(data[prec]['bs'])
    bs_list_sorted = sorted(list(all_bs))
    
    # 1. Inference Throughput vs Batch Size (Log Scale)
    plt.figure(figsize=(10, 6))
    for i, (prec, metrics) in enumerate(data.items()):
        bs_arr = np.array(metrics['bs'])
        thpt_arr = np.array(metrics['thpt'])
        valid = ~np.isnan(thpt_arr)
        if np.any(valid):
            sorted_idx = np.argsort(bs_arr[valid])
            plt.plot(bs_arr[valid][sorted_idx], thpt_arr[valid][sorted_idx], 
                     marker=markers[i % len(markers)], label=prec, linewidth=2, markersize=8)
                 
    plt.xlabel('Batch Size (Log Scale)', fontsize=12)
    plt.ylabel('Throughput (Patches/sec)', fontsize=12)
    plt.title('Inference Throughput vs. Batch Size', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xscale('log', base=2)
    plt.xticks(bs_list_sorted, [str(b) for b in bs_list_sorted])
    plt.tight_layout()
    output_thpt = folder_path + "throughput_vs_batchsize.png"
    plt.savefig(output_thpt, dpi=300)
    plt.close()
    print(f"\nSaved throughput plot to {output_thpt}")

    # 2. Peak Ram vs Batch Size (Log Scale)
    plt.figure(figsize=(10, 6))
    for i, (prec, metrics) in enumerate(data.items()):
        bs_arr = np.array(metrics['bs'])
        vram_arr = np.array(metrics['vram'])
        valid = ~np.isnan(vram_arr)
        if np.any(valid):
            sorted_idx = np.argsort(bs_arr[valid])
            plt.plot(bs_arr[valid][sorted_idx], vram_arr[valid][sorted_idx], 
                     marker=markers[i % len(markers)], label=prec, linewidth=2, markersize=8)
                 
    plt.xlabel('Batch Size (Log Scale)', fontsize=12)
    plt.ylabel('Peak VRAM (MB)', fontsize=12)
    plt.title('Peak RAM vs. Batch Size', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xscale('log', base=2)
    plt.xticks(bs_list_sorted, [str(b) for b in bs_list_sorted])
    plt.tight_layout()
    output_vram = folder_path + "peak_ram_vs_batchsize.png"
    plt.savefig(output_vram, dpi=300)
    plt.close()
    print(f"Saved Peak RAM plot to {output_vram}")

    # 3. Accuracy (averaged by batch size) vs Precision
    plt.figure(figsize=(10, 6))
    precisions = []
    avg_accuracies = []
    
    # Custom order: PyTorch, FP32, FP16, INT8 to maintain logic
    order = ["PyTorch", "FP32", "FP16", "INT8"]
    keys_sorted = sorted(list(data.keys()), key=lambda x: order.index(x) if x in order else len(order))
    
    for prec in keys_sorted:
        acc_arr = np.array(data[prec]['acc'])
        valid = ~np.isnan(acc_arr)
        if np.any(valid):
            avg_acc = np.mean(acc_arr[valid])
            precisions.append(prec)
            avg_accuracies.append(avg_acc)
            
    if precisions:
        plt.bar(precisions, avg_accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(precisions)])
        plt.xlabel('Precision', fontsize=12)
        plt.ylabel('Average Accuracy (%)', fontsize=12)
        plt.title('Average Accuracy vs. Precision', fontsize=14)
        
        # Add values on top of bars
        for i, v in enumerate(avg_accuracies):
            plt.text(i, v + (1.0 if max(avg_accuracies) > 10 else 0.1), f"{v:.2f}%", ha='center', fontsize=10)
            
        max_val = max(avg_accuracies)
        upper_lim = 100 if max_val <= 100 else max_val * 1.1
        if max_val > 90 and max_val <= 100:
            upper_lim = max(100, max_val + 5)
        else:
            upper_lim = max_val + 5
 
        if min(avg_accuracies) > 20:
            lower_lim = min(avg_accuracies) - 5
        else:
            lower_lim = 0
            
        plt.ylim(lower_lim, upper_lim)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        output_acc = folder_path + "accuracy_vs_precision.png"
        plt.savefig(output_acc, dpi=300)
        plt.close()
        print(f"Saved Accuracy vs Precision plot to {output_acc}")

if __name__ == '__main__':
    generate_all_plots("results/res1/")
    generate_all_plots("results/res2/")