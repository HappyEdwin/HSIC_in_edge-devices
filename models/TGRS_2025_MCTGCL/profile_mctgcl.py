import os
import time
import numpy as np
import torch
import tensorrt as trt
import subprocess
import csv
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from train import loadData, applyPCA, createImageCubes

def profile_preprocessing(dataset_name="Indian", all_pixels = False):
    print("\n--- PREPROCESSING PROFILING (T_pre) ---")
    print(f"{'Dataset':<13}| {'Dim Original':<15}| {'PCA Time (ms)':<14}| {'Patching Time (ms)':<19}| {'Total T_pre (ms)'}")
    
    #print(f"Loading real data: {dataset_name}...")
    X, y = loadData(dataset_name)
    #print(X.shape)
    #X = X[0:64, 0:64, :]
    #print(X.shape)
    H, W, B = X.shape
    
    # Process PCA (Initial pass to prepare dimensions/labels, warmups will do it again)
    X_pca = applyPCA(X, 30)
    
    pca_time = 0.0
    patch_time = 0.0
    
    warmup = 5
    iters = 20
    
    # Warmup and time PCA
    for _ in range(warmup):
        _ = applyPCA(X, 30)
    
    start = time.perf_counter()
    for _ in range(iters):
        X_pca = applyPCA(X, 30)
    end = time.perf_counter()
    pca_time = (end - start) * 1000 / iters

    del X, _
    #print("Checkpoint 1")
    
    # Warmup and time Patching
    for _ in range(warmup):
        _ = createImageCubes(X_pca, y, windowSize=13, removeZeroLabels=False)
    
    #print("Checkpoint 2")
    del _

    start = time.perf_counter()
    for _ in range(iters):
        patches, patches_labels = createImageCubes(X_pca, y, windowSize=13, removeZeroLabels=False)
    end = time.perf_counter()
    patch_time = (end - start) * 1000 / iters
    
    #print("Checkpoint 3")
    # We only need the patches and their corresponding valid labels for inference & accuracy
    del X_pca, _
    import gc
    gc.collect()
    
    total_pre = pca_time + patch_time
    dim_str = f"{H}x{W}x{B}"
    
    print(f"{dataset_name:<13}| {dim_str:<15}| {pca_time:<14.2f}| {patch_time:.2f}{'':<13}| {total_pre:.2f}")
    
    # Reshape testing patches to match DL model input dims
    patches = patches.reshape(-1, 13, 13, 30, 1)
    patches = patches.transpose(0, 4, 3, 1, 2).astype(np.float32)
    
    if all_pixels:
        test_patches = patches
        test_labels = patches_labels - 1  # 0-indexed classes
    else:
        # Filter only non-zero labels for accuracy metrics
        valid_mask = patches_labels > 0
        test_patches = patches[valid_mask]
        test_labels = patches_labels[valid_mask] - 1  # 0-indexed classes
    
    print(f"Test patches shape: {test_patches.shape}")

    #save_image_from_patches(patches, patches_labels, dataset_name, deleted_patches = False)
    #save_image_from_patches(patches, patches_labels, dataset_name, deleted_patches = True)

    return test_patches, test_labels

"""
def save_image_from_patches(patches, labels, dataset, deleted_patches = True):
    import matplotlib.pyplot as plt

    print(f"\nAmount original patches: {patches.shape[0]} of {patches.shape[1]}x{patches.shape[2]}x{patches.shape[3]}x{patches.shape[4]}")

    print(f"Max label: {np.max(labels)}")

    # Filter only non-zero labels for accuracy metrics
    if deleted_patches:
        valid_mask = labels > 0
        patches = patches[valid_mask]
        labels = labels[valid_mask] # 0-indexed classes
        print(f"Amount of patches: {patches.shape[0]} of {patches.shape[1]}x{patches.shape[2]}x{patches.shape[3]}x{patches.shape[4]}")
        dataset = dataset + "_deleted"
    
    print(f"Selecting 6 patches")
    print(f"Patch to print shape ({patches[0,0,5,:,:].shape})")

    plot_index = [0, 1, 2, 3121, 3122, 3123] 

    plt.figure(figsize=(12, 8))
    j = 0
    for i in plot_index:
        j += 1
        plt.subplot(2, 3, j)
        plt.imshow(patches[i, 0, 5, :, :], cmap='gray')
        plt.title(f"Patch {i}: Class {int(labels[i])}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"image_from_patches_{dataset}.png")
    plt.close()

    print("Patches preview images saved")
"""
def profile_pytorch_baseline(test_data, test_labels, dataset="Indian"):
    print(f"\n--- PYTORCH BASELINE ACCURACY & PROFILING ---")
    import mctgcl
    from sklearn.metrics import accuracy_score
    device = torch.device('cuda')
    model = mctgcl.mctgcl(num_classes=16, num_tokens=121).to(device)
    model.load_state_dict(torch.load(f"params/{dataset}.pt", map_location=device))
    model.eval()
    
    batch_sizes = [16, 32, 64, 128, 256, 512]
    VRAM_LIMIT_MB = 8000
    
    csv_results = []
    
    print(f"{'Prec':<8}| {'BS':<8}| {'Peak VRAM (MB)':<15}| {'Latency (ms/batch)':<19}| {'Throughput (Patches/sec)':<25}| {'Accuracy (%)':<15}| {'Status'}")
    
    for BS in batch_sizes:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Warmup
            dummy_input = torch.randn(BS, 1, 30, 13, 13).to(device)
            with torch.no_grad():
                for _ in range(20):
                    _ = model(dummy_input)
                    
            free, total = torch.cuda.mem_get_info()
            peak_vram = (total - free) / (1024 * 1024)
            
            if peak_vram > VRAM_LIMIT_MB:
                print(f"{'PyTorch':<8}| {BS:<8}| {peak_vram:<15.2f}| {'N/A':<19}| {'N/A':<25}| {'N/A':<15}| OOM")
                csv_results.append(["PyTorch", BS, f"{peak_vram:.2f}", "N/A", "N/A", "N/A", "OOM"])
                continue
                
            iters = 100
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            
            with torch.no_grad():
                start_evt.record()
                for _ in range(iters):
                    _ = model(dummy_input)
                end_evt.record()
            torch.cuda.synchronize()
            
            latency = start_evt.elapsed_time(end_evt) / iters
            throughput = (BS * 1000) / latency
            
            # Accuracy profiling
            total_samples = len(test_data)
            predictions = []
            
            with torch.no_grad():
                for i in range(0, total_samples, BS):
                    end_idx = min(i + BS, total_samples)
                    current_batch = test_data[i:end_idx]
                    
                    inputs = torch.from_numpy(current_batch).to(device)
                    out, _ = model(inputs)
                    preds_batch = out.argmax(dim=1).cpu().numpy()
                    predictions.extend(preds_batch)
                    
            oa = accuracy_score(test_labels, predictions)
            
            print(f"{'PyTorch':<8}| {BS:<8}| {peak_vram:<15.2f}| {latency:<19.2f}| {throughput:<25.2f}| {oa * 100:<15.2f}| OK")
            csv_results.append(["PyTorch", BS, f"{peak_vram:.2f}", f"{latency:.2f}", f"{throughput:.2f}", f"{oa * 100:.2f}", "OK"])
            
            del dummy_input
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{'PyTorch':<8}| {BS:<8}| {'> '+str(VRAM_LIMIT_MB):<15}| {'N/A':<19}| {'N/A':<25}| {'N/A':<15}| OOM")
                csv_results.append(["PyTorch", BS, f"> {VRAM_LIMIT_MB}", "N/A", "N/A", "N/A", "OOM"])
                torch.cuda.empty_cache()
            else:
                print(f"PyTorch Exception for BS {BS}: {e}")
                csv_results.append(["PyTorch", BS, "N/A", "N/A", "N/A", "N/A", f"Error: {e}"])
            
    return csv_results


def profile_trt(test_data, test_labels, csv_results_pts, dataset="Indian"):
    print("\n--- TENSORRT INFERENCE PROFILING (T_inf) ---")
    print(f"{'Prec':<8}| {'BS':<8}| {'Peak VRAM (MB)':<15}| {'Latency (ms/batch)':<19}| {'Throughput (Patches/sec)':<25}| {'Accuracy (%)':<15}| {'Status'}")
    
    device = torch.device('cuda')
    batch_sizes = [16, 32, 64, 128, 256, 512]
    precisions = ["FP32", "FP16", "INT8"]
    VRAM_LIMIT_MB = 8000
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    csv_results = csv_results_pts.copy()
    
    for prec in precisions:
        for BS in batch_sizes:
            try:
                engine_path = f'profiles/mctgcl_bs{BS}_{prec.lower()}.engine'
                
                if not os.path.exists(engine_path):
                    print(f"{prec:<8}| {BS:<8}| {'N/A':<15}| {'N/A':<19}| {'N/A':<25}| {'N/A':<15}| Engine Not Found")
                    csv_results.append([prec, BS, "N/A", "N/A", "N/A", "N/A", "Engine Not Found"])
                    continue
                    
                runtime = trt.Runtime(TRT_LOGGER)
                with open(engine_path, 'rb') as f:
                    engine = runtime.deserialize_cuda_engine(f.read())
                context = engine.create_execution_context()
                
                # Prepare Real Data Batch
                if len(test_data) < BS:
                    pad = np.zeros((BS - len(test_data), *test_data.shape[1:]), dtype=test_data.dtype)
                    batch_data = np.concatenate([test_data, pad], axis=0)
                else:
                    batch_data = test_data[:BS]
                
                # Create Dedicated stream to avoid warnings
                trt_stream = torch.cuda.Stream()
                
                allocations = []
                input_allocations = []
                output_allocations = []
                input_names = []
                output_names = []
                
                if BS == batch_sizes[0] and prec == precisions[0]:
                    # print("\n--- TENSORRT TENSOR MAPPING ---")
                    pass
                    
                for i in range(engine.num_io_tensors):
                    name = engine.get_tensor_name(i)
                    shape = context.get_tensor_shape(name)
                    dtype_trt = engine.get_tensor_dtype(name)
                    
                    # if BS == batch_sizes[0] and prec == precisions[0]:
                    #     print(f"Tensor {i}: '{name}', Shape: {shape}, Dtype: {dtype_trt}, Mode: {engine.get_tensor_mode(name)}")
                    
                    if dtype_trt == trt.float16:
                        dtype = torch.float16
                    elif dtype_trt == trt.int8:
                        dtype = torch.int8
                    elif dtype_trt == trt.int32:
                        dtype = torch.int32
                    else:
                        dtype = torch.float32
                        
                    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                        input_names.append(name)
                        alloc = torch.from_numpy(batch_data).contiguous().to(device)
                        if alloc.dtype != dtype:
                            alloc = alloc.to(dtype).contiguous()
                        input_allocations.append(alloc)
                    else:
                        output_names.append(name)
                        alloc = torch.empty(tuple(shape), dtype=dtype, device=device).contiguous()
                        output_allocations.append(alloc)
                    
                    context.set_tensor_address(name, int(alloc.data_ptr()))
                    allocations.append(alloc)
                    
                torch.cuda.reset_peak_memory_stats()
                
                # Warmup
                with torch.cuda.stream(trt_stream):
                    for _ in range(100):
                        context.execute_async_v3(stream_handle=trt_stream.cuda_stream)
                    trt_stream.synchronize()
                
                free, total = torch.cuda.mem_get_info()
                peak_vram = (total - free) / (1024 * 1024)
                
                if peak_vram > VRAM_LIMIT_MB:
                    print(f"{prec:<8}| {BS:<8}| {peak_vram:<15.2f}| {'N/A':<19}| {'N/A':<25}| {'N/A':<15}| OOM")
                    csv_results.append([prec, BS, f"{peak_vram:.2f}", "N/A", "N/A", "N/A", "OOM"])
                    break

                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                
                iters = 500
                
                with torch.cuda.stream(trt_stream):
                    start_evt.record(trt_stream)
                    for _ in range(iters):
                        context.execute_async_v3(stream_handle=trt_stream.cuda_stream)
                    end_evt.record(trt_stream)
                    trt_stream.synchronize()
                
                latency = start_evt.elapsed_time(end_evt) / iters
                throughput = (BS * 1000) / latency
                
                # print(f"{prec:<8}| {BS:<8}| {peak_vram:<15.2f}| {latency:<19.2f}| {throughput:<25.2f}| ... (waiting for acc) | OK")
                
                # print(f"\n--- MEASURING ACCURACY ({prec} - BS={BS}) ---")
                total_samples = len(test_data)
                predictions = []
                
                eval_start = time.perf_counter()
                out_idx = output_names.index('output') if 'output' in output_names else 0
                
                for i in range(0, total_samples, BS):
                    end_idx = min(i + BS, total_samples)
                    current_batch = test_data[i:end_idx].copy()
                    actual_bs = len(current_batch)
                    
                    if actual_bs < BS:
                        pad = np.zeros((BS - actual_bs, *current_batch.shape[1:]), dtype=current_batch.dtype)
                        current_batch = np.concatenate([current_batch, pad], axis=0)
                        
                    inp_tensor = torch.from_numpy(current_batch).contiguous().to(device)
                    # Use exact explicit casting
                    if input_allocations[0].dtype != inp_tensor.dtype:
                        inp_tensor = inp_tensor.to(input_allocations[0].dtype)
                        
                    input_allocations[0].copy_(inp_tensor)
                    torch.cuda.synchronize()
                    
                    with torch.cuda.stream(trt_stream):
                        if not context.execute_async_v3(stream_handle=trt_stream.cuda_stream):
                            print(f"TRT Execution failed at batch start {i}")
                        trt_stream.synchronize()
                        
                    out_tensor = output_allocations[out_idx][:actual_bs].detach().cpu()
                    preds_batch = out_tensor.argmax(dim=1).numpy()
                    predictions.extend(preds_batch)
                    
                eval_end = time.perf_counter()
                
                from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                image_inference_time_ms = (eval_end - eval_start) * 1000
                # print(f"Total Inference Time for entire image: {image_inference_time_ms:.2f} ms")
                
                predictions = np.array(predictions)
                oa = accuracy_score(test_labels, predictions)
                # print(f"[Accuracy] Overall Accuracy (OA): {oa * 100:.2f}%\n")
                
                print(f"{prec:<8}| {BS:<8}| {peak_vram:<15.2f}| {latency:<19.2f}| {throughput:<25.2f}| {oa * 100:<15.2f}| OK")
                
                csv_results.append([prec, BS, f"{peak_vram:.2f}", f"{latency:.2f}", f"{throughput:.2f}", f"{oa * 100:.2f}", "OK"])
                
                # if BS == batch_sizes[0]:
                #     print("--- TRT PREDICTIONS SNAPSHOT ---")
                #     unique, counts = np.unique(predictions, return_counts=True)
                #     print(f"TRT Predicted Classes Distribution: {dict(zip(unique, counts))}")
                #     gt_unique, gt_counts = np.unique(test_labels, return_counts=True)
                #     print(f"Ground Truth Distribution: {dict(zip(gt_unique, gt_counts))}")
                
                if dataset == 'Pavia':
                    target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted metal sheets','Bare Soil','Bitumen',
                                    'Self-Blocking Bricks','Shadows']
                else:
                    target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture', 'Grass-trees', 
                                    'Grass-pasture-mowed', 'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill', 
                                    'Soybean-clean', 'Wheat', 'Woods', 'Buildings-grass-trees-drives', 'Stone-steel towers']
                
                # if BS == max(batch_sizes):
                #     print(f"--- Classification Report ({prec} Largest BS) ---")
                #     print(classification_report(test_labels, predictions, digits=4, target_names=target_names))
                #     print("--- Confusion Matrix ---")
                #     print(confusion_matrix(test_labels, predictions))
                
                del allocations
                del context
                del engine
                if 'batch_data' in locals():
                    del batch_data
                if 'alloc' in locals():
                    del alloc
                
                torch.cuda.empty_cache()
                
            except Exception as e:
                free, total = torch.cuda.mem_get_info()
                peak_vram = (total - free) / (1024 * 1024)
                if "out of memory" in str(e).lower() or peak_vram > VRAM_LIMIT_MB:
                    print(f"{prec:<8}| {BS:<8}| {'> '+str(VRAM_LIMIT_MB):<15}| {'N/A':<19}| {'N/A':<25}| {'N/A':<15}| OOM")
                    csv_results.append([prec, BS, f"> {VRAM_LIMIT_MB}", "N/A", "N/A", "N/A", "OOM"])
                    break
                else:
                    print(f"Exception for BS {BS}, Prec {prec}: {e}")
                    csv_results.append([prec, BS, "N/A", "N/A", "N/A", "N/A", f"Error: {e}"])
                    break
                    
    # Save CSV
    with open("comparative_table.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Precision', 'Batch Size', 'Peak VRAM (MB)', 'Latency (ms/batch)', 'Throughput (Patches/sec)', 'Accuracy (%)', 'Status'])
        writer.writerows(csv_results)
    print(f"\nSaved comparative table to comparative_table.csv")

def generate_all_plots(csv_path="comparative_table.csv"):
    if not os.path.exists(csv_path):
        print(f"Skipping plots, {csv_path} not found.")
        return
        
    import csv
    import matplotlib.pyplot as plt
    import numpy as np
    
    data = {}
    
    # Read the data
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
                    
                if prec not in data:
                    data[prec] = {'bs': [], 'thpt': [], 'vram': [], 'acc': []}
                    
                data[prec]['bs'].append(bs)
                data[prec]['thpt'].append(thpt)
                data[prec]['vram'].append(vram)
                data[prec]['acc'].append(acc)
                
    if not data:
        print("No valid data to plot.")
        return

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
    output_thpt = "throughput_vs_batchsize.png"
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
    output_vram = "peak_ram_vs_batchsize.png"
    plt.savefig(output_vram, dpi=300)
    plt.close()
    print(f"Saved Peak RAM plot to {output_vram}")

    # 3. Accuracy (averaged by batch size) vs Precision
    plt.figure(figsize=(10, 6))
    precisions = []
    avg_accuracies = []
    
    # Custom order: PyTorch, FP32, FP16, INT8
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
        
        for i, v in enumerate(avg_accuracies):
            plt.text(i, v + (1.0 if max(avg_accuracies) > 10 else 0.1), f"{v:.2f}%", ha='center', fontsize=10)
            
        max_val = max(avg_accuracies)
        upper_lim = 100 if max_val <= 100 else max_val * 1.1
        if max_val > 90 and max_val <= 100:
            upper_lim = max(100, max_val + 5)
            
        plt.ylim(0, upper_lim)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        output_acc = "accuracy_vs_precision.png"
        plt.savefig(output_acc, dpi=300)
        plt.close()
        print(f"Saved Accuracy vs Precision plot to {output_acc}")

if __name__ == '__main__':
    print("==== MCTGCL Hardware Edge Profiling ====")
    print("Hardware Specs:")
    try:
        sm = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], encoding='utf-8').strip()
        print(f"GPU: {sm}")
    except:
        print("GPU: Error querying nvidia-smi")
    print("=" * 40)
    
    # -----------------------------------------------------
    # dataset
    # -----------------------------------------------------
    #dataset_name = "Pavia"
    dataset_name = "Indian"
    
    test_data, test_labels = profile_preprocessing(dataset_name)
    csv_results_pts = profile_pytorch_baseline(test_data, test_labels, dataset=dataset_name)
    profile_trt(test_data, test_labels, csv_results_pts, dataset=dataset_name)
    generate_all_plots()
    
    print("==== success ====")