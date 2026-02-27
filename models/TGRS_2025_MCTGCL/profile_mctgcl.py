import os
import time
import numpy as np
import torch
import tensorrt as trt
import subprocess

from train import applyPCA, createImageCubes
from mctgcl_onnx import mctgcl

def profile_preprocessing():
    print("--- PREPROCESSING PROFILING (T_pre) ---")
    print(f"{'Dataset':<13}| {'Dim Original':<15}| {'PCA Time (ms)':<14}| {'Patching Time (ms)':<19}| {'Total T_pre (ms)'}")
    
    datasets = [
        {"name": "Indian Pines", "shape": (145, 145, 200), "pca": True},
        {"name": "Pavia Uni.", "shape": (610, 340, 103), "pca": True},
        {"name": "Custom", "shape": (32, 32, 31), "pca": False},
    ]
    
    for ds in datasets:
        shape = ds["shape"]
        H, W, B = shape
        X = np.random.rand(H, W, B).astype(np.float32)
        y = np.random.randint(0, 9, (H, W)) # dummy labels
        
        pca_time = 0.0
        patch_time = 0.0
        
        warmup = 10
        iters = 50
        
        if ds["pca"]:
            for _ in range(warmup):
                _ = applyPCA(X, 30)
            
            start = time.perf_counter()
            for _ in range(iters):
                X_pca = applyPCA(X, 30)
            end = time.perf_counter()
            pca_time = (end - start) * 1000 / iters
        else:
            X_pca = X
        
        for _ in range(warmup):
            _ = createImageCubes(X_pca, y, windowSize=13, removeZeroLabels=False)
            
        start = time.perf_counter()
        for _ in range(iters):
            _ = createImageCubes(X_pca, y, windowSize=13, removeZeroLabels=False)
        end = time.perf_counter()
        patch_time = (end - start) * 1000 / iters
        
        total_pre = pca_time + patch_time
        pca_str = f"{pca_time:.2f}" if ds["pca"] else "N/A"
        dim_str = f"{H}x{W}x{B}"
        
        print(f"{ds['name']:<13}| {dim_str:<15}| {pca_str:<14}| {patch_time:.2f}{'':<13}| {total_pre:.2f}")

def build_engine(onnx_file_path, engine_file_path, max_batch_size):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1024 ** 3)) # 4GB workspace
    
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
            
    # Define optimization profile for dynamic batch sizes
    profile = builder.create_optimization_profile()
    # Find input tensor
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    
    opt_shape = [max_batch_size // 2 if max_batch_size > 1 else 1, 1, 30, 13, 13]
    max_shape = [max_batch_size, 1, 30, 13, 13]
    min_shape = [1, 1, 30, 13, 13]
    
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("Failed to build engine.")
        return None
    with open(engine_file_path, 'wb') as f:
        f.write(engine_bytes)
        
    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(engine_bytes)

def profile_trt():
    print("\n--- TENSORRT FP16 INFERENCE PROFILING (T_inf) ---")
    print(f"{'BS':<8}| {'Peak VRAM (MB)':<15}| {'Latency (ms/batch)':<19}| {'Throughput (Patches/sec)':<25}| {'Status'}")
    
    device = torch.device('cuda')
    model = mctgcl(num_classes=16, num_tokens=121).to(device)
    model.eval()
    
    batch_sizes = [32, 128, 512, 1024, 2048, 4096]
    VRAM_LIMIT_MB = 10000
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    for BS in batch_sizes:
        try:
            # 1. Export static ONNX
            onnx_path = f'mctgcl_bs{BS}.onnx'
            dummy_input = torch.randn(BS, 1, 30, 13, 13).to(device)
            torch.onnx.export(model, dummy_input, onnx_path, 
                              export_params=True, opset_version=17, 
                              do_constant_folding=True,
                              input_names=['input'], output_names=['output', 'features'])
            import onnx
            from onnxsim import simplify
            model_onnx = onnx.load(onnx_path)
            model_simp, check = simplify(model_onnx)
            if check:
                onnx.save(model_simp, onnx_path)
                
            # 2. Build TRT engine with static config
            engine_path = f'mctgcl_bs{BS}_fp16.engine'
            
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1024 ** 3))
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                
            with open(onnx_path, 'rb') as f:
                parser.parse(f.read())
            
            engine_bytes = builder.build_serialized_network(network, config)
            if engine_bytes is None:
                print(f"{BS:<8}| {'N/A':<15}| {'N/A':<19}| {'N/A':<25}| TRT Engine Build Failed")
                del parser, network, config, builder
                continue
                
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(engine_bytes)
            context = engine.create_execution_context()
            
            # cleanup builder objects right away
            del parser, network, config, builder
            
            # Prepare bindings
            bindings = []
            allocations = []
            for i in range(engine.num_bindings):
                shape = context.get_binding_shape(i)
                dtype = torch.float16 if engine.get_binding_dtype(i) == trt.float16 else torch.float32
                if i == 0:
                    alloc = torch.randn(tuple(shape), dtype=dtype, device=device)
                else:
                    alloc = torch.empty(tuple(shape), dtype=dtype, device=device)
                allocations.append(alloc)
                bindings.append(int(alloc.data_ptr()))
                
            torch.cuda.reset_peak_memory_stats()
            # warmup
            for _ in range(100):
                context.execute_async_v2(bindings=bindings, stream_handle=torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            
            free, total = torch.cuda.mem_get_info()
            peak_vram = (total - free) / (1024 * 1024)
            
            if peak_vram > VRAM_LIMIT_MB:
                print(f"{BS:<8}| {peak_vram:<15.2f}| {'N/A':<19}| {'N/A':<25}| OOM (Exceeds 10GB)")
                break

            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            
            start_evt.record()
            for _ in range(500):
                context.execute_async_v2(bindings=bindings, stream_handle=torch.cuda.current_stream().cuda_stream)
            end_evt.record()
            
            torch.cuda.synchronize()
            
            latency = start_evt.elapsed_time(end_evt) / 500
            throughput = (BS * 1000) / latency
            
            print(f"{BS:<8}| {peak_vram:<15.2f}| {latency:<19.2f}| {throughput:<25.2f}| OK")
            
            del allocations
            del context
            del engine
            torch.cuda.empty_cache()
            
        except Exception as e:
            free, total = torch.cuda.mem_get_info()
            peak_vram = (total - free) / (1024 * 1024)
            if "out of memory" in str(e).lower() or peak_vram > VRAM_LIMIT_MB:
                print(f"{BS:<8}| {'> '+str(VRAM_LIMIT_MB):<15}| {'N/A':<19}| {'N/A':<25}| OOM")
                break
            else:
                print(f"Exception for BS {BS}: {e}")
                break

if __name__ == '__main__':
    print("=== MCTGCL Hardware Edge Profiling ===")
    print("Hardware Specs:")
    sm = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], encoding='utf-8').strip()
    print(f"GPU: {sm}")
    print("Target Constraints: 15 FPS (Throughput: >= 15 patches/sec), VRAM <= 10GB")
    print("=" * 40 + "\n")
    
    profile_preprocessing()
    profile_trt()
