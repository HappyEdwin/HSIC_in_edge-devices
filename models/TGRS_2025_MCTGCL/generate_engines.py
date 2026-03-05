import os
import gc
import torch
import tensorrt as trt
import numpy as np
from train import loadData, applyPCA, createImageCubes
from mctgcl_onnx import mctgcl

class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batch_size, cache_file, data, device):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.batch_idx = 0
        self.max_batches = (len(data) + batch_size - 1) // batch_size
        self.device = device
        
        # Allocate device memory for calibration data
        shape = (self.batch_size, *self.data.shape[1:])
        self.device_input = torch.empty(shape, dtype=torch.float32, device=self.device)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.batch_idx < self.max_batches:
            start_idx = self.batch_idx * self.batch_size
            end_idx = min((self.batch_idx + 1) * self.batch_size, len(self.data))
            batch = self.data[start_idx:end_idx]
            
            # Pad if needed
            if len(batch) < self.batch_size:
                pad = np.zeros((self.batch_size - len(batch), *batch.shape[1:]), dtype=self.data.dtype)
                batch = np.concatenate([batch, pad], axis=0)
            
            # Copy to device tensor
            self.device_input.copy_(torch.from_numpy(batch).to(self.device))
            self.batch_idx += 1
            return [int(self.device_input.data_ptr())]
        return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def get_real_calibration_data(dataset_name="Indian", num_samples=512):
    print(f"Loading real data from {dataset_name} for INT8 calibration...")
    X, y = loadData(dataset_name)
    
    # PCA reduction and patch extraction using functions from train.py
    X_pca = applyPCA(X, numComponents=30)
    
    # Free memory of X aggressively (Jetson 8gb shared ram)
    del X
    gc.collect()
    
    patches, _ = createImageCubes(X_pca, y, windowSize=13, removeZeroLabels=False)
    
    del X_pca
    del y
    gc.collect()
    
    # Reshape to match the model's expected input (N, 1, 30, 13, 13)
    patches = patches.reshape(-1, 13, 13, 30, 1)
    patches = patches.transpose(0, 4, 3, 1, 2).astype(np.float32)
    
    # Shuffle data to get a diverse calibration set
    np.random.seed(42)
    indices = np.random.permutation(len(patches))
    patches = patches[indices]
    
    num_samples = min(num_samples, len(patches))
    return patches[:num_samples]

if __name__ == '__main__':
    # Ensure profiles directory exists
    os.makedirs("profiles", exist_ok=True)

    print("version:1")

    dataset = "Indian"
    
    device = torch.device('cuda')
    model = mctgcl(num_classes=16, num_tokens=121).to(device)

    model.load_state_dict(torch.load(f"params/{dataset}.pt", map_location=device))
    model.eval()
    
    batch_sizes = [16, 32, 64, 128, 256, 512]
    precisions = ["FP32", "FP16", "INT8"]
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # To use PaviaU, uncomment the following line and comment the Indian one
    # calib_data = get_real_calibration_data(dataset_name="Pavia", num_samples=512)
    calib_data = get_real_calibration_data(dataset_name=dataset, num_samples=512)
    
    for BS in batch_sizes:
        onnx_path = f'profiles/mctgcl_bs{BS}.onnx'
        
        # 1. Export static ONNX (once per batch size)
        print(f"\n--- Exporting ONNX model to {onnx_path} for Batch Size: {BS} ---")
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
            
        for prec in precisions:
            print(f"Building TensorRT Engine for BS {BS}, Precision {prec}...")
            engine_path = f'profiles/mctgcl_bs{BS}_{prec.lower()}.engine'
            cache_file = f"profiles/int8_calib_cache_bs{BS}.bin"
            
            # Clear pyTorch cache before allocating TensorRT workspace
            torch.cuda.empty_cache()
            gc.collect()
            
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            config = builder.create_builder_config()
            # Reduce TRT workspace to 2GB to prevent OOM
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 * (1024 ** 3))
            
            if prec == "FP16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                
            elif prec == "INT8" and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                config.set_flag(trt.BuilderFlag.FP16)
                calibrator = EntropyCalibrator(batch_size=BS, cache_file=cache_file, data=calib_data, device=device)
                config.int8_calibrator = calibrator
                
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    print(f"Failed to parse ONNX file for BS {BS} {prec}")
                    continue
            
            print("Building serialized network... this may take a while depending on hardware.")
            engine_bytes = builder.build_serialized_network(network, config)
            if engine_bytes is None:
                print(f"TRT Engine Build Failed for BS {BS} {prec}")
                continue
                
            with open(engine_path, 'wb') as f:
                f.write(engine_bytes)
            print(f"Successfully generated and saved engine: {engine_path}")
            
            # Free memory at the end of each iteration
            del parser, network, config, builder, engine_bytes
            gc.collect()
