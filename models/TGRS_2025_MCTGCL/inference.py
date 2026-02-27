import torch
import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnxruntime as ort
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
from mctgcl import mctgcl
from train import loadData, create_data_loader, patch_size, acc_reports

batch_size = 512
dataset = "Indian" # Indian or Pavia

if dataset == 'Pavia':
    num_classes = 9
elif dataset == 'Indian':
    num_classes = 16
else:
    raise ValueError("Invalid dataset name")

num_tokens = (patch_size - 2) ** 2

# -------------------------------------------------------------------
# 1. PyTorch Evaluation
# -------------------------------------------------------------------
def eval_pytorch(test_loader, model_path, device):
    print("\n--- Evaluating PyTorch (.pt) ---")
    model = mctgcl(num_classes=num_classes, num_tokens=num_tokens).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    y_true, y_pred = [], []
    total_time = 0.0
    
    with torch.no_grad():
        for data, label in test_loader:
            start = time.perf_counter() # Start timer

            data = data.to(device)
            logits, _ = model(data)
            
            # Crucial: Synchronize GPU to get accurate timing for async CUDA operations
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            pred_class = torch.argmax(logits, dim=1).cpu()

            end = time.perf_counter() # End timer
            total_time += (end - start)

            y_pred.extend(pred_class.numpy())
            y_true.extend(label.numpy())
    
    print("Successful")
    return y_true, y_pred, total_time, len(y_true)

# -------------------------------------------------------------------
# 2. ONNX Evaluation
# -------------------------------------------------------------------
def eval_onnx(test_loader, onnx_path):
    print("\n--- Evaluating ONNX (.onnx) ---")
    # Force execution on GPU
    providers = ['CUDAExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    
    y_true, y_pred = [], []
    total_time = 0.0
    
    for data, label in test_loader:
        data_np = data.numpy().astype(np.float32)
        
        # ONNX Runtime handles GPU synchronization internally for the Python API
        start = time.perf_counter()
        outputs = session.run(None, {input_name: data_np})
        end = time.perf_counter()
        
        total_time += (end - start)
        
        logits = outputs[0]
        pred_class = np.argmax(logits, axis=1)
        
        y_pred.extend(pred_class)
        y_true.extend(label.numpy())

    print("Successful")
    return y_true, y_pred, total_time, len(y_true)

# -------------------------------------------------------------------
# 3. TensorRT Evaluation
# -------------------------------------------------------------------
def allocate_trt_buffers(engine, context, batch_shape):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    
    # Modern TRT API: Use the exact tensor name (usually 'input')
    input_name = engine.get_tensor_name(0)

    # Set the dynamic shape for the current batch
    context.set_input_shape(input_name, batch_shape)

    # In TRT 8.6+, iterating over the engine yields tensor names (strings), not integers
    for tensor_name in engine:
        shape = context.get_tensor_shape(tensor_name)
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        
        # Check if the tensor is an input or output
        is_input = engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
        
        if is_input:
            inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
        else:
            outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
            
    return inputs, outputs, bindings, stream

def eval_tensorrt(test_loader, engine_path):
    print("\n--- Evaluating TensorRT (.engine) ---")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # This is mandatory for operations like InstanceNormalization or LayerNorm.
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")

    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        
    context = engine.create_execution_context()
    
    y_true, y_pred = [], []
    total_time = 0.0
    
    current_shape = None
    inputs, outputs, bindings, stream = None, None, None, None
    
    for data, label in test_loader:
        data_np = data.numpy().astype(np.float32)
        batch_shape = data_np.shape
        
        # Dynamic Batch Handling: 
        # Only reallocate memory buffers if the batch size changes (e.g., the last batch)
        if current_shape != batch_shape:
            current_shape = batch_shape
            inputs, outputs, bindings, stream = allocate_trt_buffers(engine, context, batch_shape)
        
        np.copyto(inputs[0]['host'], data_np.ravel())
        
        start = time.perf_counter()
        # Transfer data to GPU
        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
        # Execute TRT Engine
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer results back to CPU
        for out in outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
        # Synchronize GPU stream before stopping the timer
        stream.synchronize()
        end = time.perf_counter()
        
        total_time += (end - start)
        
        # Identify which output is the logits array (should match num_classes)
        out_idx = 0 if outputs[0]['shape'][1] == num_classes else 1
        logits = outputs[out_idx]['host'].reshape(*outputs[out_idx]['shape'])
        
        pred_class = np.argmax(logits, axis=1)
        y_pred.extend(pred_class)
        y_true.extend(label.numpy())
    
    print("Successful")
    return y_true, y_pred, total_time, len(y_true)

# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Load the data using your exact original pipeline (PCA included)
    print("Loading Dataset and applying PCA pipeline...")
    X, y = loadData(dataset) 
    train_loader, test_loader, all_data_loader, y_all = create_data_loader(X, y, patch_size, batch_size, dataset) 
    
    # Define Model Paths
    pt_path = f"params/{dataset}.pt"
    onnx_path = f"params/mctgcl_{dataset}.onnx"
    engine_path = f"params/mctgcl_{dataset}.engine"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    
    # Run Evaluations
    results = {}
    
    def AA_andEachClassAccuracy(confusion_matrix):
        list_diag = np.diag(confusion_matrix)
        list_raw_sum = np.sum(confusion_matrix, axis=1)
        each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
        average_acc = np.mean(each_acc)
        return each_acc, average_acc

    try:
        y_true_pt, y_pred_pt, time_pt, samples = eval_pytorch(test_loader, pt_path, device)
        results['PyTorch'] = {'Time': time_pt, 'y_true': y_true_pt, 'y_pred': y_pred_pt}
    except Exception as e:
        print(f"PyTorch eval failed: {e}")
        
    try:
        y_true_onnx, y_pred_onnx, time_onnx, _ = eval_onnx(test_loader, onnx_path)
        results['ONNX'] = {'Time': time_onnx, 'y_true': y_true_onnx, 'y_pred': y_pred_onnx}
    except Exception as e:
        print(f"ONNX eval failed: {e}")
        
    try:
        y_true_trt, y_pred_trt, time_trt, samples = eval_tensorrt(test_loader, engine_path)
        results['TensorRT'] = {'Time': time_trt, 'y_true': y_true_trt, 'y_pred': y_pred_trt}
    except Exception as e:
        print(f"TensorRT eval failed: {e}")
        
    # Print Comparative Report
    print("\n" + "="*50)
    print(f"            HARDWARE COMPARATIVE RESULTS           ")
    print("="*50)
    print(f"Total Test Samples      : {samples}")
    print(f"Dataset                 : {dataset}")
    print(f"Batch Size              : {batch_size}")
    print("-"*50)
    
    for model_name, data in results.items():
        latency = (data['Time'] / samples) * 1000 # ms per sample
        fps = samples / data['Time']
        
        classification, oa, confusion, each_acc, aa, kappa = acc_reports(data['y_true'], data['y_pred'], dataset)
        
        print(f"[{model_name.upper()}]")
        print(f"Total Inference Time : {data['Time']:.4f} seconds")
        print(f"Latency per patch    : {latency:.4f} ms")
        print(f"Throughput           : {fps:.2f} FPS")
        print(f"Overall Accuracy     : {oa:.2f} %")
        print(f"Average Accuracy     : {aa:.2f} %")
        print(f"Kappa                : {kappa:.2f} %")
        print("Classification Report:")
        print(classification)
        print("Confusion Matrix:")
        print(confusion)
        print("-" * 50)