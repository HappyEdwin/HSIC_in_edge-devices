#!/usr/bin/env python3
"""
Benchmark Runner & Hardware Telemetry for Hyperspectral Classification (SS-ResNet)
on NVIDIA Jetson Orin Nano (TensorRT FP16 / INT8).
Evaluates:
  1. Pure TensorRT GPU Latency & Throughput (patches/sec)
  2. Full Scene Classification Time & Accuracy (OA, AA, Cohen's Kappa)
  3. Power (W) via Jetson sysfs / tegrastats & Energy per Patch / Scene (mJ)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from operator import truediv
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.metrics import JetsonPowerMonitor

class TRTHSIRunner:
    def __init__(self, engine_path: str):
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bindings = []
        self.output_tensors = []

        num_io = getattr(self.engine, "num_io_tensors", None)
        if num_io is not None:
            for i in range(num_io):
                name = self.engine.get_tensor_name(i)
                shape = tuple(self.engine.get_tensor_shape(name))
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                t = torch.empty(shape, dtype=getattr(torch, np.dtype(dtype).name), device=self.device)
                self.bindings.append(t.data_ptr())
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.input_tensor = t
                    self.context.set_tensor_address(name, t.data_ptr())
                else:
                    self.output_tensor = t
                    self.context.set_tensor_address(name, t.data_ptr())
        else:
            for i in range(self.engine.num_bindings):
                shape = tuple(self.engine.get_binding_shape(i))
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                t = torch.empty(shape, dtype=getattr(torch, np.dtype(dtype).name), device=self.device)
                self.bindings.append(t.data_ptr())
                if self.engine.binding_is_input(i):
                    self.input_tensor = t
                else:
                    self.output_tensor = t

    def infer(self, patch_np: np.ndarray) -> np.ndarray:
        # patch_np: (1, 30, 13, 13)
        t_patch = torch.from_numpy(patch_np).float().to(self.device)
        self.input_tensor.copy_(t_patch)
        if hasattr(self.context, "execute_async_v3"):
            self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        elif hasattr(self.context, "execute_v2"):
            self.context.execute_v2(self.bindings)
        torch.cuda.synchronize()
        return self.output_tensor.cpu().numpy()

def load_data(name, data_dir="models/TGRS_2025_MCTGCL/data"):
    if name == 'Indian':
        data = sio.loadmat(os.path.join(data_dir, 'Indian.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_dir, 'Indian_gt.mat'))['indian_pines_gt']
    elif name == 'Pavia':
        data = sio.loadmat(os.path.join(data_dir, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_dir, 'PaviaU_gt.mat'))['paviaU_gt']
    else:
        raise ValueError(f"Unknown dataset {name}")
    return data, labels

def apply_pca(X, num_components=30):
    orig_shape = X.shape
    flat_X = np.reshape(X, (-1, orig_shape[2]))
    pca = PCA(n_components=num_components, whiten=True, random_state=42)
    pca_X = pca.fit_transform(flat_X)
    pca_X = np.reshape(pca_X, (orig_shape[0], orig_shape[1], num_components))
    return pca_X, pca

def pad_with_zeros(X, margin=2):
    padded = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]), dtype=X.dtype)
    padded[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
    return padded

def create_image_cubes(X, y, window_size=13, remove_zeros=True):
    margin = int((window_size - 1) / 2)
    zero_padded = pad_with_zeros(X, margin=margin)
    h, w, c = X.shape
    total_pixels = h * w
    patches_data = np.zeros((total_pixels, window_size, window_size, c), dtype=np.float32)
    patches_labels = np.zeros(total_pixels, dtype=np.int64)

    idx = 0
    for r in range(margin, zero_padded.shape[0] - margin):
        for col in range(margin, zero_padded.shape[1] - margin):
            patch = zero_padded[r - margin:r + margin + 1, col - margin:col + margin + 1, :]
            patches_data[idx] = patch
            patches_labels[idx] = y[r - margin, col - margin]
            idx += 1

    if remove_zeros:
        mask = patches_labels > 0
        patches_data = patches_data[mask]
        patches_labels = patches_labels[mask] - 1

    return patches_data, patches_labels

def main():
    parser = argparse.ArgumentParser(description="Jetson HSI SS-ResNet Benchmark")
    parser.add_argument("--engine", type=str, default="models/engines/ss_resnet_indian_b1_int8.engine", help="Path to TensorRT engine")
    parser.add_argument("--precision", type=str, default="INT8", choices=["FP16", "INT8"])
    parser.add_argument("--dataset", type=str, default="Indian", choices=["Indian", "Pavia"])
    parser.add_argument("--iterations", type=int, default=1000, help="Latency benchmark iterations")
    parser.add_argument("--eval_full", action="store_true", default=True, help="Evaluate full test set accuracy")
    parser.add_argument("--output_json", type=str, default="results/hsi/benchmark_jetson.json")
    args = parser.parse_args()

    os.makedirs("results/hsi", exist_ok=True)
    engine_path = os.path.abspath(args.engine)
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Engine not found: {engine_path}")

    print("=" * 75)
    print(f"🚀 NVIDIA Jetson Orin Nano Hyperspectral Benchmark (SS-ResNet)")
    print(f"   Engine: {engine_path} | Precision: {args.precision}")
    print(f"   Dataset: {args.dataset} Pines")
    print("=" * 75)

    # 1. Load Data
    raw_data, gt = load_data(args.dataset)
    t0_pca = time.time()
    pca_data, _ = apply_pca(raw_data, num_components=30)
    t_pca = (time.time() - t0_pca) * 1000.0
    print(f"[*] PCA Preprocessing Time: {t_pca:.2f} ms ({raw_data.shape} -> {pca_data.shape})")

    X_cubes, y_labels = create_image_cubes(pca_data, gt, window_size=13)
    # Transpose to (N, 30, 13, 13)
    X_cubes = np.transpose(X_cubes, (0, 3, 1, 2))
    _, X_test, _, y_test = train_test_split(X_cubes, y_labels, test_size=0.90, random_state=42, stratify=y_labels)
    print(f"[*] Total test patches: {len(X_test)}")

    # 2. Init TensorRT Runner
    runner = TRTHSIRunner(engine_path)

    # 3. Latency Benchmark
    power_mon = JetsonPowerMonitor(interval_ms=50)
    power_mon.start()

    print(f"\n[*] Running Pure TensorRT Latency Benchmark ({args.iterations} iterations)...")
    latencies_ms = []
    # Warmup
    for _ in range(25):
        runner.infer(X_test[0:1])

    num_iters = min(args.iterations, len(X_test))
    for i in range(num_iters):
        patch = X_test[i:i+1]
        t0 = time.perf_counter_ns()
        runner.infer(patch)
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1_000_000.0)

    power_stats = power_mon.stop()
    mean_lat = float(np.mean(latencies_ms))
    median_lat = float(np.median(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0
    energy_mj = mean_lat * power_stats["power_avg_watts"]

    print("=" * 75)
    print(f"⚡ JETSON ORIN NANO TENSORRT BENCHMARK RESULTS:")
    print(f"   Mean Latency: {mean_lat:.3f} ms | Median: {median_lat:.3f} ms | P95: {p95_lat:.3f} ms")
    print(f"   Throughput: {fps:.2f} patches/sec (FPS)")
    print(f"   Average Power: {power_stats['power_avg_watts']:.3f} W")
    print(f"   Energy per Patch: {energy_mj:.3f} mJ")
    print("=" * 75)

    # 4. Full Scene Evaluation (Accuracy)
    if args.eval_full:
        print(f"\n[*] Evaluating Full Test Set ({len(X_test)} patches) for Accuracy Verification...")
        y_preds = []
        t0_eval = time.time()
        for i in range(len(X_test)):
            out = runner.infer(X_test[i:i+1])
            y_preds.append(int(np.argmax(out.reshape(-1))))

        t_eval = time.time() - t0_eval
        y_preds = np.array(y_preds)
        oa = accuracy_score(y_test, y_preds) * 100.0
        cm = confusion_matrix(y_test, y_preds)
        each_acc = np.nan_to_num(truediv(np.diag(cm), np.sum(cm, axis=1))) * 100.0
        aa = float(np.mean(each_acc))
        kappa = float(cohen_kappa_score(y_test, y_preds)) * 100.0
        scene_energy_j = t_eval * power_stats["power_avg_watts"]

        print(f"📊 HARDWARE ACCURACY VERIFICATION (TensorRT {args.precision}):")
        print(f"   Overall Accuracy (OA): {oa:.2f} %")
        print(f"   Average Accuracy (AA): {aa:.2f} %")
        print(f"   Kappa Coefficient (κ): {kappa:.2f} %")
        print(f"   Full Test Classification Time: {t_eval:.2f} s ({len(X_test)/t_eval:.1f} patches/s)")
        print(f"   Total Energy for Scene: {scene_energy_j:.2f} J")
        print("=" * 75)

    results = {
        "platform": "NVIDIA Jetson Orin Nano",
        "model_name": "SS-ResNet",
        "precision": args.precision,
        "dataset": args.dataset,
        "latency_mean_ms": round(mean_lat, 3),
        "latency_median_ms": round(median_lat, 3),
        "latency_p95_ms": round(p95_lat, 3),
        "fps": round(fps, 2),
        "power_avg_w": power_stats["power_avg_watts"],
        "energy_mj_per_patch": round(energy_mj, 3),
        "overall_accuracy_oa": round(oa, 2) if args.eval_full else None,
        "average_accuracy_aa": round(aa, 2) if args.eval_full else None,
        "kappa": round(kappa, 2) if args.eval_full else None
    }

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Saved Jetson benchmark results to: {args.output_json}")

if __name__ == "__main__":
    main()
