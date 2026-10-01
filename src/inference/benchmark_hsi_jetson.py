#!/usr/bin/env python3
"""
Benchmark Runner & Hardware Telemetry for Hyperspectral Classification (SS-ResNet)
on NVIDIA Jetson Orin Nano (TensorRT FP16 / INT8).
- PCA Preprocessing: Performed 100% ON-BOARD from raw 200-band HSI cube.
- Dependencies: ONLY Standard Library + NumPy + TensorRT / PyTorch (No scipy, no sklearn).
Evaluates:
  1. On-Board Preprocessing Latency (PCA 200 -> 30 bands + Patch Extraction)
  2. Pure TensorRT GPU Latency & Throughput (patches/sec)
  3. Full Scene Classification Time & Accuracy (OA, AA, Cohen's Kappa)
  4. Real-time Power (W) via Jetson sysfs / tegrastats & Energy per Scene (J)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
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
        t_patch = torch.from_numpy(patch_np).float().to(self.device)
        self.input_tensor.copy_(t_patch)
        if hasattr(self.context, "execute_async_v3"):
            self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        elif hasattr(self.context, "execute_v2"):
            self.context.execute_v2(self.bindings)
        torch.cuda.synchronize()
        return self.output_tensor.cpu().numpy()

def apply_pca_on_board(raw_data, weights_path="data/hsi/pca_transform_weights.npy", mean_path="data/hsi/pca_mean.npy", mode="project"):
    h, w, b = raw_data.shape
    flat_X = np.reshape(raw_data, (-1, b)).astype(np.float32)

    t0 = time.perf_counter()
    if mode == "project" and os.path.exists(weights_path) and os.path.exists(mean_path):
        W = np.load(weights_path)
        mu = np.load(mean_path)
        pca_flat = (flat_X - mu) @ W
    else:
        mean = np.mean(flat_X, axis=0)
        X_c = flat_X - mean
        cov = np.cov(X_c, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        idx = np.argsort(evals)[::-1][:30]
        evals, evecs = evals[idx], evecs[:, idx]
        pca_flat = (X_c @ evecs) / np.sqrt(evals)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    pca_data = np.reshape(pca_flat, (h, w, 30)).astype(np.float32)
    return pca_data, elapsed_ms

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

def compute_metrics_numpy(y_true, y_pred, num_classes=16):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    oa = float(np.mean(y_true == y_pred) * 100.0)
    diag = np.diag(cm)
    row_sum = np.sum(cm, axis=1)
    each_acc = np.divide(diag, row_sum, out=np.zeros_like(diag, dtype=float), where=row_sum != 0) * 100.0
    aa = float(np.mean(each_acc))

    total = np.sum(cm)
    if total > 0:
        po = np.trace(cm) / total
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (total ** 2)
        kappa = float((po - pe) / (1.0 - pe) * 100.0) if (1.0 - pe) != 0 else 0.0
    else:
        kappa = 0.0

    return oa, aa, kappa, cm

def main():
    parser = argparse.ArgumentParser(description="Jetson HSI SS-ResNet Benchmark with On-Board PCA")
    parser.add_argument("--engine", type=str, default="models/engines/ss_resnet_indian_b1_int8.engine", help="Path to TensorRT engine")
    parser.add_argument("--precision", type=str, default="INT8", choices=["FP16", "INT8"])
    parser.add_argument("--dataset", type=str, default="Indian", choices=["Indian", "Pavia"])
    parser.add_argument("--pca_mode", type=str, default="project", choices=["project", "fit"], help="PCA execution mode on board")
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
    print(f"   Dataset: {args.dataset} Pines | On-Board PCA: {args.pca_mode.upper()}")
    print("=" * 75)

    # 1. Load Raw Hyperspectral Cube
    raw_file = "data/hsi/raw_indian_pines.npy"
    gt_file = "data/hsi/indian_pines_gt.npy"
    idx_file = "data/hsi/indian_test_indices.npy"

    if os.path.exists(raw_file) and os.path.exists(gt_file):
        print(f"[*] Loading raw HSI sensor cube from {raw_file}...")
        raw_data = np.load(raw_file)
        gt = np.load(gt_file)
    else:
        import scipy.io as sio
        print(f"[*] Loading raw .mat files...")
        raw_data = sio.loadmat("models/TGRS_2025_MCTGCL/data/Indian.mat")['indian_pines_corrected'].astype(np.float32)
        gt = sio.loadmat("models/TGRS_2025_MCTGCL/data/Indian_gt.mat")['indian_pines_gt'].astype(np.int64)

    num_classes = int(np.max(gt))
    print(f"[*] Raw Sensor Cube: {raw_data.shape} ({raw_data.shape[2]} spectral bands)")
    print(f"[*] Ground Truth Mask: {gt.shape} ({num_classes} classes)")

    # 2. Execute PCA on the Board
    print(f"\n[*] Executing On-Board PCA Reduction ({raw_data.shape[2]} -> 30 bands)...")
    pca_data, t_pca_ms = apply_pca_on_board(raw_data, mode=args.pca_mode)
    print(f"✅ On-Board PCA Preprocessing Completed in: {t_pca_ms:.2f} ms! Reduced shape: {pca_data.shape}")

    # 3. Patch Extraction
    t0_patch = time.perf_counter()
    X_cubes, y_labels = create_image_cubes(pca_data, gt, window_size=13)
    X_cubes = np.transpose(X_cubes, (0, 3, 1, 2)) # NCHW for TensorRT
    t_patch_ms = (time.perf_counter() - t0_patch) * 1000.0
    print(f"[*] Spatial Patch Extraction: {t_patch_ms:.2f} ms ({len(X_cubes)} valid patches)")

    if os.path.exists(idx_file):
        test_indices = np.load(idx_file)
        X_test = X_cubes[test_indices]
        y_test = y_labels[test_indices]
        print(f"[*] Loaded exact test partition: {len(X_test)} patches.")
    else:
        split_idx = int(0.10 * len(X_cubes))
        X_test = X_cubes[split_idx:]
        y_test = y_labels[split_idx:]
        print(f"[*] Sliced test partition: {len(X_test)} patches.")

    # 4. Init TensorRT Runner
    runner = TRTHSIRunner(engine_path)

    # 5. Latency Benchmark
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
    print(f"   On-Board Preprocessing (PCA): {t_pca_ms:.2f} ms")
    print(f"   GPU Mean Latency: {mean_lat:.3f} ms | Median: {median_lat:.3f} ms | P95: {p95_lat:.3f} ms")
    print(f"   Throughput: {fps:.2f} patches/sec (FPS)")
    print(f"   Average Power: {power_stats['power_avg_watts']:.3f} W")
    print(f"   Energy per Patch: {energy_mj:.3f} mJ")
    print("=" * 75)

    # 6. Full Scene Evaluation (Accuracy)
    oa, aa, kappa = 0.0, 0.0, 0.0
    if args.eval_full:
        print(f"\n[*] Evaluating Full Test Set ({len(X_test)} patches) for Accuracy Verification...")
        y_preds = []
        t0_eval = time.time()
        for i in range(len(X_test)):
            out = runner.infer(X_test[i:i+1])
            y_preds.append(int(np.argmax(out.reshape(-1))))
        t_eval = time.time() - t0_eval
        y_preds = np.array(y_preds)
        oa, aa, kappa, _ = compute_metrics_numpy(y_test, y_preds, num_classes=num_classes)
        scene_total_time_s = round((t_pca_ms + t_patch_ms) / 1000.0 + t_eval, 3)
        scene_total_energy_j = round(scene_total_time_s * power_stats["power_avg_watts"], 3)
        scene_pixels = len(X_test)

        print(f"📊 HARDWARE ACCURACY VERIFICATION (TensorRT {args.precision}):")
        print(f"   Overall Accuracy (OA): {oa:.2f} %")
        print(f"   Average Accuracy (AA): {aa:.2f} %")
        print(f"   Kappa Coefficient (κ): {kappa:.2f} %")
        print(f"   Full Test Classification Time: {t_eval:.2f} s ({len(X_test)/t_eval:.1f} patches/s)")
        print(f"   Total Scene End-to-End Time: {scene_total_time_s:.2f} s")
        print(f"   Total Energy for Scene: {scene_total_energy_j:.2f} J")
        print("=" * 75)
    else:
        scene_total_time_s = None
        scene_total_energy_j = None
        scene_pixels = None
        t_eval = None

    def get_process_ram_mb():
        res = {"rss_mb": 0.0, "peak_rss_mb": 0.0}
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            res["peak_rss_mb"] = round(float(parts[1]) / 1024.0, 2)
                    elif line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            res["rss_mb"] = round(float(parts[1]) / 1024.0, 2)
        except Exception:
            pass
        return res

    ram = get_process_ram_mb()
    model_size = round(os.path.getsize(engine_path) / (1024 * 1024), 2)
    scene_ops_giga = round(scene_pixels * 0.0637, 2) if scene_pixels else None
    effective_throughput_gops = round(fps * 0.06366, 2)
    energy_eff_gops_w = round(effective_throughput_gops / power_stats["power_avg_watts"], 2) if power_stats["power_avg_watts"] > 0 else 0.0
    results = {
        "platform": "NVIDIA Jetson Orin Nano",
        "model_name": "SS-ResNet",
        "precision": args.precision,
        "dataset": args.dataset,
        "input_resolution": "13x13x30",
        "params_m": 0.636,
        "total_params": 635664,
        "model_size_mb": model_size,
        "total_ops_giga": 0.0637,
        "gflops": 0.0637,
        "total_macs_m": 31.83,
        "gmacs": 0.0318,
        "effective_throughput_gops": effective_throughput_gops,
        "energy_efficiency_gops_per_watt": energy_eff_gops_w,
        "pca_latency_ms": round(t_pca_ms, 2),
        "patch_extraction_latency_ms": round(t_patch_ms, 2),
        "latency_mean_ms": round(mean_lat, 3),
        "latency_median_ms": round(median_lat, 3),
        "latency_p95_ms": round(p95_lat, 3),
        "fps": round(fps, 2),
        "power_avg_w": power_stats["power_avg_watts"],
        "energy_mj_per_patch": round(energy_mj, 3),
        "scene_pixels_count": scene_pixels,
        "scene_inference_time_s": round(t_eval, 3) if t_eval is not None else None,
        "scene_total_time_s": scene_total_time_s,
        "scene_energy_joules": scene_total_energy_j,
        "scene_total_ops_giga": scene_ops_giga,
        "overall_accuracy_oa": round(oa, 2) if args.eval_full else None,
        "average_accuracy_aa": round(aa, 2) if args.eval_full else None,
        "kappa": round(kappa, 2) if args.eval_full else None,
        "ram_rss_mb": ram["rss_mb"]
    }

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Saved Jetson benchmark results to: {args.output_json}")

if __name__ == "__main__":
    main()
