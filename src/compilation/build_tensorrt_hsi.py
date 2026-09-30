#!/usr/bin/env python3
"""
TensorRT Engine Builder for Hyperspectral Image Classification (SS-ResNet).
Supports FP32, FP16, and INT8 Calibration using real HSI cubes.
Target: NVIDIA Jetson Orin Nano (Ampere GPU + Tensor Cores).
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import tensorrt as trt
import torch

class CustomTRTLogger(trt.ILogger):
    def __init__(self, severity=trt.ILogger.Severity.INFO):
        super().__init__()
        self.severity = severity
        self.unaccelerated_warnings = []

    def log(self, severity, msg):
        msg_lower = msg.lower()
        unaccel_keywords = ["fallback", "could not run", "unsupported", "cannot be placed"]
        if any(kw in msg_lower for kw in unaccel_keywords):
            self.unaccelerated_warnings.append(msg)
            print(f"⚠️  [TRT FALLBACK/ALERTA]: {msg}")
        elif severity == trt.ILogger.Severity.ERROR:
            print(f"❌ [TRT ERROR]: {msg}")


class HSIEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_npy="models/vitis_ai/calib_patches_indian.npy", cache_file="models/engines/hsi_calib.cache", batch_size=1):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.data = np.load(calib_npy).astype(np.float32)
        self.current_idx = 0
        self.num_samples = len(self.data)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Preallocate device buffer for batch
        sample_shape = (self.batch_size, self.data.shape[1], self.data.shape[2], self.data.shape[3])
        self.device_input = torch.zeros(sample_shape, dtype=torch.float32, device=self.device)
        print(f"[*] HSI INT8 Calibrator initialized with {self.num_samples} patches from {calib_npy}.")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_idx >= self.num_samples:
            return None
        end_idx = min(self.current_idx + self.batch_size, self.num_samples)
        batch_np = self.data[self.current_idx:end_idx]
        self.current_idx += self.batch_size
        if len(batch_np) < self.batch_size:
            pad = np.zeros((self.batch_size - len(batch_np), *self.data.shape[1:]), dtype=np.float32)
            batch_np = np.concatenate([batch_np, pad], axis=0)

        self.device_input.copy_(torch.from_numpy(batch_np).to(self.device))
        return [int(self.device_input.data_ptr())]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"📖 Reading INT8 calibration cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_file)), exist_ok=True)
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"💾 Saved INT8 calibration cache: {self.cache_file}")


def build_hsi_engine(onnx_path: str, engine_path: str, precision: str = "FP16", workspace_gb: int = 4):
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    precision = precision.upper()

    print("=" * 75)
    print(f"⚙️  Building TensorRT Engine: {engine_path}")
    print(f"    Source ONNX: {onnx_path}")
    print(f"    Target Precision: {precision}")
    print(f"    Workspace: {workspace_gb} GB")
    print("=" * 75)

    custom_logger = CustomTRTLogger(trt.ILogger.Severity.INFO)
    builder = trt.Builder(custom_logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, custom_logger)

    with open(onnx_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                print(f"❌ [ONNX PARSER ERROR]: {parser.get_error(error)}")
            raise RuntimeError("Failed to parse ONNX file.")

    config = builder.create_builder_config()
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1024 ** 3))
    else:
        config.max_workspace_size = workspace_gb * (1024 ** 3)

    if precision == "FP16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("⚡ FP16 Precision enabled.")
    elif precision == "INT8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            calibrator = HSIEntropyCalibrator(
                calib_npy="models/vitis_ai/calib_patches_indian.npy",
                cache_file=f"models/engines/ss_resnet_indian_calib.cache"
            )
            config.int8_calibrator = calibrator
            print("⚡ INT8 Precision enabled with HSI Entropy Calibrator.")
        else:
            print("⚠️ Warning: Platform does not support fast INT8, falling back to FP16.")
            config.set_flag(trt.BuilderFlag.FP16)

    print("🔨 Optimizing graph and selecting tacticians...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine.")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"✅ TensorRT Engine successfully saved to: {engine_path} ({size_mb:.2f} MB)")
    return engine_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build TensorRT Engine for SS-ResNet")
    parser.add_argument("--onnx", type=str, default="models/onnx/ss_resnet_indian_b1.onnx")
    parser.add_argument("--output", type=str, default="models/engines/ss_resnet_indian_b1_int8.engine")
    parser.add_argument("--precision", type=str, default="INT8", choices=["FP32", "FP16", "INT8"])
    parser.add_argument("--workspace", type=int, default=4)
    args = parser.parse_args()

    build_hsi_engine(args.onnx, args.output, precision=args.precision, workspace_gb=args.workspace)
