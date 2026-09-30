#!/usr/bin/env python3
"""
Exports YOLOv11n with LeakyReLU(0.1) activations to PyTorch weights and ONNX.
Enables compiling FP32, FP16, and INT8 TensorRT engines on the Jetson Orin Nano
for fair comparison against the Kria KV260 LeakyReLU xmodel.
"""

import os
import torch
from ultralytics import YOLO

def export_leaky_model():
    weights_in = "models/weights/yolo11n.pt"
    weights_out = "models/weights/yolo11n_leaky.pt"
    onnx_out = "models/onnx/yolo11n_leaky_640.onnx"

    print("=" * 70)
    print(f"📦 Exporting YOLOv11n LeakyReLU for Jetson Orin Nano TensorRT Compilation")
    print(f"   Input:  {weights_in}")
    print(f"   Output: {weights_out}")
    print("=" * 70)

    yolo = YOLO(weights_in)
    model = yolo.model
    model.eval()

    count = 0
    for name, m in model.named_modules():
        if hasattr(m, "act") and isinstance(m.act, torch.nn.SiLU):
            m.act = torch.nn.LeakyReLU(0.1, inplace=False)
            count += 1
        elif isinstance(m, torch.nn.SiLU):
            pass

    print(f"✅ Replaced {count} SiLU activations with LeakyReLU(0.1).")

    # Save modified PyTorch model
    os.makedirs(os.path.dirname(weights_out), exist_ok=True)
    yolo.save(weights_out)
    print(f"✅ Saved PyTorch model: {weights_out}")

    # Export to ONNX for TensorRT compilation
    os.makedirs(os.path.dirname(onnx_out), exist_ok=True)
    try:
        yolo.export(format="onnx", imgsz=640, opset=17, dynamic=False, simplify=True)
        # Move generated ONNX to intended path
        gen_onnx = weights_out.replace(".pt", ".onnx")
        if os.path.exists(gen_onnx):
            os.rename(gen_onnx, onnx_out)
            print(f"✅ Exported ONNX: {onnx_out}")
    except Exception as e:
        print(f"[-] Note during ONNX export: {e}")

if __name__ == "__main__":
    export_leaky_model()
