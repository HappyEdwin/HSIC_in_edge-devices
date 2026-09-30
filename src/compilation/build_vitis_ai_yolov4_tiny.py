#!/usr/bin/env python3
"""
Vitis AI INT8 Calibration & Compilation for YOLOv4-tiny on AMD-Xilinx Kria KV260.
Target: DPUCZDX8G_ISA1_B3136 (1 single unified DPU kernel).
"""

import os
import sys
import glob
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from pytorch_nndct.apis import torch_quantizer
import subprocess

# Add yolov4_tiny path
sys.path.insert(0, os.path.abspath("src/models/yolov4_tiny"))
from yolo import YoloBody

def load_calib_dataset(dataset_dir="data/coco128/images/train2017", num_samples=64, img_size=416):
    valid_exts = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(dataset_dir, ext)))
    image_paths = sorted(image_paths)[:num_samples]
    print(f"[*] Found {len(image_paths)} images for YOLOv4-tiny INT8 calibration ({img_size}x{img_size}).")

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])

    tensors = []
    for p in image_paths:
        try:
            with Image.open(p) as img:
                img = img.convert("RGB")
                tensors.append(transform(img).unsqueeze(0))
        except Exception as e:
            print(f"[-] Warning: Failed to load {p}: {e}")
    return tensors

def main():
    weights_path = "models/weights/yolov4_tiny_coco.pth"
    output_dir = "models/xmodel"
    temp_dir = "models/vitis_ai/quant_yolov4_tiny"
    arch_json = "models/vitis_ai/arch_kv260_b3136.json"
    img_size = 416

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 75)
    print("🚀 Compiling YOLOv4-tiny for AMD Kria KV260 (DPUCZDX8G_ISA1_B3136)")
    print(f"   Input Size: {img_size}x{img_size}")
    print(f"   Architecture: Native LeakyReLU CSPDarknet53-tiny (1 DPU Kernel)")
    print("=" * 75)

    # 1. Instantiate Model & Load Weights
    anchors_mask = [[3, 4, 5], [1, 2, 3]]
    model = YoloBody(anchors_mask, num_classes=80)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Successfully loaded pretrained weights into YOLOv4-tiny.")

    dummy_input = torch.randn(1, 3, img_size, img_size)
    calib_tensors = load_calib_dataset(num_samples=64, img_size=img_size)

    # 2. Calibration
    print("\n[*] Running INT8 Calibration...")
    quantizer = torch_quantizer("calib", model, (dummy_input,), output_dir=temp_dir)
    quant_model = quantizer.quant_model
    for idx, img_t in enumerate(calib_tensors):
        quant_model(img_t)
        if (idx + 1) % 16 == 0:
            print(f"   Calibrated {idx + 1}/{len(calib_tensors)} images...")
    quantizer.export_quant_config()
    print("✅ Calibration config exported.")

    # 3. Export XModel
    print("\n[*] Exporting Quantized Model to XIR...")
    test_quantizer = torch_quantizer("test", model, (dummy_input,), output_dir=temp_dir)
    test_model = test_quantizer.quant_model
    test_model(dummy_input)
    test_quantizer.export_xmodel(output_dir=temp_dir, deploy_check=False)
    print("✅ XIR Graph created.")

    # 4. Compile with vai_c_xir
    int_xmodel = os.path.join(temp_dir, "YoloBody_int.xmodel")
    print(f"\n[*] Compiling with vai_c_xir -> {output_dir}/yolov4_tiny_kv260.xmodel...")
    cmd = [
        "vai_c_xir",
        "--xmodel", int_xmodel,
        "--arch", arch_json,
        "--net_name", "yolov4_tiny_kv260",
        "--output_dir", output_dir
    ]
    subprocess.run(cmd, check=True)

    final_xmodel = os.path.join(output_dir, "yolov4_tiny_kv260.xmodel")
    if os.path.exists(final_xmodel):
        size_mb = os.path.getsize(final_xmodel) / (1024 * 1024)
        print(f"\n🎉🎉🎉 SUCCESS: Compiled YOLOv4-tiny xmodel: {final_xmodel} ({size_mb:.2f} MB) 🎉🎉🎉")
    else:
        raise FileNotFoundError(f"Output xmodel not found at {final_xmodel}")

if __name__ == "__main__":
    main()
