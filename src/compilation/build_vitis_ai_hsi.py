#!/usr/bin/env python3
"""
Vitis AI INT8 Calibration & Compilation for SS-ResNet on AMD Kria KV260.
Target: DPUCZDX8G_ISA1_B3136 (1 single unified DPU kernel).
"""

import os
import sys
import numpy as np
import torch
import subprocess
from pytorch_nndct.apis import torch_quantizer

# Add HSI model path
sys.path.insert(0, os.path.abspath("src/models/hsi"))
from ss_resnet import SSResNet

def main():
    dataset = "indian"
    weights_path = f"models/weights/ss_resnet_{dataset}.pt"
    calib_data_path = f"models/vitis_ai/calib_patches_{dataset}.npy"
    output_dir = "models/xmodel"
    temp_dir = f"models/vitis_ai/quant_ss_resnet_{dataset}"
    arch_json = "models/vitis_ai/arch_kv260_b3136.json"
    
    in_bands = 30
    patch_size = 13
    num_classes = 16

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 75)
    print("🚀 Compiling SS-ResNet (HSI) for AMD Kria KV260 (DPUCZDX8G_ISA1_B3136)")
    print(f"   Dataset: Indian Pines | Bands: {in_bands} | Patch: {patch_size}x{patch_size}")
    print(f"   Architecture: 100% Native 2D DPU Layers (Single Unified DPU Kernel)")
    print("=" * 75)

    # 1. Instantiate Model & Load Weights
    model = SSResNet(in_bands=in_bands, num_classes=num_classes, patch_size=patch_size)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Successfully loaded pretrained weights into SS-ResNet.")

    dummy_input = torch.randn(1, in_bands, patch_size, patch_size)

    # Load calibration patches
    print(f"[*] Loading calibration patches from {calib_data_path}...")
    calib_patches = np.load(calib_data_path)
    calib_tensors = [torch.from_numpy(calib_patches[i:i+1]).float() for i in range(min(128, len(calib_patches)))]
    print(f"[*] Using {len(calib_tensors)} patches for INT8 calibration.")

    # 2. Calibration
    print("\n[*] Running Vitis AI INT8 Calibration...")
    quantizer = torch_quantizer("calib", model, (dummy_input,), output_dir=temp_dir)
    quant_model = quantizer.quant_model
    for idx, patch_t in enumerate(calib_tensors):
        quant_model(patch_t)
        if (idx + 1) % 32 == 0:
            print(f"   Calibrated {idx + 1}/{len(calib_tensors)} patches...")
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
    int_xmodel = os.path.join(temp_dir, "SSResNet_int.xmodel")
    print(f"\n[*] Compiling with vai_c_xir -> {output_dir}/ss_resnet_indian_kv260.xmodel...")
    cmd = [
        "vai_c_xir",
        "--xmodel", int_xmodel,
        "--arch", arch_json,
        "--net_name", "ss_resnet_indian_kv260",
        "--output_dir", output_dir
    ]
    subprocess.run(cmd, check=True)

    final_xmodel = os.path.join(output_dir, "ss_resnet_indian_kv260.xmodel")
    if os.path.exists(final_xmodel):
        size_kb = os.path.getsize(final_xmodel) / 1024
        print(f"\n🎉🎉🎉 SUCCESS: Compiled SS-ResNet xmodel: {final_xmodel} ({size_kb:.2f} KB) 🎉🎉🎉")
    else:
        raise FileNotFoundError(f"Output xmodel not found at {final_xmodel}")

if __name__ == "__main__":
    main()
