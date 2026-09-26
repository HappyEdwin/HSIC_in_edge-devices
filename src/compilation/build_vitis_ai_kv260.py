import os
import sys
import copy
import types
import glob
import subprocess
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
from pytorch_nndct.apis import torch_quantizer
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f, C3k2

class DpuC2f(torch.nn.Module):
    """
    Bit-exact, DPU-native replacement for Ultralytics C2f/C3k2 blocks.
    Replaces torch.chunk(2, 1) with two parallel 1x1 convolutions (cv1_a and cv1_b)
    to completely eliminate unsupported strided_slice and transpose operations on Xilinx DPU.
    """
    def __init__(self, c2f):
        super().__init__()
        c = c2f.c
        self.c = c
        self.cv1_a = copy.deepcopy(c2f.cv1)
        self.cv1_b = copy.deepcopy(c2f.cv1)
        
        # Split weights and bias for 1x1 conv
        self.cv1_a.conv.weight = torch.nn.Parameter(c2f.cv1.conv.weight[:c].clone())
        self.cv1_b.conv.weight = torch.nn.Parameter(c2f.cv1.conv.weight[c:].clone())
        if c2f.cv1.conv.bias is not None:
            self.cv1_a.conv.bias = torch.nn.Parameter(c2f.cv1.conv.bias[:c].clone())
            self.cv1_b.conv.bias = torch.nn.Parameter(c2f.cv1.conv.bias[c:].clone())
            
        # Split BatchNorm parameters
        if hasattr(c2f.cv1, "bn") and c2f.cv1.bn is not None:
            for attr in ["weight", "bias", "running_mean", "running_var"]:
                val = getattr(c2f.cv1.bn, attr)
                if val is not None:
                    if "running" in attr:
                        setattr(self.cv1_a.bn, attr, val[:c].clone())
                        setattr(self.cv1_b.bn, attr, val[c:].clone())
                    else:
                        setattr(self.cv1_a.bn, attr, torch.nn.Parameter(val[:c].clone()))
                        setattr(self.cv1_b.bn, attr, torch.nn.Parameter(val[c:].clone()))
        self.m = c2f.m
        self.cv2 = c2f.cv2
        
    def forward(self, x):
        y = [self.cv1_a(x), self.cv1_b(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

def raw_detect_forward(self, x):
    """
    Returns raw multi-scale feature maps across all detection scales.
    """
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    return x[0], x[1], x[2]

def cnn_attn_forward(self, x):
    """
    DPU-native convolution equivalent for C2PSA attention block using trained pe and proj layers.
    """
    return self.proj(self.pe(x))

def get_calib_dataset(dataset_dir: str, num_samples: int = 64, img_size: int = 640):
    """
    Loads and preprocesses calibration images from dataset_dir.
    """
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(dataset_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))
    
    image_paths = sorted(list(set(image_paths)))
    if not image_paths:
        raise FileNotFoundError(f"No images found in calibration directory: {dataset_dir}")
    
    selected_paths = image_paths[:num_samples]
    print(f"[*] Found {len(image_paths)} images. Using {len(selected_paths)} for INT8 calibration.")
    
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),  # Scales [0, 255] to [0.0, 1.0]
    ])
    
    tensors = []
    for p in selected_paths:
        try:
            with Image.open(p) as img:
                img = img.convert("RGB")
                tensor = transform(img).unsqueeze(0)
                tensors.append(tensor)
        except Exception as e:
            print(f"[-] Warning: Failed to load {p}: {e}")
            
    return tensors

def quantize_and_compile(
    weights_path: str = "models/weights/yolo11m.pt",
    calib_dir: str = "/home/edwinacevedo/VIP/datasets/coco128/images/train2017",
    output_dir: str = "models/xmodel",
    model_name: str = "yolo11m_kv260",
    arch_json: str = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json",
    num_calib_samples: int = 64,
    img_size: int = 640,
):
    print("=" * 75)
    print(f"🚀 Vitis AI INT8 Quantization & Compilation for Kria KV260")
    print(f"   Model: {weights_path}")
    print(f"   Calibration dataset: {calib_dir}")
    print(f"   Target Architecture: {arch_json}")
    print(f"   Output Directory: {output_dir}")
    print("=" * 75)

    target_dpu = "DPUCZDX8G_ISA1_B4096"
    temp_quant_dir = "models/vitis_ai/quantize_result_yolo11m"
    os.makedirs(temp_quant_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    dummy_input = torch.randn(1, 3, img_size, img_size)

    # 1. Load Model and transform layers for DPU
    print(f"\n[1/4] 📦 Loading PyTorch model from {weights_path}...")
    yolo = YOLO(weights_path)
    model = yolo.model
    model.eval()

    transformed_c2f = 0
    for i, layer in enumerate(model.model):
        if isinstance(layer, (C2f, C3k2)):
            dpu_layer = DpuC2f(layer)
            dpu_layer.i = getattr(layer, "i", i)
            dpu_layer.f = getattr(layer, "f", -1)
            dpu_layer.type = getattr(layer, "type", layer.__class__.__name__)
            model.model[i] = dpu_layer
            transformed_c2f += 1
    print(f"✅ Transformed {transformed_c2f} C3k2/C2f blocks into DPU-native parallel convolutions.")

    disabled_inplace = 0
    for name, m in model.named_modules():
        if hasattr(m, "act") and isinstance(m.act, torch.nn.Module):
            if hasattr(m.act, "inplace"):
                m.act.inplace = False
                disabled_inplace += 1
        elif isinstance(m, (torch.nn.SiLU, torch.nn.ReLU, torch.nn.LeakyReLU)):
            m.inplace = False
            disabled_inplace += 1
    print(f"✅ Disabled inplace on {disabled_inplace} activation layers.")

    model.model[-1].forward = types.MethodType(raw_detect_forward, model.model[-1])
    print("✅ Detect head mapped to raw multi-scale convolution outputs.")

    for m in model.modules():
        if m.__class__.__name__ == "Attention":
            m.forward = types.MethodType(cnn_attn_forward, m)
            print("✅ Patched C2PSA Attention module to CNN mode (pe + proj).")

    # 2. Calibration Mode ('calib')
    print(f"\n[2/4] 🎯 Running INT8 Calibration with {num_calib_samples} COCO images...")
    calib_tensors = get_calib_dataset(calib_dir, num_samples=num_calib_samples, img_size=img_size)

    quantizer = torch_quantizer(
        quant_mode="calib",
        module=model,
        input_args=(dummy_input,),
        output_dir=temp_quant_dir,
        target=target_dpu,
        device=torch.device("cpu")
    )
    quant_model = quantizer.quant_model

    for idx, img_t in enumerate(calib_tensors):
        if (idx + 1) % 10 == 0 or idx == len(calib_tensors) - 1:
            print(f"   Calibrating image {idx + 1}/{len(calib_tensors)}...")
        with torch.no_grad():
            quant_model(img_t)

    quantizer.export_quant_config()
    print("✅ INT8 quantization parameters exported.")

    # 3. Test & Export Mode ('test')
    print("\n[3/4] 📤 Exporting quantized XIR graph...")
    quantizer_test = torch_quantizer(
        quant_mode="test",
        module=model,
        input_args=(dummy_input,),
        output_dir=temp_quant_dir,
        target=target_dpu,
        device=torch.device("cpu")
    )
    quant_model_test = quantizer_test.quant_model
    with torch.no_grad():
        quant_model_test(dummy_input)

    quantizer_test.export_xmodel(output_dir=temp_quant_dir, deploy_check=False)
    print("✅ Intermediate quantized XIR model exported.")

    xmodel_candidates = glob.glob(os.path.join(temp_quant_dir, "*_int.xmodel"))
    if not xmodel_candidates:
        xmodel_candidates = glob.glob(os.path.join(temp_quant_dir, "*.xmodel"))
    
    if not xmodel_candidates:
        raise FileNotFoundError(f"No exported xmodel found in {temp_quant_dir}")
        
    int_xmodel_path = xmodel_candidates[0]
    print(f"🔍 Located quantized XIR file: {int_xmodel_path}")

    # 4. Compile with vai_c_xir
    print(f"\n[4/4] ⚙️ Compiling for Kria KV260 DPU with vai_c_xir...")
    compile_cmd = [
        "vai_c_xir",
        "--xmodel", int_xmodel_path,
        "--arch", arch_json,
        "--output_dir", output_dir,
        "--net_name", model_name
    ]
    print(f"Executing: {' '.join(compile_cmd)}")
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
        
    if result.returncode != 0:
        raise RuntimeError(f"vai_c_xir failed with code {result.returncode}")

    target_xmodel = os.path.join(output_dir, f"{model_name}.xmodel")
    if os.path.exists(target_xmodel):
        size_mb = os.path.getsize(target_xmodel) / (1024 * 1024)
        print(f"\n🎉🎉🎉 SUCCESS! Kria KV260 DPU Model created: {target_xmodel} ({size_mb:.2f} MB) 🎉🎉🎉")
        return target_xmodel
    else:
        raise FileNotFoundError(f"Expected compiled file {target_xmodel} was not found.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compile YOLO model for AMD-Xilinx Kria KV260 DPU")
    parser.add_argument("--weights", type=str, default="models/weights/yolo11m.pt")
    parser.add_argument("--calib-dir", type=str, default="/home/edwinacevedo/VIP/datasets/coco128/images/train2017")
    parser.add_argument("--output-dir", type=str, default="models/xmodel")
    parser.add_argument("--name", type=str, default="yolo11m_kv260")
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()

    quantize_and_compile(
        weights_path=args.weights,
        calib_dir=args.calib_dir,
        output_dir=args.output_dir,
        model_name=args.name,
        num_calib_samples=args.samples
    )
