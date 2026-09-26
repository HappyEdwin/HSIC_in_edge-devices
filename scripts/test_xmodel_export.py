import os
import copy
import types
import glob
import subprocess
import torch
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
    # Returns raw prediction feature maps for 3 scales
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    return x[0], x[1], x[2]

def cnn_attn_forward(self, x):
    return self.proj(self.pe(x))

def main():
    target = "DPUCZDX8G_ISA1_B4096"
    weights = "models/weights/yolo11m.pt"
    temp_dir = "models/vitis_ai/test_quant_yolo11m_clean"
    os.makedirs(temp_dir, exist_ok=True)

    print(f"[*] Loading YOLOv11m from {weights}...")
    yolo = YOLO(weights)
    model = yolo.model
    model.eval()

    # 1. Transform all C3k2 and C2f blocks into DpuC2f (removes chunk/transposes)
    transformed_c2f = 0
    for i, layer in enumerate(model.model):
        if isinstance(layer, (C2f, C3k2)):
            dpu_layer = DpuC2f(layer)
            dpu_layer.i = getattr(layer, "i", i)
            dpu_layer.f = getattr(layer, "f", -1)
            dpu_layer.type = getattr(layer, "type", layer.__class__.__name__)
            model.model[i] = dpu_layer
            transformed_c2f += 1
    print(f"[*] Transformed {transformed_c2f} C3k2/C2f blocks into DPU-native parallel convolutions.")

    # 2. Disable inplace on all activations
    disabled_inplace = 0
    for name, m in model.named_modules():
        if hasattr(m, "act") and isinstance(m.act, torch.nn.Module):
            if hasattr(m.act, "inplace"):
                m.act.inplace = False
                disabled_inplace += 1
        elif isinstance(m, (torch.nn.SiLU, torch.nn.ReLU, torch.nn.LeakyReLU)):
            m.inplace = False
            disabled_inplace += 1
    print(f"[*] Disabled inplace on {disabled_inplace} activation layers.")

    # 3. Patch Detect head for raw multi-scale feature maps
    model.model[-1].forward = types.MethodType(raw_detect_forward, model.model[-1])
    
    # 4. Patch C2PSA Attention to CNN form (pe 3x3 depthwise + proj 1x1)
    for m in model.modules():
        if m.__class__.__name__ == "Attention":
            m.forward = types.MethodType(cnn_attn_forward, m)
            print("[*] Patched Attention module to CNN mode (pe + proj)")

    dummy_input = torch.randn(1, 3, 640, 640)

    print("\n[*] Step 1: Quantization Calibration...")
    quantizer = torch_quantizer("calib", model, (dummy_input,), output_dir=temp_dir, target=target, device=torch.device("cpu"))
    quant_model = quantizer.quant_model
    with torch.no_grad():
        for _ in range(5):
            quant_model(torch.randn(1, 3, 640, 640))
    quantizer.export_quant_config()

    print("\n[*] Step 2: Export Quantized XIR Model...")
    quantizer_test = torch_quantizer("test", model, (dummy_input,), output_dir=temp_dir, target=target, device=torch.device("cpu"))
    quant_model_test = quantizer_test.quant_model
    with torch.no_grad():
        quant_model_test(dummy_input)

    quantizer_test.export_xmodel(output_dir=temp_dir, deploy_check=False)
    print("✅ export_xmodel completed successfully!")

    xmodels = glob.glob(os.path.join(temp_dir, "*_int.xmodel"))
    if not xmodels:
        xmodels = glob.glob(os.path.join(temp_dir, "*.xmodel"))
        
    if xmodels:
        int_xmodel = xmodels[0]
        print(f"\n[*] Step 3: Compiling {int_xmodel} with vai_c_xir for Kria KV260...")
        output_dir = "models/xmodel"
        os.makedirs(output_dir, exist_ok=True)
        cmd = [
            "vai_c_xir",
            "--xmodel", int_xmodel,
            "--arch", "/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json",
            "--output_dir", output_dir,
            "--net_name", "yolo11m_kv260"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print("STDERR:", res.stderr)
            
        target_path = os.path.join(output_dir, "yolo11m_kv260.xmodel")
        if os.path.exists(target_path):
            print(f"\n🎉🎉🎉 COMPILED XMODEL GENERATED FOR KRIA KV260: {target_path} ({os.path.getsize(target_path) / (1024*1024):.2f} MB) 🎉🎉🎉")
        else:
            print("[-] Target xmodel was not created.")

if __name__ == "__main__":
    main()
