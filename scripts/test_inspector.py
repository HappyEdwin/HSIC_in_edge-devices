import os
import torch
from pytorch_nndct.apis import Inspector
from ultralytics import YOLO

def main():
    target = "DPUCZDX8G_ISA1_B4096"
    print(f"[*] Initializing Vitis AI Inspector for target: {target}")
    inspector = Inspector(target)
    
    weights = "models/weights/yolo11m.pt"
    print(f"[*] Loading YOLOv11m from {weights}...")
    yolo = YOLO(weights)
    model = yolo.model
    model.eval()
    
    import types
    def raw_forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        return x[0], x[1], x[2]

    model.model[-1].forward = types.MethodType(raw_forward, model.model[-1])
    print("[*] Replaced Detect.forward with raw convolution head forward pass")

    dummy_input = torch.randn(1, 3, 640, 640)
    output_dir = "models/vitis_ai/inspect_result"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[*] Inspecting model ops and hardware compatibility...")
    inspector.inspect(model, (dummy_input,), device=torch.device("cpu"), output_dir=output_dir)
    print(f"[+] Inspection finished! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
