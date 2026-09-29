import torch
from ultralytics import YOLO

yolo = YOLO("models/weights/yolo11m.pt")
model = yolo.model
count = 0
for name, m in model.named_modules():
    if hasattr(m, "act") and isinstance(m.act, torch.nn.SiLU):
        m.act = torch.nn.LeakyReLU(0.1, inplace=False)
        count += 1
    elif isinstance(m, torch.nn.SiLU):
        count += 1
print(f"Successfully detected and replaced {count} SiLU activations with LeakyReLU!")
