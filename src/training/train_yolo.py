#!/usr/bin/env python3
import os
import argparse
import yaml
from ultralytics import YOLO

def train_model(
    config_path: str = "configs/yolo11n.yaml",
    epochs: int = 100,
    batch_size: int = 16,
    device: str = "0"
):
    """
    Etapa desacoplada de entrenamiento / fine-tuning.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    output_weights = cfg["model"]["weights"]
    dataset_yaml = cfg["dataset"]["data_yaml"]
    img_size = cfg["model"]["img_size"]

    os.makedirs(os.path.dirname(os.path.abspath(output_weights)), exist_ok=True)
    print(f"🚀 [ENTRENAMIENTO] Iniciando entrenamiento de {model_name} en {dataset_yaml}...")

    model = YOLO(f"{model_name}.pt")
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project="training_runs",
        name=f"{model_name}_run"
    )

    best_weight_path = os.path.join("training_runs", f"{model_name}_run", "weights", "best.pt")
    if os.path.exists(best_weight_path):
        import shutil
        shutil.copy(best_weight_path, output_weights)
        print(f"✅ Mejor checkpoint guardado en: {output_weights}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo YOLO")
    parser.add_argument("--config", type=str, default="configs/yolo11n.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    train_model(args.config, args.epochs, args.batch, args.device)
