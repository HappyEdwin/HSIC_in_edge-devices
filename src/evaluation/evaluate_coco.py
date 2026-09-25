import os
from typing import Dict, Any
from ultralytics import YOLO

def evaluate_model_coco(
    model_path: str,
    data_yaml: str = "coco128.yaml",
    img_size: int = 640,
    device: str = "0"
) -> Dict[str, float]:
    """
    Evalúa la precisión de detección de un modelo (PyTorch .pt o TensorRT .engine / ONNX)
    sobre el dataset COCO128 usando el validador oficial de Ultralytics.
    """
    print(f"\n🎯 [EVALUACIÓN COCO128] Evaluando modelo: {model_path} en dataset {data_yaml}...")
    
    # Ultralytics permite pasar directamente .pt, .onnx o .engine a YOLO()
    model = YOLO(model_path)
    
    # Ejecutar validación
    metrics = model.val(
        data=data_yaml,
        imgsz=img_size,
        device=device,
        verbose=False,
        split="val"
    )

    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)

    print("\n📈 --- REPORTE DE PRECISIÓN ---")
    print(f"   • mAP@50:     {map50:.4f}")
    print(f"   • mAP@50-95:  {map50_95:.4f}")
    print(f"   • Precision:  {precision:.4f}")
    print(f"   • Recall:     {recall:.4f}")

    return {
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4)
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluar modelo en COCO128")
    parser.add_argument("--model", type=str, default="models/weights/yolo11n.pt")
    parser.add_argument("--data", type=str, default="coco128.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    evaluate_model_coco(
        model_path=args.model,
        data_yaml=args.data,
        img_size=args.imgsz,
        device=args.device
    )
