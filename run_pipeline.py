#!/usr/bin/env python3
import os
import sys
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO

# Agregar la raíz del repositorio al sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.metrics import count_parameters, estimate_flops, save_benchmark_result
from src.compilation.export_onnx import export_to_onnx
from src.compilation.build_tensorrt import build_engine
from src.evaluation.evaluate_coco import evaluate_model_coco

def run_pipeline(config_path: str = "configs/yolo11n.yaml"):
    print("=" * 70)
    print("🚀 PIPELINE INICIAL: YOLOv11 NANO (HASTA COMPILACIÓN TENSORRT)")
    print("=" * 70)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    weights_path = cfg["model"]["weights"]
    img_size = cfg["model"]["img_size"]
    dataset_yaml = cfg["dataset"]["data_yaml"]
    onnx_out = cfg["export"]["onnx"]["output_path"]
    opset = cfg["export"]["onnx"]["opset"]
    simplify = cfg["export"]["onnx"]["simplify"]
    engine_out = cfg["export"]["tensorrt"]["output_path"]
    precision = cfg["export"]["tensorrt"]["precision"]
    workspace_gb = cfg["export"]["tensorrt"]["workspace_gb"]

    # 1. Cargar pesos preentrenados y calcular métricas estáticas
    print(f"\n[PASO 1/5] 📥 Verificando pesos base de {model_name}...")
    os.makedirs(os.path.dirname(os.path.abspath(weights_path)), exist_ok=True)
    if not os.path.exists(weights_path):
        print(f"Descargando checkpoint preentrenado {model_name}.pt...")
        temp = YOLO(f"{model_name}.pt")
        import shutil
        if os.path.exists(f"{model_name}.pt"):
            shutil.move(f"{model_name}.pt", weights_path)
    
    pt_model = YOLO(weights_path)
    
    print("\n[PASO 2/5] 🧮 Análisis de Complejidad Inicial del Modelo:")
    num_params = count_parameters(pt_model)
    params_m = round(num_params / 1e6, 3)
    gflops = round(estimate_flops(pt_model, input_size=(1, 3, img_size, img_size)), 3)
    
    print(f"   • Cantidad de Parámetros: {num_params:,} ({params_m} M)")
    print(f"   • Operaciones estimadas:  {gflops} GFLOPs (a resolución {img_size}x{img_size})")

    # 2. Evaluación de Precisión Baseline en COCO128
    print(f"\n[PASO 3/5] 🎯 Reporte de Precisión Inicial (PyTorch Baseline en {dataset_yaml}):")
    coco_metrics = evaluate_model_coco(
        model_path=weights_path,
        data_yaml=dataset_yaml,
        img_size=img_size
    )

    # 3. Exportación a ONNX
    print(f"\n[PASO 4/5] 📦 Exportando a Formato Intermedio ONNX:")
    onnx_path = export_to_onnx(
        weights_path=weights_path,
        output_path=onnx_out,
        img_size=img_size,
        opset=opset,
        simplify=simplify
    )

    # 4. Compilación a TensorRT Engine
    print(f"\n[PASO 5/5] ⚡ Compilando TensorRT Engine ({precision}):")
    calib_cache = cfg.get("export", {}).get("tensorrt", {}).get("calib_cache")
    engine_path, trt_report = build_engine(
        onnx_path=onnx_path,
        engine_path=engine_out,
        precision=precision,
        workspace_gb=workspace_gb,
        calib_cache=calib_cache,
        dataset_yaml=dataset_yaml,
        img_size=img_size
    )

    # Resumen y Almacenamiento
    print("\n" + "=" * 70)
    print("📋 RESUMEN DE LA COMPILACIÓN")
    print("=" * 70)
    print(f"• Modelo:                    {model_name}")
    print(f"• Parámetros:                {params_m} M")
    print(f"• Operaciones Iniciales:     {gflops} GFLOPs")
    print(f"• Precisión Baseline mAP50:  {coco_metrics['mAP50']}")
    print(f"• Precisión Baseline mAP50-95: {coco_metrics['mAP50_95']}")
    print(f"• Grafo ONNX:                {trt_report['network_layers_count']} capas")
    print(f"• Tamaño Engine TensorRT:    {trt_report['engine_size_mb']} MB")
    print(f"• Capas no aceleradas/Alertas: {trt_report['unaccelerated_layers_count']}")
    
    # Guardar en JSON / CSV para posteriores comparativas gráficas
    export_summary = {
        "model_name": model_name,
        "platform": "compilation_host",
        "precision": precision.lower(),
        "input_resolution": f"{img_size}x{img_size}",
        "params_m": params_m,
        "gflops": gflops,
        "mAP50": coco_metrics["mAP50"],
        "mAP50_95": coco_metrics["mAP50_95"],
        "engine_size_mb": trt_report["engine_size_mb"],
        "unaccelerated_layers_count": trt_report["unaccelerated_layers_count"],
        "unaccelerated_warnings": trt_report["unaccelerated_warnings"],
        "latency_mean_ms": 0.0, # Se llenará en la etapa de inferencia en la Jetson
        "fps": 0.0,
        "peak_vram_mb": 0.0,
        "power_avg_watts": 0.0
    }
    save_benchmark_result(export_summary, results_dir="results")
    print("\n✨ Pipeline completado exitosamente hasta la exportación a TensorRT.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar Pipeline YOLOv11 hasta TensorRT")
    parser.add_argument("--config", type=str, default="configs/yolo11n.yaml", help="Ruta al archivo de configuración")
    args = parser.parse_args()
    run_pipeline(args.config)
