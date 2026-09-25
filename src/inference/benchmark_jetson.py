#!/usr/bin/env python3
import os
import sys
import time
import glob
import numpy as np
import yaml
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.metrics import JetsonPowerMonitor, save_benchmark_result, count_parameters, estimate_flops
from src.evaluation.evaluate_coco import evaluate_model_coco
from ultralytics import YOLO
import torch

def benchmark_jetson(
    config_path: str = "configs/yolo11n.yaml",
    engine_path: str = None
):
    print("=" * 70)
    print("⚡ BENCHMARK JETSON: INFERENCIA END-TO-END Y TELEMETRÍA")
    print("=" * 70)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    img_size = cfg["model"]["img_size"]
    dataset_yaml = cfg["dataset"]["data_yaml"]
    warmup_runs = cfg["benchmark"].get("warmup_runs", 50)
    engine_path = engine_path or cfg["export"]["tensorrt"]["output_path"]
    precision = cfg["export"]["tensorrt"]["precision"].lower()

    if not os.path.exists(engine_path):
        raise FileNotFoundError(
            f"No se encontró el motor TensorRT: {engine_path}.\n"
            f"Asegúrate de haberlo compilado en la Jetson con:\n"
            f"python3 src/compilation/build_tensorrt.py --onnx models/onnx/{model_name}_{img_size}.onnx --output {engine_path}"
        )

    print(f"📦 Cargando modelo compilado: {engine_path}")
    model = YOLO(engine_path, task="detect")

    # 1. Warmup
    print(f"\n🔥 [WARMUP] Ejecutando {warmup_runs} pasadas de calentamiento...")
    dummy_img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    for _ in range(warmup_runs):
        _ = model(dummy_img, imgsz=img_size, verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # 2. Iniciar monitor de potencia tegrastats
    print("\n🔋 Iniciando monitor de consumo de potencia (tegrastats)...")
    power_monitor = JetsonPowerMonitor(interval_ms=100)
    power_monitor.start()

    # 3. Medición de Latencia End-to-End sobre el dataset
    print(f"⏱️  Midiendo latencia End-to-End (Preproceso + TRT + Postproceso NMS)...")
    
    # Obtener imágenes del dataset mediante check_det_dataset
    coco_images = []
    try:
        from ultralytics.data.utils import check_det_dataset
        ds = check_det_dataset(dataset_yaml)
        val_path = ds.get('val') or ds.get('train')
        if isinstance(val_path, list) and len(val_path) > 0:
            coco_images = val_path
        elif isinstance(val_path, str) and os.path.isdir(val_path):
            coco_images = sorted(glob.glob(os.path.join(val_path, "*.jpg")))
        elif isinstance(val_path, str) and os.path.isfile(val_path):
            with open(val_path, 'r') as vf:
                coco_images = [line.strip() for line in vf if line.strip()]
    except Exception:
        pass

    if not coco_images:
        coco_images = glob.glob("datasets/coco128/images/train2017/*.jpg")

    num_samples = min(len(coco_images), 128) if coco_images else 128
    latencies = []

    start_total_time = time.perf_counter()
    for i in range(num_samples):
        img_input = coco_images[i] if coco_images else dummy_img
        
        t0 = time.perf_counter()
        _ = model(img_input, imgsz=img_size, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        latencies.append((t1 - t0) * 1000.0) # ms

    total_duration = time.perf_counter() - start_total_time
    power_stats = power_monitor.stop()

    # 4. Memoria Peak
    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        # En caso de que se use memoria unificada de Jetson:
        if peak_vram_mb == 0.0:
            free_b, total_b = torch.cuda.mem_get_info()
            peak_vram_mb = round((total_b - free_b) / (1024 * 1024), 2)

    # 5. Estadísticas de Latencia
    latencies = np.array(latencies)
    lat_mean = float(np.mean(latencies))
    lat_median = float(np.median(latencies))
    lat_std = float(np.std(latencies))
    lat_p95 = float(np.percentile(latencies, 95))
    lat_min = float(np.min(latencies))
    lat_max = float(np.max(latencies))
    fps = round(1000.0 / lat_mean, 2) if lat_mean > 0 else 0.0

    print("\n" + "=" * 70)
    print("📊 RESULTADOS DEL BENCHMARK EN JETSON")
    print("=" * 70)
    print(f"• Latencia Media End-to-End: {lat_mean:.2f} ms")
    print(f"• Latencia Mediana:          {lat_median:.2f} ms")
    print(f"• Latencia P95:              {lat_p95:.2f} ms")
    print(f"• Latencia Min / Max:        {lat_min:.2f} ms / {lat_max:.2f} ms")
    print(f"• Throughput (FPS):          {fps} FPS")
    print(f"• Memoria Peak GPU:          {peak_vram_mb} MB")
    print(f"• Consumo Promedio:          {power_stats['power_avg_watts']} W (Pico: {power_stats['power_max_watts']} W)")

    # 6. Evaluación de Precisión del Engine en COCO128
    print("\n🎯 Evaluando precisión mAP del motor TensorRT en COCO128...")
    coco_metrics = {"mAP50": 0.0, "mAP50_95": 0.0}
    try:
        coco_metrics = evaluate_model_coco(
            model_path=engine_path,
            data_yaml=dataset_yaml,
            img_size=img_size
        )
    except Exception as e:
        print(f"⚠️  No se pudo calcular mAP completo en el engine: {e}")

    # 7. Guardar resultados
    result_data = {
        "model_name": model_name,
        "platform": "jetson_orin_nano",
        "precision": precision,
        "input_resolution": f"{img_size}x{img_size}",
        "latency_mean_ms": round(lat_mean, 2),
        "latency_median_ms": round(lat_median, 2),
        "latency_p95_ms": round(lat_p95, 2),
        "latency_std_ms": round(lat_std, 2),
        "fps": fps,
        "peak_vram_mb": peak_vram_mb,
        "power_avg_watts": power_stats["power_avg_watts"],
        "power_max_watts": power_stats["power_max_watts"],
        "mAP50": coco_metrics["mAP50"],
        "mAP50_95": coco_metrics["mAP50_95"],
        "samples_evaluated": len(latencies)
    }

    save_benchmark_result(result_data, results_dir=cfg["benchmark"].get("results_dir", "results"))
    print("\n✅ Benchmark finalizado y resultados guardados para graficar.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar Benchmark en NVIDIA Jetson")
    parser.add_argument("--config", type=str, default="configs/yolo11n.yaml")
    parser.add_argument("--engine", type=str, default=None)
    args = parser.parse_args()

    benchmark_jetson(args.config, args.engine)
