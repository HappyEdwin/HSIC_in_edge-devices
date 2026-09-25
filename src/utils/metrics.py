import os
import json
import csv
import time
import subprocess
import threading
from typing import Dict, Any, Optional

def count_parameters(model) -> int:
    """Calcula el número total de parámetros de un modelo PyTorch."""
    if hasattr(model, 'parameters'):
        return sum(p.numel() for p in model.parameters())
    elif hasattr(model, 'model') and hasattr(model.model, 'parameters'):
        return sum(p.numel() for p in model.model.parameters())
    return 0

def estimate_flops(model, input_size=(1, 3, 640, 640)) -> float:
    """
    Estima los GFLOPs del modelo.
    Usa la introspección de Ultralytics si está disponible, o thop/torchinfo como fallback.
    """
    try:
        # Si es un objeto YOLO de ultralytics o contiene .model
        target = model.model if hasattr(model, 'model') else model
        if hasattr(target, 'info'):
            info = target.info(detailed=False, verbose=False)
            if isinstance(info, (tuple, list)) and len(info) >= 4:
                return float(info[3])
    except Exception:
        pass

    try:
        import torch
        from thop import profile
        dummy_input = torch.randn(*input_size)
        device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
        dummy_input = dummy_input.to(device)
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return float(macs * 2) / 1e9  # 1 MAC ~= 2 FLOPs
    except Exception:
        return 0.0

class JetsonPowerMonitor:
    """
    Monitorea el consumo de potencia (mW) y memoria en Jetson usando tegrastats en un thread de fondo.
    """
    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self.process: Optional[subprocess.Popen] = None
        self.power_readings = []
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

    def _reader(self):
        try:
            self.process = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            while not self.stop_event.is_set() and self.process.poll() is None:
                line = self.process.stdout.readline()
                if not line:
                    break
                # Parsear VDD_IN o POM_5V_IN o VDD_GPU según modelo Jetson
                # Formato típico Orin: VDD_IN 4500mW/4500mW o similar
                import re
                match = re.search(r'VDD_IN\s+(\d+)mW', line) or re.search(r'POM_5V_IN\s+(\d+)mW', line)
                if match:
                    self.power_readings.append(float(match.group(1)) / 1000.0) # Watts
        except Exception:
            pass

    def start(self):
        self.power_readings = []
        self.stop_event.clear()
        if os.path.exists("/usr/bin/tegrastats"):
            self.worker_thread = threading.Thread(target=self._reader, daemon=True)
            self.worker_thread.start()

    def stop(self) -> Dict[str, float]:
        self.stop_event.set()
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except Exception:
                self.process.kill()
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)

        if self.power_readings:
            avg_w = sum(self.power_readings) / len(self.power_readings)
            max_w = max(self.power_readings)
            return {"power_avg_watts": round(avg_w, 3), "power_max_watts": round(max_w, 3)}
        return {"power_avg_watts": 0.0, "power_max_watts": 0.0}

def save_benchmark_result(result_data: Dict[str, Any], results_dir: str = "results"):
    """
    Guarda el resultado en:
    1. Un archivo JSON detallado por modelo/ejecución
    2. Lo anexa a benchmark_summary.csv para fácil graficado cruzado
    """
    os.makedirs(results_dir, exist_ok=True)
    
    model_name = result_data.get("model_name", "unknown")
    platform = result_data.get("platform", "jetson")
    precision = result_data.get("precision", "fp16")
    timestamp = int(time.time())
    
    # 1. Guardar JSON individual
    json_filename = f"{model_name}_{platform}_{precision}_{timestamp}.json"
    json_path = os.path.join(results_dir, json_filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)
    print(f"📄 Resultado detallado guardado en: {json_path}")

    # 2. Anexar a CSV general
    csv_path = os.path.join(results_dir, "benchmark_summary.csv")
    file_exists = os.path.isfile(csv_path)

    # Campos principales para el CSV comparativo
    csv_row = {
        "timestamp": timestamp,
        "model_name": model_name,
        "platform": platform,
        "precision": precision,
        "input_resolution": result_data.get("input_resolution", "640x640"),
        "params_m": result_data.get("params_m", 0.0),
        "gflops": result_data.get("gflops", 0.0),
        "mAP50": result_data.get("mAP50", 0.0),
        "mAP50_95": result_data.get("mAP50_95", 0.0),
        "latency_mean_ms": result_data.get("latency_mean_ms", 0.0),
        "latency_median_ms": result_data.get("latency_median_ms", 0.0),
        "latency_p95_ms": result_data.get("latency_p95_ms", 0.0),
        "fps": result_data.get("fps", 0.0),
        "peak_vram_mb": result_data.get("peak_vram_mb", 0.0),
        "power_avg_watts": result_data.get("power_avg_watts", 0.0),
        "unaccelerated_layers": result_data.get("unaccelerated_layers_count", 0),
    }

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(csv_row)
    print(f"📊 Fila agregada a tabla comparativa: {csv_path}")
