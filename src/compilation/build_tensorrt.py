import os
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any
import tensorrt as trt

class CustomTRTLogger(trt.ILogger):
    """
    Logger personalizado de TensorRT para capturar advertencias sobre
    capas u operadores no acelerados o que caen en fallback.
    """
    def __init__(self, severity=trt.ILogger.Severity.INFO):
        super().__init__()
        self.severity = severity
        self.logs = []
        self.unaccelerated_warnings = []

    def log(self, severity, msg):
        self.logs.append((severity, msg))
        msg_lower = msg.lower()
        # Detectar patrones comunes de capas no aceleradas o fallbacks
        unaccel_keywords = ["fallback", "could not run", "unsupported", "cannot be placed", "cannot run on"]
        if any(kw in msg_lower for kw in unaccel_keywords):
            self.unaccelerated_warnings.append(msg)
            print(f"⚠️  [TRT FALLBACK/ALERTA]: {msg}")
        elif severity == trt.ILogger.Severity.ERROR:
            print(f"❌ [TRT ERROR]: {msg}")
        elif severity == trt.ILogger.Severity.WARNING:
            # Filtrar warnings informativos rutinarios si se desea
            if "myelin" in msg_lower or "tactician" in msg_lower or "fallback" in msg_lower:
                print(f"⚠️  [TRT WARN]: {msg}")

def build_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "FP16",
    workspace_gb: int = 4
) -> Tuple[str, Dict[str, Any]]:
    """
    Compila un modelo ONNX a motor TensorRT (.engine) usando TensorRT Builder API.
    Monitorea capas, fusiones y detecta operadores no acelerados.
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Archivo ONNX no encontrado: {onnx_path}")

    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    precision = precision.upper()

    print(f"\n⚙️  [TENSORRT BUILD] Compilando Engine desde: {onnx_path}")
    print(f"    - Precisión objetivo: {precision}")
    print(f"    - Workspace máximo: {workspace_gb} GB")

    custom_logger = CustomTRTLogger(trt.ILogger.Severity.INFO)
    builder = trt.Builder(custom_logger)
    if hasattr(builder, 'max_threads'):
        builder.max_threads = os.cpu_count() or 4
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, custom_logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("❌ Error al parsear ONNX con TensorRT:")
            for i in range(parser.num_errors):
                print(f"   - {parser.get_error(i)}")
            raise RuntimeError("Fallo en el parser ONNX de TensorRT.")

    num_network_layers = network.num_layers
    print(f"📋 Grafo ONNX importado con éxito: {num_network_layers} capas detectadas en la definición de red.")

class YOLOInt8Calibrator(trt.IInt8EntropyCalibrator2):
    """
    Calibrador Entropy para cuantización INT8 de modelos YOLO.
    """
    def __init__(self, image_paths, cache_file, img_size=640, batch_size=1):
        super().__init__()
        self.image_paths = image_paths
        self.cache_file = cache_file
        self.img_size = img_size
        self.batch_size = batch_size
        self.current_index = 0
        import torch
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device_input = torch.empty((self.batch_size, 3, self.img_size, self.img_size), dtype=torch.float32, device=self.device)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.image_paths):
            return None
        
        batch_paths = self.image_paths[self.current_index : self.current_index + self.batch_size]
        self.current_index += self.batch_size

        import cv2
        import numpy as np
        import torch

        batch_imgs = []
        for p in batch_paths:
            img = None
            if os.path.isfile(p):
                img = cv2.imread(p)
            if img is None:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1)) # HWC -> CHW
            batch_imgs.append(img)

        while len(batch_imgs) < self.batch_size:
            batch_imgs.append(np.zeros((3, self.img_size, self.img_size), dtype=np.float32))

        batch_tensor = torch.from_numpy(np.ascontiguousarray(np.array(batch_imgs))).to(self.device)
        self.device_input.copy_(batch_tensor)
        return [int(self.device_input.data_ptr())]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"📖 Leyendo caché de calibración INT8 existente desde: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_file)), exist_ok=True)
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"💾 Caché de calibración INT8 guardado en: {self.cache_file}")

def get_calibration_images(dataset_yaml="coco128.yaml", num_samples=128):
    import glob
    candidates = [
        "datasets/coco128/images/train2017/*.jpg",
        "/workspace/datasets/coco128/images/train2017/*.jpg",
        "/home/edwinacevedo/VIP/datasets/coco128/images/train2017/*.jpg",
        "datasets/coco128/images/val2017/*.jpg",
        "datasets/coco/images/val2017/*.jpg"
    ]
    for pattern in candidates:
        imgs = sorted(glob.glob(pattern))
        if imgs:
            return imgs[:num_samples]
    # Fallback usando check_det_dataset de ultralytics
    try:
        from ultralytics.data.utils import check_det_dataset
        ds = check_det_dataset(dataset_yaml)
        val_path = ds.get('train') or ds.get('val')
        if isinstance(val_path, str) and os.path.isdir(val_path):
            imgs = sorted(glob.glob(os.path.join(val_path, "*.jpg")))
            if imgs:
                return imgs[:num_samples]
    except Exception:
        pass
    return [f"dummy_{i}.jpg" for i in range(num_samples)]

def build_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "FP16",
    workspace_gb: int = 4,
    calib_cache: str = None,
    dataset_yaml: str = "coco128.yaml",
    img_size: int = 640
) -> Tuple[str, Dict[str, Any]]:
    """
    Compila un modelo ONNX a motor TensorRT (.engine) usando TensorRT Builder API.
    Soporta FP32, FP16 e INT8 con calibrador de entropía.
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Archivo ONNX no encontrado: {onnx_path}")

    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    precision = precision.upper()

    print(f"\n⚙️  [TENSORRT BUILD] Compilando Engine desde: {onnx_path}")
    print(f"    - Precisión objetivo: {precision}")
    print(f"    - Workspace máximo: {workspace_gb} GB")

    custom_logger = CustomTRTLogger(trt.ILogger.Severity.INFO)
    builder = trt.Builder(custom_logger)
    if hasattr(builder, 'max_threads'):
        builder.max_threads = os.cpu_count() or 4
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, custom_logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("❌ Error al parsear ONNX con TensorRT:")
            for i in range(parser.num_errors):
                print(f"   - {parser.get_error(i)}")
            raise RuntimeError("Fallo en el parser ONNX de TensorRT.")

    num_network_layers = network.num_layers
    print(f"📋 Grafo ONNX importado con éxito: {num_network_layers} capas detectadas en la definición de red.")

    config = builder.create_builder_config()
    
    # Asignar workspace pool limit (compatible con TRT 8.x y 10.x)
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    elif hasattr(config, 'max_workspace_size'):
        config.max_workspace_size = workspace_gb * (1 << 30)

    # Configurar precisión FP16 / INT8
    if precision == "FP16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("⚡ Aceleración FP16 (Tensor Cores) activada.")
        else:
            print("⚠️  Advertencia: Esta plataforma no tiene aceleración nativa rápida de FP16. Se compilará en modo estándar.")
    elif precision == "INT8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            print("⚡ Cuantización INT8 activada con Tensor Cores.")
            
            calib_cache_path = calib_cache or engine_path.replace(".engine", "_calib.cache")
            calib_images = get_calibration_images(dataset_yaml=dataset_yaml, num_samples=128)
            calibrator = YOLOInt8Calibrator(calib_images, cache_file=calib_cache_path, img_size=img_size, batch_size=1)
            config.int8_calibrator = calibrator
            print(f"🎯 Calibrador INT8 configurado con {len(calib_images)} muestras de calibración.")
        else:
            print("⚠️  Plataforma sin hardware INT8 nativo rápido.")

    print("🔨 Optimizando grafo y seleccionando tacticians (esto puede tomar varios minutos)...")
    
    # Compilar serialized network
    serialized_engine = None
    if hasattr(builder, 'build_serialized_network'):
        serialized_engine = builder.build_serialized_network(network, config)
    else:
        engine = builder.build_engine(network, config)
        if engine:
            serialized_engine = engine.serialize()

    if serialized_engine is None:
        raise RuntimeError("No se pudo compilar el motor TensorRT (serialized_engine is None).")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    file_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"✅ TensorRT Engine guardado en: {engine_path} ({file_size_mb:.2f} MB)")

    # Reporte de capas no aceleradas
    unaccelerated = custom_logger.unaccelerated_warnings
    if unaccelerated:
        print(f"\n⚠️  ATENCIÓN: Se detectaron {len(unaccelerated)} posibles advertencias de aceleración/fallback:")
        for w in unaccelerated:
            print(f"   • {w}")
    else:
        print("🎉 Todas las capas y operadores fueron optimizados y acelerados para la arquitectura sin fallbacks detectados.")

    report = {
        "engine_path": engine_path,
        "engine_size_mb": round(file_size_mb, 2),
        "network_layers_count": num_network_layers,
        "unaccelerated_layers_count": len(unaccelerated),
        "unaccelerated_warnings": unaccelerated,
        "precision": precision
    }
    return engine_path, report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compilar ONNX a TensorRT Engine")
    parser.add_argument("--onnx", type=str, default="models/onnx/yolo11n_640.onnx")
    parser.add_argument("--output", type=str, default="models/engines/yolo11n_640_fp16.engine")
    parser.add_argument("--precision", type=str, default="FP16", choices=["FP16", "FP32", "INT8"])
    parser.add_argument("--workspace", type=int, default=4)
    args = parser.parse_args()

    build_engine(
        onnx_path=args.onnx,
        engine_path=args.output,
        precision=args.precision,
        workspace_gb=args.workspace
    )
