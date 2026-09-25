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

    config = builder.create_builder_config()
    
    # Asignar workspace pool limit (compatible con TRT 8.x y 10.x)
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    elif hasattr(config, 'max_workspace_size'):
        config.max_workspace_size = workspace_gb * (1 << 30)

    # Configurar precisión FP16
    if precision == "FP16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("⚡ Aceleración FP16 (Tensor Cores) activada.")
        else:
            print("⚠️  Advertencia: Esta plataforma no tiene aceleración nativa rápida de FP16. Se compilará en modo estándar.")
    elif precision == "INT8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("⚡ Cuantización INT8 activada.")
        else:
            print("⚠️  Plataforma sin hardware INT8 nativo rápido.")

    print("🔨 Optimizando grafo y seleccionando tacticians (esto puede tomar 1-3 minutos)...")
    
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
