#!/usr/bin/env bash
set -e

echo "========================================================="
echo "🚀 JETSON ORIN NANO - BENCHMARK RUNNER (DOCKER CONTAINER)"
echo "========================================================="

# 1. Asegurar directorio de trabajo
if [ -d "/workspace" ]; then
    cd /workspace
    echo "📂 Directorio de trabajo establecido en: /workspace"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR"
    echo "📂 Directorio de trabajo: $SCRIPT_DIR"
fi

# 2. Verificar y aplicar fix de DLA en JetPack 36.4 si es necesario
DLA_DEB="models/TGRS_2025_MCTGCL/nvidia-l4t-dla-compiler_36.4.3-20250107174145_arm64.deb"
if [ -f "$DLA_DEB" ] && [ ! -f "/usr/lib/aarch64-linux-gnu/libnvdla_compiler.so" ]; then
    echo "🔧 Aplicando fix de librerías DLA para L4T 36.4.x..."
    mkdir -p dla_fix
    dpkg-deb -x "$DLA_DEB" dla_fix/
    cp -a dla_fix/usr/lib/aarch64-linux-gnu/* /usr/lib/aarch64-linux-gnu/ 2>/dev/null || true
    ldconfig
    rm -rf dla_fix
    echo "✅ Fix DLA aplicado."
fi

# 3. Verificar dependencias de Python (sin romper PyTorch/TensorRT nativos de JetPack)
echo ""
echo "📦 Verificando dependencias de Python..."
python3 -c "import ultralytics" 2>/dev/null || {
    echo "Instalando ultralytics y dependencias ligeras..."
    pip3 install ultralytics --no-deps --index-url https://pypi.org/simple
    pip3 install pyyaml onnx onnxsim matplotlib scipy tqdm pillow psutil --index-url https://pypi.org/simple
}
python3 -c "import tensorrt; print('TensorRT Version:', tensorrt.__version__)"
python3 -c "import torch; print('PyTorch CUDA disponible:', torch.cuda.is_available(), '| Dispositivo:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# 4. Leer configuración del modelo dinámicamente
CONFIG_FILE="configs/yolo11n.yaml"
REBUILD_FLAG=""

for arg in "$@"; do
    if [[ "$arg" == *.yaml ]] || [[ "$arg" == *.yml ]]; then
        CONFIG_FILE="$arg"
    elif [[ "$arg" == "--rebuild" ]]; then
        REBUILD_FLAG="--rebuild"
    fi
done

echo "⚙️  Usando archivo de configuración: $CONFIG_FILE"

# Extraer parámetros con Python
PARSE_CMD="
import yaml
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
print(cfg['export']['onnx']['output_path'])
print(cfg['export']['tensorrt']['output_path'])
print(cfg['export']['tensorrt']['precision'])
print(cfg['model']['weights'])
print(cfg['model']['img_size'])
"
PARSED_VALUES=($(python3 -c "$PARSE_CMD"))
ONNX_MODEL="${PARSED_VALUES[0]}"
ENGINE_MODEL="${PARSED_VALUES[1]}"
PRECISION="${PARSED_VALUES[2]}"
WEIGHTS_PATH="${PARSED_VALUES[3]}"
IMG_SIZE="${PARSED_VALUES[4]:-640}"

mkdir -p models/onnx models/engines models/weights results

# Si no existe el ONNX, exportarlo
if [ ! -f "$ONNX_MODEL" ]; then
    echo ""
    echo "⚙️  ONNX no encontrado en $ONNX_MODEL. Generando..."
    python3 src/compilation/export_onnx.py --weights "$WEIGHTS_PATH" --output "$ONNX_MODEL" --imgsz "$IMG_SIZE" --opset 17
fi

# 5. Compilación del TensorRT Engine nativo para Orin Nano
if [ ! -f "$ENGINE_MODEL" ] || [ "$REBUILD_FLAG" == "--rebuild" ]; then
    echo ""
    echo "========================================================="
    echo "🔨 COMPILANDO TENSORRT ENGINE NATIVO PARA JETSON ($PRECISION)"
    echo "========================================================="
    python3 src/compilation/build_tensorrt.py \
        --onnx "$ONNX_MODEL" \
        --output "$ENGINE_MODEL" \
        --precision "$PRECISION" \
        --workspace 4
else
    echo ""
    echo "ℹ️  Engine TensorRT ya existente en: $ENGINE_MODEL"
    echo "    (Para forzar recompilación, ejecuta con --rebuild)"
fi

# 6. Inferencia y Benchmarking End-to-End con telemetría
echo ""
echo "========================================================="
echo "⏱️  EJECUTANDO BENCHMARK END-TO-END Y TELEMETRÍA"
echo "========================================================="
python3 src/inference/benchmark_jetson.py --config "$CONFIG_FILE" --engine "$ENGINE_MODEL"

echo ""
echo "========================================================="
echo "🎉 BENCHMARK COMPLETADO EXITOSAMENTE"
echo "========================================================="
echo "Resultados registrados en:"
echo " • Detallado JSON:  results/"
echo " • Tabla resumen:   results/benchmark_summary.csv"
