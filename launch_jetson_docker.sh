#!/usr/bin/env bash
# Script para ejecutar en el HOST de la NVIDIA Jetson Orin Nano
# Lanza el contenedor Docker con acceso a GPU, memoria compartida y sensores de potencia

set -e

# Imagen por defecto si no se pasa como primer argumento o variable de entorno
DEFAULT_IMAGE="dustynv/l4t-pytorch:r36.4.0"
CONTAINER_IMAGE="${DOCKER_IMAGE:-$DEFAULT_IMAGE}"

if [ -n "$1" ] && [[ "$1" != --* ]]; then
    CONTAINER_IMAGE="$1"
    shift
fi

echo "========================================================="
echo "🐳 LANZANDO CONTENEDOR DOCKER EN JETSON ORIN NANO"
echo "========================================================="
echo "Imagen objetivo: $CONTAINER_IMAGE"
echo "Montando workspace: $(pwd) -> /workspace"
echo ""

docker run --runtime nvidia --gpus all -it --rm \
    --network host \
    --ipc=host \
    -v /sys/devices:/sys/devices:ro \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v "$(pwd)":/workspace \
    -w /workspace \
    "$CONTAINER_IMAGE" \
    /bin/bash /workspace/run_jetson_docker.sh "$@"
