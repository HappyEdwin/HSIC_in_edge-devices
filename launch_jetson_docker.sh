#!/usr/bin/env bash
# Script para ejecutar en el HOST de la NVIDIA Jetson Orin Nano
# Lanza el contenedor Docker con acceso a GPU, memoria compartida y sensores de potencia

set -e

DEFAULT_IMAGE="dustynv/l4t-pytorch:r36.4.0"
CONTAINER_IMAGE="${DOCKER_IMAGE:-$DEFAULT_IMAGE}"

if [ -n "$1" ] && [[ "$1" != --* ]] && [[ "$1" != *.yaml ]] && [[ "$1" != *.yml ]] && [[ "$1" != "bash" ]] && [[ "$1" != "sh" ]] && [[ "$1" != "hsi" ]]; then
    CONTAINER_IMAGE="$1"
    shift
fi

echo "========================================================="
echo "🐳 LANZANDO CONTENEDOR DOCKER EN JETSON ORIN NANO"
echo "========================================================="
echo "Imagen objetivo: $CONTAINER_IMAGE"
echo "Montando workspace: $(pwd) -> /workspace"
EXTRA_MOUNTS=""
if [ -f "/usr/bin/tegrastats" ]; then
    EXTRA_MOUNTS="-v /usr/bin/tegrastats:/usr/bin/tegrastats:ro"
fi

# Soporte para abrir bash interactivo o ejecutar run_jetson_docker.sh
if [ "$1" == "bash" ] || [ "$1" == "sh" ]; then
    RUN_CMD="/bin/bash"
else
    RUN_CMD="/bin/bash /workspace/run_jetson_docker.sh $@"
fi

docker run --runtime nvidia --gpus all --privileged -it --rm \
    --network host \
    --ipc=host \
    -v /sys:/sys:ro \
    -v /dev:/dev \
    $EXTRA_MOUNTS \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v "$(pwd)":/workspace \
    -w /workspace \
    "$CONTAINER_IMAGE" \
    $RUN_CMD
