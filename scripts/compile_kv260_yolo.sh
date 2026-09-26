#!/usr/bin/env bash
set -e

# Default arguments
WEIGHTS=${1:-"models/weights/yolo11m.pt"}
MODEL_NAME=${2:-"yolo11m_kv260"}
SAMPLES=${3:-64}

WORKSPACE="/home/edwinacevedo/hsi/Misc/ewin/projects/grenoble2"
IMAGE_NAME="vitis-ai-kv260:latest"

echo "======================================================================"
echo "⚡ Launching Vitis AI Container for Kria KV260 INT8 Compilation"
echo "   Weights: ${WEIGHTS}"
echo "   Output Name: ${MODEL_NAME}"
echo "   Calibration Samples: ${SAMPLES}"
echo "======================================================================"

sg docker -c "docker run --rm \
  -v /home/edwinacevedo:/home/edwinacevedo \
  -w ${WORKSPACE} \
  ${IMAGE_NAME} \
  bash -c 'source /opt/vitis_ai/conda/etc/profile.d/conda.sh && \
           conda activate vitis-ai-pytorch && \
           python3 src/compilation/build_vitis_ai_kv260.py \
             --weights ${WEIGHTS} \
             --name ${MODEL_NAME} \
             --samples ${SAMPLES}'"
