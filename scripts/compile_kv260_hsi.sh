#!/usr/bin/env bash
set -e

WORKSPACE="/home/edwinacevedo/hsi/Misc/ewin/projects/grenoble2"
IMAGE_NAME="vitis-ai-kv260:latest"

echo "======================================================================"
echo "⚡ Launching Vitis AI Container for SS-ResNet INT8 KV260 Compilation"
echo "   Target Architecture: DPUCZDX8G_ISA1_B3136"
echo "   Output: models/xmodel/ss_resnet_indian_kv260.xmodel"
echo "======================================================================"

docker run --rm \
  -v /home/edwinacevedo:/home/edwinacevedo \
  -w ${WORKSPACE} \
  ${IMAGE_NAME} \
  bash -c "source /opt/vitis_ai/conda/etc/profile.d/conda.sh && \
           conda activate vitis-ai-pytorch && \
           python3 src/compilation/build_vitis_ai_hsi.py"
