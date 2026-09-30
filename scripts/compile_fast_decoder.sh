#!/usr/bin/env bash
# Compila la librería de posprocesamiento C++ acelerada para YOLOv4-tiny
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CPP_SRC="$ROOT_DIR/src/evaluation/yolo_fast_decoder.cpp"
SO_OUT="$ROOT_DIR/src/evaluation/libyolo_fast.so"

echo "⚡ Compilando acelerador C++: $CPP_SRC -> $SO_OUT"
g++ -O3 -shared -fPIC -std=c++17 -static-libstdc++ -static-libgcc "$CPP_SRC" -o "$SO_OUT"

echo "✅ Compilación exitosa: $(ls -lh $SO_OUT)"
