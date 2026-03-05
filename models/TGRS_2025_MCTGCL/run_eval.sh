#!/bin/bash

echo ""
echo "========================================================="
echo "Starting Environment Setup in Docker"
echo "========================================================="
echo ""

cd /workspace/models/TGRS_2025_MCTGCL/

echo "Extracting and moving DLA compiler library..."
dpkg-deb -x nvidia-l4t-dla-compiler_36.4.3-20250107174145_arm64.deb dla_fix/ 
cp -a dla_fix/usr/lib/aarch64-linux-gnu/* /usr/lib/aarch64-linux-gnu/ 

echo "Running ldconfig..."
ldconfig

echo "Installing required Python libraries..."
pip3 install scikit-learn einops scipy matplotlib onnx onnxsim tqdm --index-url https://pypi.org/simple

#echo "Running Test..."
#python3 test.py

#echo ""
#echo "========================================================="
#echo "Generating TensorRT Engines"
#echo "========================================================="
#python3 generate_engines.py

#echo ""
#echo "========================================================="
#echo "Running Profiling"
#echo "========================================================="
#python3 profile_mctgcl.py