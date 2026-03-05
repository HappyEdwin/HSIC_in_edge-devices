cd /workspace/models/TGRS_2025_MCTGCL/

#echo "Running Test..."
#python3 test.py

#echo ""
#echo "========================================================="
#echo "Generating TensorRT Engines"
#echo "========================================================="
#python3 generate_engines.py


echo ""
echo "========================================================="
echo "Running Profiling"
echo "========================================================="
#python3 profile_mctgcl.py

python3 plot.py
