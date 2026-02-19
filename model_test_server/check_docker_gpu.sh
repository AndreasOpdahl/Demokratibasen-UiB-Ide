#!/bin/bash
# Check if Docker container has GPU access

echo "="*70
echo "Docker GPU Access Diagnostics"
echo "="*70
echo ""

# Check if we're in a container
if [ -f /.dockerenv ]; then
    echo "✓ Running inside Docker container"
else
    echo "⚠ Not running inside Docker container (or /.dockerenv not found)"
fi

echo ""
echo "Checking GPU access:"
echo ""

# Check nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi is available"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "✗ nvidia-smi not found"
    echo "  This usually means GPU access is not enabled in the container"
fi

echo ""
echo "Checking CUDA libraries:"
if [ -d "/usr/local/cuda" ]; then
    echo "  ✓ /usr/local/cuda exists"
    if [ -f "/usr/local/cuda/lib64/libcudart.so" ]; then
        echo "  ✓ CUDA runtime library found"
    else
        echo "  ✗ CUDA runtime library not found"
    fi
else
    echo "  ✗ /usr/local/cuda not found"
fi

# Check common CUDA library locations
echo ""
echo "CUDA library locations:"
for path in "/usr/local/cuda/lib64" "/usr/lib/x86_64-linux-gnu" "/usr/local/cuda-12.2/lib64" "/usr/local/cuda-12.1/lib64"; do
    if [ -d "$path" ] && [ -f "$path/libcudart.so" ]; then
        echo "  ✓ Found: $path"
        ls -lh "$path/libcudart.so"* 2>/dev/null | head -1
    fi
done

echo ""
echo "Environment variables:"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-Not set}"
echo "  CUDA_HOME: ${CUDA_HOME:-Not set}"
echo "  CUDA_PATH: ${CUDA_PATH:-Not set}"

echo ""
echo "="*70
echo "If GPU is not accessible, ensure container is run with:"
echo "  docker run --gpus all ..."
echo "  or"
echo "  docker run --runtime=nvidia ..."
echo "="*70
