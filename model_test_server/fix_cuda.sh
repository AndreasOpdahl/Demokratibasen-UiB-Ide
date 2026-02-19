#!/bin/bash
# Script to help fix CUDA detection issues

echo "="*70
echo "CUDA Fix Script"
echo "="*70
echo ""

# Check if we're in a virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠ No virtual environment detected"
fi

echo ""
echo "Step 1: Checking LD_LIBRARY_PATH..."
if [ -z "$LD_LIBRARY_PATH" ]; then
    echo "  LD_LIBRARY_PATH is not set"
    echo "  Trying to find CUDA libraries..."
    
    # Common CUDA library locations
    CUDA_PATHS=(
        "/usr/local/cuda/lib64"
        "/usr/lib/x86_64-linux-gnu"
        "/usr/local/cuda-12.2/lib64"
        "/usr/local/cuda-12.1/lib64"
        "/usr/local/cuda-11.8/lib64"
    )
    
    for path in "${CUDA_PATHS[@]}"; do
        if [ -d "$path" ] && [ -f "$path/libcudart.so" ]; then
            echo "  ✓ Found CUDA libraries in: $path"
            echo ""
            echo "  Add this to your environment:"
            echo "    export LD_LIBRARY_PATH=$path:\$LD_LIBRARY_PATH"
            echo ""
            echo "  Or add to ~/.bashrc:"
            echo "    echo 'export LD_LIBRARY_PATH=$path:\$LD_LIBRARY_PATH' >> ~/.bashrc"
            break
        fi
    done
else
    echo "  LD_LIBRARY_PATH is set: $LD_LIBRARY_PATH"
fi

echo ""
echo "Step 2: Checking PyTorch installation..."
python3 -c "import torch; print(f'  PyTorch version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')" 2>&1

echo ""
echo "Step 3: Recommendations..."
echo ""
echo "If CUDA is still not available, try:"
echo ""
echo "1. Set LD_LIBRARY_PATH and retry:"
echo "   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
echo "   python check_cuda.py"
echo ""
echo "2. Reinstall PyTorch with CUDA 12.1 (closer match to system CUDA 12.2):"
echo "   pip uninstall torch torchvision torchaudio -y"
echo "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
echo ""
echo "3. If in a container, ensure GPU access is enabled:"
echo "   - Docker: Use --gpus all"
echo "   - Apptainer/Singularity: Use --nv flag"
echo ""
echo "="*70
