#!/usr/bin/env python3
"""
Quick script to check CUDA availability and PyTorch installation.
"""

import sys
import os
import subprocess

print("="*70)
print("CUDA/GPU Diagnostics")
print("="*70)

# Check environment variables
print("\nEnvironment Variables:")
print(f"  CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"  CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
print(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')[:100]}...")
print(f"  PATH: {os.environ.get('PATH', 'Not set')[:100]}...")

# Check nvidia-smi
print("\nSystem CUDA Check:")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("  ✓ nvidia-smi works")
        # Extract CUDA version from nvidia-smi output
        for line in result.stdout.split('\n'):
            if 'CUDA Version' in line:
                print(f"  {line.strip()}")
    else:
        print("  ✗ nvidia-smi failed")
except FileNotFoundError:
    print("  ✗ nvidia-smi not found")
except Exception as e:
    print(f"  ✗ Error running nvidia-smi: {e}")

# Check CUDA libraries
print("\nCUDA Library Check:")
cuda_libs = ['libcudart.so', 'libcublas.so', 'libcudnn.so']
for lib in cuda_libs:
    try:
        result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True, timeout=5)
        if lib in result.stdout:
            print(f"  ✓ {lib} found")
        else:
            print(f"  ✗ {lib} not found in ldconfig")
    except:
        pass

print("\n" + "="*70)

try:
    import torch
    print(f"✓ PyTorch installed: {torch.__version__}")
    
    print(f"\nCUDA Availability:")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  ✓ CUDA is available!")
        print(f"  CUDA version (PyTorch): {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n  GPU {i}:")
            print(f"    Name: {props.name}")
            print(f"    Compute Capability: {props.major}.{props.minor}")
            print(f"    Total Memory: {props.total_memory / 1e9:.2f} GB")
            
            # Test memory allocation
            try:
                test_tensor = torch.zeros(1).cuda(i)
                print(f"    ✓ Can allocate memory on GPU {i}")
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    ✗ Cannot allocate memory: {e}")
    else:
        print(f"  ✗ CUDA is NOT available")
        print(f"\n  PyTorch version: {torch.__version__}")
        if '+cu' in torch.__version__:
            print(f"  ⚠ PyTorch was built with CUDA support, but can't detect GPU!")
            print(f"\n  This usually means:")
            print(f"    1. CUDA runtime libraries not in LD_LIBRARY_PATH")
            print(f"    2. CUDA version mismatch (PyTorch built for different CUDA version)")
            print(f"    3. Missing CUDA runtime libraries on system")
            print(f"\n  Solutions to try:")
            print(f"    1. Set LD_LIBRARY_PATH (run: ./fix_cuda.sh for help):")
            print(f"       export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH")
            print(f"       # Or try:")
            print(f"       export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH")
            print(f"    2. Reinstall PyTorch with CUDA 12.1 (closer to system CUDA 12.2):")
            print(f"       pip uninstall torch torchvision torchaudio -y")
            print(f"       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print(f"    3. If in container, ensure GPU access:")
            print(f"       - Docker: --gpus all")
            print(f"       - Apptainer: --nv flag")
        else:
            print(f"  ✗ PyTorch was installed without CUDA support (CPU-only)")
            print(f"\n  To fix:")
            print(f"    1. Reinstall PyTorch with CUDA:")
            print(f"       pip uninstall torch torchvision torchaudio -y")
            print(f"       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
        # Try to get more debug info
        try:
            if hasattr(torch.version, 'cuda') and torch.version.cuda:
                print(f"\n  Debug: PyTorch CUDA version: {torch.version.cuda}")
            # Try to initialize CUDA
            try:
                torch.cuda.init()
                print(f"  Debug: After init: {torch.cuda.is_available()}")
            except Exception as e:
                print(f"  Debug: CUDA init error: {e}")
        except:
            pass
        
except ImportError:
    print("✗ PyTorch is not installed")
    print("  Install with: pip install torch")
    sys.exit(1)

print("\n" + "="*70)
