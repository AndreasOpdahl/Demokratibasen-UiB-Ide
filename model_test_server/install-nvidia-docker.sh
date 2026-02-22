#!/bin/bash
# Install NVIDIA Container Toolkit for GPU support in Docker
# Based on official NVIDIA Container Toolkit installation instructions

set -e

echo "Installing NVIDIA Container Toolkit..."

# Add NVIDIA Container Toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker

echo ""
echo "✓ NVIDIA Container Toolkit installed successfully!"
echo ""
echo "To verify GPU access in Docker, run:"
echo "  docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi"
echo ""
echo "If this works, you can now use --gpus all in your docker run commands."
