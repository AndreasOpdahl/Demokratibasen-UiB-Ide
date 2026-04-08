# Running Model Server in Docker

## Goal

Run the same Docker image on different machines with GPU acceleration by only changing:
- where the adapter directory is mounted from
- optional Hugging Face token

The adapter directory must contain:
- `adapter_config.json`
- `adapter_model.safetensors` (or `adapter_model.bin`)

## Quick Start (Recommended)

### Option A: Single command script

```bash
cd model_test_server

export ADAPTER_DIR="$HOME/OneDrive/Shared/Demokratibasen-UiB-Ide/TrainingRuns/olivia/winners/checkpoint-5000"
export MODEL_NAME="gemma-2-9b"
export HUGGINGFACE_TOKEN="your_token_if_needed"

./build_and_run.sh
```

### Option B: Docker Compose

```bash
cd model_test_server
cp .env.example .env
# Edit .env and set ADAPTER_DIR (+ optional HUGGINGFACE_TOKEN)

docker compose up -d --build
docker compose logs -f
docker compose down
```

## GPU Access

### Prerequisites

1. **Install Docker (if not already installed):**
   ```bash
   ./install_docker.sh
   ```

2. **Install NVIDIA Container Toolkit for GPU support:**
   ```bash
   ./install-nvidia-docker.sh
   ```
   
   Or manually:
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

3. **Verify GPU access:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
   ```

### Troubleshooting GPU Access

If GPU is not detected inside the container:

1. **Check if container has GPU access:**
   ```bash
   # Inside container
   ./check_docker_gpu.sh
   python check_cuda.py
   ```

2. **Verify Docker GPU runtime:**
   ```bash
   # On host
   docker info | grep -i runtime
   # Should show: nvidia
   ```

3. **Check container is run with --gpus:**
   ```bash
   # Verify your docker run command includes --gpus all
   docker ps --format "table {{.Names}}\t{{.Command}}"
   ```

4. **Set LD_LIBRARY_PATH if needed:**
   ```bash
   # In Dockerfile or docker-compose.yml
   ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

## Building and pushing for reuse on other machines

```bash
# Example with Docker Hub (replace with your registry/tag)
docker build -t your-user/demokratibasen-oppsummering:latest .
docker push your-user/demokratibasen-oppsummering:latest
```

On another machine:
```bash
docker pull your-user/demokratibasen-oppsummering:latest
docker run --gpus all \
  -p 8000:8000 \
  -v "$ADAPTER_DIR:/app/adapter:ro" \
  -v "$PWD/cache/huggingface:/cache/huggingface" \
  -e HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
  -e HF_HOME=/cache/huggingface \
  your-user/demokratibasen-oppsummering:latest \
  python app.py --adapter_dir /app/adapter --model_name gemma-2-9b --port 8000 --use_multi_gpu
```

## Environment Variables

- `HUGGINGFACE_TOKEN`: Your Hugging Face token (required for private models)

## Volume Mounts

- Adapter directory: mount to `/app/adapter` (read-only recommended)
- Hugging Face cache: mount to `/cache/huggingface` for faster restarts
- Logs (optional): mount to `/app/logs`

## Ports

- `8000`: HTTP API port (map to host port as needed)

## Example docker-compose.yml

See `docker-compose.yml` in this directory. It uses:
- `gpus: all`
- env-driven adapter path (`ADAPTER_DIR`)
- env-driven model name (`MODEL_NAME`)

## Notes

- The container uses Python 3.11 slim base image
- All dependencies are installed from `requirements.txt`
- GPU access requires `--gpus all` flag or docker-compose GPU configuration
- Model loading can take several minutes on first start
