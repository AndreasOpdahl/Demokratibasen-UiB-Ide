# Running Model Server in Docker

## Quick Start

### Using Docker Run

```bash
# Step 1: Build the image (name it whatever you want)
docker build -t test-checkpoint .

# Step 2: Run with GPU access
docker run --gpus all \
  -p 8000:8000 \
  -v $(pwd)/../checkpoint-5700:/app/checkpoint:ro \
  -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \
  test-checkpoint \
  python app.py --checkpoint_path /app/checkpoint --model_name gemma-2-9b --port 8000
```

**Note:** Make sure to:
- Build the image first with `docker build -t test-checkpoint .`
- Use the correct checkpoint path (mounted at `/app/checkpoint` in container)
- Set `HUGGINGFACE_TOKEN` environment variable before running

### Using Docker Compose

```bash
# Set your Hugging Face token
export HUGGINGFACE_TOKEN=your_token_here

# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
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

2. **Verify GPU access:**
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

## Building the Image

```bash
docker build -t model-server .
```

## Environment Variables

- `HUGGINGFACE_TOKEN`: Your Hugging Face token (required for private models)

## Volume Mounts

- Checkpoint directory: Mount your checkpoint to `/app/checkpoint`
- Logs: Optional, mount a logs directory to `/app/logs`

## Ports

- `8000`: HTTP API port (map to host port as needed)

## Example docker-compose.yml

See `docker-compose.yml` in this directory for a complete example.

## Notes

- The container uses Python 3.11 slim base image
- All dependencies are installed from `requirements.txt`
- GPU access requires `--gpus all` flag or docker-compose GPU configuration
- Model loading can take several minutes on first start
