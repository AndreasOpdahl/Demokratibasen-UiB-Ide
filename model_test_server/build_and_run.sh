#!/usr/bin/env bash
# Build and run the model server Docker container with GPU.

set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-model-test-server:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-model-test-server}"
ADAPTER_DIR="${ADAPTER_DIR:-}"
MODEL_NAME="${MODEL_NAME:-gemma-2-9b}"
PORT="${PORT:-8000}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$PWD/cache/huggingface}"
ENABLE_MULTI_GPU="${ENABLE_MULTI_GPU:-true}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-false}"

if [[ -z "${ADAPTER_DIR}" ]]; then
  echo "Error: ADAPTER_DIR is required."
  echo "Example:"
  echo "  ADAPTER_DIR=\"$HOME/OneDrive/Shared/Demokratibasen-UiB-Ide/TrainingRuns/olivia/winners/checkpoint-5000\" ./build_and_run.sh"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: Docker is not installed or not in PATH."
  exit 1
fi

ADAPTER_DIR_ABS="$(realpath "${ADAPTER_DIR}")"
if [[ ! -d "${ADAPTER_DIR_ABS}" ]]; then
  echo "Error: ADAPTER_DIR does not exist: ${ADAPTER_DIR_ABS}"
  exit 1
fi

if [[ ! -f "${ADAPTER_DIR_ABS}/adapter_config.json" ]]; then
  echo "Error: adapter_config.json is missing in ${ADAPTER_DIR_ABS}"
  exit 1
fi

if [[ ! -f "${ADAPTER_DIR_ABS}/adapter_model.safetensors" && ! -f "${ADAPTER_DIR_ABS}/adapter_model.bin" ]]; then
  echo "Error: Missing adapter weights in ${ADAPTER_DIR_ABS}"
  echo "Expected one of: adapter_model.safetensors or adapter_model.bin"
  exit 1
fi

mkdir -p "${HF_CACHE_DIR}" "$PWD/logs"

if [[ "${SKIP_GPU_CHECK}" != "true" ]]; then
  echo "Checking Docker GPU runtime..."
  docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi >/dev/null
fi

echo "Building image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" .

if docker ps -a --format "{{.Names}}" | rg -x "${CONTAINER_NAME}" >/dev/null; then
  echo "Removing existing container: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

EXTRA_ARGS=()
if [[ "${ENABLE_MULTI_GPU}" == "true" ]]; then
  EXTRA_ARGS+=(--use_multi_gpu)
fi

echo "Starting container: ${CONTAINER_NAME}"
echo "  Adapter: ${ADAPTER_DIR_ABS}"
echo "  Model:   ${MODEL_NAME}"
echo "  Port:    ${PORT}"

docker run --gpus all \
  --name "${CONTAINER_NAME}" \
  -p "${PORT}:8000" \
  -v "${ADAPTER_DIR_ABS}:/app/checkpoint:ro" \
  -v "${HF_CACHE_DIR}:/cache/huggingface" \
  -v "$PWD/logs:/app/logs" \
  -e HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}" \
  -e HF_HOME="/cache/huggingface" \
  -e TRANSFORMERS_CACHE="/cache/huggingface/transformers" \
  "${IMAGE_NAME}" \
  python app.py --checkpoint_path /app/checkpoint --model_name "${MODEL_NAME}" --port 8000 "${EXTRA_ARGS[@]}"
