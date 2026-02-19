#!/bin/bash
# Build and run the model server Docker container

set -e

# Configuration
IMAGE_NAME="${IMAGE_NAME:-test-checkpoint}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-../checkpoint-5700}"
MODEL_NAME="${MODEL_NAME:-gemma-2-9b}"
PORT="${PORT:-8000}"

echo "="*70
echo "Building and Running Model Server Docker Container"
echo "="*70
echo ""

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint path does not exist: $CHECKPOINT_PATH"
    echo "Please set CHECKPOINT_PATH environment variable or update the script"
    exit 1
fi

# Check if HUGGINGFACE_TOKEN is set
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Warning: HUGGINGFACE_TOKEN not set"
    echo "Set it with: export HUGGINGFACE_TOKEN=your_token"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build the image
echo "Step 1: Building Docker image '$IMAGE_NAME'..."
docker build -t "$IMAGE_NAME" .

if [ $? -ne 0 ]; then
    echo "Error: Docker build failed"
    exit 1
fi

echo ""
echo "✓ Image built successfully!"
echo ""

# Run the container
echo "Step 2: Running container..."
echo "  Image: $IMAGE_NAME"
echo "  Checkpoint: $CHECKPOINT_PATH -> /app/checkpoint"
echo "  Model: $MODEL_NAME"
echo "  Port: $PORT"
echo ""

docker run --gpus all \
  -p "$PORT:8000" \
  -v "$(pwd)/$CHECKPOINT_PATH:/app/checkpoint:ro" \
  -e HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
  "$IMAGE_NAME" \
  python app.py --checkpoint_path /app/checkpoint --model_name "$MODEL_NAME" --port 8000
