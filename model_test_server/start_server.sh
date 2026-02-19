#!/bin/bash
# Simple startup script for the model summary server

# Default values
CHECKPOINT_PATH=""
MODEL_NAME="gemma-2-9b"
PORT=8000
HOST="0.0.0.0"
USE_MULTI_GPU=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --use_multi_gpu)
            USE_MULTI_GPU=true
            shift
            ;;
        --help)
            echo "Usage: $0 --checkpoint_path PATH [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint_path PATH    Path to checkpoint directory (required)"
            echo "  --model_name NAME         Model name (default: gemma-2-9b)"
            echo "  --port PORT               Port to run server on (default: 8000)"
            echo "  --host HOST               Host to bind to (default: 0.0.0.0)"
            echo "  --use_multi_gpu           Use multiple GPUs if available"
            echo "  --help                    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: --checkpoint_path is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check if checkpoint path exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint path does not exist: $CHECKPOINT_PATH"
    exit 1
fi

# Build command
CMD="python app.py --checkpoint_path \"$CHECKPOINT_PATH\" --model_name \"$MODEL_NAME\" --port $PORT --host \"$HOST\""

if [ "$USE_MULTI_GPU" = true ]; then
    CMD="$CMD --use_multi_gpu"
fi

# Run the server
echo "Starting model summary server..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Model: $MODEL_NAME"
echo "Server: $HOST:$PORT"
echo ""
exec $CMD
