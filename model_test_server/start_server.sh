#!/bin/bash
# Simple startup script for the model summary server

# Default values
ADAPTER_DIR=""
MODEL_NAME="gemma-2-9b"
PORT=8000
HOST="0.0.0.0"
USE_MULTI_GPU=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --adapter_dir)
            ADAPTER_DIR="$2"
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
            echo "Usage: $0 --adapter_dir PATH [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --adapter_dir PATH        Path to adapter directory (required)"
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
if [ -z "$ADAPTER_DIR" ]; then
    echo "Error: --adapter_dir is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check if adapter directory exists
if [ ! -d "$ADAPTER_DIR" ]; then
    echo "Error: Adapter directory does not exist: $ADAPTER_DIR"
    exit 1
fi

# Build command
CMD="python app.py --adapter_dir \"$ADAPTER_DIR\" --model_name \"$MODEL_NAME\" --port $PORT --host \"$HOST\""

if [ "$USE_MULTI_GPU" = true ]; then
    CMD="$CMD --use_multi_gpu"
fi

# Run the server
echo "Starting model summary server..."
echo "Adapter directory: $ADAPTER_DIR"
echo "Model: $MODEL_NAME"
echo "Server: $HOST:$PORT"
echo ""
exec $CMD
