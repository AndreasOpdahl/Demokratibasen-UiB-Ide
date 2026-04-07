# Summarisarion Modl Server

A FastAPI-based HTTP server for summarising public documents using fine-tuned adapters.

## Features

- RESTful API for text summarization
- Supports PEFT (LoRA) fine-tuned models
- Multi-GPU support via model parallelism
- Configurable generation parameters
- Health check endpoint
- CORS enabled for web applications
- Docker-first deployment with GPU support

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your Hugging Face token (if needed by the base model):
```bash
export HUGGINGFACE_TOKEN=your_token_here
```

## Usage

### Basic Usage

```bash
python app.py \
    --adapter_dir ../model_fine_tuning_olivia/models/gemma-2-9b-apptainer-fsdp/checkpoint-5000 \
    --model_name gemma-2-9b \
    --port 8000
```

### With Multiple GPUs

```bash
python app.py \
    --adapter_dir ../model_fine_tuning_olivia/models/gemma-2-9b-apptainer-fsdp/checkpoint-5000 \
    --model_name gemma-2-9b \
    --use_multi_gpu \
    --port 8000
```

### Docker (portable across machines)

```bash
cd model_test_server
cp .env.example .env
# Edit .env and set ADAPTER_DIR, MODEL_NAME (and HUGGINGFACE_TOKEN if required)
docker compose up -d --build
```

Or with the helper script:

```bash
cd model_test_server
export ADAPTER_DIR="$HOME/OneDrive/Shared/Demokratibasen-UiB-Ide/TrainingRuns/olivia/winners/checkpoint-5000"
export MODEL_NAME="gemma-2-9b"
./build_and_run.sh
```

### Command Line Arguments

- `--adapter_dir`: Path to the adapter directory (required)
- `--model_name`: Model name (default: `gemma-2-9b`)
- `--hf_token`: Hugging Face token (or set `HUGGINGFACE_TOKEN` env var)
- `--port`: Port to run the server on (default: 8000)
- `--host`: Host to bind to (default: `0.0.0.0`)
- `--use_multi_gpu`: Use multiple GPUs if available

## API Endpoints

### GET `/`

Root endpoint with server status.

**Response:**
```json
{
  "message": "Model Summary Server",
  "status": "ready",
  "model": "gemma-2-9b"
}
```

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### POST `/summarize`

Generate a summary for the given text.

**Request Body:**
```json
{
  "text": "Your text to summarize here...",
  "doc_type": "tekst",
  "max_length": 512,
  "min_length": 50,
  "temperature": 0.7,
  "top_p": 0.9,
  "do_sample": true
}
```

**Response:**
```json
{
  "summary": "Generated summary text...",
  "processing_time": 2.34,
  "model_name": "gemma-2-9b",
  "adapter_dir": "checkpoint-5000"
}
```

**Parameters:**
- `text` (required): The text to summarize
- `doc_type` (optional): Document type for prompt formatting (default: "tekst")
- `max_length` (optional): Maximum tokens for summary (default: 512)
- `min_length` (optional): Minimum tokens for summary (default: 50)
- `temperature` (optional): Sampling temperature (default: 0.7)
- `top_p` (optional): Nucleus sampling parameter (default: 0.9)
- `do_sample` (optional): Whether to use sampling (default: true)

## Example Usage

### Using curl

```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your long text to summarize here...",
    "doc_type": "vedtak",
    "max_length": 256
  }'
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/summarize",
    json={
        "text": "Your long text to summarize here...",
        "doc_type": "vedtak",
        "max_length": 256
    }
)

result = response.json()
print(result["summary"])
print(f"Processing time: {result['processing_time']:.2f}s")
```

### Using JavaScript/TypeScript

```javascript
const response = await fetch('http://localhost:8000/summarize', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'Your long text to summarize here...',
    doc_type: 'vedtak',
    max_length: 256
  })
});

const result = await response.json();
console.log(result.summary);
console.log(`Processing time: ${result.processing_time}s`);
```

## Document Types

The server supports different document types that affect the prompt formatting:

- `tekst` (default)
- `vedtak`
- `saksforelegg`
- `møtereferat`
- `saksliste`

## Notes

- The server loads the model on startup, which may take a few minutes
- Model is kept in memory for fast inference
- For production use, consider using a process manager like `systemd` or `supervisor`
- For high-traffic scenarios, consider using a reverse proxy like `nginx` in front of the server

## Troubleshooting

### GPU not detected in Docker container

If you're running in a Docker container and GPU is not detected:

1. **Ensure container is run with GPU access:**
   ```bash
   # Docker with --gpus flag (recommended)
   docker run --gpus all -p 8000:8000 your-image
   
   # Or with docker-compose, add to your docker-compose.yml:
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all
             capabilities: [gpu]
   ```

2. **Check GPU access inside container:**
   ```bash
   # Inside the container
   ./check_docker_gpu.sh
   python check_cuda.py
   ```

3. **Set LD_LIBRARY_PATH if needed:**
   ```bash
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

4. **Verify nvidia-container-runtime is installed on host:**
   ```bash
   # On the host (not in container)
   docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
   ```

### Model not loading

- Check that the adapter directory path is correct
- Verify that the adapter directory contains `adapter_config.json` and `adapter_model.safetensors` (or `adapter_model.bin`)
- Ensure you have sufficient GPU memory
- Check that the Hugging Face token is set correctly

### Out of memory errors

- Reduce `max_length` parameter
- Use CPU instead of GPU (remove `--use_multi_gpu`)
- Use a smaller model or adapter

### Slow inference

- Ensure you're using GPU (check with `nvidia-smi`)
- Reduce `max_length` parameter
- Use `do_sample=false` for faster greedy decoding
