"""
FastAPI server for serving model summaries.

Usage:
    python app.py --adapter_dir path/to/adapter --port 8000
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn

# Import model configs from local file
from model_configs import get_model_config, PROMPT_PLAIN

app = FastAPI(title="Model Summary Server", version="1.0.0")

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
model = None
tokenizer = None
model_config = None
device = None
adapter_dir = None


def normalize_hf_token(raw_token: Optional[str]) -> Optional[str]:
    """Normalize HF token values so empty strings are treated as missing."""
    if raw_token is None:
        return None
    token = raw_token.strip()
    return token if token else None


def validate_adapter_dir(path: str) -> None:
    """Validate that adapter directory contains PEFT adapter artifacts."""
    adapter_path = Path(path).expanduser().resolve()
    if not adapter_path.exists() or not adapter_path.is_dir():
        raise ValueError(f"Adapter directory does not exist or is not a directory: {adapter_path}")

    config_file = adapter_path / "adapter_config.json"
    safetensors_file = adapter_path / "adapter_model.safetensors"
    bin_file = adapter_path / "adapter_model.bin"

    if not config_file.exists():
        raise ValueError(
            f"Missing adapter_config.json in adapter directory: {adapter_path}"
        )

    if not safetensors_file.exists() and not bin_file.exists():
        raise ValueError(
            "Missing adapter weights. Expected one of "
            f"{safetensors_file.name} or {bin_file.name} in {adapter_path}"
        )


class SummaryRequest(BaseModel):
    """Request model for summary generation."""
    text: str
    doc_type: Optional[str] = "tekst"  # Default document type
    max_length: Optional[int] = 150  # Maximum tokens for summary (reduced for speed)
    min_length: Optional[int] = 20  # Minimum tokens for summary (reduced for speed)
    temperature: Optional[float] = 0.3  # Lower default for faster generation
    top_p: Optional[float] = 0.9
    do_sample: Optional[bool] = False  # Default to greedy for speed


class SummaryResponse(BaseModel):
    """Response model for summary generation."""
    summary: str
    processing_time: float
    model_name: str
    adapter_dir: str


def load_model(adapter_dir: str, model_name: str = "gemma-2-9b", hf_token: Optional[str] = None, use_multi_gpu: bool = False):
    """Load the base model and PEFT adapter."""
    global model, tokenizer, model_config, device
    
    hf_token = normalize_hf_token(hf_token)

    print(f"Loading model: {model_name}")
    print(f"Adapter directory: {adapter_dir}")
    validate_adapter_dir(adapter_dir)
    
    # Get model configuration
    try:
        model_config = get_model_config(model_name)
        print(f"Model config loaded: {model_config.short_name}")
    except KeyError:
        print(f"Warning: Model config not found for {model_name}, using defaults")
        model_config = None
    
    # Determine device with detailed diagnostics
    print("\n" + "="*70)
    print("CUDA/GPU Diagnostics:")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device = "cuda"
        num_gpus = torch.cuda.device_count()
        print(f"✓ CUDA available: {num_gpus} GPU(s)")
        print(f"CUDA version (PyTorch): {torch.version.cuda}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Compute Capability: {props.major}.{props.minor}")
            print(f"    Total Memory: {props.total_memory / 1e9:.2f} GB")
    else:
        device = "cpu"
        print("✗ CUDA not available in PyTorch")
        print("\nTroubleshooting:")
        print("  1. Check if PyTorch was installed with CUDA support:")
        print("     Run: python -c 'import torch; print(torch.cuda.is_available())'")
        print("  2. Verify PyTorch CUDA version matches system CUDA:")
        print("     PyTorch CUDA: Run: python -c 'import torch; print(torch.version.cuda)'")
        print("     System CUDA: Check nvidia-smi output")
        print("  3. Reinstall PyTorch with CUDA support:")
        print("     For CUDA 12.2: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("     For CUDA 11.8: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("\n⚠ WARNING: Model will run on CPU (will be VERY slow!)")
    print("="*70 + "\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.hf_name if model_config else f"google/{model_name}",
        token=hf_token,
        trust_remote_code=True,
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print("Loading base model...")
    if use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using model parallelism across {torch.cuda.device_count()} GPUs")
        model = AutoModelForCausalLM.from_pretrained(
            model_config.hf_name if model_config else f"google/{model_name}",
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        # Single GPU or CPU
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_config.hf_name if model_config else f"google/{model_name}",
                torch_dtype=torch.float16,
                device_map="cuda:0",
                token=hf_token,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_config.hf_name if model_config else f"google/{model_name}",
                torch_dtype=torch.float32,  # CPU typically uses float32
                token=hf_token,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            model = model.to(device)
    
    # Load PEFT adapter
    print(f"Loading PEFT adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Print model info and verify GPU usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i) / 1e9
            memory_reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")
            print(f"  Memory allocated: {memory_allocated:.2f} GB")
            print(f"  Memory reserved: {memory_reserved:.2f} GB")
        
        # Verify model is on GPU
        try:
            first_param = next(model.parameters())
            if first_param.device.type == 'cuda':
                print(f"✓ Model is on GPU: {first_param.device}")
            else:
                print(f"⚠ Warning: Model is on {first_param.device}, not GPU!")
        except Exception as e:
            print(f"⚠ Could not verify model device: {e}")
    else:
        print("⚠ CUDA not available - model is running on CPU (will be very slow!)")


def format_prompt(input_text: str, doc_type: str = "tekst") -> str:
    """Format the input text with the appropriate prompt template.
    
    Uses the same eval_template as during fine-tuning evaluation to ensure consistency.
    This matches the format used in evaluate_distributed_checkpoints_multigpu.py
    and wandb_finetune.py via model_config.prompt_config.eval_template.
    """
    if model_config and model_config.prompt_config:
        # Use the same eval_template that was used during fine-tuning
        prompt_template = model_config.prompt_config.eval_template
        return prompt_template.format(input=input_text, doc_type=doc_type)
    else:
        # Default plain prompt (same as PROMPT_PLAIN.eval_template)
        return PROMPT_PLAIN.eval_template.format(input=input_text, doc_type=doc_type)


def generate_summary(
    text: str,
    doc_type: str = "tekst",
    max_length: int = 512,
    min_length: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Generate a summary for the given text."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Format prompt
    prompt = format_prompt(text, doc_type)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    # Move to device
    # Check if model uses device_map (model parallelism)
    if hasattr(model, 'hf_device_map') or (hasattr(model, 'base_model') and hasattr(model.base_model, 'hf_device_map')):
        # Model is already on device via device_map - find the first device
        first_param = next(model.parameters())
        inputs = {k: v.to(first_param.device) for k, v in inputs.items()}
    else:
        # Model is on a single device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate with optimized settings for speed
    with torch.no_grad():
        # Use greedy decoding (faster) if do_sample is False or temperature is very low
        use_greedy = not do_sample or temperature < 0.1
        
        generation_kwargs = {
            **inputs,
            'max_new_tokens': max_length,
            'min_new_tokens': min(min_length, 10),  # Cap min_new_tokens at 10 for speed
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'repetition_penalty': 1.1,
            'use_cache': True,  # Enable KV cache for speed
        }
        
        if use_greedy:
            generation_kwargs['do_sample'] = False
            generation_kwargs['num_beams'] = 1
        else:
            generation_kwargs['do_sample'] = True
            generation_kwargs['temperature'] = temperature
            generation_kwargs['top_p'] = top_p
        
        outputs = model.generate(**generation_kwargs)
    
    # Decode
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_length:]
    summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Clean up summary
    summary = summary.replace('[/INST]', '').replace('[INST]', '')
    summary = summary.replace('</s>', '').replace('<s>', '')
    summary = summary.replace('\\', '')
    summary = ' '.join(summary.split())
    summary = summary.strip()
    
    return summary


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Model Summary Server",
        "status": "ready" if model is not None else "not_loaded",
        "model": model_config.short_name if model_config else "unknown",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "cuda_available": True,
            "gpu_count": torch.cuda.device_count(),
            "gpus": []
        }
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i) / 1e9  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / 1e9  # GB
            memory_total = props.total_memory / 1e9  # GB
            gpu_info["gpus"].append({
                "index": i,
                "name": props.name,
                "memory_allocated_gb": round(memory_allocated, 2),
                "memory_reserved_gb": round(memory_reserved, 2),
                "memory_total_gb": round(memory_total, 2),
                "memory_usage_percent": round((memory_reserved / memory_total) * 100, 1) if memory_total > 0 else 0
            })
    else:
        gpu_info = {"cuda_available": False}
    
    # Check if model is on GPU
    model_on_gpu = False
    if model is not None:
        try:
            first_param = next(model.parameters())
            model_on_gpu = first_param.device.type == 'cuda'
            if model_on_gpu:
                gpu_info["model_device"] = str(first_param.device)
        except:
            pass
    
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_on_gpu": model_on_gpu,
        "gpu_info": gpu_info,
    }


@app.post("/summarize", response_model=SummaryResponse)
async def summarize(request: SummaryRequest):
    """Generate a summary for the given text."""
    start_time = time.time()
    
    try:
        summary = generate_summary(
            text=request.text,
            doc_type=request.doc_type,
            max_length=request.max_length,
            min_length=request.min_length,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample,
        )
        
        processing_time = time.time() - start_time
        
        return SummaryResponse(
            summary=summary,
            processing_time=processing_time,
            model_name=model_config.short_name if model_config else "unknown",
            adapter_dir=os.path.basename(adapter_dir) if adapter_dir else "unknown",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Model Summary Server")
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="Path to the model adapter directory",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemma-2-9b",
        help="Model name (default: gemma-2-9b)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token (or set HUGGINGFACE_TOKEN env var)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--use_multi_gpu",
        action="store_true",
        help="Use multiple GPUs if available",
    )
    
    args = parser.parse_args()
    
    # Get HF token from CLI/env and treat blank values as missing.
    # Supports both HUGGINGFACE_TOKEN and HF_TOKEN naming.
    hf_token = normalize_hf_token(
        args.hf_token or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    )
    
    # Load model
    print("=" * 70)
    print("Loading model...")
    print("=" * 70)
    load_model(
        adapter_dir=args.adapter_dir,
        model_name=args.model_name,
        hf_token=hf_token,
        use_multi_gpu=args.use_multi_gpu,
    )
    
    # Store adapter_dir globally for response
    global adapter_dir
    adapter_dir = args.adapter_dir
    
    print("=" * 70)
    print(f"Server starting on {args.host}:{args.port}")
    print("=" * 70)
    
    # Run server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
