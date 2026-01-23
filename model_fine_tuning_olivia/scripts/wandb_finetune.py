"""
Unified fine-tuning script for both quantized (GTX3090) and non-quantized (GH200) training.
Supports single-GPU, multi-GPU DDP (Distributed Data Parallel), and FSDP (Fully Sharded Data Parallel).

Usage:
  # Single GPU with 4-bit quantization (GTX3090):
  python finetune.py \\
    --model gemma-2b \\
    --quantization 4bit \\
    --train_dataset data/train.jsonl \\
    --val_dataset data/val.jsonl \\
    --output_dir models/gemma_2b_4bit \\
    --hf_token YOUR_TOKEN
  
  # Single GPU without quantization with custom hyperparameters:
  python finetune.py \\
    --model gemma-7b \\
    --quantization none \\
    --train_dataset data/train.jsonl \\
    --val_dataset data/val.jsonl \\
    --output_dir models/gemma_7b \\
    --max_steps 1200 \\
    --train_batch_size 4 \\
    --val_steps 100 \\
    --hf_token YOUR_TOKEN
  
  # Multi-GPU DDP training with torchrun:
  torchrun --nproc_per_node=2 finetune.py \\
    --model gemma-2b \\
    --quantization none \\
    --ddp \\
    --train_dataset data/train.jsonl \\
    --val_dataset data/val.jsonl \\
    --output_dir models/gemma_2b_ddp \\
    --hf_token YOUR_TOKEN
  
  # Multi-GPU FSDP training for large models:
  torchrun --nproc_per_node=4 finetune.py \\
    --model gemma-7b \\
    --quantization none \\
    --fsdp \\
    --train_dataset data/train.jsonl \\
    --val_dataset data/val.jsonl \\
    --output_dir models/gemma_7b_fsdp \\
    --hf_token YOUR_TOKEN

Before running:
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Important Notes:
  - DDP/FSDP training removes device_map="auto" (handled automatically)
  - Quantization + DDP/FSDP is not well supported - use single GPU or no quantization
  - FSDP shards the model across GPUs (saves memory) vs DDP (full replicas)
  - FSDP is better for very large models (e.g., 7B+ parameters)
  - CRITICAL: DDP/FSDP training with LoRA CANNOT resume from checkpoints due to PEFT limitations
    Checkpoints will be automatically ignored in distributed mode to prevent errors.
    If interrupted, training will restart from scratch (checkpoints are still saved for recovery).
"""

import argparse
import glob
import json
import os
import random
import sys
import importlib
import math
import time
from typing import Any, Dict, Optional, Tuple, Union
import wandb
import numpy as np
import pandas as pd

import evaluate
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import torch
import torch.serialization
from safetensors.torch import load_file
import pynvml
# Fix for PyTorch 2.6+ weights_only security issue
# Patch torch.load to disable weights_only for checkpoint files (we trust our own checkpoints)
_original_torch_load = torch.load

def _torch_load_with_weights_only_false(path, *args, **kwargs):
    """Wrapper around torch.load that disables weights_only for checkpoint compatibility."""
    # Only disable weights_only for checkpoint-related files
    if 'rng_state' in str(path) or 'optimizer' in str(path) or 'scheduler' in str(path):
        kwargs['weights_only'] = False
    return _original_torch_load(path, *args, **kwargs)

torch.load = _torch_load_with_weights_only_false

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Trainer,
    TrainingArguments,
)
try:
    TrainerState = importlib.import_module("transformers.trainer_state").TrainerState  # type: ignore[attr-defined]
except (ImportError, AttributeError):  # pragma: no cover - environment without transformers installed
    TrainerState = None  # type: ignore[assignment]

# Optional imports for quantization
try:
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    print("Warning: BitsAndBytesConfig/prepare_model_for_kbit_training not available.")
    print("Quantization will be disabled. This is expected on ARM-based systems (e.g., GH200).")

# Import model configurations
from model_configs import (
    get_model_config,
    get_model_config_by_hf_name,
    get_doc_type_norwegian,
    get_model_name_mapping,
)

# Default values when command-line args are not supplied
MAX_INPUT_TEXT_TOKENS = 2048  # max tokens for input to summarisation
MAX_EXTRA_PROMPT_TOKENS = 40  # max extra tokens for input prompt (the task description)
MAX_INPUT_PROMPT_TOKENS = MAX_INPUT_TEXT_TOKENS + MAX_EXTRA_PROMPT_TOKENS
MAX_OUTPUT_SUMMARY_TOKENS = 512  # max tokens for output from summarisation
MAX_EPOCHS = 5
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 5
VAL_DATA_SIZE = 5  # WAS 20, number of examples to use for validation
VAL_BEAM_SIZE = 4  # beam size for evaluation
VAL_STEPS = 100  # Reduce checkpoint frequency (was 20)



class GPUMemoryCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"Step {state.global_step} | GPU {i}: {mem_info.used / 1024**2:.2f} MB")

def check_early_stopping_signal(output_dir: str) -> bool:
    """Check if early stopping signal exists from evaluation monitor."""
    signal_file = os.path.join(output_dir, ".early_stop")
    return os.path.exists(signal_file)


class EarlyStoppingMonitorCallback(TrainerCallback):
    """Callback to check for early stopping signal from evaluation monitor.
    
    This allows the evaluation monitor script to signal early stopping
    during FSDP training when evaluation is disabled.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.check_interval = 100  # Check every 100 steps to avoid too much I/O
    
    def on_step_end(self, args, state, control, **kwargs):
        """Check for early stopping signal periodically."""
        # Only check periodically to avoid too much I/O
        if state.global_step % self.check_interval == 0:
            if check_early_stopping_signal(self.output_dir):
                print("\n" + "="*70)
                print("Early stopping signal detected from evaluation monitor!")
                print(f"Best checkpoint will be selected from evaluated checkpoints.")
                print("Stopping training...")
                print("="*70 + "\n")
                control.should_training_stop = True
                
class EvalDataCollator:
    """Custom data collator for evaluation that pads both input_ids and labels.
    
    Uses left-padding for input_ids (decoder-only models) and right-padding for labels.
    """
    
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        # Separate input_ids and labels
        input_ids = [f['input_ids'] for f in features]
        labels = [f['labels'] for f in features]
        
        # Pad input_ids (LEFT padding for decoder-only models)
        max_input_length = max(len(ids) for ids in input_ids)
        if self.pad_to_multiple_of:
            max_input_length = ((max_input_length + self.pad_to_multiple_of - 1) 
                               // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        padded_input_ids = []
        attention_mask = []
        for ids in input_ids:
            padding_length = max_input_length - len(ids)
            # LEFT padding: pad tokens go BEFORE the actual tokens
            padded_input_ids.append([self.tokenizer.pad_token_id] * padding_length + ids)
            attention_mask.append([0] * padding_length + [1] * len(ids))
        
        # Pad labels (RIGHT padding - standard for targets)
        max_label_length = max(len(lbl) for lbl in labels)
        if self.pad_to_multiple_of:
            max_label_length = ((max_label_length + self.pad_to_multiple_of - 1) 
                               // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        padded_labels = []
        for lbl in labels:
            padding_length = max_label_length - len(lbl)
            # RIGHT padding for labels: pad with -100 so they're ignored
            padded_labels.append(lbl + [-100] * padding_length)
        
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long),
        }


class CausalLMTrainer(Trainer):
    def __init__(self, *args, 
                 generation_max_length: Optional[int] = None,
                 generation_num_beams: Optional[int] = None,
                 eval_data_collator: Optional[Any] = None,
                 **kwargs) -> None:
        # 1. Store generation parameters
        self.generation_max_length = generation_max_length
        self.generation_num_beams = generation_num_beams
        self.eval_data_collator = eval_data_collator
        # 2. Call parent constructor
        super().__init__(*args, **kwargs)
        # 3. Store reference to tokenizer for compatibility
        self._processing_class = self.tokenizer
    
    def get_eval_dataloader(self, eval_dataset=None):
        """Override to use a different data collator for evaluation."""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        # Temporarily swap the data collator
        original_collator = self.data_collator
        if self.eval_data_collator is not None:
            self.data_collator = self.eval_data_collator
        
        # Get the dataloader using parent's method
        dataloader = super().get_eval_dataloader(eval_dataset)
        
        # Restore original collator
        self.data_collator = original_collator
        
        return dataloader

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        # 1. Compute Loss (Standard behavior)
        if prediction_loss_only:
            # For loss-only, we can't compute it properly since eval data is prompt-only
            # Just return None and skip loss calculation
            return (None, None, None)

        # If we are here, we are evaluating the model
        print('*** evaluation: prediction_step ***')
        torch.cuda.empty_cache()

        # 2. Get Input IDs and Labels
        # input_ids contains ONLY the prompt (no target answer)
        if 'input_ids' in inputs:
            input_ids = inputs["input_ids"]
        else:
            raise KeyError("input_ids not found in batch. Check your DataCollator setup.")

        # labels contains the tokenized target summary (for ROUGE)
        labels = inputs.get("labels")
        
        # Remove padding from labels (-100 will be there from DataCollator)
        # We need clean label token IDs for ROUGE
        if labels is not None:
            # Create a copy and replace -100 with pad_token_id for proper shape
            labels = labels.clone()
            labels[labels == -100] = self._processing_class.pad_token_id

        print('*** evaluation: input_ids (prompt only) ***', input_ids.shape)
        if labels is not None:
            print('*** evaluation: labels (target summary) ***', labels.shape)

        # 3. Autoregressive Generation (The Key Step)
        # Generate from the prompt only
        generated_ids = model.generate(
            input_ids=input_ids,
            use_cache=True,
            max_new_tokens=self.generation_max_length,  # Generate up to 512 new tokens
            num_beams=self.generation_num_beams,
            do_sample=False,  # Greedy/beam search
            pad_token_id=self._processing_class.pad_token_id,
            eos_token_id=self._processing_class.eos_token_id,
        )
        
        # CRITICAL: Remove the input prompt from generated_ids
        # model.generate() returns [input_ids + new_tokens], we only want the new tokens for ROUGE
        input_length = input_ids.shape[1]
        generated_ids = generated_ids[:, input_length:]
        
        print('*** evaluation: generated_ids (generated summary only) ***', generated_ids.shape)
        
        torch.cuda.empty_cache()

        # No loss calculation during evaluation (we can't compute it from prompt-only data)
        loss = None
        
        # 5. Return Results
        # The Trainer expects: (loss, predictions, labels)
        # predictions: generated summary tokens
        # labels: target summary tokens
        return (loss, generated_ids, labels)


def load_model_with_optional_quantization(
    model_name: str,
    quantization: str,
    hf_token: Optional[str] = None,
    use_ddp: bool = False,
    use_fsdp: bool = False
):
    """Load model with optional quantization.
    
    Args:
        model_name: Model identifier (e.g., 'google/gemma-2b')
        quantization: One of 'none', '4bit', '8bit'
        hf_token: Hugging Face token for private models
        use_ddp: Whether to use DDP (Distributed Data Parallel) training (removes device_map)
        use_fsdp: Whether to use FSDP (Fully Sharded Data Parallel) training (removes device_map)
    
    Returns:
        Loaded model
    """
    if model_name == 'google/mt5-base':
        return MT5ForConditionalGeneration.from_pretrained(model_name)
    
    if quantization == 'none':
        # No quantization (GH200/Cray path)
        print("Loading model without quantization (FP16)...")
        
        # For DDP/FSDP training, we must NOT use device_map="auto"
        # The DDP/FSDP launcher will handle device placement
        # Keep model on CPU - Trainer will move it to the correct device for each rank
        use_distributed = use_ddp or use_fsdp
        if use_distributed:
            mode_str = "FSDP" if use_fsdp else "DDP"
            print(f"{mode_str} enabled - loading model without device_map (keeping on CPU)")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                token=hf_token,
                # DO NOT use device_map for distributed training
                # Let torchrun/FSDP handle device placement
                low_cpu_mem_usage=True  # Efficient loading for large models
            )
            # Explicitly keep on CPU - do NOT move to CUDA yet
            # The Trainer will handle device placement after DDP/FSDP wrapping
            print(f"Model loaded on CPU ({mode_str} mode)")
        else:
            # Single GPU path with device_map
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    token=hf_token
                )
            except Exception as e:
                print(f"Error loading model with device_map: {e}")
                print("Trying fallback without device_map...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    token=hf_token
                ).cuda()
        return model
    
    elif quantization == '4bit':
        # 4-bit quantization (GTX3090 path)
        if not QUANTIZATION_AVAILABLE:
            raise ImportError(
                "BitsAndBytesConfig not available. Install bitsandbytes for quantization support:\n"
                "  pip install bitsandbytes"
            )
        
        if use_ddp or use_fsdp:
            print("WARNING: Quantization with DDP/FSDP is not well supported.")
            print("Consider single-GPU training with quantization.")
        
        print("Loading model with 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto" if not (use_ddp or use_fsdp) else None,
            token=hf_token
        )
        return model
    
    elif quantization == '8bit':
        # 8-bit quantization (optional)
        if not QUANTIZATION_AVAILABLE:
            raise ImportError(
                "BitsAndBytesConfig not available. Install bitsandbytes for quantization support:\n"
                "  pip install bitsandbytes"
            )
        
        if use_ddp or use_fsdp:
            print("WARNING: Quantization with DDP/FSDP is not well supported.")
            print("Consider single-GPU training with quantization.")
        
        print("Loading model with 8-bit quantization...")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto" if not (use_ddp or use_fsdp) else None,
            token=hf_token
        )
        return model
    
    else:
        raise ValueError(f"Unknown quantization method: {quantization}")


def prepare_model_for_lora(model, use_quantization: bool):
    """Prepare model for LoRA training with or without quantization.
    
    Args:
        model: The model to prepare
        use_quantization: Whether the model uses quantization
    
    Returns:
        Prepared model (note: model is modified in-place)
    """
    if use_quantization:
        # Use peft's built-in function for quantized models
        if not QUANTIZATION_AVAILABLE:
            raise ImportError("prepare_model_for_kbit_training not available. Install peft.")
        
        print("Preparing quantized model for LoRA training...")
        model = prepare_model_for_kbit_training(model)
    else:
        # Manual preparation for non-quantized models
        print("Preparing non-quantized model for LoRA training...")
        
        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Enable gradients for input embeddings (critical for LoRA!)
        # Without this, the embeddings stay frozen and cause "does not require grad" errors
        if hasattr(model, 'get_input_embeddings'):
            input_embeddings = model.get_input_embeddings()
            if input_embeddings is not None:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                input_embeddings.register_forward_hook(make_inputs_require_grad)
        
        # Disable cache for gradient checkpointing
        model.config.use_cache = False
    
    return model


def fine_tune_model(
    model_name: str,
    dataset_path: str,
    val_dataset_path: str,
    output_dir: str,
    quantization: str = 'none',
    max_steps: Optional[int] = None,
    num_train_epochs: Optional[int] = None,
    hf_token: Optional[str] = None,
    use_ddp: bool = False,
    use_fsdp: bool = False,
    max_input_text_tokens: int = MAX_INPUT_TEXT_TOKENS,
    max_extra_prompt_tokens: int = MAX_EXTRA_PROMPT_TOKENS,
    max_output_summary_tokens: int = MAX_OUTPUT_SUMMARY_TOKENS,
    train_batch_size: int = TRAIN_BATCH_SIZE,
    val_batch_size: int = VAL_BATCH_SIZE,
    val_data_size: int = VAL_DATA_SIZE,
    val_beam_size: int = VAL_BEAM_SIZE,
    val_steps: int = VAL_STEPS,
    resume_checkpoint: Optional[str] = None,
):
    """Fine-tune a language model with LoRA.
    
    Args:
        model_name: Model identifier
        dataset_path: Path to training dataset (JSONL)
        val_dataset_path: Path to validation dataset (JSONL)
        output_dir: Directory to save the fine-tuned model
        quantization: Quantization method ('none', '4bit', '8bit')
        max_steps: Maximum training steps (overrides num_train_epochs if set)
        num_train_epochs: Number of training epochs
        hf_token: Hugging Face authentication token
        use_ddp: Whether to enable DDP (Distributed Data Parallel) multi-GPU training
        use_fsdp: Whether to enable FSDP (Fully Sharded Data Parallel) multi-GPU training
    """
    
    def compute_metrics(eval_pred):
        print('*** evaluation: compute_metrics ***')
        
        # Load ROUGE metric (lazy loading after cache paths are set)
        # This avoids loading at module level before environment is configured
        rouge = evaluate.load("rouge")
        
        preds, labels = eval_pred  # preds: generated summary ids; labels: target summary ids
        print('*** evaluation: preds ***', preds.shape)
        print('*** evaluation: labels ***', labels.shape)
        
        # Replace -100 and pad tokens so we can decode properly
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # Fix for 4-bit quantization: clip token IDs to valid vocabulary range
        # This prevents OverflowError during decoding when quantization causes out-of-range values
        vocab_size = tokenizer.vocab_size
        print(f'*** Vocab size: {vocab_size} ***')
        
        # Clip predictions to valid token ID range [0, vocab_size)
        # Replace any invalid values with pad_token_id
        preds = np.clip(preds, 0, vocab_size - 1)
        
        # Also ensure labels are in valid range
        labels = np.clip(labels, 0, vocab_size - 1)
        
        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Strip/normalize for ROUGE
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        
        # Debug: print first example
        if len(decoded_preds) > 0:
            print(f'\n*** Example 1 ***')
            print(f'Prediction: {decoded_preds[0][:200]}...')
            print(f'Reference:  {decoded_labels[0][:200]}...\n')

        scores = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        print('*** evaluation: computed_metrics ***', scores)
        
        # Only log to wandb from main process
        if wandb.run is not None and int(os.environ.get('RANK', 0)) == 0:
            wandb.log({
                "eval/rouge1": scores['rouge1'] * 100,
                "eval/rouge2": scores['rouge2'] * 100,
                "eval/rougeL": scores['rougeL'] * 100,
                "eval/rougeLsum": scores['rougeLsum'] * 100,
            })

        return {k: v * 100 for k, v in scores.items()}  # % values

    # Set HF token via environment variable
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token

    # Determine rank for distributed training
    rank = int(os.environ.get('RANK', 0))
    is_main_process = (rank == 0)
    
    # Only initialize wandb on rank 0
    if is_main_process:
        print("Initializing Weights & Biases...")
        wandb.init(
            project="lm-finetuning",
            name=f"{os.path.basename(output_dir)}_{quantization}",
            config={
                "model_name": model_name,
                "quantization": quantization,
                "max_input_text_tokens": max_input_text_tokens,
                "max_output_summary_tokens": max_output_summary_tokens,
                "train_batch_size": train_batch_size,
                "val_batch_size": val_batch_size,
                "use_ddp": use_ddp,
                "use_fsdp": use_fsdp,
            }
        )
        print(f">>> wandb run initialized: {wandb.run.name}")
        print(f">>> wandb run URL: {wandb.run.get_url()}")
        # Get the run name for TrainingArguments
        wandb_run_name = wandb.run.name
    else:
        # Disable wandb for non-rank-0 processes
        os.environ['WANDB_DISABLED'] = 'true'
        wandb_run_name = None

    # Load tokenizer
    try:
        if model_name == 'google/mt5-base':
            tokenizer = MT5Tokenizer.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token if hf_token else None
            )
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Make sure you have the correct model name and authentication if needed.")
        return

    # Set padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # For TRAINING: use RIGHT padding (DataCollatorForLanguageModeling needs this)
    # For GENERATION: we'll switch to left padding later if needed
    tokenizer.padding_side = 'right'  # Changed from 'left' - needed for training
    
    # Note: If you need left padding for generation, set it in the generation code
    # or use a separate tokenizer instance for evaluation

    # Validate DDP/FSDP flags (mutually exclusive)
    if use_ddp and use_fsdp:
        raise ValueError("Cannot use both --ddp and --fsdp. Choose one distributed training strategy.")
    
    # Detect if we're in a distributed environment
    # If launched with torchrun/accelerate, environment variables will be set
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    rank = int(os.environ.get('RANK', -1))
    
    use_distributed = use_ddp or use_fsdp
    
    print(f"=== Environment Check ===")
    print(f"use_ddp flag: {use_ddp}, use_fsdp flag: {use_fsdp}")
    print(f"WORLD_SIZE={world_size}, RANK={rank}, LOCAL_RANK={local_rank}")
    
    if world_size > 1 or use_distributed:
        mode = "FSDP" if use_fsdp else "DDP"
        print(f"=== {mode} Training Detected ===")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        if use_fsdp:
            use_fsdp = True
        else:
            use_ddp = True
    else:
        print("=== Single-GPU Training Mode ===")
        # CRITICAL: Clean up any DDP environment variables
        # These can be set by SLURM or previous runs and confuse TrainingArguments/Accelerate
        # We must clean these BEFORE creating TrainingArguments
        ddp_vars = [
            'WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT',
            'LOCAL_WORLD_SIZE', 'NODE_RANK', 'GROUP_RANK', 
            'TORCHELASTIC_RUN_ID', 'TORCHELASTIC_RESTART_COUNT', 'TORCHELASTIC_MAX_RESTARTS',
            'NCCL_ASYNC_ERROR_HANDLING', 'TORCH_DISTRIBUTED_DEBUG',
            # Accelerate-specific variables
            'ACCELERATE_USE_CPU', 'ACCELERATE_MIXED_PRECISION', 'ACCELERATE_USE_FSDP',
            'ACCELERATE_USE_DEEPSPEED', 'ACCELERATE_DYNAMO_BACKEND'
        ]
        cleaned = []
        for var in ddp_vars:
            if var in os.environ:
                cleaned.append(f"{var}={os.environ[var]}")
                del os.environ[var]
        if cleaned:
            print(f"Cleaned DDP env vars: {cleaned}")
        else:
            print("No DDP env vars found to clean")
        
        # Double-check cleanup worked
        remaining = {k: v for k, v in os.environ.items() 
                    if any(x in k.upper() for x in ['RANK', 'WORLD', 'DIST', 'TORCH_DIST'])}
        if remaining:
            print(f"WARNING: Some DDP vars still present: {list(remaining.keys())}")
            for k in list(remaining.keys()):
                del os.environ[k]
                print(f"  Force-deleted: {k}")
        
        # Explicitly disable DDP for Accelerate/Transformers
        # This prevents PartialState from trying to initialize DDP
        # Do NOT set LOCAL_RANK - that triggers DDP detection!
        
        # Force single-process mode for Accelerate
        # These variables tell Accelerate to NOT use DDP
        os.environ['ACCELERATE_USE_CPU'] = 'false'
        os.environ['ACCELERATE_NUM_PROCESSES'] = '1'
        
        # CRITICAL FIX: Restrict CUDA visibility to single GPU
        # This prevents accelerate from detecting multi-GPU and assuming DDP mode
        # If CUDA_VISIBLE_DEVICES is not already set, set it to GPU 0
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            print("Set CUDA_VISIBLE_DEVICES=0 to restrict to single GPU")
        else:
            print(f"CUDA_VISIBLE_DEVICES already set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
        
        # Verify GPU count after setting CUDA_VISIBLE_DEVICES
        # NOTE: This won't take effect until we import torch again, but good to log
        print(f"CUDA device count: {torch.cuda.device_count()}")
        
        # Destroy any existing DDP process group
        if torch.distributed.is_initialized():
            print("WARNING: torch.distributed was already initialized - destroying it")
            torch.distributed.destroy_process_group()
        
        print("DDP environment variables cleaned for single-GPU mode")
    
    # Load model with optional quantization
    try:
        model = load_model_with_optional_quantization(
            model_name, quantization, hf_token, use_ddp=use_ddp, use_fsdp=use_fsdp
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load and preprocess dataset
    print(f"Loading dataset from: {dataset_path}")

    # Read JSONL file manually
    train_data = []
    try:
        # Check if file exists and is readable
        if not os.path.exists(dataset_path):
            print(f"ERROR: Training dataset file does not exist: {dataset_path}")
            return
        
        # Check file size (Git LFS pointers are typically < 200 bytes)
        file_size = os.path.getsize(dataset_path)
        if file_size < 200:
            print(f"WARNING: Training dataset file is very small ({file_size} bytes).")
            print(f"         This might be a Git LFS pointer file. Please ensure the actual file is downloaded.")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            # Check if it's a Git LFS pointer
            if first_line.strip().startswith('version https://git-lfs.github.com/spec/v1'):
                print(f"ERROR: Training dataset file appears to be a Git LFS pointer, not actual data.")
                print(f"       Please download the actual file using: git lfs pull")
                print(f"       Or ensure the file at {dataset_path} contains actual JSONL data.")
                return
            
            # Reset file pointer and read all lines
            f.seek(0)
            line_num = 0
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    train_data.append(json.loads(line))
                except json.JSONDecodeError as json_err:
                    print(f"ERROR: Invalid JSON on line {line_num} of training dataset:")
                    print(f"       {str(json_err)}")
                    print(f"       Line content (first 200 chars): {line[:200]}")
                    return
            
        if len(train_data) == 0:
            print(f"ERROR: Training dataset file is empty or contains no valid JSON lines: {dataset_path}")
            return
            
        print(f"Successfully loaded {len(train_data)} training examples")
    except Exception as e:
        print(f"Error reading training dataset: {e}")
        print(f"File path: {dataset_path}")
        import traceback
        traceback.print_exc()
        return

    # Read validation JSONL file
    val_data = []
    try:
        # Check if file exists and is readable
        if not os.path.exists(val_dataset_path):
            print(f"ERROR: Validation dataset file does not exist: {val_dataset_path}")
            return
        
        # Check file size (Git LFS pointers are typically < 200 bytes)
        file_size = os.path.getsize(val_dataset_path)
        if file_size < 200:
            print(f"WARNING: Validation dataset file is very small ({file_size} bytes).")
            print(f"         This might be a Git LFS pointer file. Please ensure the actual file is downloaded.")
        
        with open(val_dataset_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            # Check if it's a Git LFS pointer
            if first_line.strip().startswith('version https://git-lfs.github.com/spec/v1'):
                print(f"ERROR: Validation dataset file appears to be a Git LFS pointer, not actual data.")
                print(f"       Please download the actual file using: git lfs pull")
                print(f"       Or ensure the file at {val_dataset_path} contains actual JSONL data.")
                return
            
            # Reset file pointer and read all lines
            f.seek(0)
            line_num = 0
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    val_data.append(json.loads(line))
                except json.JSONDecodeError as json_err:
                    print(f"ERROR: Invalid JSON on line {line_num} of validation dataset:")
                    print(f"       {str(json_err)}")
                    print(f"       Line content (first 200 chars): {line[:200]}")
                    return
            
        if len(val_data) == 0:
            print(f"ERROR: Validation dataset file is empty or contains no valid JSON lines: {val_dataset_path}")
            return
            
        print(f"Successfully loaded {len(val_data)} validation examples")
    except Exception as e:
        print(f"Error reading validation dataset: {e}")
        print(f"File path: {val_dataset_path}")
        import traceback
        traceback.print_exc()
        return

    # Create training dataset
    train_df = pd.DataFrame(train_data)
    dataset = Dataset.from_pandas(train_df)
    
    # Randomly sample validation_data_size examples from val_data
    val_data = random.sample(val_data, min(val_data_size, len(val_data)))
    
    # Filter out examples with missing input or output
    val_data = [ex for ex in val_data if ex.get('input') and ex.get('output')]
    
    val_df = pd.DataFrame(val_data)
    val_dataset = Dataset.from_pandas(val_df)
    print(f'*** validation dataset size: {len(val_dataset)} examples ***')

    def format_example_train(example):
        """Format training example with model-specific prompt template."""
        # Extract doc_type from metadata if available
        doc_type = None
        if 'metadata' in example and isinstance(example['metadata'], dict):
            doc_type = example['metadata'].get('doc_type')
        
        model_config = get_model_config_by_hf_name(model_name)
        if model_config:
            return {"text": model_config.prompt_config.format_train(
                input_text=example['input'],
                output_text=example['output'],
                doc_type=doc_type
            )}
        else:
            # Fallback to default format with doc_type
            from model_configs import get_doc_type_norwegian
            doc_type_nor = get_doc_type_norwegian(doc_type)
            text = f"Oppgave: Oppsummer følgende {doc_type_nor}:\n\n###\n\n{example['input']}\n\n###\n\nOppsummering:\n\n###\n\n{example['output']}\n\n###\n"
            return {"text": text}

    def format_example_eval(example):
        """Format evaluation example with model-specific prompt template."""
        # Extract doc_type from metadata if available
        doc_type = None
        if 'metadata' in example and isinstance(example['metadata'], dict):
            doc_type = example['metadata'].get('doc_type')
        
        model_config = get_model_config_by_hf_name(model_name)
        if model_config:
            prompt = model_config.prompt_config.format_eval(input_text=example['input'], doc_type=doc_type)
        else:
            # Fallback to default format with doc_type
            from model_configs import get_doc_type_norwegian
            doc_type_nor = get_doc_type_norwegian(doc_type)
            prompt = f"Oppgave: Oppsummer følgende {doc_type_nor}:\n\n###\n\n{example['input']}\n\n###\n\nOppsummering:\n\n###\n\n"
        
        target = example['output'] if example.get('output') is not None else ""
        return {
            "prompt": prompt,
            "target_summary": str(target)  # Ensure it's a string
        }

    def tokenize_function_train(examples):
        # Tokenize the formatted text for training
        max_input_prompt_tokens = max_input_text_tokens + max_extra_prompt_tokens
        # Note: padding=False here - DataCollatorForLanguageModeling handles padding
        # This approach is compatible with both tokenizers 0.20.0 and 0.22.0
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_input_prompt_tokens + max_output_summary_tokens,
            padding=False  # Padding done by data collator for compatibility across tokenizer versions
        )
        return tokenized

    def tokenize_function_eval(examples):
        # Tokenize ONLY the prompt (without answer) for evaluation
        max_input_prompt_tokens = max_input_text_tokens + max_extra_prompt_tokens
        tokenized_prompts = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=max_input_prompt_tokens,
            padding=False
        )
        # Tokenize target summaries for labels
        tokenized_targets = tokenizer(
            examples["target_summary"],
            truncation=True,
            max_length=max_output_summary_tokens,
            padding=False
        )
        # Store target token IDs as labels
        tokenized_prompts["labels"] = tokenized_targets["input_ids"]
        return tokenized_prompts

    formatted_dataset = dataset.map(format_example_train)
    
    # Log example prompts to wandb (lightweight - just a few examples to verify prompt formatting)
    if is_main_process and wandb.run is not None:
        # Collect example prompts with different doc_types
        example_prompts = []
        doc_types_seen = set()
        
        # Sample from training dataset (first 5 examples to show variety)
        for i in range(min(5, len(formatted_dataset))):
            example = formatted_dataset[i]
            original_example = train_data[i] if i < len(train_data) else {}
            
            # Extract doc_type
            doc_type = None
            if 'metadata' in original_example and isinstance(original_example['metadata'], dict):
                doc_type = original_example['metadata'].get('doc_type')
            
            doc_type_nor = get_doc_type_norwegian(doc_type) if doc_type else "tekst"
            doc_types_seen.add(doc_type_nor)
            
            # Get formatted text (full prompt + output for training)
            formatted_text = example.get('text', '')
            # Extract just the prompt part (before the output summary)
            if 'Oppsummering:\n\n###\n\n' in formatted_text:
                prompt_part = formatted_text.split('Oppsummering:\n\n###\n\n')[0] + 'Oppsummering:\n\n###\n\n'
            elif 'Oppsummering:' in formatted_text:
                prompt_part = formatted_text.split('Oppsummering:')[0] + 'Oppsummering:'
            else:
                # For other prompt formats, take first 400 chars
                prompt_part = formatted_text[:400] + "..." if len(formatted_text) > 400 else formatted_text
            
            example_prompts.append({
                "example_num": i + 1,
                "doc_type": doc_type or "unknown",
                "doc_type_norwegian": doc_type_nor,
                "prompt_preview": prompt_part[:300] + "..." if len(prompt_part) > 300 else prompt_part
            })
        
        # Log to wandb config (lightweight - just metadata)
        # Get model_config for template type
        model_config = get_model_config_by_hf_name(model_name)
        wandb.config.update({
            "prompt_examples": example_prompts,
            "doc_types_in_training": sorted(list(doc_types_seen)),
            "prompt_template_type": model_config.prompt_config.template_type if model_config else "plain"
        })
        
        # Also print to console for visibility
        print("\n" + "=" * 70)
        print("PROMPT EXAMPLES (logged to wandb config):")
        print("=" * 70)
        for ex in example_prompts:
            print(f"\nExample {ex['example_num']}:")
            print(f"  Doc Type: {ex['doc_type']} -> {ex['doc_type_norwegian']}")
            print(f"  Prompt Preview: {ex['prompt_preview']}")
        print("=" * 70 + "\n")
    
    tokenized_dataset = formatted_dataset.map(
        tokenize_function_train, 
        batched=True,
        load_from_cache_file=False,  # ADD THIS - keeps data in memory
    )

    # Format and tokenize the VALIDATION dataset differently
    formatted_val_dataset = val_dataset.map(format_example_eval)
    tokenized_val_dataset = formatted_val_dataset.map(
        tokenize_function_eval, 
        batched=True,
        load_from_cache_file=False,  # ADD THIS
    )
    
    # Data collators
    # For TRAINING: use DataCollatorForLanguageModeling which creates labels
    # IMPORTANT: Must use RIGHT padding for this to work correctly
    train_data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Debug: Verify labels are created correctly (only on rank 0)
    if is_main_process:
        print("\n=== Verifying data collator ===")
        try:
            # Get a sample - ensure it's a dict with proper structure
            sample = tokenized_dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Sample type: {type(sample)}")
            
            # Check if input_ids exists and is the right type
            if 'input_ids' in sample:
                input_ids = sample['input_ids']
                print(f"input_ids type: {type(input_ids)}, length: {len(input_ids) if hasattr(input_ids, '__len__') else 'N/A'}")
                if isinstance(input_ids, list):
                    print(f"input_ids first few: {input_ids[:5] if len(input_ids) > 5 else input_ids}")
                elif hasattr(input_ids, 'tolist'):
                    print(f"input_ids first few: {input_ids[:5].tolist() if len(input_ids) > 5 else input_ids.tolist()}")
            
            # Convert sample to proper format if needed (HuggingFace datasets sometimes return lists)
            # Ensure all values are lists of integers (token IDs), not nested structures
            sample_dict = {}
            for key, value in sample.items():
                if isinstance(value, list):
                    # If it's already a list, check if it contains token IDs (integers)
                    if len(value) > 0 and isinstance(value[0], (int, np.integer)):
                        sample_dict[key] = value
                    else:
                        # If it's a list of strings or other types, skip this sample
                        print(f"WARNING: {key} is a list but not token IDs, skipping debug collation")
                        continue
                elif hasattr(value, 'tolist'):
                    # If it's a tensor/array, convert to list
                    sample_dict[key] = value.tolist()
                elif isinstance(value, (int, np.integer)):
                    # Single integer token ID - wrap in list
                    sample_dict[key] = [value]
                else:
                    # Skip non-integer values to avoid tokenization errors
                    print(f"WARNING: {key} has unsupported type {type(value)}, skipping")
                    continue
            
            # Only try collating if we have valid token IDs
            if 'input_ids' in sample_dict and isinstance(sample_dict['input_ids'], list):
                collated = train_data_collator([sample_dict])
            else:
                print("WARNING: Could not create valid sample_dict for collation test")
                # Ensure all values are lists/tensors, not nested structures
                sample_dict = {}
                for key, value in sample.items():
                    if isinstance(value, list):
                        # If it's already a list, use it
                        sample_dict[key] = value
                    elif hasattr(value, 'tolist'):
                        # If it's a tensor/array, convert to list
                        sample_dict[key] = value.tolist()
                    else:
                        # Otherwise, wrap in list
                        sample_dict[key] = [value] if not isinstance(value, list) else value
                
                # Now try collating
                collated = train_data_collator([sample_dict])
            
            if 'labels' in collated:
                labels = collated['labels']
                if hasattr(labels, 'numel'):
                    non_ignored = (labels != -100).sum().item()
                    total = labels.numel()
                    print(f"Labels shape: {labels.shape}")
                    print(f"Non-ignored labels: {non_ignored}/{total} ({100*non_ignored/total:.1f}%)")
                    if non_ignored == 0:
                        print("ERROR: All labels are ignored (-100)! This will cause loss=0.0")
                else:
                    print(f"Labels type: {type(labels)}")
            else:
                print("WARNING: No 'labels' key in collated output")
            print("=" * 50 + "\n")
        except Exception as e:
            print(f"ERROR during data collator verification: {e}")
            print(f"Sample structure: {sample if 'sample' in locals() else 'N/A'}")
            import traceback
            traceback.print_exc()
            print("=" * 50 + "\n")
            # Don't fail training, just warn
            print("WARNING: Data collator verification failed, but continuing training...")
    
    # For EVALUATION: use custom collator that pads both input_ids and labels
    eval_data_collator = EvalDataCollator(tokenizer=tokenizer)

    print("--- VIKING DEBUGGING ---")
    # Method 1: Print the model and look for the layer class names
    print(model)

    # Method 2: Check if the model has a '_no_split_modules' attribute
    if hasattr(model, "_no_split_modules"):
        print("_no_split_modules:", model._no_split_modules)

    # Method 3: Check the model's configuration and source class
    print("\nModel class:", model.__class__)
    print("\nModel config:\n", model.config)
    # Prepare model for LoRA training
    use_quantization = (quantization != 'none')
    model = prepare_model_for_lora(model, use_quantization)
    
    # Get model config for LoRA settings
    model_config = get_model_config_by_hf_name(model_name)
    if model_config:
        lora_config = model_config.get_lora_config()
        print(f"Using LoRA config for {model_config.short_name}: r={model_config.lora_r}, alpha={model_config.lora_alpha}")
    else:
        print(f"WARNING: Unknown model {model_name}, using default LoRA config")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
    # Apply LoRA adapters
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Resolve resume checkpoint directory (if provided)
    resolved_resume_checkpoint: Optional[str] = None
    if resume_checkpoint:
        resume_checkpoint = resume_checkpoint.strip()
        print(resume_checkpoint)
        if resume_checkpoint.lower() == "latest":
            candidate_paths = glob.glob(os.path.join(output_dir, "checkpoint-*"))
            print("LATEST CANDIDATE PATHS", candidate_paths)
            if not candidate_paths:
                print(f"WARNING: resume_checkpoint=latest requested but no checkpoints found in {output_dir}")
            else:
                def _checkpoint_step(path: str) -> int:
                    base = os.path.basename(path.rstrip(os.sep))
                    try:
                        return int(base.split("-")[-1])
                    except (IndexError, ValueError):
                        return -1
                candidate_paths.sort(key=_checkpoint_step)
                resolved_resume_checkpoint = os.path.abspath(candidate_paths[-1])
                print("LATEST RESUME CHECKPOINT: ", resolved_resume_checkpoint)
        else:
            candidate_path = resume_checkpoint
            if not os.path.isabs(candidate_path):
                candidate_path = os.path.join(output_dir, candidate_path)
            candidate_path = os.path.abspath(candidate_path)
            print("NOT LATEST", candidate_path)
            if os.path.isdir(candidate_path):
                resolved_resume_checkpoint = candidate_path
            else:
                print(f"ERROR: resume checkpoint directory not found: {candidate_path}")
                return
        if resolved_resume_checkpoint:
            print(f"Resume checkpoint resolved to: {resolved_resume_checkpoint}")

    # Check if we're resuming from checkpoint
    checkpoints_exist = len(glob.glob(os.path.join(output_dir, "checkpoint-*"))) > 0
    force_restart = False

    # CRITICAL: PEFT + DDP/FSDP + checkpoint resumption is problematic
    # The adapter loading conflicts with distributed device management
    if checkpoints_exist and (use_ddp or use_fsdp) and not resolved_resume_checkpoint:
        mode = "FSDP" if use_fsdp else "DDP"
        print("\n" + "=" * 70)
        print(f"CRITICAL: Checkpoint resumption with PEFT + {mode} is not supported!")
        print("=" * 70)
        print(f"Found existing checkpoints, but {mode} training with LoRA adapters")
        print("cannot resume from checkpoints due to PEFT device management conflicts.")
        print("")
        print("This is a known limitation: PEFT's load_adapter() tries to load weights")
        print(f"to CUDA before {mode} is properly coordinated across ranks.")
        print("")
        print("Solutions:")
        print(f"  1. Use --force_restart to ignore checkpoints and start fresh")
        print(f"  2. Delete checkpoints: rm -rf {output_dir}/checkpoint-*")
        print(f"  3. Use a different output_dir")
        print(f"  4. Train on single GPU (no --ddp or --fsdp flag)")
        print("=" * 70)
        print("FORCING RESTART to avoid checkpoint loading errors...")
        print("=" * 70 + "\n")
        force_restart = True  # Force it to avoid the error
    elif checkpoints_exist and (use_ddp or use_fsdp) and resolved_resume_checkpoint:
        print("\n" + "=" * 70)
        print("Manual checkpoint resumption requested.")
        print("Proceeding with user-specified resume checkpoint despite PEFT + FSDP limitation.")
        print("Ensure all ranks see the same checkpoint directory.")
        print("=" * 70 + "\n")
    
    # In distributed mode (DDP/FSDP), ensure model stays on CPU
    # The Trainer will move it to the correct device for each rank
    if use_ddp or use_fsdp:
        # Verify model is on CPU
        device_str = str(next(model.parameters()).device)
        mode = "FSDP" if use_fsdp else "DDP"
        print(f"After LoRA, model device: {device_str}")
        if device_str != "cpu":
            print(f"WARNING: Model is on {device_str}, moving to CPU for {mode} training")
            model = model.cpu()
    
    if resolved_resume_checkpoint:
        adapter_path = os.path.join(resolved_resume_checkpoint, "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            print(f"ERROR: Expected adapter weights not found at {adapter_path}")
            return
        print(f"Loading LoRA adapter weights from {adapter_path}")
        adapter_state = load_file(adapter_path, device="cpu")
        set_peft_model_state_dict(model, adapter_state)
        if use_fsdp and torch.distributed.is_initialized():
            torch.distributed.barrier()
        print("LoRA adapter weights loaded successfully.")

    # Determine training duration
    if max_steps is not None and max_steps > 0:
        train_epochs = 1  # Set to 1 instead of None to avoid Trainer comparison errors
        train_steps = max_steps
        print(f"Training for {max_steps} steps (epochs ignored)")
    else:
        train_epochs = num_train_epochs if num_train_epochs is not None else MAX_EPOCHS
        train_steps = -1  # -1 means "use epochs instead"
        print(f"Training for {train_epochs} epochs")

    # Training arguments
    # CRITICAL: FSDP + generation during evaluation causes errors
    # Disable evaluation for FSDP training
    if use_fsdp:
        print("\n" + "=" * 70)
        print("WARNING: FSDP mode detected - disabling evaluation")
        print("=" * 70)
        print("FSDP parameter sharding is incompatible with model.generate()")
        print("during evaluation. Training will proceed without ROUGE metrics.")
        print("You can evaluate checkpoints later using load_distributed_peft_checkpoint.py")
        print("=" * 70 + "\n")
        eval_enabled = False
    else:
        eval_enabled = True
    
    # Get model config for hyperparameters (if not already loaded)
    if 'model_config' not in locals():
        model_config = get_model_config_by_hf_name(model_name)
    
    training_args_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=val_batch_size,
        gradient_accumulation_steps=4,
        # Model-specific learning rate from config
        learning_rate=model_config.learning_rate if model_config else 1e-5,
        num_train_epochs=train_epochs,
        max_steps=train_steps,
        fp16=False,
        bf16=True,
        logging_steps=10,
        
        # Validate + save on a schedule (disabled for FSDP due to generation incompatibility)
        eval_strategy="steps" if eval_enabled else "no",
        eval_steps=val_steps if eval_enabled else None,
        save_strategy="steps",
        save_steps=val_steps,
        save_total_limit=10,  # keep disk usage sane

        # Pick the best checkpoint and restore it at the end (only if eval enabled)
        load_best_model_at_end=eval_enabled,
        metric_for_best_model="rougeLsum" if eval_enabled else None,
        greater_is_better=True if eval_enabled else None,
        
        # Numerical stability improvements
        max_grad_norm=1.0,  # Increased from 0.5 - too aggressive clipping can cause NaN
        warmup_steps=500,
        warmup_ratio=0.0,
        weight_decay=0.05,
        adam_epsilon=1e-8,
        adam_beta1=0.9,
        adam_beta2=0.999,
                
        optim="adamw_torch",
        report_to="wandb" if is_main_process else "none",
        run_name=wandb_run_name,  # ADD THIS - link to manually initialized wandb run
        gradient_checkpointing=not use_fsdp,  # Disable for FSDP, use activation_checkpointing instead
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        
        # fsdp config
        #fsdp_min_num_params=1e8,
        #cpu_offload=True,
    )
    
    # Add distributed or single-GPU-specific parameters
    if use_ddp:
        # DDP training settings (only when actually using DDP)
        training_args_kwargs['ddp_find_unused_parameters'] = False
        training_args_kwargs['ddp_backend'] = 'nccl'
        print("Added DDP parameters for multi-GPU training")
    elif use_fsdp:
        # FSDP training settings - Compatible with transformers 4.45.2 and PEFT/LoRA
        # IMPORTANT: With PEFT/LoRA, we MUST use "full_shard" WITHOUT "auto_wrap"
        # because PEFT wraps the model and breaks auto-wrapping detection
        
        print("Configuring FSDP for PEFT/LoRA training...")
        print("Using 'full_shard' mode (without auto_wrap) for PEFT compatibility")
        
        # Use full_shard only - NO auto_wrap when using PEFT/LoRA
        # This avoids the "Could not find transformer layer class" error
        training_args_kwargs['fsdp'] = "full_shard"
        
        # Do NOT set fsdp_transformer_layer_cls_to_wrap when not using auto_wrap
        # It will cause errors and deprecation warnings
        
        # Disable gradient_checkpointing when using FSDP
        # FSDP has its own memory optimization mechanisms
        if 'gradient_checkpointing' in training_args_kwargs:
            training_args_kwargs['gradient_checkpointing'] = False
            print("Disabled gradient_checkpointing for FSDP (FSDP handles memory optimization)")
        
        print("FSDP configured: full_shard mode (compatible with PEFT/LoRA)")
    
    if not (use_ddp or use_fsdp):
        # Single-GPU mode - explicitly prevent distributed detection
        training_args_kwargs['local_rank'] = -1
        print("Added local_rank=-1 to TrainingArguments for single-GPU mode")
    
    training_args = TrainingArguments(**training_args_kwargs)
    
    # Only add EarlyStoppingCallback when:
    # 1. Starting fresh (not resuming)
    # 2. Evaluation is enabled (FSDP disables eval, so no early stopping)
    callbacks = []
    
    if not checkpoints_exist and eval_enabled:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=10,  # stop if no improvement for n evals
            early_stopping_threshold=0.0  # require strictly better than best
        )
        callbacks.append(early_stopping)
        print("Adding EarlyStoppingCallback (fresh training with evaluation enabled)")
    else:
        if checkpoints_exist:
            print("Resuming from checkpoint - skipping EarlyStoppingCallback to avoid state errors")
        elif not eval_enabled:
            print("Evaluation disabled (FSDP mode) - skipping EarlyStoppingCallback")

    # Add early stopping monitor callback for FSDP (when eval is disabled)
    if use_fsdp and not eval_enabled:
        early_stopping_monitor = EarlyStoppingMonitorCallback(output_dir)
        callbacks.append(early_stopping_monitor)
        print("Added EarlyStoppingMonitorCallback for FSDP (checks for external early stopping signal)")

    # Initialize Trainer
    # Prepare trainer kwargs
    trainer_kwargs = dict(
        # Generation settings (important so ROUGE is computed on model outputs)
        generation_max_length=max_output_summary_tokens,
        generation_num_beams=val_beam_size,
        eval_data_collator=eval_data_collator,  # Use separate collator for eval
        # General Trainer settings
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=train_data_collator,  # Training collator
        callbacks=callbacks
    )
    
    # Only add eval_dataset and compute_metrics if evaluation is enabled
    if eval_enabled:
        trainer_kwargs['eval_dataset'] = tokenized_val_dataset
        trainer_kwargs['compute_metrics'] = compute_metrics
    
    trainer = CausalLMTrainer(**trainer_kwargs)

    if resolved_resume_checkpoint:
        training_args.resume_from_checkpoint = resolved_resume_checkpoint
        if not use_fsdp:
            print(f"Restoring optimizer/scheduler state from {resolved_resume_checkpoint}")
            total_training_steps = training_args.max_steps if training_args.max_steps and training_args.max_steps > 0 else None
            if total_training_steps is None:
                # Fallback estimate for epoch-based training
                effective_batch = training_args.per_device_train_batch_size * max(training_args.gradient_accumulation_steps, 1)
                steps_per_epoch = math.ceil(len(tokenized_dataset) / max(effective_batch, 1))
                total_training_steps = steps_per_epoch * max(training_args.num_train_epochs, 1)
            trainer.create_optimizer_and_scheduler(num_training_steps=total_training_steps)

            optimizer_state_path = os.path.join(resolved_resume_checkpoint, "optimizer.bin")
            scheduler_state_path = os.path.join(resolved_resume_checkpoint, "scheduler.pt")
            trainer_state_path = os.path.join(resolved_resume_checkpoint, "trainer_state.json")

            if os.path.exists(optimizer_state_path) or os.path.exists(scheduler_state_path):
                trainer._load_optimizer_and_scheduler(resolved_resume_checkpoint)
            else:
                print("WARNING: Optimizer or scheduler state not found; continuing without optimizer resume.")

            if os.path.exists(trainer_state_path):
                if TrainerState is None:
                    raise ImportError("TrainerState is unavailable. Ensure transformers is installed in the runtime environment.")
                trainer.state = TrainerState.load_from_json(trainer_state_path)
                trainer.state.is_local_process_zero = trainer.args.process_index == 0
                trainer._globalstep_last_logged = trainer.state.global_step
                print(f"Trainer state restored (global_step={trainer.state.global_step}).")
            else:
                print("WARNING: trainer_state.json not found; starting trainer state from scratch.")

            try:
                trainer._load_rng_state(resolved_resume_checkpoint)
                print("RNG state restored.")
            except Exception as rng_error:
                print(f"WARNING: Failed to restore RNG state: {rng_error}")
        else:
            print("FSDP detected - skipping manual optimizer/scheduler restore. Trainer will handle resumption internally.")

    # Start training
    manual_resume = resolved_resume_checkpoint is not None and not use_fsdp

    if manual_resume:
        print("Continuing training with manually restored checkpoint state...")
        trainer.train()
    elif checkpoints_exist and not force_restart:
        resume_arg = resolved_resume_checkpoint if resolved_resume_checkpoint else True
        if isinstance(resume_arg, str):
            print(f"Resuming training from checkpoint path: {resume_arg}")
        else:
            print("Resuming training from latest checkpoint in output directory...")
        trainer.train(resume_from_checkpoint=resume_arg)
    else:
        if force_restart and checkpoints_exist:
            print("Force restart enabled - ignoring existing checkpoints and starting fresh...")
        else:
            print("Starting training from scratch...")
        trainer.train()
    
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"Training completed. Model saved to {output_dir}")

    # After trainer.train() finishes
    if is_main_process and wandb.run is not None:
        print(f">>> wandb final sync...")
        wandb.finish()  # Ensure all logs are synced
    
    # Write training completion signal for monitor script
    if is_main_process:
        completion_file = os.path.join(output_dir, ".training_complete")
        with open(completion_file, 'w') as f:
            f.write(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"Training completion signal written to {completion_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fine-tune a language model with optional quantization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GTX3090 with 4-bit quantization:
  python finetune.py --model gemma-2b --quantization 4bit --hf_token YOUR_TOKEN

  # GH200/Cray without quantization, fixed steps:
  python finetune.py --model gemma-2b --quantization none --max_steps 1200 --hf_token YOUR_TOKEN

  # Without quantization, train for epochs:
  python finetune.py --model gemma-2b --quantization none --num_train_epochs 3 --hf_token YOUR_TOKEN
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                    choices=['viking-7b', 'viking-13b', 'viking-33b',
                             'gemma-2b', 'gemma-7b', 'gemma-2-9b', 'gemma-2-27b',
                             'gemma-3-12b', 'gemma-3-27b',
                             'normistral-7b', 'normistral-11b',
                             'norskgpt-llama3-8b', 'llama-2-13b-chat-norwegian', 'mt5'],
                    help='Model to fine-tune')
    parser.add_argument('--quantization', type=str, default='none',
                       choices=['none', '4bit', '8bit'],
                       help='Quantization method (default: none). Use "4bit" for GTX3090, "none" for GH200.')
    parser.add_argument('--train_dataset', type=str, default='/app/data/output/processed_data_train.jsonl',
                       help='Path to training dataset (JSONL format)')
    parser.add_argument('--val_dataset', type=str, default='/app/data/output/processed_data_val.jsonl',
                       help='Path to validation dataset (JSONL format)')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for the fine-tuned model')
    parser.add_argument('--max_steps', type=int, default=None,
                       help='Maximum training steps (overrides num_train_epochs if set)')
    parser.add_argument('--num_train_epochs', type=int, default=None,
                       help=f'Number of training epochs (default: {MAX_EPOCHS}, ignored if max_steps is set)')
    parser.add_argument('--hf_token', type=str,
                       help='Hugging Face authentication token for private models')
    parser.add_argument('--ddp', action='store_true',
                       help='Enable DDP (Distributed Data Parallel) multi-GPU training. Auto-detected if launched with torchrun.')
    parser.add_argument('--fsdp', action='store_true',
                       help='Enable FSDP (Fully Sharded Data Parallel) multi-GPU training for large models. Shards model across GPUs to save memory.')
    parser.add_argument('--force_restart', action='store_true',
                       help='Ignore existing checkpoints and start training from scratch.')
    
    # Hyperparameters
    parser.add_argument('--max_input_text_tokens', type=int, default=MAX_INPUT_TEXT_TOKENS,
                       help=f'Maximum tokens for input text (default: {MAX_INPUT_TEXT_TOKENS})')
    parser.add_argument('--max_extra_prompt_tokens', type=int, default=MAX_EXTRA_PROMPT_TOKENS,
                       help=f'Maximum extra tokens for input prompt/task description (default: {MAX_EXTRA_PROMPT_TOKENS})')
    parser.add_argument('--max_output_summary_tokens', type=int, default=MAX_OUTPUT_SUMMARY_TOKENS,
                       help=f'Maximum tokens for output summary (default: {MAX_OUTPUT_SUMMARY_TOKENS})')
    parser.add_argument('--train_batch_size', type=int, default=TRAIN_BATCH_SIZE,
                       help=f'Training batch size per device (default: {TRAIN_BATCH_SIZE})')
    parser.add_argument('--val_batch_size', type=int, default=VAL_BATCH_SIZE,
                       help=f'Validation batch size per device (default: {VAL_BATCH_SIZE})')
    parser.add_argument('--val_data_size', type=int, default=VAL_DATA_SIZE,
                       help=f'Number of examples to use for validation (default: {VAL_DATA_SIZE})')
    parser.add_argument('--val_beam_size', type=int, default=VAL_BEAM_SIZE,
                       help=f'Beam size for validation generation (default: {VAL_BEAM_SIZE})')
    parser.add_argument('--val_steps', type=int, default=VAL_STEPS,
                       help=f'Validate and save every N steps (default: {VAL_STEPS})')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                       help='Path to a checkpoint directory to resume from. '
                            'Use "latest" to automatically pick the newest checkpoint in the output_dir.')
    parser.add_argument('--timeout_minutes', type=int, default=30,
                       help='Stop monitoring if no new checkpoints appear for this many minutes (default: 30)')

    args = parser.parse_args()
    
    # Store force_restart in globals for access in fine_tune_model
    globals()['force_restart'] = args.force_restart

    # Validate quantization availability
    if args.quantization != 'none' and not QUANTIZATION_AVAILABLE:
        print(f"ERROR: Quantization requested ({args.quantization}) but BitsAndBytesConfig is not available.")
        print("Please install bitsandbytes: pip install bitsandbytes")
        print("Or use --quantization none to run without quantization.")
        exit(1)

    # Model mapping from configs
    model_mapping = get_model_name_mapping()
    try:
        model_name = model_mapping[args.model]
    except Exception as e:
        print(f"Error mapping model name: {e}")
        sys.exit(1)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Clean model name for directory (replace / with _)
        clean_model_name = model_name.replace('/', '_').replace('\\', '_')
        output_dir = 'models/' + clean_model_name

    # Run fine-tuning
    fine_tune_model(
        model_name=model_name,
        dataset_path=args.train_dataset,
        val_dataset_path=args.val_dataset,
        output_dir=output_dir,
        quantization=args.quantization,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        hf_token=args.hf_token,
        use_ddp=args.ddp,
        use_fsdp=args.fsdp,
        max_input_text_tokens=args.max_input_text_tokens,
        max_extra_prompt_tokens=args.max_extra_prompt_tokens,
        max_output_summary_tokens=args.max_output_summary_tokens,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        val_data_size=args.val_data_size,
        val_beam_size=args.val_beam_size,
        val_steps=args.val_steps,
        resume_checkpoint=args.resume_checkpoint,
    )

