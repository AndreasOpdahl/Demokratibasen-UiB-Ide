"""
Unified fine-tuning script for both quantized (GTX3090) and non-quantized (GH200) training.
Supports single-GPU, multi-GPU DDP (Distributed Data Parallel), and FSDP (Fully Sharded Data Parallel).
Supports both quantized (GTX3090 with AMD64-architecture) and non-quantized (GH200 with ARM64-architecture) training.

This was forked off of wandb_finetune.py to test different ways of doing FSDP training.

Before running:
  export YOUR_TOKEN=...your huggingface token here, or source it from an .env file...
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Usage:
  # Single GPU with 4-bit quantization (GTX3090):
  python finetune.py \\
    --model gemma-2b \\
    --quantization 4bit \\
    --train_dataset data/train.jsonl \\
    --val_dataset data/val.jsonl \\
    --output_dir models/gemma_2b_4bit
  
  # Single GPU without quantization with custom hyperparameters:
  python finetune.py \\
    --model gemma-7b \\
    --quantization none \\
    --train_dataset data/train.jsonl \\
    --val_dataset data/val.jsonl \\
    --output_dir models/gemma_7b \\
    --max_steps 1200 \\
    --train_batch_size 4 \\
    --val_steps 100
  
  # Multi-GPU DDP training with torchrun:
  torchrun --nproc_per_node=2 \\
    finetune.py \\
      --model gemma-2b \\
      --quantization none \\
      --ddp \\
      --train_dataset data/train.jsonl \\
      --val_dataset data/val.jsonl \\
      --output_dir models/gemma_2b_ddp \\
      --hf_token YOUR_TOKEN
  
  # Multi-GPU FSDP training for large models:
  torchrun --nproc_per_node=4 \\
    finetune.py \\
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
import importlib
import json
import math
import os
import sys
from typing import Any, Dict, Optional, Tuple, Union

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


# Default TEST VALUES for hyperparameters when command-line args are not supplied
TRAIN_BATCH_SIZE = 1
MAX_TRAIN_STEPS = 60  # ignored if NUM_TRAIN_EPOCHS is set
NUM_TRAIN_EPOCHS = None  # ignored if MAX_TRAIN_STEPS is set
VAL_BATCH_SIZE = 4
VAL_DATA_SIZE = 8  # multiple of batch size
VAL_BEAM_SIZE = 4  # beam size for evaluation
VAL_STEPS = 20  # multiple of save_steps
SAVE_STRATEGY = "steps"
SAVE_STEPS = 10
SAVE_TOTAL_LIMIT = 10


# Default values for training data preparation
MAX_INPUT_TEXT_TOKENS = 2048  # max tokens for input to summarisation
MAX_EXTRA_PROMPT_TOKENS = 40  # max extra tokens for input prompt (the task description)
MAX_INPUT_PROMPT_TOKENS = MAX_INPUT_TEXT_TOKENS + MAX_EXTRA_PROMPT_TOKENS
MAX_OUTPUT_SUMMARY_TOKENS = 512  # max tokens for output from summarisation
MAX_EPOCHS = 5
TRAIN_BATCH_SIZE = 1
VAL_BATCH_SIZE = 5
VAL_DATA_SIZE = 5  # WAS 20, number of examples to use for validation
VAL_BEAM_SIZE = 4  # beam size for evaluation
VAL_STEPS = 20  # WAS 200



class GPUMemoryCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"Step {state.global_step} | GPU {i}: {mem_info.used / 1024**2:.2f} MB")


class TrainDataCollator:
    """Custom data collator for training that pads both input_ids and labels.
    
    For causal LM training, pads sequences to the same length.
    Labels are padded with -100 so they're ignored in loss computation.
    """
    
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        # Extract input_ids, attention_mask, and labels
        input_ids = [f['input_ids'] for f in features]
        labels = [f['labels'] for f in features]
        # Create attention_masks if not present
        attention_masks = []
        for f, ids in zip(features, input_ids):
            if 'attention_mask' in f:
                attention_masks.append(f['attention_mask'])
            else:
                attention_masks.append([1] * len(ids))
        
        # Find max length (should be same for input_ids and labels)
        max_length = max(len(ids) for ids in input_ids)
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) 
                         // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        # Pad all sequences
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for ids, mask, lbl in zip(input_ids, attention_masks, labels):
            # Ensure input_ids and labels have the same length
            if len(ids) != len(lbl):
                raise ValueError(f"input_ids length ({len(ids)}) != labels length ({len(lbl)})")
            
            padding_length = max_length - len(ids)
            
            # Pad input_ids with pad_token_id (RIGHT padding for training)
            padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * padding_length)
            
            # Pad attention_mask with 0
            padded_attention_mask.append(mask + [0] * padding_length)
            
            # Pad labels with -100 (so they're ignored in loss)
            padded_labels.append(lbl + [-100] * padding_length)
        
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long),
        }


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
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Override to ensure loss is extracted correctly from model output.
        
        This is especially important for FSDP where model outputs might be wrapped.
        """
        # Get model outputs directly
        outputs = model(**inputs)
        
        # Extract or compute loss from outputs
        loss = None
        
        # Handle dict-like outputs (including FSDP-wrapped outputs)
        # Check if outputs is dict-like (has 'get' method and 'keys' method)
        is_dict_like = isinstance(outputs, dict) or (hasattr(outputs, 'get') and hasattr(outputs, 'keys'))
        
        if is_dict_like:
            # Try to get loss from outputs
            loss = outputs.get("loss") if hasattr(outputs, 'get') else (outputs["loss"] if "loss" in outputs else None)
            
            # Safety check: ensure loss is a tensor, not a dict or other type
            if loss is not None and not isinstance(loss, torch.Tensor):
                # If loss exists but is not a tensor, ignore it and compute from logits
                loss = None
            
            # If loss is not in outputs or is not a tensor, compute it from logits and labels
            if loss is None:
                if "logits" not in outputs:
                    raise ValueError(f"Cannot compute loss: 'logits' not found in outputs. Output keys: {list(outputs.keys())}")
                
                # Get labels - use input_ids if labels are not provided (standard for causal LM)
                if "labels" in inputs:
                    labels = inputs["labels"]
                elif "input_ids" in inputs:
                    # For causal LM, labels are typically the same as input_ids (shifted)
                    labels = inputs["input_ids"]
                else:
                    raise ValueError(f"Cannot compute loss: neither 'labels' nor 'input_ids' found in inputs. Input keys: {list(inputs.keys())}")
                
                # Compute loss manually using cross entropy
                logits = outputs["logits"]
                
                # Shift labels and logits for causal LM (next token prediction)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten for cross entropy
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Verify loss was computed correctly
                if not isinstance(loss, torch.Tensor):
                    raise RuntimeError(f"Loss computation failed: expected tensor, got {type(loss)}")
        elif isinstance(outputs, tuple):
            # Outputs is a tuple, first element is typically loss
            loss = outputs[0] if len(outputs) > 0 else None
            if loss is None or not isinstance(loss, torch.Tensor):
                raise ValueError("Model output tuple does not contain loss tensor")
        else:
            # Unexpected output type
            raise TypeError(f"Unexpected model output type: {type(outputs)}. Expected dict or tuple.")
        
        # Final validation - ensure loss is a tensor
        if loss is None:
            raise ValueError("Could not extract or compute loss from model outputs")
        if not isinstance(loss, torch.Tensor):
            # This should never happen, but if it does, provide detailed error
            raise TypeError(
                f"Expected loss to be a tensor, got {type(loss)}: {loss}. "
                f"Outputs type: {type(outputs)}, Outputs keys (if dict): {list(outputs.keys()) if isinstance(outputs, dict) else 'N/A'}"
            )
        
        return (loss, outputs) if return_outputs else loss

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
    use_ddp: bool = False,
    use_fsdp: bool = False
):
    """Load model with optional quantization.
    
    Args:
        model_name: Model identifier (e.g., 'google/gemma-2b')
        quantization: One of 'none', '4bit', '8bit'
        use_ddp: Whether to use DDP (Distributed Data Parallel) training (removes device_map)
        use_fsdp: Whether to use FSDP (Fully Sharded Data Parallel) training (removes device_map)
    
    Note:
        HuggingFace token is read from HF_TOKEN environment variable
    
    Returns:
        Loaded model
    """            
            # If loss is not in outputs or is not a tensor, compute it from logits and labels
            if loss is None:
                if "logits" not in outputs:
                    raise ValueError(f"Cannot compute loss: 'logits' not found in outputs. Output keys: {list(outputs.keys())}")
                
                # Get labels - use input_ids if labels are not provided (standard for causal LM)
                if "labels" in inputs:
                    labels = inputs["labels"]
                elif "input_ids" in inputs:
                    # For causal LM, labels are typically the same as input_ids (shifted)
                    labels = inputs["input_ids"]
                else:
                    raise ValueError(f"Cannot compute loss: neither 'labels' nor 'input_ids' found in inputs. Input keys: {list(inputs.keys())}")
                
                # Compute loss manually using cross entropy
                logits = outputs["logits"]
                
                # Shift labels and logits for causal LM (next token prediction)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten for cross entropy
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Verify loss was computed correctly
                if not isinstance(loss, torch.Tensor):
                    raise RuntimeError(f"Loss computation failed: expected tensor, got {type(loss)}")
        elif isinstance(outputs, tuple):
            # Outputs is a tuple, first element is typically loss
            loss = outputs[0] if len(outputs) > 0 else None
            if loss is None or not isinstance(loss, torch.Tensor):
                raise ValueError("Model output tuple does not contain loss tensor")
        else:
            # Unexpected output type
            raise TypeError(f"Unexpected model output type: {type(outputs)}. Expected dict or tuple.")
        
        # Final validation - ensure loss is a tensor
        if loss is None:
            raise ValueError("Could not extract or compute loss from model outputs")
        if not isinstance(loss, torch.Tensor):
            # This should never happen, but if it does, provide detailed error
            raise TypeError(
                f"Expected loss to be a tensor, got {type(loss)}: {loss}. "
                f"Outputs type: {type(outputs)}, Outputs keys (if dict): {list(outputs.keys()) if isinstance(outputs, dict) else 'N/A'}"
            )

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
    save_strategy: str = SAVE_STRATEGY,
    save_steps: int = SAVE_STEPS,
    save_total_limit: int = SAVE_TOTAL_LIMIT,
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
        use_ddp: Whether to enable DDP (Distributed Data Parallel) multi-GPU training
        use_fsdp: Whether to enable FSDP (Fully Sharded Data Parallel) multi-GPU training
    
    Note:
        HuggingFace token is read from HF_TOKEN or HUGGINGFACE_TOKEN environment variable
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
        
        # Are we using 4-bit quantization?
        if quantization == '4bit':
            print('*** evaluation: using 4-bit quantization ***')

            # Fix for 4-bit quantization: clip token IDs to valid vocabulary range
            # This prevents OverflowError during decoding when quantization causes out-of-range values
            vocab_size = tokenizer.vocab_size
            print(f'*** evaluation: vocab size: {vocab_size} ***')
            
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
        
        # Print first 5 examples
        num_examples_to_show = min(5, len(decoded_preds))
        if num_examples_to_show > 0:
            print(f'\n*** Showing first {num_examples_to_show} examples ***')
            for i in range(num_examples_to_show):
                print(f'\n--- Example {i+1} ---')
                print(f'Prediction: {decoded_preds[i][:200]}...')
                print(f'Reference:  {decoded_labels[i][:200]}...')
            print()

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

    
    # Login to Hugging Face if token is provided
    if hf_token:
        print("Logging in to Hugging Face Hub...")
        login(token=hf_token)

    print("Initializing Weights & Biases...")
    wandb.init(
        project="lm-finetuning",  # Change to your project name
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
            model_name, quantization, use_ddp=use_ddp, use_fsdp=use_fsdp
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load and preprocess dataset
    print(f"Loading dataset from: {dataset_path}")

    # Read JSONL file manually
    train_data = []
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                train_data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading training dataset: {e}")
        return

    # Read validation JSONL file
    val_data = []
    try:
        with open(val_dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                val_data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading validation dataset: {e}")
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
        # Format the full text (prompt + target) for tokenization
        text = f"Oppgave: Oppsummer følgende tekst:\n\n###\n\n{example['input']}\n\n###\n\nOppsummering:\n\n###\n\n{example['output']}\n\n###\n"
        return {"text": text}

    def tokenize_function_train(examples):
        # Tokenize the full text first to ensure consistent tokenization
        max_input_prompt_tokens = max_input_text_tokens + max_extra_prompt_tokens
        max_total_length = max_input_prompt_tokens + max_output_summary_tokens
        
        # Tokenize full sequences
        tokenized_full = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_total_length,
            padding=False,
            add_special_tokens=True
        )
        
        # Find prompt boundaries by matching prompt tokens in full sequence
        # Extract prompt portions to tokenize separately
        prompts_only = []
        for text in examples["text"]:
            # Extract prompt portion (everything before the output)
            prompt = text.split("Oppsummering:\n\n###\n\n")[0] + "Oppsummering:\n\n###\n\n"
            prompts_only.append(prompt)
        
        tokenized_prompts = tokenizer(
            prompts_only,
            truncation=True,
            max_length=max_input_prompt_tokens,
            padding=False,
            add_special_tokens=True
        )
        
        # Create labels: mask prompt tokens, keep target tokens
        input_ids = []
        attention_mask = []
        labels = []
        
        for i in range(len(tokenized_full["input_ids"])):
            full_ids = tokenized_full["input_ids"][i]
            prompt_ids = tokenized_prompts["input_ids"][i]
            
            # Find where prompt ends in full sequence by actually matching tokens
            # Tokenize separately can differ, so we need to find the actual match point
            prompt_len = len(prompt_ids)
            
            # Try to find where prompt tokens match in full sequence
            # Check if prompt tokens match at the start of full_ids
            if prompt_len <= len(full_ids):
                # Try to match tokens from the beginning
                match_count = 0
                for j in range(min(prompt_len, len(full_ids))):
                    if j < len(prompt_ids) and j < len(full_ids) and prompt_ids[j] == full_ids[j]:
                        match_count += 1
                    else:
                        break
                
                # Use the matched length (or minimum if mismatch found)
                # This handles cases where tokenization differs slightly
                prompt_len = match_count if match_count > 0 else min(prompt_len, len(full_ids))
            else:
                # Full sequence was truncated more than prompt, mask everything
                prompt_len = len(full_ids)
            
            # Create labels: mask prompt tokens with -100, keep target tokens
            combined_labels = [-100] * prompt_len + full_ids[prompt_len:]
            
            # Safety check: ensure labels length matches input_ids length
            if len(combined_labels) != len(full_ids):
                # This should never happen, but handle it gracefully
                if len(combined_labels) < len(full_ids):
                    # Pad labels if somehow shorter
                    combined_labels.extend([-100] * (len(full_ids) - len(combined_labels)))
                else:
                    # Truncate if somehow longer (shouldn't happen)
                    combined_labels = combined_labels[:len(full_ids)]
            
            input_ids.append(full_ids)
            attention_mask.append([1] * len(full_ids))
            labels.append(combined_labels)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    formatted_dataset = train_dataset.map(format_example_train)
    tokenized_dataset = formatted_dataset.map(tokenize_function_train, batched=True)

    def format_example_eval(example):
        # Format for EVALUATION: only the input prompt, not the answer
        prompt = f"Oppgave: Oppsummer følgende tekst:\n\n###\n\n{example['input']}\n\n###\n\nOppsummering:\n\n###\n\n"
        # Keep the target output separate for ROUGE calculation
        # Handle None or missing output
        target = example['output'] if example['output'] is not None else ""
        return {
            "prompt": prompt,
            "target_summary": str(target)  # Ensure it's a string
        }

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

    formatted_dataset = train_dataset.map(format_example_train)
    tokenized_dataset = formatted_dataset.map(tokenize_function_train, batched=True)

    # Format and tokenize the VALIDATION dataset differently
    formatted_val_dataset = val_dataset.map(format_example_eval)
    tokenized_val_dataset = formatted_val_dataset.map(
        tokenize_function_eval, 
        batched=True,
        load_from_cache_file=False,  # ADD THIS
    )
    
    # Data collators
    # For TRAINING: use standard causal LM collator (creates labels by shifting)
    train_data_collator = DataCollatorWithPadding(
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
    
    if model_name == "google/gemma-7b" or model_name == "google/gemma-2b":    
        # Define LoRA config
        lora_config = LoraConfig(
            r=16,  # Increased rank for better capacity
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # depends on model architecture
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
    
    if model_name == "LumiOpen/Viking-7B" or model_name == "LumiOpen/Viking-13B":
        # Define LoRA config
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Mistral architecture
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
    elif 'llama' in model_name.lower() or 'norskgpt' in model_name.lower():
        # Define LoRA config for Llama-based models (NorskGPT, Llama-2-13b-chat-norwegian)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Llama architecture
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
    else:
        # Default LoRA config for unknown models
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
    
    # Debug: Print layer class names for FSDP configuration
    print("\n=== Model Layer Classes (for FSDP debugging) ===")
    layer_classes = set()
    for name, module in model.named_modules():
        layer_classes.add(type(module).__name__)
    decoder_layers = [cls for cls in layer_classes if 'Decoder' in cls or 'Layer' in cls]
    print(f"Decoder/Layer classes found: {sorted(decoder_layers)}")
    print("=" * 50 + "\n")

    # Resolve resume checkpoint directory (if provided)
    resolved_resume_checkpoint: Optional[str] = None
    print(f"\n=== CHECKPOINT RESUMPTION DEBUG ===")
    print(f"resume_checkpoint argument: {repr(resume_checkpoint)}")
    print(f"output_dir: {output_dir}")
    print(f"Current working directory: {os.getcwd()}")
    if resume_checkpoint:
        resume_checkpoint = resume_checkpoint.strip()
        print(f"After strip: {repr(resume_checkpoint)}")
        if resume_checkpoint.lower() == "latest":
            search_path = os.path.join(output_dir, "checkpoint-*")
            print(f"Searching for checkpoints at: {search_path}")
            candidate_paths = glob.glob(search_path)
            print(f"LATEST CANDIDATE PATHS: {candidate_paths}")
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
        train_epochs = num_train_epochs if num_train_epochs is not None else 1
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
    
    training_args_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=val_batch_size,
        gradient_accumulation_steps=4,
        # Model-specific learning rate for stability
        learning_rate=2e-5 if 'viking-13b' in model_name.lower() or 'llama-2-13b' in model_name.lower() else 1e-5,
        num_train_epochs=train_epochs,
        max_steps=train_steps,
        fp16=False,
        bf16=True,
        logging_steps=10,
        
        # Validate + save on a schedule (disabled for FSDP due to generation incompatibility)
        eval_strategy="steps" if eval_enabled else "no",
        eval_steps=val_steps if eval_enabled else None,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,  # keep disk usage sane

        # Pick the best checkpoint and restore it at the end (only if eval enabled)
        load_best_model_at_end=eval_enabled,
        metric_for_best_model="rougeLsum" if eval_enabled else None,
        greater_is_better=True if eval_enabled else None,
        
        # Numerical stability improvements
        max_grad_norm=0.5,  # More aggressive gradient clipping
        warmup_steps=500,  # Much longer warmup (was 100)
        warmup_ratio=0.0,  # Ensure warmup_steps is used
        weight_decay=0.05,  # Increased regularization (was 0.01)
        adam_epsilon=1e-8,  # Standard Adam epsilon
        adam_beta1=0.9,
        adam_beta2=0.999,
                
        optim="adamw_torch",
        report_to="wandb" if is_main_process else "none",
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
        # FSDP training settings
        # FSDP shards model parameters, gradients, and optimizer states across GPUs
        training_args_kwargs['fsdp'] = "full_shard auto_wrap"
        
        # Configure FSDP using fsdp_config (not deprecated parameters)
        fsdp_config = {
            "activation_checkpointing": True,  # Use activation_checkpointing instead of gradient_checkpointing
        }
        
        # Set transformer_layer_cls_to_wrap based on model type
        if model_name == 'google/gemma-2b' or model_name == 'google/gemma-7b':
            print("GEMMA model utilized")
            fsdp_config['transformer_layer_cls_to_wrap'] = "GemmaDecoderLayer"
        elif model_name == 'LumiOpen/Viking-7B':
            print("VIKING model utilized")
            fsdp_config['transformer_layer_cls_to_wrap'] = "LlamaDecoderLayer"
        
        training_args_kwargs['fsdp_config'] = fsdp_config
        
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fine-tune a language model with optional quantization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # AMD64-architecture (like GTX3090) with 4-bit quantization:
  python finetune.py --model gemma-2b --quantization 4bit --hf_token YOUR_TOKEN

  # ARM64-architecture (like GH200/Cray) without quantization, fixed steps:
  python finetune.py --model gemma-2b --quantization none --max_steps 1200 --hf_token YOUR_TOKEN

  # Without quantization, train for epochs:
  python finetune.py --model gemma-2b --quantization none --num_train_epochs 3
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       choices=['viking-7b', 'gemma-2b', 'mt5', 'gemma-7b'],
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
    # HF_TOKEN is now read from environment variable (HF_TOKEN or HUGGINGFACE_TOKEN)
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
    parser.add_argument('--max_steps', type=int, default=MAX_TRAIN_STEPS,
                       help='Maximum training steps (overrides num_train_epochs if set)')
    parser.add_argument('--num_train_epochs', type=int, default=None,
                       help=f'Number of training epochs (default: {NUM_TRAIN_EPOCHS}, ignored if max_steps is set)')
    parser.add_argument('--val_batch_size', type=int, default=VAL_BATCH_SIZE,
                       help=f'Validation batch size per device (default: {VAL_BATCH_SIZE})')
    parser.add_argument('--val_data_size', type=int, default=VAL_DATA_SIZE,
                       help=f'Number of examples to use for validation (default: {VAL_DATA_SIZE})')
    parser.add_argument('--val_beam_size', type=int, default=VAL_BEAM_SIZE,
                       help=f'Beam size for validation generation (default: {VAL_BEAM_SIZE})')
    parser.add_argument('--val_steps', type=int, default=VAL_STEPS,
                       help=f'Validate and save every N steps (default: {VAL_STEPS})')
    parser.add_argument('--save_strategy', type=str, default=SAVE_STRATEGY,
                       choices=['no', 'steps', 'epoch'],
                       help=f'Checkpoint save strategy (default: {SAVE_STRATEGY})')
    parser.add_argument('--save_steps', type=int, default=SAVE_STEPS,
                       help=f'Save a checkpoint every N steps (default: {SAVE_STEPS})')
    parser.add_argument('--save_total_limit', type=int, default=SAVE_TOTAL_LIMIT,
                       help=f'Maximum number of checkpoints to keep (default: {SAVE_TOTAL_LIMIT})')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                       help='Path to a checkpoint directory to resume from. '
                            'Use "latest" to automatically pick the newest checkpoint in the output_dir.')

    args = parser.parse_args()
    
    # Store force_restart in globals for access in fine_tune_model
    globals()['force_restart'] = args.force_restart

    # Validate quantization availability
    if args.quantization != 'none' and not QUANTIZATION_AVAILABLE:
        print(f"ERROR: Quantization requested ({args.quantization}) but BitsAndBytesConfig is not available.")
        print("Please install bitsandbytes: pip install bitsandbytes")
        print("Or use --quantization none to run without quantization.")
        exit(1)

    # Model mapping
    model_mapping = {
        'viking-7b': 'LumiOpen/Viking-7B',
        'gemma-2b': 'google/gemma-2b',
        'viking-13b': 'LumiOpen/Viking-13B',
        'mt5': 'google/mt5-base',
        'gemma-7b': 'google/gemma-7b'
    }
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
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        resume_checkpoint=args.resume_checkpoint,
    )

