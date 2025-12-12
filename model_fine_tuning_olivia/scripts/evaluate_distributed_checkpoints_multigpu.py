"""
Multi-GPU evaluation script for PEFT checkpoints using model parallelism.

This script uses model parallelism (device_map="auto") to split large models across GPUs,
avoiding the FSDP/DDP synchronization issues that occur with model.generate().

NOTE: FSDP is incompatible with model.generate() - the training script disables
evaluation when using FSDP for this reason. This script uses model parallelism instead.

Usage:
  # Multi-GPU evaluation with model parallelism:
  python evaluate_distributed_checkpoints_multigpu.py \
    --model gemma-7b \
    --checkpoint_dir models/gemma-7b_fsdp/checkpoint-100 \
    --val_dataset data/output/processed_data_val.jsonl \
    --hf_token YOUR_TOKEN \
    --wandb_project lm-evaluation \
    --use_multi_gpu

Before running:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""

import argparse
import json
import os
import random
import time  # ADD THIS for staggered loading
from typing import Any, Dict, Optional, Tuple, Union

# Disable tokenizer parallelism to avoid fork warnings in multi-process environment
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import wandb

import evaluate
from datasets import Dataset
from huggingface_hub import login
from peft import PeftModel
import torch
import torch.distributed as dist
import torch.serialization

# Fix for PyTorch 2.6+ weights_only security issue
_original_torch_load = torch.load

def _torch_load_with_weights_only_false(path, *args, **kwargs):
    """Wrapper around torch.load that disables weights_only for checkpoint compatibility."""
    if 'rng_state' in str(path) or 'optimizer' in str(path) or 'scheduler' in str(path):
        kwargs['weights_only'] = False
    return _original_torch_load(path, *args, **kwargs)

torch.load = _torch_load_with_weights_only_false

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Import model configurations
from model_configs import get_model_config_by_hf_name, get_model_name_mapping

# Evaluation parameters
MAX_INPUT_TEXT_TOKENS = 2048
MAX_EXTRA_PROMPT_TOKENS = 40
MAX_INPUT_PROMPT_TOKENS = MAX_INPUT_TEXT_TOKENS + MAX_EXTRA_PROMPT_TOKENS
MAX_OUTPUT_SUMMARY_TOKENS = 512
VAL_BATCH_SIZE = 32
VAL_DATA_SIZE = 500
VAL_BEAM_SIZE = 4

# Add a custom exception at the top of the file (near other imports)
class AlreadyEvaluatedError(Exception):
    """Raised when a checkpoint has already been evaluated (only contains eval_results)."""
    pass


def setup_distributed_evaluation():
    """Check if we should use multi-GPU model parallelism.
    
    Returns:
        tuple: (use_multi_gpu, rank, world_size, local_rank)
    """
    # For evaluation, we use model parallelism (device_map="auto") not DDP/FSDP
    # This avoids synchronization issues with model.generate()
    num_gpus = torch.cuda.device_count()
    
    if num_gpus > 1:
        print(f"Multiple GPUs detected ({num_gpus}) - will use model parallelism")
        return True, 0, 1, 0  # Not actually distributed, just multi-GPU
    else:
        return False, 0, 1, 0


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
            padded_input_ids.append([self.tokenizer.pad_token_id] * padding_length + ids)
            attention_mask.append([0] * padding_length + [1] * len(ids))
        
        # Pad labels (RIGHT padding)
        max_label_length = max(len(lbl) for lbl in labels)
        if self.pad_to_multiple_of:
            max_label_length = ((max_label_length + self.pad_to_multiple_of - 1) 
                               // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        padded_labels = []
        for lbl in labels:
            padding_length = max_label_length - len(lbl)
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
                 use_greedy: bool = True,
                 **kwargs) -> None:
        self.generation_max_length = generation_max_length
        self.generation_num_beams = generation_num_beams
        self.eval_data_collator = eval_data_collator
        self.use_greedy = use_greedy
        super().__init__(*args, **kwargs)
        self._processing_class = self.tokenizer
    
    def get_eval_dataloader(self, eval_dataset=None):
        """Override to use a different data collator for evaluation."""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        original_collator = self.data_collator
        if self.eval_data_collator is not None:
            self.data_collator = self.eval_data_collator
        
        dataloader = super().get_eval_dataloader(eval_dataset)
        self.data_collator = original_collator
        
        return dataloader

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        if prediction_loss_only:
            return (None, None, None)

        print('*** evaluation: prediction_step ***')
        
        # Clear cache before generation
        torch.cuda.empty_cache()

        if 'input_ids' in inputs:
            input_ids = inputs["input_ids"]
        else:
            raise KeyError("input_ids not found in batch")

        labels = inputs.get("labels")
        
        if labels is not None:
            labels = labels.clone()
            labels[labels == -100] = self._processing_class.pad_token_id

        print('*** evaluation: input_ids (prompt only) ***', input_ids.shape)
        if labels is not None:
            print('*** evaluation: labels (target summary) ***', labels.shape)

        # Generate with memory-efficient settings
        with torch.amp.autocast('cuda'):
            # Get special token IDs for better stopping
            inst_token_id = None
            if hasattr(self._processing_class, 'convert_tokens_to_ids'):
                try:
                    inst_token_id = self._processing_class.convert_tokens_to_ids('[/INST]')
                except:
                    pass
            
            generation_kwargs = {
                'input_ids': input_ids,
                'use_cache': True,
                'max_new_tokens': self.generation_max_length,
                'num_beams': 1 if self.use_greedy else self.generation_num_beams,
                'do_sample': False,
                'pad_token_id': self._processing_class.pad_token_id,
                'eos_token_id': self._processing_class.eos_token_id,
            }
            
            # Add stop token if found (for chat models)
            if inst_token_id is not None:
                # Don't stop on [/INST] during generation, but we'll clean it later
                pass
            
            generated_ids = model.generate(**generation_kwargs)
        
        input_length = input_ids.shape[1]
        generated_ids = generated_ids[:, input_length:]
        
        print('*** evaluation: generated_ids (generated summary only) ***', generated_ids.shape)
        
        # Clear cache after generation
        torch.cuda.empty_cache()
        
        # Log GPU memory usage if wandb is active
        # Note: No distributed mode, so always main process
        if wandb.run is not None:
            gpu_memory = {}
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9  # GB
                reserved = torch.cuda.memory_reserved(i) / 1e9  # GB
                max_allocated = torch.cuda.max_memory_allocated(i) / 1e9  # GB
                gpu_memory[f"gpu_{i}_allocated_gb"] = allocated
                gpu_memory[f"gpu_{i}_reserved_gb"] = reserved
                gpu_memory[f"gpu_{i}_max_allocated_gb"] = max_allocated  # Track peak usage
            
            # Log periodically (every 10 steps to avoid spam)
            if hasattr(self, '_eval_step_count'):
                self._eval_step_count += 1
            else:
                self._eval_step_count = 1
            
            if self._eval_step_count % 10 == 0:
                wandb.log(gpu_memory, step=self._eval_step_count)
                # Print warning if getting close to OOM
                for i in range(torch.cuda.device_count()):
                    if reserved > 80:  # >80GB used out of 102GB
                        print(f"WARNING: GPU {i} using {reserved:.1f}GB - consider reducing batch size")

        loss = None
        
        return (loss, generated_ids, labels)


def load_model_and_peft_checkpoint(
    model_name: str,
    checkpoint_dir: str,
    hf_token: Optional[str] = None,
    use_multi_gpu: bool = False,
):
    """Load base model and PEFT checkpoint for inference.
    
    Uses model parallelism (device_map="auto") for multi-GPU, not FSDP/DDP.
    This is compatible with model.generate() unlike FSDP.
    
    Args:
        model_name: Base model identifier
        checkpoint_dir: Path to PEFT checkpoint directory
        hf_token: Hugging Face token for private models
        use_multi_gpu: If True, use device_map="auto" to split model across GPUs
    
    Returns:
        Loaded model with PEFT adapters
    
    Raises:
        AlreadyEvaluatedError: If checkpoint was already evaluated (only contains eval_results)
        ValueError: If checkpoint directory is invalid or missing adapter files
    """
    print(f"Loading base model: {model_name}")
    
    num_gpus = torch.cuda.device_count()
    
    # Use model parallelism (device_map="auto") for multi-GPU
    # This is compatible with generation, unlike FSDP
    if use_multi_gpu and num_gpus > 1:
        print(f"Using model parallelism across {num_gpus} GPUs")
        
        # Print GPU info
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")
        
        # Try using accelerate for better device_map control
        try:
            from accelerate import infer_auto_device_map, dispatch_model
            from accelerate.utils import get_balanced_memory
            
            print("Using accelerate for optimized device_map...")
            
            # First load model to CPU to get its structure
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cpu",  # Load to CPU first
                token=hf_token,
                low_cpu_mem_usage=True,
            )
            
            # Get balanced memory across GPUs
            max_memory = get_balanced_memory(
                base_model,
                max_memory=None,  # Use all available memory
                no_split_module_classes=base_model._no_split_modules if hasattr(base_model, '_no_split_modules') else None,
            )
            
            # Infer device map
            device_map = infer_auto_device_map(
                base_model,
                max_memory=max_memory,
                no_split_module_classes=base_model._no_split_modules if hasattr(base_model, '_no_split_modules') else None,
            )
            
            print(f"Device map computed: {len(device_map)} layers")
            # Print device map summary
            device_summary = {}
            for layer_name, device in device_map.items():
                if isinstance(device, (int, str)):
                    device_str = f"cuda:{device}" if isinstance(device, int) else str(device)
                else:
                    device_str = str(device)
                device_summary[device_str] = device_summary.get(device_str, 0) + 1
            print(f"Device map distribution: {device_summary}")
            
            # Reload with device_map
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device_map,  # Use accelerate's device_map
                token=hf_token,
                low_cpu_mem_usage=True,
            )
            
        except (ImportError, Exception) as e:
            print(f"Accelerate not available or failed ({e}), using device_map='auto'")
            device_map_strategy = "auto"
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device_map_strategy,
                token=hf_token,
                low_cpu_mem_usage=True,
            )
    else:
        print("Using single GPU")
        device_map_strategy = "cuda:0"
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map_strategy,
            token=hf_token,
            low_cpu_mem_usage=True,
        )
    
    # Verify device_map worked for base model
    if use_multi_gpu and num_gpus > 1:
        print("Verifying base model is split across GPUs...")
        device_usage = {}
        device_param_count = {}
        for name, param in base_model.named_parameters():
            device = str(param.device)
            device_usage[device] = device_usage.get(device, 0) + param.numel()
            device_param_count[device] = device_param_count.get(device, 0) + 1
        print(f"Base model device distribution:")
        total_params = sum(device_usage.values())
        for device, num_params in sorted(device_usage.items()):
            pct = (num_params / total_params * 100) if total_params > 0 else 0
            print(f"  {device}: {num_params:,} parameters ({pct:.1f}%), {device_param_count[device]} layers")
        
        unique_devices = set(str(p.device) for p in base_model.parameters())
        if len(unique_devices) > 1:
            print(f"✓ Base model successfully split across {len(unique_devices)} devices: {unique_devices}")
        else:
            print(f"⚠ WARNING: Base model is only on {unique_devices} - device_map may not have worked!")
    
    # Convert checkpoint_dir to absolute path to avoid PEFT interpreting it as a repo ID
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    
    # Verify checkpoint directory exists
    if not os.path.isdir(checkpoint_dir):
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    # Check if this checkpoint was already evaluated (only has eval_results)
    dir_contents = os.listdir(checkpoint_dir)
    if len(dir_contents) == 1 and 'eval_results' in dir_contents:
        eval_results_file = os.path.join(checkpoint_dir, 'eval_results', 'eval_results.json')
        if os.path.exists(eval_results_file):
            raise AlreadyEvaluatedError(
                f"Checkpoint {checkpoint_dir} appears to be already evaluated "
                f"(only contains 'eval_results' with results file). "
                f"The adapter files may have been cleaned up. Skipping evaluation."
            )
    
    adapter_config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    adapter_model_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    
    if not os.path.exists(adapter_config_path):
        # Check if maybe adapter files are in a parent directory (FSDP might save differently)
        parent_dir = os.path.dirname(checkpoint_dir)
        parent_adapter_config = os.path.join(parent_dir, "adapter_config.json")
        
        if os.path.exists(parent_adapter_config):
            print(f"Warning: adapter_config.json not found in {checkpoint_dir}, but found in parent directory.")
            print(f"Using parent directory: {parent_dir}")
            checkpoint_dir = parent_dir
            adapter_config_path = parent_adapter_config
            adapter_model_path = os.path.join(parent_dir, "adapter_model.safetensors")
        else:
            raise ValueError(
                f"PEFT adapter config not found at {adapter_config_path}. "
                f"This checkpoint may not be a valid PEFT checkpoint. "
                f"Files in checkpoint directory: {dir_contents}. "
                f"Expected files: adapter_config.json, adapter_model.safetensors (or adapter_model.bin)"
            )
    
    if not os.path.exists(adapter_model_path):
        # Try alternative name
        adapter_model_path = os.path.join(checkpoint_dir, "adapter_model.bin")
        if not os.path.exists(adapter_model_path):
            raise ValueError(
                f"PEFT adapter weights not found. Expected one of: "
                f"{os.path.join(checkpoint_dir, 'adapter_model.safetensors')} or "
                f"{os.path.join(checkpoint_dir, 'adapter_model.bin')}. "
                f"Files in directory: {os.listdir(checkpoint_dir)}"
            )
    
    print(f"Found PEFT adapter at: {checkpoint_dir}")
    print(f"  - adapter_config.json: {os.path.exists(adapter_config_path)}")
    print(f"  - adapter_model: {os.path.basename(adapter_model_path)}")
    
    print(f"Loading PEFT checkpoint from: {checkpoint_dir}")
    
    # Load PEFT adapter
    # Note: PEFT adapters are small, but the base model should remain split
    # We explicitly pass device_map=None to let PEFT preserve the base model's device_map
    # Use absolute path to ensure PEFT treats it as a local path, not a repo ID
    try:
        model = PeftModel.from_pretrained(
            base_model,
            checkpoint_dir,  # Now an absolute path
            is_trainable=False,
        )
    except ValueError as e:
        if "Can't find 'adapter_config.json'" in str(e):
            raise ValueError(
                f"Failed to load PEFT adapter from {checkpoint_dir}. "
                f"Make sure this is a valid PEFT checkpoint directory containing adapter_config.json. "
                f"Original error: {e}"
            )
        raise
    
    # CRITICAL: After loading PEFT, verify base model is still split
    # PEFT might have moved things around, so we need to check
    if use_multi_gpu and num_gpus > 1:
        print("Verifying model device placement after PEFT loading...")
        
        # Check base model device distribution (should still be split)
        base_devices = set()
        base_param_count = {}
        base_layer_count = {}
        for name, param in model.base_model.named_parameters():
            device = str(param.device)
            base_devices.add(device)
            base_param_count[device] = base_param_count.get(device, 0) + param.numel()
            base_layer_count[device] = base_layer_count.get(device, 0) + 1
        
        # Check adapter device distribution  
        adapter_devices = set()
        adapter_param_count = {}
        for name, param in model.named_parameters():
            if 'lora' in name.lower() or 'adapter' in name.lower():
                device = str(param.device)
                adapter_devices.add(device)
                adapter_param_count[device] = adapter_param_count.get(device, 0) + param.numel()
        
        print(f"Base model devices: {base_devices}")
        if base_param_count:
            total_base = sum(base_param_count.values())
            print("Base model parameter distribution after PEFT:")
            for device, count in sorted(base_param_count.items()):
                pct = (count / total_base * 100) if total_base > 0 else 0
                print(f"  {device}: {count:,} parameters ({pct:.1f}%), {base_layer_count[device]} layers")
        
        if adapter_devices:
            print(f"Adapter devices: {adapter_devices}")
            if adapter_param_count:
                total_adapter = sum(adapter_param_count.values())
                print("Adapter parameter distribution:")
                for device, count in sorted(adapter_param_count.items()):
                    pct = (count / total_adapter * 100) if total_adapter > 0 else 0
                    print(f"  {device}: {count:,} parameters ({pct:.1f}%)")
        
        if len(base_devices) > 1:
            print(f"✓ Base model is still split across {len(base_devices)} GPUs after PEFT loading")
        else:
            print(f"⚠ WARNING: Base model is only on {base_devices} after PEFT loading")
            print("This means device_map='auto' didn't work properly with PEFT")
            print("The model might be too small to split, or PEFT moved it to one device")
            print("This will result in low GPU utilization!")
    
    print("Successfully loaded PEFT checkpoint!")
    model.print_trainable_parameters()
    
    return model


def get_model_batch_size(model_name: str, default_batch_size: int) -> int:
    """Get appropriate batch size based on model size."""
    model_name_lower = model_name.lower()
    
    # Very large models (27B+) - need very small batches even with model parallelism
    # These models take ~85GB per GPU even when split, leaving little room for batch processing
    if 'gemma-2-27b' in model_name_lower or 'gemma-3-27b' in model_name_lower or 'viking-33b' in model_name_lower:
        return min(2, default_batch_size)  # Very small batch for 27B+ models
    # Large models (13B-20B) - can use larger batches with model parallelism
    # Peak memory usage was only ~19GB, so we have ~80GB headroom
    elif 'viking-13b' in model_name_lower or 'llama-2-13b' in model_name_lower:
        return min(8, default_batch_size)  # Increase from 4 to 8 (2x faster)
    # Large models (7B-11B)
    elif 'gemma-7b' in model_name_lower:
        return min(8, default_batch_size)
    elif 'normistral-11b' in model_name_lower:
        return min(6, default_batch_size)  # Increase from 4 to 6
    # Medium models (2-7B)
    elif 'gemma-2b' in model_name_lower or 'viking-7b' in model_name_lower or 'normistral-7b' in model_name_lower:
        return min(16, default_batch_size)
    # Small models
    else:
        return default_batch_size


def evaluate_checkpoint(
    model_name: str,
    checkpoint_dir: str,
    val_dataset_path: str,
    hf_token: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_input_text_tokens: int = MAX_INPUT_TEXT_TOKENS,
    max_extra_prompt_tokens: int = MAX_EXTRA_PROMPT_TOKENS,
    max_output_summary_tokens: int = MAX_OUTPUT_SUMMARY_TOKENS,
    val_batch_size: int = VAL_BATCH_SIZE,
    val_data_size: int = VAL_DATA_SIZE,
    val_beam_size: int = VAL_BEAM_SIZE,
    use_greedy: bool = True,
    use_multi_gpu: bool = False,  # Changed from use_fsdp
    wandb_project: Optional[str] = "lm-evaluation",
    wandb_entity: Optional[str] = None,
    wandb_disabled: bool = False,  # ADD THIS PARAMETER
):
    """Load a PEFT checkpoint and run evaluation with model parallelism support."""
    
    # Auto-enable multi-GPU if multiple GPUs available and not explicitly disabled
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1 and not use_multi_gpu:
        print(f"Auto-enabling model parallelism: {num_gpus} GPUs detected")
        use_multi_gpu = True
    
    is_main_process = True  # No distributed mode, so always main process
    
    # Adjust batch size based on model size
    adjusted_batch_size = get_model_batch_size(model_name, val_batch_size)
    if adjusted_batch_size != val_batch_size:
        print(f"Adjusted batch size from {val_batch_size} to {adjusted_batch_size} for {model_name}")
        val_batch_size = adjusted_batch_size
    
    print(f"Using batch size: {val_batch_size} for evaluation")

    def compute_metrics(eval_pred):
        if is_main_process:
            print('*** evaluation: compute_metrics ***')
        
        # Load ROUGE metric (lazy loading after cache paths are set)
        rouge = evaluate.load("rouge")
        
        preds, labels = eval_pred
        if is_main_process:
            print('*** evaluation: preds ***', preds.shape)
            print('*** evaluation: labels ***', labels.shape)
        
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        vocab_size = tokenizer.vocab_size
        if is_main_process:
            print(f'*** Vocab size: {vocab_size} ***')
        
        preds = np.clip(preds, 0, vocab_size - 1)
        labels = np.clip(labels, 0, vocab_size - 1)
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Clean up decoded predictions - remove special tokens and backslashes
        def clean_text(text):
            """Clean decoded text by removing special tokens and unwanted characters."""
            # Remove common chat format tokens
            text = text.replace('[/INST]', '').replace('[INST]', '')
            text = text.replace('</s>', '').replace('<s>', '')
            # Remove backslashes (common issue with Llama-2 chat models)
            text = text.replace('\\', '')
            # Remove multiple spaces
            text = ' '.join(text.split())
            return text.strip()
        
        decoded_preds = [clean_text(p) for p in decoded_preds]
        decoded_labels = [clean_text(l) for l in decoded_labels]
        
        # Additional strip (keep the existing strip)
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        if len(decoded_preds) > 0 and is_main_process:
            print(f'\n*** Example 1 ***')
            print(f'Prediction: {decoded_preds[0][:200]}...')
            print(f'Reference:  {decoded_labels[0][:200]}...\n')

        scores = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        if is_main_process:
            print('*** evaluation: computed_metrics ***', scores)
        
        # Log to wandb if initialized and not disabled (only on main process)
        if wandb.run is not None and is_main_process and not wandb_disabled:
            wandb.log({
                "eval/rouge1": scores['rouge1'] * 100,
                "eval/rouge2": scores['rouge2'] * 100,
                "eval/rougeL": scores['rougeL'] * 100,
                "eval/rougeLsum": scores['rougeLsum'] * 100,
            })
        
        return {k: v * 100 for k, v in scores.items()}  # % values

    
    # Login to Hugging Face if token is provided
    if hf_token and is_main_process:
        print("Logging in to Hugging Face Hub...")
        login(token=hf_token)
    
    # Synchronize all processes before loading tokenizer
    if use_multi_gpu: # No distributed mode, so no barrier needed
        pass

    # Load tokenizer
    if is_main_process:
        print(f"Loading tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token if hf_token else None
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = 'left'

    # Extract checkpoint step number for run naming
    checkpoint_name = os.path.basename(checkpoint_dir.rstrip('/'))
    checkpoint_step = checkpoint_name.replace('checkpoint-', '') if 'checkpoint-' in checkpoint_name else 'unknown'
    
    # Create a clean model name for display
    clean_model_name = model_name.split('/')[-1].replace('-', '_')
    
    # Only initialize wandb if not disabled and not already initialized
    if wandb_project and not wandb_disabled and wandb.run is None and is_main_process:
        print(f"Initializing Weights & Biases for evaluation...")
        
        # Collect GPU information
        gpu_info = {}
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            gpu_info[f"gpu_{i}_name"] = props.name
            gpu_info[f"gpu_{i}_memory_total_gb"] = props.total_memory / 1e9
        
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=f"{clean_model_name}_ckpt-{checkpoint_step}",
            tags=[
                "evaluation",
                "multi-gpu" if use_multi_gpu else "single-gpu",
                clean_model_name,
                f"checkpoint-{checkpoint_step}",
                os.path.basename(checkpoint_dir).replace('checkpoint-', 'step-')
            ],
            config={
                "model_name": model_name,
                "checkpoint_dir": checkpoint_dir,
                "checkpoint_step": checkpoint_step,
                "val_dataset_path": val_dataset_path,
                "val_data_size": val_data_size,
                "val_batch_size": val_batch_size,
                "val_beam_size": val_beam_size,
                "max_input_text_tokens": max_input_text_tokens,
                "max_output_summary_tokens": max_output_summary_tokens,
                "num_gpus": num_gpus,
                "use_multi_gpu": use_multi_gpu,
                "world_size": 1, # No distributed mode
                "is_distributed": False, # No distributed mode
                **gpu_info,  # ADD GPU INFO
            },
            reinit=True,
        )
        print(f">>> wandb run initialized: {wandb.run.name}")
        print(f">>> wandb run URL: {wandb.run.get_url()}")
    elif wandb_project and not is_main_process:
        # Disable wandb on non-main processes
        os.environ['WANDB_DISABLED'] = 'true'

    # Synchronize before loading model (all ranks need to load)
    if use_multi_gpu: # No distributed mode, so no barrier needed
        pass

    # Load model with PEFT checkpoint
    if is_main_process:
        print("Loading model with PEFT checkpoint...")
    try:
        model = load_model_and_peft_checkpoint(
            model_name, checkpoint_dir, hf_token, 
            use_multi_gpu=use_multi_gpu
        )
    except AlreadyEvaluatedError:
        # Re-raise to be caught by the main block
        raise
    
    # DISABLE gradient checkpointing for evaluation - it's only for training
    # It trades speed for memory, but during inference we want speed
    # Model parallelism already saves memory by splitting the model
    if '13b' in model_name.lower() or '11b' in model_name.lower():
        if is_main_process:
            print("Skipping gradient checkpointing for evaluation (disabled for speed - not needed for inference)...")
        # Do NOT enable - it slows down inference without benefit
        # if hasattr(model, 'gradient_checkpointing_enable'):
        #     model.gradient_checkpointing_enable()
        # elif hasattr(model, 'base_model') and hasattr(model.base_model, 'gradient_checkpointing_enable'):
        #     model.base_model.gradient_checkpointing_enable()

    # Load validation dataset (only on main process, then broadcast)
    if is_main_process:
        print(f"Loading validation dataset from: {val_dataset_path}")
    val_data = []
    with open(val_dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            val_data.append(json.loads(line))

    # Sample validation examples
    val_data = random.sample(val_data, min(val_data_size, len(val_data)))
    
    # Filter out examples with missing input or output
    val_data = [ex for ex in val_data if ex.get('input') and ex.get('output')]
    
    val_df = pd.DataFrame(val_data)
    val_dataset = Dataset.from_pandas(val_df)
    
    if is_main_process:
        print(f'*** validation dataset size: {len(val_dataset)} examples ***')

    def format_example_eval(example):
        """Format evaluation example with model-specific prompt template."""
        model_config = get_model_config_by_hf_name(model_name)
        if model_config:
            prompt = model_config.prompt_config.format_eval(input_text=example['input'])
        else:
            # Fallback to default format
            prompt = f"Oppgave: Oppsummer følgende tekst:\n\n###\n\n{example['input']}\n\n###\n\nOppsummering:\n\n###\n\n"
        
        target = example['output'] if example.get('output') is not None else ""
        return {
            "prompt": prompt,
            "target_summary": str(target)  # Ensure it's a string
        }

    def tokenize_function_eval(examples):
        max_input_prompt_tokens = max_input_text_tokens + max_extra_prompt_tokens
        tokenized_prompts = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=max_input_prompt_tokens,
            padding=False
        )
        tokenized_targets = tokenizer(
            examples["target_summary"],
            truncation=True,
            max_length=max_output_summary_tokens,
            padding=False
        )
        tokenized_prompts["labels"] = tokenized_targets["input_ids"]
        return tokenized_prompts

    formatted_val_dataset = val_dataset.map(format_example_eval)
    tokenized_val_dataset = formatted_val_dataset.map(tokenize_function_eval, batched=True)
    
    eval_data_collator = EvalDataCollator(tokenizer=tokenizer)

    # Set up evaluation-only training args
    if output_dir is None:
        output_dir = os.path.join(checkpoint_dir, "eval_results")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=val_batch_size,
        fp16=False,
        bf16=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,  # Can use workers with model parallelism
        dataloader_prefetch_factor=2,
        report_to="none",
        local_rank=-1,  # Explicitly disable distributed training
    )
    
    # No FSDP/DDP - we use model parallelism (device_map="auto") instead
    print("Using model parallelism (device_map='auto') for multi-GPU evaluation")
    print("This is compatible with model.generate() unlike FSDP")

    # Initialize Trainer for evaluation only
    trainer = CausalLMTrainer(
        generation_max_length=max_output_summary_tokens,
        generation_num_beams=val_beam_size,
        eval_data_collator=eval_data_collator,
        use_greedy=use_greedy,
        model=model,
        args=training_args,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Run evaluation
    if is_main_process:
        print("\n" + "=" * 70)
        print("Running evaluation on checkpoint...")
        print("=" * 70 + "\n")
    
    eval_results = trainer.evaluate()
    
    if is_main_process:
        print("\n" + "=" * 70)
        print("Evaluation Results:")
        print("=" * 70)
        for key, value in eval_results.items():
            print(f"{key}: {value:.4f}")
        print("=" * 70 + "\n")
    
    # Log final summary to wandb if initialized and not disabled
    if wandb_project and wandb.run is not None and is_main_process and not wandb_disabled:
        wandb.summary.update({
            "rouge1": eval_results.get("eval_rouge1", 0),
            "rouge2": eval_results.get("eval_rouge2", 0),
            "rougeL": eval_results.get("eval_rougeL", 0),
            "rougeLsum": eval_results.get("eval_rougeLsum", 0),
        })
        
        wandb.finish()
        print(">>> Evaluation results logged to wandb")
    elif wandb_disabled and is_main_process:
        print(">>> Wandb disabled - skipping wandb logging")
    
    # Save results to file (only on main process)
    if is_main_process:
        results_file = os.path.join(output_dir, "eval_results.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"Results saved to: {results_file}")
    
    # Clean up distributed process group
    if use_multi_gpu: # No distributed mode, so no barrier needed
        pass
    
    return eval_results, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Load PEFT checkpoint from distributed training for multi-GPU evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-GPU evaluation with model parallelism:
  python evaluate_distributed_checkpoints_multigpu.py \\
    --model gemma-7b \\
    --checkpoint_dir models/gemma-7b_fsdp/checkpoint-100 \\
    --val_dataset data/output/processed_data_val.jsonl \\
    --hf_token YOUR_TOKEN \\
    --wandb_project lm-evaluation \\
    --use_multi_gpu

  # Single-GPU fallback:
  python evaluate_distributed_checkpoints_multigpu.py \\
    --model gemma-7b \\
    --checkpoint_dir models/gemma-7b_fsdp/checkpoint-100 \\
    --val_dataset data/output/processed_data_val.jsonl \\
    --hf_token YOUR_TOKEN
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       choices=['viking-7b', 'viking-13b', 'viking-33b',
                                'gemma-2b', 'gemma-7b', 'gemma-2-9b', 'gemma-2-27b',
                                'gemma-3-12b', 'gemma-3-27b',
                                'normistral-7b', 'normistral-11b',
                                'norskgpt-llama3-8b', 'llama-2-13b-chat-norwegian', 'mt5'],
                       help='Base model that was fine-tuned')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Path to PEFT checkpoint directory')
    parser.add_argument('--val_dataset', type=str, default=None,
                       help='Path to validation dataset (JSONL format). Required unless --skip_eval is used.')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for evaluation results (default: checkpoint_dir/eval_results)')
    parser.add_argument('--hf_token', type=str,
                       help='Hugging Face authentication token for private models')
    parser.add_argument('--skip_eval', action='store_true',
                       help='Skip evaluation, only load the model')
    
    # Hyperparameters
    parser.add_argument('--max_input_text_tokens', type=int, default=MAX_INPUT_TEXT_TOKENS,
                       help=f'Maximum tokens for input text (default: {MAX_INPUT_TEXT_TOKENS})')
    parser.add_argument('--max_extra_prompt_tokens', type=int, default=MAX_EXTRA_PROMPT_TOKENS,
                       help=f'Maximum extra tokens for input prompt (default: {MAX_EXTRA_PROMPT_TOKENS})')
    parser.add_argument('--max_output_summary_tokens', type=int, default=MAX_OUTPUT_SUMMARY_TOKENS,
                       help=f'Maximum tokens for output summary (default: {MAX_OUTPUT_SUMMARY_TOKENS})')
    parser.add_argument('--val_batch_size', type=int, default=VAL_BATCH_SIZE,
                       help=f'Validation batch size per device (default: {VAL_BATCH_SIZE})')
    parser.add_argument('--val_data_size', type=int, default=VAL_DATA_SIZE,
                       help=f'Number of examples to use for validation (default: {VAL_DATA_SIZE})')
    parser.add_argument('--val_beam_size', type=int, default=VAL_BEAM_SIZE,
                       help=f'Beam size for validation generation (default: {VAL_BEAM_SIZE})')
    parser.add_argument('--use_greedy', action='store_true',
                       help='Use greedy decoding instead of beam search for faster evaluation')
    parser.add_argument('--use_multi_gpu', action='store_true',
                       help='Use model parallelism (device_map="auto") to split model across multiple GPUs. Compatible with generation.')
    
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='lm-evaluation',
                       help='Wandb project name for evaluation runs (default: lm-evaluation)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Wandb entity/team name (default: uses your default entity)')
    parser.add_argument('--wandb_disabled', action='store_true',
                       help='Disable wandb logging for this evaluation')

    args = parser.parse_args()
    
    # Validate arguments
    if not args.skip_eval and args.val_dataset is None:
        parser.error("--val_dataset is required when evaluation is enabled (use --skip_eval to skip evaluation)")

    # Model mapping from configs
    model_mapping = get_model_name_mapping()
    try:
        model_name = model_mapping[args.model]
    except Exception as e:
        print(f"Error mapping model name: {e}")
        sys.exit(1)

    if args.skip_eval:
        # Just load the model
        print("Loading model without evaluation...")
        if args.hf_token:
            login(token=args.hf_token)
        model = load_model_and_peft_checkpoint(
            model_name, args.checkpoint_dir, args.hf_token, 
            use_multi_gpu=args.use_multi_gpu
        )
        print("Model loaded successfully! Ready for inference.")
    else:
        # Run evaluation
        try:
            evaluate_checkpoint(
                model_name=model_name,
                checkpoint_dir=args.checkpoint_dir,
                val_dataset_path=args.val_dataset,
                hf_token=args.hf_token,
                output_dir=args.output_dir if args.output_dir else os.path.join(args.checkpoint_dir, "eval_results"),
                use_multi_gpu=args.use_multi_gpu,
                wandb_project=args.wandb_project if not args.wandb_disabled else None,
                wandb_entity=args.wandb_entity,
                wandb_disabled=args.wandb_disabled,
            )
        except AlreadyEvaluatedError as e:
            print(f"⚠ SKIPPING: {e}")
            print(f"Checkpoint {args.checkpoint_dir} was already evaluated. Moving to next checkpoint.")
            sys.exit(0)  # Exit with success code so bash loop continues
