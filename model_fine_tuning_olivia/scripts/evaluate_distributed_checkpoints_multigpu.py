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
import shutil
import sys
import time  # ADD THIS for staggered loading
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union, List

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

# Ensure scripts directory is in Python path for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Import model configurations
from model_configs import get_model_config_by_hf_name, get_model_name_mapping, get_doc_type_norwegian

# Import shared utilities
from utils import (
    EvalDataCollator,
    compute_rouge_metrics,
    extract_checkpoint_step,
    get_checkpoint_name_and_step,
    is_major_checkpoint,
    get_model_dir_from_checkpoint,
    get_eval_results_path,
    get_old_eval_results_path,
    load_eval_results,
    save_eval_results,
    update_evaluation_summary,
    load_jsonl_dataset,
    tokenize_eval_examples,
    get_or_create_fixed_nli_subset,
    apply_fixed_subset,
    NLI_FIXED_SUBSET_SIZE,
    format_eval_example,
)

# Import extended evaluation metrics
try:
    from extended_evaluation import extended_evaluate
    EXTENDED_EVAL_AVAILABLE = True
except ImportError as e:
    EXTENDED_EVAL_AVAILABLE = False
    extended_evaluate = None  # type: ignore
    print(f"Warning: extended_evaluation.py could not be imported: {e}")
    print("Only ROUGE metrics will be computed.")

# Helper function to calculate examples from steps
def calculate_examples_from_steps(steps, batch_size, gradient_accumulation_steps, num_gpus):
    """Calculate total number of examples processed given training parameters."""
    if steps is None or steps <= 0:
        return None
    return steps * batch_size * gradient_accumulation_steps * num_gpus

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


def _update_summary_statistics(summary: Dict[str, Any]) -> None:
    """Update summary statistics in the summary dictionary."""
    successful = [c for c in summary["checkpoints"] if c.get("status") == "success"]
    
    if successful:
        # Extract ROUGE scores
        rouge1_scores = [c.get("rouge1", 0) for c in successful if "rouge1" in c]
        rouge2_scores = [c.get("rouge2", 0) for c in successful if "rouge2" in c]
        rougeL_scores = [c.get("rougel", 0) for c in successful if "rougel" in c]
        rougeLsum_scores = [c.get("rougelsum", 0) for c in successful if "rougelsum" in c]
        
        summary["statistics"] = {
            "total_checkpoints": len(summary["checkpoints"]),
            "successful": len(successful),
            "failed": len([c for c in summary["checkpoints"] if c.get("status") == "failed"]),
            "no_results": len([c for c in summary["checkpoints"] if c.get("status") == "no_results_file"]),
            "rouge1": {
                "mean": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0,
                "max": max(rouge1_scores) if rouge1_scores else 0,
                "min": min(rouge1_scores) if rouge1_scores else 0
            },
            "rouge2": {
                "mean": sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0,
                "max": max(rouge2_scores) if rouge2_scores else 0,
                "min": min(rouge2_scores) if rouge2_scores else 0
            },
            "rougeL": {
                "mean": sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0,
                "max": max(rougeL_scores) if rougeL_scores else 0,
                "min": min(rougeL_scores) if rougeL_scores else 0
            },
            "rougeLsum": {
                "mean": sum(rougeLsum_scores) / len(rougeLsum_scores) if rougeLsum_scores else 0,
                "max": max(rougeLsum_scores) if rougeLsum_scores else 0,
                "min": min(rougeLsum_scores) if rougeLsum_scores else 0
            }
        }
        
        # Find best checkpoint by ROUGE-Lsum
        if rougeLsum_scores:
            best_idx = rougeLsum_scores.index(max(rougeLsum_scores))
            summary["best_checkpoint"] = successful[best_idx]["checkpoint"]
            summary["best_rouge_lsum"] = max(rougeLsum_scores)
    else:
        summary["statistics"] = {
            "total_checkpoints": len(summary["checkpoints"]),
            "successful": 0,
            "failed": len([c for c in summary["checkpoints"] if c.get("status") == "failed"]),
            "no_results": len([c for c in summary["checkpoints"] if c.get("status") == "no_results_file"])
        }


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


# EvalDataCollator is now imported from utils.data_collators


class CausalLMTrainer(Trainer):
    def __init__(self, *args, 
                 generation_max_length: Optional[int] = None,
                 generation_num_beams: Optional[int] = None,
                 eval_data_collator: Optional[Any] = None,
                 use_greedy: bool = True,
                 checkpoint_dir: Optional[str] = None,
                 **kwargs) -> None:
        self.generation_max_length = generation_max_length
        self.generation_num_beams = generation_num_beams
        self.eval_data_collator = eval_data_collator
        self.use_greedy = use_greedy
        self.checkpoint_dir = checkpoint_dir  # Store checkpoint directory
        super().__init__(*args, **kwargs)
        self._processing_class = self.tokenizer
        # Store predictions for saving to JSONL
        self._eval_predictions = []
    
    def _move_model_to_device(self, model, device):
        """Override to prevent moving models that are already dispatched with device_map.
        
        When using model parallelism (device_map="auto"), the model is already
        distributed across devices and cannot be moved. This prevents the error:
        "You can't move a model that has some modules offloaded to cpu or disk."
        
        We detect dispatched models and skip device movement, but ensure the
        Trainer's device attribute is set to the primary GPU to avoid overloading.
        """
        # Check if model is already dispatched (has device_map)
        is_dispatched = False
        
        # Check for hf_device_map attribute (accelerate's indicator)
        if hasattr(model, 'hf_device_map') or (hasattr(model, 'base_model') and hasattr(model.base_model, 'hf_device_map')):
            is_dispatched = True
        
        # Also check if model parameters are on multiple devices
        if not is_dispatched:
            devices = set(str(p.device) for p in model.parameters() if p.device.type == 'cuda')
            if len(devices) > 1:
                is_dispatched = True
        
        if is_dispatched:
            # Model is already dispatched - don't try to move it
            print("Model is already dispatched with device_map - skipping device movement")
            
            # Find the primary device (GPU with most parameters) for Trainer operations
            device_counts = {}
            for p in model.parameters():
                if p.device.type == 'cuda':
                    dev_str = str(p.device)
                    device_counts[dev_str] = device_counts.get(dev_str, 0) + p.numel()
            
            if device_counts:
                # Use the GPU with the most parameters as the "primary" device
                primary_device = max(device_counts.items(), key=lambda x: x[1])[0]
                print(f"Using {primary_device} as primary device for Trainer operations")
                # Note: We can't set self.args.device directly (it's read-only)
                # The Trainer will automatically use the correct device for inputs
                # based on where the model parameters are located
            else:
                # Fallback to first available CUDA device
                if torch.cuda.is_available():
                    print("Using cuda:0 as primary device for Trainer operations")
            
            # Return model without moving it - Trainer will handle input device placement
            return model
        
        # For single-device models, use the parent implementation
        return super()._move_model_to_device(model, device)
    
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
        
        # Store predictions for JSONL output
        # Decode predictions (generated summary only, without special tokens)
        decoded_predictions = self._processing_class.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Clean up decoded predictions - remove special tokens and backslashes (same as in compute_metrics)
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
        
        cleaned_predictions = [clean_text(p) for p in decoded_predictions]
        
        # Store for later saving to JSONL (predictions only - inputs/references will come from original dataset)
        self._eval_predictions.extend(cleaned_predictions)
        
        # Monitor GPU memory usage and log peak usage
        # Track peak memory to identify optimal batch size
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            peak_memory = {}
            for i in range(num_gpus):
                max_allocated = torch.cuda.max_memory_allocated(i) / 1e9  # GB
                reserved = torch.cuda.memory_reserved(i) / 1e9  # GB
                total = torch.cuda.get_device_properties(i).total_memory / 1e9  # GB
                peak_memory[f"gpu_{i}_peak_gb"] = max_allocated
                peak_memory[f"gpu_{i}_reserved_gb"] = reserved
                peak_memory[f"gpu_{i}_total_gb"] = total
                peak_memory[f"gpu_{i}_utilization_pct"] = (reserved / total * 100) if total > 0 else 0
                
                # Print warning if getting close to OOM
                if reserved > 80:  # >80GB used out of 102GB
                    print(f"⚠ WARNING: GPU {i} using {reserved:.1f}GB / {total:.1f}GB ({reserved/total*100:.1f}%) - consider reducing batch size")
            
            # Log peak memory to wandb (once per evaluation, not every step)
            if wandb.run is not None and not hasattr(self, '_peak_memory_logged'):
                wandb.log(peak_memory, step=0)  # Log at step 0 for this checkpoint
                self._peak_memory_logged = True

        loss = None
        
        return (loss, generated_ids, labels)


def load_model_and_peft_checkpoint(
    model_name: str,
    checkpoint_dir: str,
    hf_token: Optional[str] = None,
    use_multi_gpu: bool = False,
    major_checkpoint_interval: int = 500,
    include_nli_faithfulness: bool = False,  # Check for missing NLI metrics if True
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
    
    # Extract checkpoint step number and name using utility function
    checkpoint_name, checkpoint_step_int = get_checkpoint_name_and_step(checkpoint_dir)
    if checkpoint_step_int is None:
        checkpoint_step_int = 0
    
    # Determine model directory using utility function
    model_dir = get_model_dir_from_checkpoint(checkpoint_dir)

    # Check if this is a major checkpoint
    is_major = checkpoint_step_int > 0 and checkpoint_step_int % major_checkpoint_interval == 0
    
    # ------------------------------------------------------------------
    # Backup: Regular checkpoints (non-major checkpoints only)
    # ------------------------------------------------------------------
    # For non-major checkpoints, create a copy under model_dir/regular_checkpoints
    # This preserves all evaluated checkpoints even if training later deletes
    # the original checkpoint directories.
    # Major checkpoints are NOT backed up here - they go to major_checkpoints/ only
    # Only backup if backup doesn't already exist (to avoid unnecessary I/O)
    if checkpoint_step_int > 0 and not is_major:
        regular_ckpt_dir = os.path.join(model_dir, "regular_checkpoints")
        os.makedirs(regular_ckpt_dir, exist_ok=True)
        regular_ckpt_name = f"regular-checkpoint-{checkpoint_step_int}"
        regular_ckpt_path = os.path.join(regular_ckpt_dir, regular_ckpt_name)

        # Check if backup already exists and has adapter files
        backup_adapter_file = os.path.join(regular_ckpt_path, "adapter_model.safetensors")
        if os.path.exists(regular_ckpt_path) and os.path.exists(backup_adapter_file):
            print(f"✓ Regular checkpoint backup already exists: {regular_ckpt_path} (skipping backup)")
        else:
            # Remove existing incomplete copy if it exists
            if os.path.exists(regular_ckpt_path):
                print(f"Removing incomplete regular checkpoint copy: {regular_ckpt_path}")
                try:
                    shutil.rmtree(regular_ckpt_path)
                except Exception as e:
                    print(f"⚠ Warning: Failed to remove incomplete backup: {e}")

            print(f"Copying regular checkpoint to: {regular_ckpt_path}")
            try:
                shutil.copytree(checkpoint_dir, regular_ckpt_path)
                print(f"✓ Successfully copied regular checkpoint to {regular_ckpt_path}")
            except Exception as e:
                print(f"⚠ Warning: Failed to copy regular checkpoint: {e}")
                print("   Continuing with evaluation, but regular checkpoint copy was not created.")

    # ------------------------------------------------------------------
    # Backup: Major checkpoints (only in major_checkpoints, not in regular_checkpoints)
    # ------------------------------------------------------------------
    # For major checkpoints, create a copy under model_dir/major_checkpoints
    # Format: major-checkpoint-nnn (copy of checkpoint-nnn)
    # Major checkpoints are NOT backed up to regular_checkpoints/ to save space
    # Only backup if backup doesn't already exist (to avoid unnecessary I/O)
    if is_major:
        major_ckpt_dir = os.path.join(model_dir, "major_checkpoints")
        os.makedirs(major_ckpt_dir, exist_ok=True)
        major_ckpt_name = f"major-checkpoint-{checkpoint_step_int}"
        major_ckpt_path = os.path.join(major_ckpt_dir, major_ckpt_name)
        
        # Check if backup already exists and has adapter files
        backup_adapter_file = os.path.join(major_ckpt_path, "adapter_model.safetensors")
        if os.path.exists(major_ckpt_path) and os.path.exists(backup_adapter_file):
            print(f"✓ Major checkpoint backup already exists: {major_ckpt_path} (skipping backup)")
        else:
            # Remove existing incomplete copy if it exists
            if os.path.exists(major_ckpt_path):
                print(f"Removing incomplete major checkpoint copy: {major_ckpt_path}")
                try:
                    shutil.rmtree(major_ckpt_path)
                except Exception as e:
                    print(f"⚠ Warning: Failed to remove incomplete backup: {e}")
            
            # Copy checkpoint to major_checkpoints directory
            # This ensures major checkpoints persist even if original checkpoints are deleted
            print(f"Copying major checkpoint to: {major_ckpt_path}")
            try:
                shutil.copytree(checkpoint_dir, major_ckpt_path)
                print(f"✓ Successfully copied major checkpoint to {major_ckpt_path}")
            except Exception as e:
                print(f"⚠ Warning: Failed to copy major checkpoint: {e}")
                print("   Continuing with evaluation, but major checkpoint copy was not created.")
    
    # Check if this checkpoint was already evaluated using utility functions
    eval_results_file = get_eval_results_path(checkpoint_dir, model_dir)
    old_eval_results_file = get_old_eval_results_path(checkpoint_dir)
    
    # Check if results exist in new location
    if os.path.exists(eval_results_file):
        # Check if extended metrics are missing (if extended evaluation is available)
        # If extended metrics should be computed but aren't present, allow re-evaluation
        if EXTENDED_EVAL_AVAILABLE:
            try:
                existing_results = load_eval_results(checkpoint_dir, model_dir)
                if existing_results is None:
                    # Can't parse, allow re-evaluation
                    print(f"⚠ Warning: Could not parse existing results file. Re-evaluating...")
                    # Don't raise AlreadyEvaluatedError - allow re-evaluation
                else:
                    # Check if this is a major checkpoint that should have BERTScore
                    is_major = is_major_checkpoint(checkpoint_step_int, major_checkpoint_interval)
                    
                    # Check if extended metrics are missing
                    has_extended_metrics = any(
                        key.startswith("eval_reference_") or 
                        key.startswith("eval_hygiene_") or 
                        key.startswith("eval_faithfulness_") or
                        key == "eval_faithfulness"
                        for key in existing_results.keys()
                    )
                    
                    # If it's a major checkpoint and should have BERTScore but doesn't, re-evaluate
                    if is_major and "eval_reference_bertscore_f1_mean" not in existing_results:
                        print(f"⚠ Checkpoint {checkpoint_name} was evaluated but missing BERTScore (major checkpoint).")
                        print(f"   Re-evaluating to compute extended metrics...")
                        # Don't raise AlreadyEvaluatedError - allow re-evaluation
                    elif not has_extended_metrics:
                        print(f"⚠ Checkpoint {checkpoint_name} was evaluated but missing extended metrics.")
                        print(f"   Re-evaluating to compute extended metrics...")
                        # Don't raise AlreadyEvaluatedError - allow re-evaluation
                    # Check if NLI faithfulness is requested but missing
                    elif include_nli_faithfulness:
                        has_nli_metrics = (
                            any(key.startswith("eval_faithfulness_") for key in existing_results.keys()) or
                            ("eval_faithfulness" in existing_results and existing_results.get("eval_faithfulness") is not None)
                        )
                        # Also check for eval_faithfulness key (could be set to null)
                        has_nli_results = (
                            has_nli_metrics or 
                            ("eval_faithfulness" in existing_results and existing_results.get("eval_faithfulness") is not None)
                        )
                        if not has_nli_results:
                            print(f"⚠ Checkpoint {checkpoint_name} was evaluated but missing NLI faithfulness metrics (--include_nli_faithfulness was requested).")
                            print(f"   Re-evaluating to compute NLI metrics...")
                            # Don't raise AlreadyEvaluatedError - allow re-evaluation
                        else:
                            # All metrics present, skip evaluation
                            raise AlreadyEvaluatedError(
                                f"Checkpoint {checkpoint_dir} appears to be already evaluated "
                                f"(results file exists at {eval_results_file}). "
                                f"Skipping evaluation."
                            )
                    else:
                        # Extended metrics are present, skip evaluation
                        raise AlreadyEvaluatedError(
                            f"Checkpoint {checkpoint_dir} appears to be already evaluated "
                            f"(results file exists at {eval_results_file}). "
                            f"Skipping evaluation."
                        )
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # If we can't parse the file, allow re-evaluation
                print(f"⚠ Warning: Could not parse existing results file: {e}")
                print(f"   Re-evaluating checkpoint {checkpoint_name}...")
                # Don't raise AlreadyEvaluatedError - allow re-evaluation
        else:
            # Extended evaluation not available, skip if results exist
            raise AlreadyEvaluatedError(
                f"Checkpoint {checkpoint_dir} appears to be already evaluated "
                f"(results file exists at {eval_results_file}). "
                f"Skipping evaluation."
            )
    
    # Also check old location for backwards compatibility
    if os.path.exists(old_eval_results_file):
        # Check if extended metrics are missing (same logic as above)
        if EXTENDED_EVAL_AVAILABLE:
            try:
                existing_results_old = load_eval_results(checkpoint_dir, model_dir)
                if existing_results_old is not None:
                    is_major_old = is_major_checkpoint(checkpoint_step_int, major_checkpoint_interval)
                    has_extended_metrics_old = any(
                        key.startswith("eval_reference_") or 
                        key.startswith("eval_hygiene_") or 
                        key.startswith("eval_faithfulness_") or
                        key == "eval_faithfulness"
                        for key in existing_results_old.keys()
                    )
                    
                    if is_major_old and "eval_reference_bertscore_f1_mean" not in existing_results_old:
                        print(f"⚠ Checkpoint {checkpoint_name} was evaluated but missing BERTScore (major checkpoint).")
                        print(f"   Re-evaluating to compute extended metrics...")
                        # Don't raise AlreadyEvaluatedError - allow re-evaluation
                    elif not has_extended_metrics_old:
                        print(f"⚠ Checkpoint {checkpoint_name} was evaluated but missing extended metrics.")
                        print(f"   Re-evaluating to compute extended metrics...")
                        # Don't raise AlreadyEvaluatedError - allow re-evaluation
                    # Check if NLI faithfulness is requested but missing
                    elif include_nli_faithfulness:
                        has_nli_metrics_old = (
                            any(key.startswith("eval_faithfulness_") for key in existing_results_old.keys()) or
                            ("eval_faithfulness" in existing_results_old and existing_results_old.get("eval_faithfulness") is not None)
                        )
                        # Also check for eval_faithfulness key (could be set to null)
                        has_nli_results_old = (
                            has_nli_metrics_old or 
                            ("eval_faithfulness" in existing_results_old and existing_results_old.get("eval_faithfulness") is not None)
                        )
                        if not has_nli_results_old:
                            print(f"⚠ Checkpoint {checkpoint_name} was evaluated but missing NLI faithfulness metrics (--include_nli_faithfulness was requested).")
                            print(f"   Re-evaluating to compute NLI metrics...")
                            # Don't raise AlreadyEvaluatedError - allow re-evaluation
                        else:
                            # All metrics present, skip evaluation
                            raise AlreadyEvaluatedError(
                                f"Checkpoint {checkpoint_dir} appears to be already evaluated "
                                f"(old results file exists at {old_eval_results_file}). "
                                f"Skipping evaluation."
                            )
                    else:
                        raise AlreadyEvaluatedError(
                            f"Checkpoint {checkpoint_dir} appears to be already evaluated "
                            f"(old results file exists at {old_eval_results_file}). "
                            f"Skipping evaluation."
                        )
                else:
                    # Can't parse, allow re-evaluation
                    print(f"⚠ Warning: Could not parse existing results file. Re-evaluating...")
                    # Don't raise AlreadyEvaluatedError - allow re-evaluation
            except (json.JSONDecodeError, ValueError, KeyError):
                # If we can't parse, allow re-evaluation
                print(f"⚠ Warning: Could not parse existing results file. Re-evaluating...")
                # Don't raise AlreadyEvaluatedError - allow re-evaluation
        else:
            raise AlreadyEvaluatedError(
                f"Checkpoint {checkpoint_dir} appears to be already evaluated "
                f"(old results file exists at {old_eval_results_file}). "
                f"Skipping evaluation."
            )
    
    # Also check if checkpoint directory only contains eval_results (old structure)
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
    except (ValueError, TypeError) as e:
        error_str = str(e)
        # Handle corrupted adapter_config.json files (e.g., typos like 'corda_config' instead of 'lora_config')
        if "Can't find 'adapter_config.json'" in error_str:
            raise ValueError(
                f"Failed to load PEFT adapter from {checkpoint_dir}. "
                f"Make sure this is a valid PEFT checkpoint directory containing adapter_config.json. "
                f"Original error: {e}"
            )
        elif "unexpected keyword argument" in error_str and ("corda_config" in error_str or "adapter_config" in error_str):
            # Try to fix corrupted adapter_config.json
            print(f"⚠ Warning: Detected corrupted adapter_config.json with error: {e}")
            print(f"Attempting to fix adapter_config.json...")
            try:
                with open(adapter_config_path, 'r') as f:
                    config = json.load(f)
                
                # Fix common typos - remove invalid keys
                fixed = False
                if 'corda_config' in config:
                    print(f"  Fixing typo: removing 'corda_config' (invalid key)")
                    del config['corda_config']
                    fixed = True
                
                if fixed:
                    # Save fixed config
                    backup_path = adapter_config_path + '.backup'
                    shutil.copy2(adapter_config_path, backup_path)
                    print(f"  Created backup: {backup_path}")
                    
                    with open(adapter_config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    print(f"  Fixed adapter_config.json, retrying load...")
                    
                    # Retry loading
                    model = PeftModel.from_pretrained(
                        base_model,
                        checkpoint_dir,
                        is_trainable=False,
                    )
                else:
                    # Couldn't fix it automatically
                    raise ValueError(
                        f"Failed to load PEFT adapter from {checkpoint_dir}. "
                        f"The adapter_config.json appears to be corrupted with error: {e}. "
                        f"Please check the adapter_config.json file manually."
                    )
            except Exception as fix_error:
                raise ValueError(
                    f"Failed to load PEFT adapter from {checkpoint_dir}. "
                    f"The adapter_config.json appears to be corrupted. "
                    f"Original error: {e}. "
                    f"Fix attempt error: {fix_error}"
                )
        else:
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
    # Large models (12B-13B) - can use larger batches with model parallelism
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


def check_gpu_memory_utilization(num_gpus: Optional[int] = None) -> Dict[str, Any]:
    """Check current GPU memory utilization and return statistics.
    
    Returns:
        Dictionary with memory stats per GPU and recommendations
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        return {"error": "No GPUs available"}
    
    # Ensure num_gpus is an int for range()
    num_gpus_int = int(num_gpus) if num_gpus is not None else 0
    if num_gpus_int == 0:
        return {"error": "No GPUs available"}
    
    memory_stats = {
        "gpus": [],
        "total_memory_gb": 0,
        "total_allocated_gb": 0,
        "total_reserved_gb": 0,
        "total_free_gb": 0,
        "utilization_pct": 0.0,
    }
    
    for i in range(num_gpus_int):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1e9  # GB
        allocated = torch.cuda.memory_allocated(i) / 1e9  # GB
        reserved = torch.cuda.memory_reserved(i) / 1e9  # GB
        free = total_memory - reserved
        
        utilization = (reserved / total_memory * 100) if total_memory > 0 else 0
        
        memory_stats["gpus"].append({
            "gpu_id": i,
            "name": props.name,
            "total_gb": total_memory,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": free,
            "utilization_pct": utilization,
        })
        
        memory_stats["total_memory_gb"] += total_memory
        memory_stats["total_allocated_gb"] += allocated
        memory_stats["total_reserved_gb"] += reserved
        memory_stats["total_free_gb"] += free
    
    if memory_stats["total_memory_gb"] > 0:
        memory_stats["utilization_pct"] = (memory_stats["total_reserved_gb"] / memory_stats["total_memory_gb"] * 100)
    
    # Add recommendations
    avg_utilization = memory_stats["utilization_pct"]
    avg_free = memory_stats["total_free_gb"] / num_gpus_int if num_gpus_int > 0 else 0
    
    recommendations = []
    if avg_utilization < 50:
        recommendations.append(f"Low GPU utilization ({avg_utilization:.1f}%) - consider increasing batch size")
        if avg_free > 20:
            recommendations.append(f"High free memory ({avg_free:.1f}GB per GPU) - batch size can likely be increased")
    elif avg_utilization > 90:
        recommendations.append(f"High GPU utilization ({avg_utilization:.1f}%) - consider decreasing batch size")
    elif avg_utilization > 80:
        recommendations.append(f"Moderate-high GPU utilization ({avg_utilization:.1f}%) - monitor for OOM")
    
    memory_stats["recommendations"] = recommendations
    
    return memory_stats


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
    use_multi_gpu: bool = False,
    wandb_project: Optional[str] = "lm-evaluation",
    wandb_entity: Optional[str] = None,
    wandb_disabled: bool = False,
    wandb_run_name: Optional[str] = None,
    wandb_group: Optional[str] = None,
    major_checkpoint_interval: int = 500,  # Every Nth step is major (gets BERTScore). Default: 500 (every 500 steps = checkpoint-500, checkpoint-1000, etc.)
    include_nli_faithfulness: bool = False,  # Enable NLI faithfulness evaluation (uses fixed 500-example subset for consistency)
):
    """Load a PEFT checkpoint and run evaluation with model parallelism support."""
    
    # Convert checkpoint_dir to absolute path
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    
    # Extract checkpoint step number early (needed for checking existing results)
    checkpoint_name = os.path.basename(checkpoint_dir.rstrip('/'))
    checkpoint_step = checkpoint_name.replace('checkpoint-', '') if 'checkpoint-' in checkpoint_name else 'unknown'
    
    try:
        checkpoint_step_int = int(checkpoint_step)
    except ValueError:
        checkpoint_step_int = 0
    
    # Determine if this is a "major" checkpoint for tiered evaluation
    # Major checkpoints: every Nth checkpoint (based on major_checkpoint_interval)
    # Normal checkpoints: all others
    # This allows selective computation of expensive metrics (BERTScore, NLI)
    # Tiered evaluation strategy:
    #   - Normal checkpoints: ROUGE + Hygiene only (~2 min)
    #   - Major checkpoints: ROUGE + Hygiene + BERTScore (~3-4 min)
    #   - NLI Faithfulness: Optional, controlled by include_nli_faithfulness parameter
    is_major_checkpoint_bool = is_major_checkpoint(checkpoint_step_int, major_checkpoint_interval)
    
    # Debug: Print NLI flag status
    print(f"\n{'='*70}")
    print(f"NLI FAITHFULNESS CONFIGURATION:")
    print(f"{'='*70}")
    print(f"  include_nli_faithfulness parameter: {include_nli_faithfulness}")
    print(f"  NLI fixed subset size: {NLI_FIXED_SUBSET_SIZE} examples (consistent across all checkpoints)")
    print(f"  EXTENDED_EVAL_AVAILABLE: {EXTENDED_EVAL_AVAILABLE}")
    print(f"{'='*70}\n")
    
    # Default training parameters for example calculation (may vary - check training config for exact values)
    DEFAULT_TRAIN_BATCH_SIZE = 4
    DEFAULT_GRADIENT_ACCUMULATION = 4
    DEFAULT_TRAIN_NUM_GPUS = 2  # Typical for multi-node training
    
    # Calculate estimated examples from checkpoint step
    estimated_examples = calculate_examples_from_steps(
        checkpoint_step_int, DEFAULT_TRAIN_BATCH_SIZE, DEFAULT_GRADIENT_ACCUMULATION, DEFAULT_TRAIN_NUM_GPUS
    ) if checkpoint_step_int > 0 else None
    
    # Determine output directory: save to model/all_eval_results/checkpoint-nnn-eval-results.json
    # Get model directory first (needed for results_file path)
    model_dir_eval = get_model_dir_from_checkpoint(checkpoint_dir)
    
    if output_dir is None:
        # Create all_eval_results directory in model directory
        output_dir = os.path.join(model_dir_eval, "all_eval_results")
    else:
        output_dir = os.path.abspath(output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get results file path using utility function
    results_file = get_eval_results_path(checkpoint_dir, model_dir_eval)
    
    # If results file exists, load and log to Wandb without re-evaluating
    if os.path.exists(results_file):
        print(f"⚠ Checkpoint {checkpoint_name} already evaluated. Loading existing results...")
        
        try:
            existing_results = load_eval_results(checkpoint_dir, model_dir_eval)
            if existing_results is None:
                print(f"⚠ Warning: Could not load existing results from {results_file}")
                # Fall through to normal evaluation
            else:
                print(f"✓ Loaded existing results from {results_file}")
                
                # Extract ROUGE scores (handle both eval_rouge* and rouge* keys)
                rouge1 = existing_results.get("eval_rouge1", existing_results.get("rouge1", 0))
                rouge2 = existing_results.get("eval_rouge2", existing_results.get("rouge2", 0))
                rougeL = existing_results.get("eval_rougeL", existing_results.get("rougeL", 0))
                rougeLsum = existing_results.get("eval_rougeLsum", existing_results.get("rougeLsum", 0))
                
                # Check if all ROUGE scores are zero (indicates failed evaluation)
                all_zeros = (rouge1 == 0.0 and rouge2 == 0.0 and rougeL == 0.0 and rougeLsum == 0.0)
                
                # Check if extended metrics are missing (if extended evaluation is available)
                missing_extended_metrics = False
                if EXTENDED_EVAL_AVAILABLE:
                    has_extended_metrics = any(
                        key.startswith("eval_reference_") or 
                        key.startswith("eval_hygiene_") or 
                        key.startswith("eval_faithfulness_") or
                        key == "eval_faithfulness"
                        for key in existing_results.keys()
                    )
                    
                    is_major_val = is_major_checkpoint(checkpoint_step_int, major_checkpoint_interval)
                    if is_major_val and "eval_reference_bertscore_f1_mean" not in existing_results:
                        missing_extended_metrics = True
                        print(f"⚠ Checkpoint {checkpoint_name} missing BERTScore (major checkpoint). Re-evaluating...")
                    elif not has_extended_metrics:
                        missing_extended_metrics = True
                        print(f"⚠ Checkpoint {checkpoint_name} missing extended metrics. Re-evaluating...")
                    
                    # Check if NLI faithfulness is requested but missing (separate check, not elif)
                    # Always check NLI if requested, regardless of other extended metrics status
                    if include_nli_faithfulness:
                        has_nli_metrics = (
                            any(key.startswith("eval_faithfulness_") for key in existing_results.keys()) or
                            ("eval_faithfulness" in existing_results and existing_results.get("eval_faithfulness") is not None)
                        )
                        # Also check for eval_faithfulness key (could be set to null)
                        has_nli_results = (
                            has_nli_metrics or 
                            ("eval_faithfulness" in existing_results and existing_results.get("eval_faithfulness") is not None)
                        )
                        if not has_nli_results:
                            missing_extended_metrics = True
                            print(f"⚠ Checkpoint {checkpoint_name} missing NLI faithfulness metrics (--include_nli_faithfulness was requested). Re-evaluating...")
                
                if all_zeros or missing_extended_metrics:
                    if all_zeros:
                        print(f"⚠ Warning: All ROUGE scores are 0.00 - this indicates a failed evaluation.")
                        print(f"   Re-evaluating checkpoint {checkpoint_name}...")
                    # Fall through to normal evaluation instead of returning
                else:
                    # Initialize Wandb if needed (even for already-evaluated checkpoints)
                    is_main_process_cached = True
                    if wandb_project and not wandb_disabled and wandb.run is None and is_main_process_cached:
                        print(f"Initializing Weights & Biases to log existing results...")
                    
                    # Create a clean model name for display
                    clean_model_name = model_name.split('/')[-1].replace('-', '_')
                    
                    # Collect GPU information
                    gpu_info = {}
                    num_gpus = torch.cuda.device_count()
                    for i in range(num_gpus):
                        props = torch.cuda.get_device_properties(i)
                        gpu_info[f"gpu_{i}_name"] = props.name
                        gpu_info[f"gpu_{i}_memory_total_gb"] = props.total_memory / 1e9
                    
                        # Use provided run_name or create one based on model and checkpoint
                        # Include checkpoint step and type (major/normal) in run name for better identification
                        if wandb_run_name:
                            run_name = wandb_run_name
                            is_major_cached = is_major_checkpoint(checkpoint_step_int, major_checkpoint_interval)
                        else:
                            # Default: include checkpoint step and type for uniqueness
                            is_major_cached = is_major_checkpoint(checkpoint_step_int, major_checkpoint_interval)
                            checkpoint_type = "major" if is_major_cached else "normal"
                            run_name = f"{clean_model_name}_eval_{checkpoint_type}-{checkpoint_step_int}"
                        
                        wandb.init(
                        project=wandb_project,
                        entity=wandb_entity,
                        name=run_name,
                        group=wandb_group or clean_model_name,
                        tags=[
                            "evaluation",
                            "cached",  # Indicate this was loaded from cache
                            "major" if is_major_cached else "normal",
                        ],
                        config={
                            "model": model_name,
                            "checkpoint_step": checkpoint_step_int,
                            "checkpoint_type": "major" if is_major_cached else "normal",
                            "val_dataset": val_dataset_path,
                            "val_size": val_data_size,
                            "val_batch_size": val_batch_size,
                            "max_input_tokens": max_input_text_tokens,
                            "max_output_tokens": max_output_summary_tokens,
                            "num_gpus": num_gpus,
                        },
                        reinit=True,
                    )
                    print(f">>> wandb run initialized: {wandb.run.name}")
                    print(f">>> wandb run URL: {wandb.run.get_url()}")
                
                    # Log existing results to Wandb
                    if wandb.run is not None and is_main_process_cached and not wandb_disabled:
                        # Log metrics with checkpoint step as the x-axis
                        wandb.log({
                        "rouge1": rouge1,
                        "rouge2": rouge2,
                        "rougeL": rougeL,
                        "rougeLsum": rougeLsum,
                    }, step=checkpoint_step_int)
                    
                    print(f">>> Logged existing results to wandb at step {checkpoint_step_int}")
                    print(f"   ROUGE-1: {rouge1:.2f}, ROUGE-2: {rouge2:.2f}, ROUGE-L: {rougeL:.2f}, ROUGE-Lsum: {rougeLsum:.2f}")
                    
                    # Update evaluation summary JSON file even for cached results
                    summary_file = os.path.join(output_dir, "evaluation_summary.json")
                    
                    # Load existing summary or create new one
                    if os.path.exists(summary_file):
                        with open(summary_file, 'r') as f:
                            summary = json.load(f)
                    else:
                        # Initialize new summary
                        summary = {
                            "model": model_name,
                            "checkpoint_base_dir": os.path.dirname(checkpoint_dir.rstrip('/')),
                            "val_dataset": val_dataset_path,
                            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                            "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "checkpoints": []
                        }
                    
                    # Create checkpoint entry with all metrics from existing results
                    checkpoint_entry = {
                        "checkpoint": checkpoint_name,
                        "checkpoint_number": checkpoint_step_int,
                        "status": "success",
                        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "result_file": results_file,
                        "from_cache": True
                    }
                    
                    # Add all metrics from existing_results
                    for key, value in existing_results.items():
                        normalized_key = key.replace("eval_", "").lower()
                        if isinstance(value, (int, float)):
                            checkpoint_entry[normalized_key] = float(value)
                        elif isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, (int, float)):
                                    checkpoint_entry[f"{normalized_key}_{sub_key}"] = float(sub_value)
                    
                    # Remove existing entry for this checkpoint if it exists, then add new one
                    summary["checkpoints"] = [c for c in summary["checkpoints"] if c.get("checkpoint") != checkpoint_name]
                    summary["checkpoints"].append(checkpoint_entry)
                    
                    # Sort and update statistics (same logic as in main save section)
                    summary["checkpoints"].sort(key=lambda x: x.get("checkpoint_number", 0))
                    _update_summary_statistics(summary)
                    
                    # Save updated summary
                    with open(summary_file, 'w') as f:
                        json.dump(summary, f, indent=2)
                    print(f"Evaluation summary updated: {summary_file}")
                    
                    # Return existing results (no model needed since we're not evaluating)
                    return existing_results, None
            
        except Exception as e:
            print(f"⚠ Warning: Failed to load existing results from {results_file}: {e}")
            print("   Proceeding with full evaluation...")
            # Fall through to normal evaluation
    
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
    
    # Check GPU memory utilization before evaluation
    if is_main_process and torch.cuda.is_available():
        print("\n" + "=" * 70)
        print("GPU MEMORY UTILIZATION CHECK")
        print("=" * 70)
        memory_stats = check_gpu_memory_utilization(num_gpus)
        for gpu_info in memory_stats.get("gpus", []):
            print(f"GPU {gpu_info['gpu_id']}: {gpu_info['name']}")
            print(f"  Total: {gpu_info['total_gb']:.1f} GB")
            print(f"  Reserved: {gpu_info['reserved_gb']:.1f} GB ({gpu_info['utilization_pct']:.1f}%)")
            print(f"  Free: {gpu_info['free_gb']:.1f} GB")
        
        if memory_stats.get("recommendations"):
            print("\nRecommendations:")
            for rec in memory_stats["recommendations"]:
                print(f"  • {rec}")
        
        print(f"\nUsing batch size: {val_batch_size} for evaluation")
        print("=" * 70 + "\n")
    else:
        print(f"Using batch size: {val_batch_size} for evaluation")

    def compute_metrics(eval_pred):
        """Compute ROUGE metrics using shared utility function."""
        return compute_rouge_metrics(
            eval_pred=eval_pred,
            tokenizer=tokenizer,
            log_to_wandb=True,
            step=checkpoint_step_int,  # Use checkpoint step as x-axis
            is_main_process=is_main_process,
            verbose=True
        )

    
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
        
        # Use provided run_name or create one based on model and checkpoint
        # Include checkpoint step and type (major/normal) in run name for better identification
        is_major_eval = is_major_checkpoint(checkpoint_step_int, major_checkpoint_interval)
        if wandb_run_name:
            run_name = wandb_run_name
        else:
            # Default: include checkpoint step and type for uniqueness
            checkpoint_type = "major" if is_major_eval else "normal"
            run_name = f"{clean_model_name}_eval_{checkpoint_type}-{checkpoint_step_int}"
        
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            group=wandb_group or clean_model_name,  # Group all checkpoints together
            tags=[
                "evaluation",
                "major" if is_major_eval else "normal",
            ],
            config={
                "model": model_name,
                "checkpoint_step": checkpoint_step_int,
                "checkpoint_type": "major" if is_major_eval else "normal",
                "val_dataset": val_dataset_path,
                "val_size": val_data_size,
                "val_batch_size": val_batch_size,
                "max_input_tokens": max_input_text_tokens,
                "max_output_tokens": max_output_summary_tokens,
                "num_gpus": num_gpus,
                "major_checkpoint_interval": major_checkpoint_interval,
                "metrics": {
                    "rouge": True,
                    "hygiene": True,
                    "bertscore": is_major_checkpoint,
                    "faithfulness": include_nli_faithfulness,
                },
            },
            reinit=True,
        )
        print(f">>> wandb run initialized: {wandb.run.name}")
        print(f">>> wandb run URL: {wandb.run.get_url()}")
        
        # Display estimated examples from checkpoint step
        if estimated_examples:
            print(f"\n{'='*70}")
            print(f"CHECKPOINT TRAINING INFO:")
            print(f"{'='*70}")
            print(f"  Checkpoint step: {checkpoint_step_int:,}")
            print(f"  Estimated examples (using defaults: batch=4, grad_acc=4, gpus=2): {estimated_examples:,} ({estimated_examples/1000:.1f}k)")
            print(f"  Note: Actual values may vary - check training config for exact parameters")
            print(f"{'='*70}\n")
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
            use_multi_gpu=use_multi_gpu,
            major_checkpoint_interval=major_checkpoint_interval,
            include_nli_faithfulness=include_nli_faithfulness
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
    
    # Load validation dataset using shared utility
    val_data = load_jsonl_dataset(val_dataset_path, dataset_type="validation", raise_on_error=True)
    if val_data is None:
        raise ValueError(f"Failed to load validation dataset from {val_dataset_path}")
    
    if is_main_process:
        print(f"Successfully loaded {len(val_data)} validation examples")

    # Sample validation examples
    val_data = random.sample(val_data, min(val_data_size, len(val_data)))
    
    # Filter out examples with missing input or output
    val_data = [ex for ex in val_data if ex.get('input') and ex.get('output')]
    
    val_df = pd.DataFrame(val_data)
    val_dataset = Dataset.from_pandas(val_df)
    
    if is_main_process:
        print(f'*** validation dataset size: {len(val_dataset)} examples ***')

    # Use shared formatting and tokenization functions
    def format_example_eval_wrapper(example):
        """Wrapper to call shared format_eval_example with model_name."""
        return format_eval_example(example, model_name)
    
    def tokenize_function_eval_wrapper(examples):
        """Wrapper to call shared tokenize_eval_examples with tokenizer and config."""
        return tokenize_eval_examples(
            examples=examples,
            tokenizer=tokenizer,
            max_input_text_tokens=max_input_text_tokens,
            max_extra_prompt_tokens=max_extra_prompt_tokens,
            max_output_summary_tokens=max_output_summary_tokens
        )

    formatted_val_dataset = val_dataset.map(format_example_eval_wrapper)
    
    # Log example prompts to wandb (lightweight - just a few examples to verify prompt formatting)
    if is_main_process and wandb.run is not None:
        # Collect example prompts with different doc_types (only first 3 examples to keep it lightweight)
        example_prompts = []
        doc_types_seen = set()
        
        for i in range(min(3, len(formatted_val_dataset))):
            example = formatted_val_dataset[i]
            original_example = val_data[i] if i < len(val_data) else {}
            
            # Extract doc_type
            doc_type = None
            if 'metadata' in original_example and isinstance(original_example['metadata'], dict):
                doc_type = original_example['metadata'].get('doc_type')
            
            doc_type_nor = get_doc_type_norwegian(doc_type) if doc_type else "tekst"
            doc_types_seen.add(doc_type_nor)
            
            # Get prompt
            prompt = example.get('prompt', '')
            prompt_preview = prompt[:300] + "..." if len(prompt) > 300 else prompt
            
            example_prompts.append({
                "example_num": i + 1,
                "doc_type": doc_type or "unknown",
                "doc_type_norwegian": doc_type_nor,
                "prompt_preview": prompt_preview
            })
        
        # Log doc types to wandb config (lightweight metadata)
        # Merge with existing doc_types if present, otherwise set new value
        # This prevents WandB errors when doc_types differ between checkpoints
        existing_doc_types = set(wandb.config.get("doc_types", []))
        all_doc_types = sorted(list(existing_doc_types.union(doc_types_seen)))
        
        if all_doc_types != wandb.config.get("doc_types", []):
            wandb.config.update({
                "doc_types": all_doc_types,
            }, allow_val_change=True)
        
        # Print to console (lightweight - just once)
        print("\n" + "=" * 70)
        print("EVALUATION PROMPT EXAMPLES (logged to wandb config):")
        print("=" * 70)
        for ex in example_prompts:
            print(f"\nExample {ex['example_num']}:")
            print(f"  Doc Type: {ex['doc_type']} -> {ex['doc_type_norwegian']}")
            print(f"  Prompt Preview: {ex['prompt_preview']}")
        print("=" * 70 + "\n")
    
    tokenized_val_dataset = formatted_val_dataset.map(tokenize_function_eval_wrapper, batched=True)
    
    # Store original examples for JSONL output (before tokenization)
    # We need both the raw input text and the formatted prompt
    original_examples_for_jsonl = []
    for i, example in enumerate(formatted_val_dataset):
        # Get the original raw input from the original dataset (before formatting)
        raw_input = val_data[i].get('input', '') if i < len(val_data) else ''
        original_examples_for_jsonl.append({
            "input_text": raw_input,  # Raw input text (human-readable)
            "prompt": example.get("prompt", ""),  # Formatted prompt with template (human-readable)
            "reference": example.get("target_summary", "")  # Target summary (human-readable)
        })
    
    eval_data_collator = EvalDataCollator(tokenizer=tokenizer)

    # Set up evaluation-only training args
    # Note: output_dir was already set earlier to model_dir/all_eval_results
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
        checkpoint_dir=checkpoint_dir,  # Pass checkpoint directory to Trainer
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
    
    # Check GPU memory utilization after evaluation
    if is_main_process and torch.cuda.is_available():
        print("\n" + "=" * 70)
        print("GPU MEMORY UTILIZATION AFTER EVALUATION")
        print("=" * 70)
        memory_stats = check_gpu_memory_utilization(num_gpus)
        for gpu_info in memory_stats.get("gpus", []):
            print(f"GPU {gpu_info['gpu_id']}: {gpu_info['name']}")
            print(f"  Peak allocated: {gpu_info['allocated_gb']:.1f} GB")
            print(f"  Reserved: {gpu_info['reserved_gb']:.1f} GB ({gpu_info['utilization_pct']:.1f}%)")
            print(f"  Free: {gpu_info['free_gb']:.1f} GB")
        
        if memory_stats.get("recommendations"):
            print("\nRecommendations:")
            for rec in memory_stats["recommendations"]:
                print(f"  • {rec}")
        print("=" * 70 + "\n")
    
    # ------------------------------------------------------------------
    # Enrich eval_results with clearer runtime and cardinality metadata
    # ------------------------------------------------------------------
    # HF Trainer reports wall-clock time in eval_runtime and per-second rates.
    # Make this explicit and add simple counts for easier inspection.
    eval_num_examples = len(tokenized_val_dataset)
    eval_results.setdefault("eval_num_examples", eval_num_examples)
    # For evaluation we run exactly one pass over the dataset
    eval_results.setdefault("eval_num_epochs", 1)
    
    # Clarify that these are wall-clock metrics by adding explicit aliases
    if "eval_runtime" in eval_results:
        eval_results.setdefault("eval_wall_runtime", eval_results["eval_runtime"])
    if "eval_samples_per_second" in eval_results:
        eval_results.setdefault("eval_wall_samples_per_second", eval_results["eval_samples_per_second"])
    if "eval_steps_per_second" in eval_results:
        eval_results.setdefault("eval_wall_steps_per_second", eval_results["eval_steps_per_second"])

    # ------------------------------------------------------------------
    # Doc-type distribution for evaluated examples
    # ------------------------------------------------------------------
    # Summarise how many examples of each doc_type were used in this evaluation.
    doc_type_counts: Dict[str, int] = {}
    doc_type_nor_counts: Dict[str, int] = {}
    for ex in val_data:
        meta = ex.get("metadata") if isinstance(ex, dict) else None
        if isinstance(meta, dict):
            raw_doc_type = meta.get("doc_type") or "unknown"
        else:
            raw_doc_type = "unknown"
        doc_type_counts[raw_doc_type] = doc_type_counts.get(raw_doc_type, 0) + 1

        # Also track Norwegian label where possible
        try:
            doc_type_nor = get_doc_type_norwegian(raw_doc_type)
        except Exception:
            doc_type_nor = "tekst"
        doc_type_nor_counts[doc_type_nor] = doc_type_nor_counts.get(doc_type_nor, 0) + 1

    eval_results.setdefault("eval_num_examples_by_doc_type", doc_type_counts)
    eval_results.setdefault("eval_num_examples_by_doc_type_norwegian", doc_type_nor_counts)
    
    # Save inputs, references, and predictions to JSONL file
    # Only save for major checkpoints to save disk space
    predictions_file = None
    if is_main_process and is_major_checkpoint:
        predictions_file = os.path.join(output_dir, f"{checkpoint_name}-inputs-refs-preds.jsonl")
        
        # Write to JSONL file
        # Predictions are generated in the same order as the dataset
        with open(predictions_file, 'w', encoding='utf-8') as f:
            num_examples = len(original_examples_for_jsonl)
            num_predictions = len(trainer._eval_predictions)
            
            # If counts don't match, use the minimum (shouldn't happen, but safety check)
            num_to_save = min(num_examples, num_predictions)
            
            for i in range(num_to_save):
                # Match predictions with original examples (same order)
                # All fields are human-readable text
                entry = {
                    "input_text": original_examples_for_jsonl[i].get("input_text", ""),  # Raw input text
                    "prompt": original_examples_for_jsonl[i].get("prompt", ""),  # Formatted prompt with template
                    "reference": original_examples_for_jsonl[i].get("reference", ""),  # Target summary (ground truth)
                    "prediction": trainer._eval_predictions[i] if i < len(trainer._eval_predictions) else ""  # Model prediction (cleaned)
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"Saved predictions to: {predictions_file}")
        print(f"  - {num_to_save} examples saved")
    elif is_main_process and not is_major_checkpoint:
        print(f"Skipping predictions file (not a major checkpoint - only saved for major checkpoints to save disk space)")
    
    # Run extended evaluation metrics (reference-based, hygiene, faithfulness)
    # Note: For normal checkpoints, we still run extended evaluation but use predictions from memory
    # For major checkpoints, we can load from the saved JSONL file
    if is_main_process:
        # Debug: Print why extended evaluation might not run
        if not EXTENDED_EVAL_AVAILABLE:
            print("\n" + "=" * 70)
            print("WARNING: Extended evaluation NOT available")
            print("=" * 70)
            print("Reason: extended_evaluation.py could not be imported")
            print("Only ROUGE metrics will be saved to JSON.")
            print("=" * 70 + "\n")
        elif not is_major_checkpoint and not predictions_file:
            # For normal checkpoints, we'll use predictions from memory (trainer._eval_predictions)
            # This is fine - we don't need the file for normal checkpoints
            pass
        elif is_major_checkpoint and not predictions_file:
            print("\n" + "=" * 70)
            print("WARNING: Extended evaluation may be limited")
            print("=" * 70)
            print("Reason: predictions_file is None for major checkpoint")
            print("Will use predictions from memory instead.")
            print("=" * 70 + "\n")
        elif predictions_file and not os.path.exists(predictions_file):
            print("\n" + "=" * 70)
            print("WARNING: Extended evaluation may be limited")
            print("=" * 70)
            print(f"Reason: predictions_file does not exist: {predictions_file}")
            print("Will use predictions from memory instead.")
            print("=" * 70 + "\n")
    
    # Run extended evaluation metrics (reference-based, hygiene, faithfulness)
    # For major checkpoints: load from saved JSONL file
    # For normal checkpoints: use predictions from memory (trainer._eval_predictions)
    include_faithfulness = False  # Initialize to avoid unbound variable errors
    
    if is_main_process and EXTENDED_EVAL_AVAILABLE:
        print("\n" + "=" * 70)
        print("Running Extended Evaluation Metrics...")
        print("=" * 70)
        
        try:
            # Load texts from JSONL file if available (major checkpoints), otherwise use memory
            input_texts = []
            prediction_texts = []
            reference_texts = []
            
            if predictions_file and os.path.exists(predictions_file):
                # Major checkpoint: load from saved file
                with open(predictions_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        entry = json.loads(line)
                        input_texts.append(entry.get("input_text", ""))
                        prediction_texts.append(entry.get("prediction", ""))
                        reference_texts.append(entry.get("reference", ""))
            else:
                # Normal checkpoint: use predictions from memory
                # We have original_examples_for_jsonl and trainer._eval_predictions in memory
                num_examples = len(original_examples_for_jsonl)
                num_predictions = len(trainer._eval_predictions)
                num_to_use = min(num_examples, num_predictions)
                
                for i in range(num_to_use):
                    input_texts.append(original_examples_for_jsonl[i].get("input_text", ""))
                    prediction_texts.append(trainer._eval_predictions[i] if i < len(trainer._eval_predictions) else "")
                    reference_texts.append(original_examples_for_jsonl[i].get("reference", ""))
            
            if len(input_texts) > 0 and len(prediction_texts) > 0 and len(reference_texts) > 0:
                if not EXTENDED_EVAL_AVAILABLE:
                    print("Warning: Extended evaluation not available (extended_evaluation.py not found). Skipping extended metrics.")
                else:
                    # Determine which metrics to compute based on checkpoint type and user settings
                    # Normal checkpoints: ROUGE + Hygiene only (fast, ~2 min)
                    # Major checkpoints: ROUGE + Hygiene + BERTScore (moderate, ~3-4 min)
                    # NLI Faithfulness: Optional, can be enabled via include_nli_faithfulness parameter
                    is_major_extended = is_major_checkpoint(checkpoint_step_int, major_checkpoint_interval)
                    include_bertscore = is_major_extended
                    include_faithfulness = include_nli_faithfulness  # User-controlled
                    
                    # Prepare fixed subset for NLI if requested
                    # For ROUGE, Hygiene, and BERTScore: use full dataset
                    # For NLI: use fixed 500-example subset for consistency across all checkpoints
                    nli_input_texts = input_texts
                    nli_prediction_texts = prediction_texts
                    nli_reference_texts = reference_texts
                    
                    if include_faithfulness:
                        # Get or create fixed NLI subset (500 examples, same across all checkpoints)
                        model_dir_eval = get_model_dir_from_checkpoint(checkpoint_dir)
                        nli_indices = get_or_create_fixed_nli_subset(
                            total_examples=len(input_texts),
                            model_dir=model_dir_eval,
                            subset_size=NLI_FIXED_SUBSET_SIZE
                        )
                        
                        # Safety check: ensure subset is not empty
                        if not nli_indices or len(nli_indices) == 0:
                            raise ValueError(
                                f"Fixed NLI subset is empty! This should not happen. "
                                f"Total examples: {len(input_texts)}, subset size: {NLI_FIXED_SUBSET_SIZE}"
                            )
                        
                        nli_input_texts, nli_prediction_texts, nli_reference_texts = apply_fixed_subset(
                            input_texts, prediction_texts, reference_texts, nli_indices
                        )
                        
                        # Verify subset was applied correctly
                        if len(nli_input_texts) == 0:
                            raise ValueError(
                                f"After applying fixed subset, NLI input texts are empty! "
                                f"Indices: {len(nli_indices)}, Applied: {len(nli_input_texts)}"
                            )
                        
                        # Ensure we're using the subset, not the full dataset
                        assert len(nli_input_texts) <= len(input_texts), \
                            "NLI subset should never be larger than full dataset"
                        assert len(nli_input_texts) == len(nli_indices) or len(nli_input_texts) <= NLI_FIXED_SUBSET_SIZE, \
                            f"NLI subset size ({len(nli_input_texts)}) should match expected subset size ({NLI_FIXED_SUBSET_SIZE}) or be smaller"
                        
                        print(f"  → NLI subset: Using fixed {len(nli_input_texts)} examples from {len(input_texts)} total (consistent across all checkpoints)")
                        print(f"  → Subset indices saved to: {os.path.join(model_dir_eval, 'all_eval_results', 'nli_fixed_subset_indices.json')}")
                    
                    is_major_extended = is_major_checkpoint(checkpoint_step_int, major_checkpoint_interval)
                    checkpoint_type = "MAJOR" if is_major_extended else "NORMAL"
                    print(f"Computing extended metrics on {len(input_texts)} examples...")
                    print(f"Checkpoint type: {checkpoint_type} (step {checkpoint_step_int})")
                    if is_major_extended:
                        print(f"  → Computing: ROUGE + Hygiene + BERTScore (~3-4 min)")
                    else:
                        print(f"  → Computing: ROUGE + Hygiene only (~2 min)")
                    
                    if include_faithfulness:
                        nli_examples = len(nli_input_texts)
                        nli_time_estimate = (nli_examples * 4.5) / 60  # ~4.5 seconds per example
                        print(f"  → Computing: NLI Faithfulness on {nli_examples} examples (~{nli_time_estimate:.1f} min)")
                    else:
                        print(f"  → Skipping: NLI Faithfulness (enable with --include_nli_faithfulness)")
                    
                    # Run extended evaluation with selective metrics
                    # Note: extended_evaluate computes all metrics on the same input set
                    # For NLI, we pass the subset if specified; for other metrics, we need to handle separately
                    # However, extended_evaluate expects all inputs to be the same length
                    # So we'll run it twice: once for non-NLI metrics (full set), once for NLI (subset if specified)
                    
                    # First run: ROUGE + Hygiene + BERTScore (on full set)
                    assert extended_evaluate is not None, "extended_evaluate should be available when EXTENDED_EVAL_AVAILABLE is True"
                    extended_results = extended_evaluate(
                        input_texts=input_texts,
                        prediction_texts=prediction_texts,
                        reference_texts=reference_texts,
                        print_output=False,  # We'll print formatted results ourselves
                        include_bertscore=include_bertscore,
                        include_faithfulness=False  # Skip NLI in first run
                    )
                    # Initialize faithfulness to None if not present (will be set in second run if requested)
                    if "faithfulness" not in extended_results:
                        extended_results["faithfulness"] = None
                    
                    # Second run: NLI only (on fixed subset - always uses same 500 examples)
                    if include_faithfulness:
                        assert extended_evaluate is not None, "extended_evaluate should be available when EXTENDED_EVAL_AVAILABLE is True"
                        
                        # Safety check: ensure we're using the subset, not the full dataset
                        assert len(nli_input_texts) <= len(input_texts), \
                            "NLI evaluation must use subset, not full dataset"
                        assert len(nli_input_texts) <= NLI_FIXED_SUBSET_SIZE, \
                            f"NLI subset size ({len(nli_input_texts)}) exceeds fixed subset size ({NLI_FIXED_SUBSET_SIZE})"
                        
                        try:
                            nli_results = extended_evaluate(
                                input_texts=nli_input_texts,  # Fixed subset (500 examples)
                                prediction_texts=nli_prediction_texts,  # Fixed subset
                                reference_texts=nli_reference_texts,  # Fixed subset
                                print_output=False,
                                include_bertscore=False,  # Skip BERTScore in second run (already computed)
                                include_faithfulness=True  # Only compute NLI on fixed subset
                            )
                            # Merge NLI results into extended_results
                            faithfulness_result = nli_results.get("faithfulness") if nli_results else None
                            extended_results["faithfulness"] = faithfulness_result
                        except Exception as nli_error:
                            print(f"\n{'='*70}")
                            print(f"ERROR: NLI faithfulness evaluation failed: {nli_error}")
                            print(f"{'='*70}")
                            import traceback
                            traceback.print_exc()
                            print(f"{'='*70}\n")
                            extended_results["faithfulness"] = None
                    
                    # Note: If NLI was run on a subset, the NLI results are for that subset only
                    # The other metrics (ROUGE, Hygiene, BERTScore) are computed on the full set
                    # This is intentional - NLI is expensive, so we sample for it
                    
                    # Merge extended results into eval_results
                    # Flatten nested structure for easier access
                    for category, metrics in extended_results.items():
                        if isinstance(metrics, dict):
                            # For faithfulness, preserve the full dict only (nested structure is cleaner)
                            if category == "faithfulness" and metrics is not None:
                                # Save full faithfulness dict
                                eval_results["eval_faithfulness"] = metrics
                            else:
                                # For other categories, flatten for easier access
                                for key, value in metrics.items():
                                    eval_key = f"eval_{category}_{key}"
                                    eval_results[eval_key] = value
                        else:
                            eval_key = f"eval_{category}"
                            eval_results[eval_key] = metrics
                    
                    # Print extended metrics summary
                    print("\nExtended Evaluation Results:")
                    print("-" * 70)
                    
                    # Reference-based metrics (ROUGE always, BERTScore if major checkpoint)
                    if "reference" in extended_results:
                        ref_metrics = extended_results["reference"]
                        print("Reference-based Metrics:")
                        for key, value in ref_metrics.items():
                            if isinstance(value, (int, float)):
                                print(f"  {key}: {value:.4f}")
                        if not is_major_checkpoint and "bertscore_f1_mean" not in ref_metrics:
                            print("  (BERTScore skipped - not a major checkpoint)")
                    
                    # Hygiene metrics
                    if "hygiene" in extended_results:
                        hygiene_metrics = extended_results["hygiene"]
                        print("\nHygiene Metrics:")
                        for key, value in hygiene_metrics.items():
                            if isinstance(value, (int, float)):
                                print(f"  {key}: {value:.4f}")
                            elif value is not None:
                                print(f"  {key}: {value}")
                    
                    # Faithfulness metrics (skipped for checkpoints)
                    if "faithfulness" in extended_results and extended_results["faithfulness"] is not None:
                        faith_metrics = extended_results["faithfulness"]
                        print("\nFaithfulness Metrics:")
                        for key, value in faith_metrics.items():
                            if isinstance(value, (int, float)):
                                print(f"  {key}: {value:.4f}")
                            elif isinstance(value, list):
                                # Skip reasons_failed list (too verbose)
                                if key != "reasons_failed":
                                    print(f"  {key}: {len(value)} items")
                            elif value is not None:
                                print(f"  {key}: {value}")
                    elif "faithfulness" in extended_results:
                        print("\nFaithfulness Metrics: (skipped - too slow for checkpoint evaluation)")
                    
                    print("=" * 70 + "\n")
            else:
                print("Warning: No valid examples found in predictions file for extended evaluation")
        except Exception as e:
            print(f"\n{'='*70}")
            print(f"ERROR: Extended evaluation failed: {e}")
            print(f"{'='*70}")
            import traceback
            traceback.print_exc()
            print("Continuing with ROUGE metrics only...")
            print(f"{'='*70}\n")
    # Note: Diagnostic messages for why extended evaluation didn't run are printed earlier
    
    if is_main_process:
        print("\n" + "=" * 70)
        print("Evaluation Results (ROUGE + Extended Metrics):")
        print("=" * 70)
        # Print ROUGE metrics first
        rouge_keys = [k for k in eval_results.keys() if 'rouge' in k.lower() and isinstance(eval_results[k], (int, float))]
        if rouge_keys:
            print("ROUGE Metrics:")
            for key in sorted(rouge_keys):
                value = eval_results[key]
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        # Print extended metrics if available
        if EXTENDED_EVAL_AVAILABLE:
            extended_keys = [k for k in eval_results.keys() if any(prefix in k for prefix in ['reference_', 'hygiene_', 'faithfulness_'])]
            if extended_keys:
                print("\nExtended Metrics:")
                # Group by category
                reference_keys = [k for k in extended_keys if 'reference_' in k]
                hygiene_keys = [k for k in extended_keys if 'hygiene_' in k]
                faithfulness_keys = [k for k in extended_keys if 'faithfulness_' in k]
                
                if reference_keys:
                    print("  Reference-based:")
                    for key in sorted(reference_keys):
                        value = eval_results[key]
                        if isinstance(value, (int, float)):
                            print(f"    {key}: {value:.4f}")
                        else:
                            print(f"    {key}: {value}")
                if hygiene_keys:
                    print("  Hygiene:")
                    for key in sorted(hygiene_keys):
                        value = eval_results[key]
                        if isinstance(value, (int, float)):
                            print(f"    {key}: {value:.4f}")
                        else:
                            print(f"    {key}: {value}")
                if faithfulness_keys:
                    print("  Faithfulness:")
                    for key in sorted(faithfulness_keys):
                        value = eval_results[key]
                        if isinstance(value, (int, float)):
                            print(f"    {key}: {value:.4f}")
                        else:
                            print(f"    {key}: {value}")
        print("=" * 70 + "\n")
    
    # Log final summary to wandb if initialized and not disabled
    # Log with checkpoint step so it appears in the timeline
    if wandb_project and wandb.run is not None and is_main_process and not wandb_disabled:
        # Log ROUGE metrics
        # Log extended metrics if available (ROUGE already logged in compute_metrics)
        if EXTENDED_EVAL_AVAILABLE:
            log_dict = {}
            
            # Reference-based metrics (BERTScore)
            if "eval_reference_bertscore_f1_mean" in eval_results:
                log_dict["bertscore_f1"] = eval_results.get("eval_reference_bertscore_f1_mean", 0)
            
            # Hygiene metrics
            if "eval_hygiene_mean_compression_ratio" in eval_results:
                log_dict["compression_ratio"] = eval_results.get("eval_hygiene_mean_compression_ratio", 0)
            if "eval_hygiene_mean_rep_3gram" in eval_results:
                log_dict["rep_3gram"] = eval_results.get("eval_hygiene_mean_rep_3gram", 0)
            if "eval_hygiene_ratio_ends_with_punct" in eval_results:
                log_dict["ends_with_punct"] = eval_results.get("eval_hygiene_ratio_ends_with_punct", 0)
            
            # Faithfulness metrics (from nested dict)
            if "eval_faithfulness" in eval_results and isinstance(eval_results["eval_faithfulness"], dict):
                faithfulness = eval_results["eval_faithfulness"]
                if "mean_entailment_score" in faithfulness:
                    log_dict["entailment_mean"] = faithfulness.get("mean_entailment_score", 0)
                if "ratio_passed_documents" in faithfulness:
                    log_dict["faithfulness_passed"] = faithfulness.get("ratio_passed_documents", 0)
            
            if log_dict:
                wandb.log(log_dict, step=checkpoint_step_int)
        
        # Update summary with best metrics only (for quick overview)
        best_rouge_lsum = eval_results.get("eval_rougeLsum", 0)
        current_best = wandb.summary.get("best_rouge_lsum", 0)
        if current_best < best_rouge_lsum:
            wandb.summary.update({
                "best_checkpoint": checkpoint_step_int,
                "best_rouge_lsum": best_rouge_lsum,
                "best_rouge1": eval_results.get("eval_rouge1", 0),
                "best_rouge2": eval_results.get("eval_rouge2", 0),
                "best_rougeL": eval_results.get("eval_rougeL", 0),
            })
        
        # DON'T call wandb.finish() here - keep the run open for multiple checkpoints
        # Only finish if explicitly requested or at the very end
        print(">>> Evaluation results logged to wandb (run kept open for additional checkpoints)")
    elif wandb_disabled and is_main_process:
        print(">>> Wandb disabled - skipping wandb logging")
    
    # Save results to file (only on main process)
    if is_main_process:
        # Save results using utility function (saves to both new and old locations)
        save_eval_results(
            results=eval_results,
            checkpoint_dir=checkpoint_dir,
            model_dir=model_dir_eval,
            save_to_old_location=True  # Keep backwards compatibility
        )
        print(f"Results saved to: {results_file}")
        
        # Update evaluation summary using utility function
        update_evaluation_summary(
            results=eval_results,
            checkpoint_dir=checkpoint_dir,
            model_name=model_name,
            val_dataset_path=val_dataset_path,
            model_dir=model_dir_eval
        )
    
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

  # With NLI faithfulness evaluation on a subset:
  python evaluate_distributed_checkpoints_multigpu.py \\
    --model gemma-7b \\
    --checkpoint_dir models/gemma-7b_fsdp/checkpoint-100 \\
    --val_dataset data/output/processed_data_val.jsonl \\
    --hf_token YOUR_TOKEN \\
    --include_nli_faithfulness
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
                       help='Output directory for evaluation results (default: model_dir/all_eval_results)')
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
                       help=f'Validation batch size per device (default: {VAL_BATCH_SIZE}). The script will automatically adjust based on model size and provide memory utilization reports.')
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
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name (if not provided, defaults to {model}_eval_{major|normal}-{step}). Use same name for all checkpoints to combine them.')
    parser.add_argument('--wandb_group', type=str, default=None,
                       help='Wandb group name to combine multiple runs (default: model name)')
    parser.add_argument('--major_checkpoint_interval', type=int, default=500,
                       help='Every Nth step is considered "major" for BERTScore evaluation (default: 500). Major checkpoints: checkpoint-500, checkpoint-1000, checkpoint-1500, etc.')
    parser.add_argument('--include_nli_faithfulness', action='store_true',
                       help='Enable NLI-based faithfulness evaluation using fixed 500-example subset (slow: ~4.5s per example, ~37 min for 500 examples). The same 500 examples are used for all checkpoints for consistency.')

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
            use_multi_gpu=args.use_multi_gpu,
            major_checkpoint_interval=args.major_checkpoint_interval,
            include_nli_faithfulness=args.include_nli_faithfulness,
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
                output_dir=args.output_dir,  # None will trigger default: model_dir/all_eval_results
                use_multi_gpu=args.use_multi_gpu,
                wandb_project=args.wandb_project if not args.wandb_disabled else None,
                wandb_entity=args.wandb_entity,
                wandb_disabled=args.wandb_disabled,
                wandb_run_name=args.wandb_run_name,
                wandb_group=args.wandb_group,
                major_checkpoint_interval=args.major_checkpoint_interval,
                include_nli_faithfulness=args.include_nli_faithfulness,
            )
        except AlreadyEvaluatedError as e:
            print(f"⚠ SKIPPING: {e}")
            print(f"Checkpoint {args.checkpoint_dir} was already evaluated. Moving to next checkpoint.")
            sys.exit(0)  # Exit with success code so bash loop continues
