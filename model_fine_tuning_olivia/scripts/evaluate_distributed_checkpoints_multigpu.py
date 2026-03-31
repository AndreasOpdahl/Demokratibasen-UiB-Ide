"""
Multi-GPU evaluation script for PEFT checkpoints using model parallelism.

This script uses model parallelism (device_map="auto") to split large models across GPUs,
avoiding the FSDP/DDP synchronization issues that occur with model.generate().

NOTE: FSDP is incompatible with model.generate() - the training script disables
evaluation when using FSDP for this reason. This script uses model parallelism instead.

Usage:
  # Multi-GPU evaluation with model parallelism:
  python evaluate_distributed_checkpoints_multigpu.py \
    --model gemma-7b-it \
    --checkpoint_dir models/gemma-7b-it_fsdp/checkpoint-100 \
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
import re
import shutil
import sys
import time  # ADD THIS for staggered loading
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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
    get_predictions_file_path,
    get_faithfulness_details_path,
    get_old_eval_results_path,
    load_eval_results,
    save_eval_results,
    should_skip_faithfulness_update,
    nli_faithfulness_aggregate_present,
    update_evaluation_summary,
    load_jsonl_dataset,
    tokenize_eval_examples,
    get_or_create_fixed_nli_subset,
    apply_fixed_subset,
    NLI_DEFAULT_SUBSET_SIZE,
    format_eval_example,
)

# Import extended evaluation metrics (hygiene, BERTScore, NLI faithfulness)
try:
    from utils.metrics import extended_evaluate
    EXTENDED_EVAL_AVAILABLE = True
except ImportError as e:
    EXTENDED_EVAL_AVAILABLE = False
    extended_evaluate = None  # type: ignore
    print(f"Warning: utils.metrics.extended_evaluate not available ({e}). Only ROUGE metrics will be computed.")

# Helper function to calculate examples from steps
def calculate_examples_from_steps(steps, batch_size, gradient_accumulation_steps, num_gpus):
    """Calculate total number of examples processed given training parameters."""
    if steps is None or steps <= 0:
        return None
    return steps * batch_size * gradient_accumulation_steps * num_gpus

def load_predictions_jsonl(predictions_file: str):
    """Load input_texts, prediction_texts, reference_texts from a predictions JSONL file.

    Returns:
        Tuple of (input_texts, prediction_texts, reference_texts) as lists of strings.
    """
    input_texts, prediction_texts, reference_texts = [], [], []
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            input_texts.append(entry.get("input_text", ""))
            prediction_texts.append(entry.get("prediction", ""))
            reference_texts.append(entry.get("reference", ""))
    return input_texts, prediction_texts, reference_texts


# Evaluation parameters
MAX_INPUT_TEXT_TOKENS = 2048
MAX_EXTRA_PROMPT_TOKENS = 40
MAX_INPUT_PROMPT_TOKENS = MAX_INPUT_TEXT_TOKENS + MAX_EXTRA_PROMPT_TOKENS
MAX_OUTPUT_SUMMARY_TOKENS = 512
VAL_BATCH_SIZE = 32
VAL_DATA_SIZE = 500
VAL_DATA_SEED = 42  # Fixed seed for reproducible validation sampling
VAL_BEAM_SIZE = 4

def sample_validation_data_reproducibly(
    val_data: list,
    val_data_size: int,
    seed: int = VAL_DATA_SEED
) -> list:
    """Sample validation data with reproducible, backward-compatible selection.
    
    When val_data_size=500: uses first 500 from seed-based sample (canonical set).
    When val_data_size=1000: canonical 500 + 500 more (from remaining indices).
    This ensures the 1000-example set contains the same 500 used for 500-example runs.
    
    Args:
        val_data: Full validation dataset
        val_data_size: Number of examples to sample (500 or 1000)
        seed: Random seed for reproducibility
        
    Returns:
        Sampled validation data list
    """
    n = len(val_data)
    if val_data_size >= n:
        return val_data
    
    if val_data_size <= 500:
        random.seed(seed)
        indices = sorted(random.sample(range(n), val_data_size))
        return [val_data[i] for i in indices]
    
    # val_data_size > 500 (e.g. 1000): canonical 500 + additional from remaining
    random.seed(seed)
    first_500_idx = set(random.sample(range(n), 500))
    remaining = [i for i in range(n) if i not in first_500_idx]
    random.seed(seed + 1)
    add_count = min(val_data_size - 500, len(remaining))
    extra_idx = sorted(random.sample(remaining, add_count))
    all_indices = sorted(first_500_idx) + extra_idx
    return [val_data[i] for i in all_indices]


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
                 model_name: Optional[str] = None,  # Store model name for prompt format detection
                 **kwargs) -> None:
        self.generation_max_length = generation_max_length
        self.generation_num_beams = generation_num_beams
        self.eval_data_collator = eval_data_collator
        self.use_greedy = use_greedy
        self.checkpoint_dir = checkpoint_dir  # Store checkpoint directory
        self.model_name = model_name  # Store model name for prompt format detection
        super().__init__(*args, **kwargs)
        self._processing_class = self.tokenizer
        # Store predictions for saving to JSONL
        self._eval_predictions = []
        self._empty_warning_shown = False  # Throttle per-batch empty-prediction warning to once per eval
    
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

        # Generate with memory-efficient settings (use BF16 to match model's native dtype)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Get special token IDs for better stopping
            inst_token_id = None
            if hasattr(self._processing_class, 'convert_tokens_to_ids'):
                try:
                    candidate_inst_id = self._processing_class.convert_tokens_to_ids('[/INST]')
                    unk_token_id = getattr(self._processing_class, 'unk_token_id', None)
                    # Some tokenizers map unknown strings to unk_token_id (often 0).
                    # Treat that as "not found" so we don't mis-handle generation boundaries.
                    if candidate_inst_id is not None and candidate_inst_id != unk_token_id:
                        inst_token_id = candidate_inst_id
                except:
                    pass
            
            # Set minimum length to prevent model from outputting EOS immediately.
            # Keep it low (5–15): high min_new_tokens (e.g. 51) forces model past natural stop and causes gibberish.
            gen_max_len = self.generation_max_length if self.generation_max_length is not None else 512
            min_new_tokens = max(5, 15)
            
            generation_kwargs = {
                'input_ids': input_ids,
                'use_cache': True,
                'max_new_tokens': self.generation_max_length,
                'num_beams': 1 if self.use_greedy else self.generation_num_beams,
                'do_sample': False,
                'pad_token_id': self._processing_class.pad_token_id,
                'eos_token_id': self._processing_class.eos_token_id,
                'repetition_penalty': 1.1,  # Slight penalty to prevent repetition
            }
            # Critical: pass attention_mask so model does not attend to left-padding.
            # Without this, left-padded prompts cause the model to see pad tokens and often emit pad/unk.
            if 'attention_mask' in inputs:
                generation_kwargs['attention_mask'] = inputs['attention_mask']
            
            # Add min_new_tokens if supported (newer transformers versions)
            # Fallback: use min_length (total length including input) if min_new_tokens not available
            try:
                generation_kwargs['min_new_tokens'] = min_new_tokens
            except Exception as e:
                # Fallback: set min_length to input_length + min_new_tokens
                generation_kwargs['min_length'] = input_ids.shape[1] + min_new_tokens
            
            # Add stop token if found (for chat models)
            if inst_token_id is not None:
                # Don't stop on [/INST] during generation, but we'll clean it later
                pass
            
            # GPT-J specific validation: Check input length and token IDs to prevent CUDA device-side asserts
            # GPT-J has a 2048 token context window and can crash with device-side asserts if exceeded
            input_length = input_ids.shape[1]
            # Try to detect GPT-J model from model config or name
            model_name_lower = ''
            if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
                model_name_lower = model.config.model_type.lower()
            elif hasattr(model, 'name_or_path'):
                model_name_lower = str(model.name_or_path).lower()
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'config'):
                model_name_lower = getattr(model.base_model.config, 'model_type', '').lower()
            
            if 'gpt-j' in model_name_lower or 'gptj' in model_name_lower or 'gpt_j' in model_name_lower:
                # Get model's max position embeddings (GPT-J typically has 2048)
                max_pos_embeddings = getattr(model.config, 'max_position_embeddings', 2048) if hasattr(model, 'config') else 2048
                vocab_size = getattr(self._processing_class, 'vocab_size', None) or (len(self._processing_class) if hasattr(self._processing_class, '__len__') else None)
                
                # Validate input length and token IDs for GPT-J (2048 context)
                # Validate input length doesn't exceed max position embeddings; leave room for generation
                if input_length > max_pos_embeddings:
                    room_for_new = min(256, generation_kwargs["max_new_tokens"])
                    max_input_allowed = max_pos_embeddings - room_for_new - 10
                    if not getattr(self, '_gptj_context_warned', False):
                        print(f'⚠ WARNING: Input length ({input_length}) exceeds max position embeddings ({max_pos_embeddings})')
                        print(f'  Truncating prompt to {max_input_allowed} tokens to reserve {room_for_new}+ for generation')
                        self._gptj_context_warned = True
                    input_ids = input_ids[:, :max_input_allowed]
                    input_length = input_ids.shape[1]
                    generation_kwargs['input_ids'] = input_ids
                    if 'attention_mask' in generation_kwargs and generation_kwargs['attention_mask'] is not None:
                        generation_kwargs['attention_mask'] = generation_kwargs['attention_mask'][:, :max_input_allowed]
                
                # Validate total sequence length (input + generation) doesn't exceed max
                total_sequence_length = input_length + generation_kwargs["max_new_tokens"]
                if total_sequence_length > max_pos_embeddings:
                    buffer = 10
                    # Prefer truncating the prompt so we keep room for real generation (min 256 tokens)
                    min_new_tokens_room = min(256, generation_kwargs["max_new_tokens"])
                    max_input_allowed = max_pos_embeddings - min_new_tokens_room - buffer
                    if input_length > max_input_allowed and max_input_allowed > 100:
                        # Truncate input so we keep instruction + start of document; reserve room for generation
                        if not getattr(self, '_gptj_context_warned', False):
                            print(f'⚠ WARNING: Total sequence length ({total_sequence_length}) would exceed max position embeddings ({max_pos_embeddings})')
                            print(f'  Truncating prompt from {input_length} to {max_input_allowed} tokens to reserve {min_new_tokens_room}+ for generation')
                            self._gptj_context_warned = True
                        input_ids = input_ids[:, :max_input_allowed].clone()
                        input_length = input_ids.shape[1]
                        generation_kwargs['input_ids'] = input_ids
                        if 'attention_mask' in generation_kwargs and generation_kwargs['attention_mask'] is not None:
                            generation_kwargs['attention_mask'] = generation_kwargs['attention_mask'][:, :max_input_allowed].clone()
                        generation_kwargs["max_new_tokens"] = min(generation_kwargs["max_new_tokens"], max_pos_embeddings - input_length - buffer)
                    else:
                        # Prompt already short; reduce max_new_tokens to fit
                        max_new_tokens_safe = max(1, max_pos_embeddings - input_length - buffer)
                        if not getattr(self, '_gptj_context_warned', False):
                            print(f'⚠ WARNING: Total sequence length ({total_sequence_length}) would exceed max position embeddings ({max_pos_embeddings})')
                            print(f'  Reducing max_new_tokens from {generation_kwargs["max_new_tokens"]} to {max_new_tokens_safe}')
                            self._gptj_context_warned = True
                        generation_kwargs["max_new_tokens"] = max_new_tokens_safe
                    # Update min_new_tokens if it was set
                    if 'min_new_tokens' in generation_kwargs:
                        generation_kwargs['min_new_tokens'] = min(generation_kwargs['min_new_tokens'], generation_kwargs["max_new_tokens"])
                    elif 'min_length' in generation_kwargs:
                        generation_kwargs['min_length'] = min(generation_kwargs['min_length'], max_pos_embeddings)
                
                # Validate token IDs are within valid range
                if vocab_size is not None:
                    # Check for invalid token IDs (outside vocab range)
                    invalid_mask = (input_ids < 0) | (input_ids >= vocab_size)
                    num_invalid = invalid_mask.sum().item()
                    if num_invalid > 0:
                        print(f'⚠ WARNING: Found {num_invalid} invalid token IDs (outside vocab range [0, {vocab_size}))')
                        print(f'  Clamping invalid token IDs to valid range')
                        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
                        generation_kwargs['input_ids'] = input_ids
                    
                    # Check for NaN or Inf values
                    if torch.isnan(input_ids).any() or torch.isinf(input_ids).any():
                        print(f'⚠ WARNING: Found NaN or Inf values in input_ids')
                        print(f'  Replacing with pad_token_id')
                        input_ids = torch.where(torch.isnan(input_ids) | torch.isinf(input_ids), 
                                               torch.tensor(generation_kwargs['pad_token_id'], device=input_ids.device, dtype=input_ids.dtype),
                                               input_ids)
                        generation_kwargs['input_ids'] = input_ids
            
            try:
                generated_ids = model.generate(**generation_kwargs)
            except RuntimeError as e:
                error_str = str(e)
                # Check if this is a CUDA device-side assert (common with GPT-J)
                if "CUDA error" in error_str or "device-side assert" in error_str:
                    print(f'⚠ CUDA device-side assert detected during generation: {e}')
                    print(f'  This is often caused by:')
                    print(f'    1. Input sequence length exceeding model max position embeddings')
                    print(f'    2. Invalid token IDs (outside vocab range)')
                    print(f'    3. Total sequence (input + generation) exceeding context window')
                    
                    # For GPT-J models, try with more aggressive truncation
                    # Re-check model type in case validation didn't run
                    error_model_type = ''
                    if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
                        error_model_type = model.config.model_type.lower()
                    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'config'):
                        error_model_type = getattr(model.base_model.config, 'model_type', '').lower()
                    
                    if 'gpt-j' in error_model_type or 'gptj' in error_model_type or 'gpt_j' in error_model_type or 'gpt-j' in model_name_lower or 'gptj' in model_name_lower:
                        print(f'  Attempting recovery for GPT-J model...')
                        
                        # Clear CUDA cache and reset state
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        
                        # Further reduce input length if needed
                        max_pos_embeddings = getattr(model.config, 'max_position_embeddings', 2048) if hasattr(model, 'config') else 2048
                        safe_input_length = min(input_length, max_pos_embeddings - 100)  # Leave 100 token buffer
                        
                        if safe_input_length < input_length:
                            print(f'  Truncating input from {input_length} to {safe_input_length} tokens')
                            input_ids_truncated = input_ids[:, :safe_input_length]
                            generation_kwargs['input_ids'] = input_ids_truncated
                            generation_kwargs['max_new_tokens'] = min(generation_kwargs['max_new_tokens'], 100)  # Very conservative
                            if 'min_new_tokens' in generation_kwargs:
                                generation_kwargs['min_new_tokens'] = min(generation_kwargs['min_new_tokens'], 10)
                        
                        # Try again with truncated input
                        try:
                            print(f'  Retrying generation with truncated input...')
                            generated_ids = model.generate(**generation_kwargs)
                            print(f'  ✓ Recovery succeeded with truncated input')
                        except Exception as recovery_error:
                            print(f'  ✗ Recovery failed: {recovery_error}')
                            print(f'  This checkpoint may be corrupted or incompatible with the model')
                            raise RuntimeError(
                                f"CUDA device-side assert for GPT-J model could not be recovered. "
                                f"Original error: {e}\nRecovery error: {recovery_error}\n"
                                f"Try: 1) Check if checkpoint is corrupted, 2) Verify model and checkpoint compatibility, "
                                f"3) Run with CUDA_LAUNCH_BLOCKING=1 for detailed error info"
                            ) from recovery_error
                    else:
                        # For non-GPT-J models, try standard fallback
                        print(f'  Trying with do_sample=True and temperature=0.7 as fallback...')
                        generation_kwargs_fallback = generation_kwargs.copy()
                        generation_kwargs_fallback['do_sample'] = True
                        generation_kwargs_fallback['temperature'] = 0.7
                        generation_kwargs_fallback['top_p'] = 0.9
                        try:
                            generated_ids = model.generate(**generation_kwargs_fallback)
                            print(f'  Fallback generation succeeded')
                        except Exception as e2:
                            print(f'  Fallback also failed: {e2}')
                            raise
                else:
                    # Non-CUDA error, try standard fallback
                    print(f'⚠ ERROR during generation: {e}')
                    print(f'  Trying with do_sample=True and temperature=0.7 as fallback...')
                    generation_kwargs_fallback = generation_kwargs.copy()
                    generation_kwargs_fallback['do_sample'] = True
                    generation_kwargs_fallback['temperature'] = 0.7
                    generation_kwargs_fallback['top_p'] = 0.9
                    try:
                        generated_ids = model.generate(**generation_kwargs_fallback)
                        print(f'  Fallback generation succeeded')
                    except Exception as e2:
                        print(f'  Fallback also failed: {e2}')
                        raise
            except Exception as e:
                print(f'⚠ ERROR during generation: {e}')
                print(f'  Trying with do_sample=True and temperature=0.7 as fallback...')
                # Try with sampling as fallback
                generation_kwargs_fallback = generation_kwargs.copy()
                generation_kwargs_fallback['do_sample'] = True
                generation_kwargs_fallback['temperature'] = 0.7
                generation_kwargs_fallback['top_p'] = 0.9
                try:
                    generated_ids = model.generate(**generation_kwargs_fallback)
                    print(f'  Fallback generation succeeded')
                except Exception as e2:
                    print(f'  Fallback also failed: {e2}')
                    raise
        
        input_length = input_ids.shape[1]

        # Check if model generated anything at all (before slicing)
        if generated_ids.shape[1] <= input_length:
            print(f'⚠ WARNING: Model generated nothing or only input! generated_ids.shape={generated_ids.shape}, input_length={input_length}')
            print(f'  This suggests the model is immediately outputting EOS or not generating.')
            print(f'  Trying with more aggressive generation parameters...')
            
            # Try again with more permissive settings
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                generation_kwargs_retry = {
                    'input_ids': input_ids,
                    'use_cache': True,
                    'max_new_tokens': self.generation_max_length if self.generation_max_length is not None else 512,
                    'min_new_tokens': max(20, (self.generation_max_length if self.generation_max_length is not None else 512) // 5),  # More aggressive minimum
                    'do_sample': True,  # Try sampling instead of greedy
                    'temperature': 0.8,
                    'top_p': 0.95,
                    'pad_token_id': self._processing_class.pad_token_id,
                    'eos_token_id': self._processing_class.eos_token_id,
                    'repetition_penalty': 1.05,  # Lower penalty
                }
                try:
                    generated_ids_retry = model.generate(**generation_kwargs_retry)
                    if generated_ids_retry.shape[1] > input_length:
                        print(f'  Retry succeeded! New shape: {generated_ids_retry.shape}')
                        generated_ids = generated_ids_retry
                    else:
                        print(f'  Retry also failed - model still not generating (shape: {generated_ids_retry.shape})')
                except Exception as e:
                    print(f'  Retry generation failed: {e}')
        
        # Slice to get only generated tokens (after input)
        # Different prompt formats use different markers:
        # - Mistral: [/INST] (token-based)
        # - Llama-3: <|eot_id|> or <|start_header_id|>assistant<|end_header_id|> (token-based)
        # - Alpaca: "Response:" (text-based, need to find in decoded text)
        # - Plain: "Oppsummering:" (text-based)
        # - ChatML: <|im_start|>assistant (token-based)
        
        # Detect prompt format from model config if available
        prompt_format_type = None
        model_name_for_config = None
        
        # First try to use stored model_name
        if hasattr(self, 'model_name') and self.model_name:
            model_name_for_config = self.model_name
        # Otherwise try to get from model
        elif hasattr(self, 'model') and hasattr(self.model, 'config'):
            if hasattr(self.model, 'name_or_path'):
                model_name_for_config = self.model.name_or_path
            elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'name_or_path'):
                model_name_for_config = self.model.base_model.name_or_path
        
        if model_name_for_config:
            try:
                from model_configs import get_model_config_by_hf_name
                model_config = get_model_config_by_hf_name(model_name_for_config)
                if model_config:
                    prompt_format_type = model_config.prompt_config.template_type
            except Exception:
                pass
        
        # Strategy 1: Try token-based extraction for Mistral/Llama/ChatML formats
        extraction_successful = False
        
        # First, try to find [/INST] token position (Mistral format)
        inst_token_id = None
        if hasattr(self._processing_class, 'convert_tokens_to_ids'):
            try:
                candidate_inst_id = self._processing_class.convert_tokens_to_ids('[/INST]')
                unk_token_id = getattr(self._processing_class, 'unk_token_id', None)
                # If [/INST] resolves to UNK, token-based [/INST] extraction is unsafe.
                if candidate_inst_id is not None and candidate_inst_id != unk_token_id:
                    inst_token_id = candidate_inst_id
            except:
                pass
        
        # If we found [/INST] token, try to extract only tokens after it
        if inst_token_id is not None and generated_ids.shape[0] > 0 and (prompt_format_type is None or prompt_format_type == 'mistral'):
            # Find the last occurrence of [/INST] in the input
            input_ids_list = input_ids[0].cpu().tolist()
            inst_positions = [i for i, token_id in enumerate(input_ids_list) if token_id == inst_token_id]
            
            if inst_positions:
                last_inst_pos = inst_positions[-1]
                # Extract only tokens after the last [/INST]
                # But we need to account for the fact that generated_ids includes the full sequence
                # So we need to find where [/INST] is in the full generated_ids
                full_sequence = generated_ids[0].cpu().tolist()
                full_inst_positions = [i for i, token_id in enumerate(full_sequence) if token_id == inst_token_id]
                
                if full_inst_positions:
                    last_full_inst_pos = full_inst_positions[-1]
                    # Extract only after the last [/INST] in the full sequence
                    # Skip any whitespace/padding tokens immediately after
                    start_pos = last_full_inst_pos + 1
                    # Skip pad/eos tokens at the start
                    while start_pos < len(full_sequence) and full_sequence[start_pos] in [self._processing_class.pad_token_id, self._processing_class.eos_token_id]:
                        start_pos += 1
                    # With left-padding, generated content always starts at input_length; never include prompt tokens.
                    use_pos = max(start_pos, input_length)
                    if use_pos < generated_ids.shape[1]:
                        generated_ids = generated_ids[:, use_pos:]
                        extraction_successful = True
                    else:
                        generated_ids = generated_ids[:, input_length:]
                else:
                    # No [/INST] in full sequence, use original slice
                    generated_ids = generated_ids[:, input_length:]
            else:
                # No [/INST] in input, use original slice
                generated_ids = generated_ids[:, input_length:]
        else:
            # No [/INST] token found or not Mistral format, try other formats
            if not extraction_successful and generated_ids.shape[0] > 0:
                # Strategy 2: Try Llama-3/Llama-3.1 format (<|start_header_id|>assistant<|end_header_id|>)
                if prompt_format_type in ['llama3', 'llama3.1'] or prompt_format_type is None:
                    # For Llama-3.1, we need to find the assistant header and extract after it
                    # The format is: ...<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n[SUMMARY]
                    # We should extract everything after the assistant header
                    
                    # First, try to find the assistant header tokens
                    assistant_header_start_id = None
                    assistant_header_end_id = None
                    eot_token_id = None
                    
                    if hasattr(self._processing_class, 'convert_tokens_to_ids'):
                        try:
                            assistant_header_start_id = self._processing_class.convert_tokens_to_ids('<|start_header_id|>')
                            assistant_header_end_id = self._processing_class.convert_tokens_to_ids('<|end_header_id|>')
                            eot_token_id = self._processing_class.convert_tokens_to_ids('<|eot_id|>')
                        except:
                            pass
                    
                    # Try token-based extraction first (more reliable)
                    if assistant_header_start_id is not None and assistant_header_end_id is not None:
                        full_sequence = generated_ids[0].cpu().tolist()
                        
                        # Find all occurrences of <|start_header_id|>assistant<|end_header_id|>
                        # This is a sequence: <|start_header_id|> (token) + "assistant" (token) + <|end_header_id|> (token)
                        # We need to find where "assistant" appears between start and end header tokens
                        assistant_token_id = None
                        try:
                            # Try to get the token ID for "assistant"
                            if hasattr(self._processing_class, 'encode'):
                                assistant_encoded = self._processing_class.encode('assistant', add_special_tokens=False)
                                if len(assistant_encoded) > 0:
                                    assistant_token_id = assistant_encoded[0]
                        except:
                            pass
                        
                        # Look for the pattern: <|start_header_id|> ... assistant ... <|end_header_id|>
                        # Find the last occurrence in the input (should be right before generation)
                        header_start_positions = [i for i, token_id in enumerate(full_sequence) if token_id == assistant_header_start_id]
                        
                        if header_start_positions:
                            # Find the last header that's in the input (before generation)
                            last_header_in_input = None
                            for pos in reversed(header_start_positions):
                                if pos < input_length:
                                    # Check if this is followed by assistant token and end header
                                    # Look ahead a few tokens to find the pattern
                                    if pos + 2 < len(full_sequence):
                                        # Check if next token is "assistant" (or skip to find it)
                                        found_assistant = False
                                        found_end = False
                                        search_pos = pos + 1
                                        
                                        # Search for assistant token and end header within next 5 tokens
                                        for i in range(min(5, len(full_sequence) - pos - 1)):
                                            if assistant_token_id and full_sequence[search_pos + i] == assistant_token_id:
                                                found_assistant = True
                                            if full_sequence[search_pos + i] == assistant_header_end_id:
                                                found_end = True
                                                if found_assistant:
                                                    # Found the complete pattern
                                                    last_header_in_input = pos
                                                    # Extract after the end header token
                                                    start_pos = search_pos + i + 1
                                                    # Skip any newline/whitespace tokens (usually \n\n after assistant header)
                                                    while start_pos < len(full_sequence) and full_sequence[start_pos] in [self._processing_class.pad_token_id, self._processing_class.eos_token_id]:
                                                        start_pos += 1
                                                    if start_pos < generated_ids.shape[1]:
                                                        generated_ids = generated_ids[:, start_pos:]
                                                        extraction_successful = True
                                                        break
                                                break
                                        
                                        if extraction_successful:
                                            break
                            
                            # Fallback: If token-based extraction failed, try text-based
                            if not extraction_successful:
                                try:
                                    # Decode the full output to find the assistant header in text
                                    full_decoded = self._processing_class.decode(generated_ids[0].cpu().tolist(), skip_special_tokens=False)
                                    input_decoded = self._processing_class.decode(input_ids[0].cpu().tolist(), skip_special_tokens=False)
                                    
                                    # Find the last occurrence of assistant header in input
                                    assistant_header_text = '<|start_header_id|>assistant<|end_header_id|>'
                                    header_pos_in_input = input_decoded.rfind(assistant_header_text)
                                    
                                    if header_pos_in_input >= 0:
                                        # Find position after the header
                                        after_header_pos = header_pos_in_input + len(assistant_header_text)
                                        # Skip whitespace/newlines
                                        while after_header_pos < len(input_decoded) and input_decoded[after_header_pos] in [' ', '\n', '\t']:
                                            after_header_pos += 1
                                        
                                        # Now find this position in the full output
                                        if len(full_decoded) > after_header_pos:
                                            # Tokenize up to this point to find token position
                                            text_up_to_header = full_decoded[:after_header_pos]
                                            tokens_up_to_header = self._processing_class.encode(text_up_to_header, add_special_tokens=False)
                                            token_start = len(tokens_up_to_header)
                                            
                                            if token_start < generated_ids.shape[1]:
                                                generated_ids = generated_ids[:, token_start:]
                                                extraction_successful = True
                                except Exception as e:
                                    pass
                    
                    # Fallback to <|eot_id|> extraction if assistant header not found
                    if not extraction_successful and eot_token_id is not None:
                        full_sequence = generated_ids[0].cpu().tolist()
                        eot_positions = [i for i, token_id in enumerate(full_sequence) if token_id == eot_token_id]
                        if eot_positions:
                            # Find the last <|eot_id|> before the generation starts (should be in input)
                            last_eot_in_input = max([p for p in eot_positions if p < input_length], default=None)
                            if last_eot_in_input is not None:
                                start_pos = last_eot_in_input + 1
                                # Skip pad/eos tokens
                                while start_pos < len(full_sequence) and full_sequence[start_pos] in [self._processing_class.pad_token_id, self._processing_class.eos_token_id]:
                                    start_pos += 1
                                if start_pos < generated_ids.shape[1]:
                                    generated_ids = generated_ids[:, start_pos:]
                                    extraction_successful = True
                
                # Strategy 2.5: ChatML format (<|im_start|>assistant) - token-based so it works with left-padding
                if not extraction_successful and (prompt_format_type == 'chatml' or prompt_format_type is None):
                    try:
                        marker_text = "<|im_start|>assistant\n"
                        marker_ids = self._processing_class.encode(marker_text, add_special_tokens=False)
                        if not marker_ids:
                            marker_ids = self._processing_class.encode("<|im_start|>assistant", add_special_tokens=False)
                        if marker_ids and generated_ids.shape[0] > 0:
                            full_sequence = generated_ids[0].cpu().tolist()
                            last_start = -1
                            search_end = min(input_length, len(full_sequence) - len(marker_ids) + 1)
                            for i in range(0, search_end):
                                if i + len(marker_ids) <= len(full_sequence) and full_sequence[i:i + len(marker_ids)] == marker_ids:
                                    last_start = i
                            if last_start >= 0:
                                start_pos = last_start + len(marker_ids)
                                while start_pos < len(full_sequence) and full_sequence[start_pos] in [self._processing_class.pad_token_id, self._processing_class.eos_token_id]:
                                    start_pos += 1
                                # Use the later of (after assistant header) and (end of input) so we never include prompt tokens
                                use_pos = max(start_pos, input_length)
                                if use_pos < generated_ids.shape[1]:
                                    generated_ids = generated_ids[:, use_pos:]
                                    extraction_successful = True
                    except Exception as e:
                        pass
                
                # Strategy 3: Try Alpaca format ("Response:" text marker)
                if not extraction_successful and (prompt_format_type == 'alpaca' or prompt_format_type is None):
                    # Decode both input and full output to find "Response:" marker
                    try:
                        input_decoded = self._processing_class.decode(input_ids[0].cpu().tolist(), skip_special_tokens=False)
                        full_output_decoded = self._processing_class.decode(generated_ids[0].cpu().tolist(), skip_special_tokens=False)
                        
                        response_marker = "Response:"
                        
                        # First, check if "Response:" exists in the input (should be at the end)
                        response_pos_in_input = input_decoded.rfind(response_marker)
                        
                        # Also check if model generated "Response:" again (might happen if model is confused)
                        response_pos_in_output = full_output_decoded.find(response_marker)
                        
                        # Use the last occurrence of "Response:" in the full output
                        # This handles both cases: Response: in input, or Response: generated by model
                        all_response_positions = []
                        if response_pos_in_input >= 0:
                            all_response_positions.append(response_pos_in_input)
                        if response_pos_in_output >= 0:
                            all_response_positions.append(response_pos_in_output)
                        
                        if all_response_positions:
                            # Use the last occurrence (most likely the one after which generation should start)
                            last_response_pos = max(all_response_positions)
                            
                            # Find the position after "Response:" and any whitespace
                            after_response_pos = last_response_pos + len(response_marker)
                            while after_response_pos < len(full_output_decoded) and full_output_decoded[after_response_pos] in [' ', '\n', '\t']:
                                after_response_pos += 1
                            
                            # If the Response: is in the output (model generated it), we need to find it in token space
                            # Tokenize the part before the Response: marker
                            text_before_response = full_output_decoded[:after_response_pos]
                            tokens_before_response = self._processing_class.encode(text_before_response, add_special_tokens=False)
                            token_pos_after_response = len(tokens_before_response)
                            
                            # Also check if model included "Instruction:" in output (model copying input)
                            instruction_marker = "Instruction:"
                            if instruction_marker in full_output_decoded and instruction_marker not in input_decoded[-200:]:
                                # Model generated "Instruction:" - this is wrong, find where actual response starts
                                # Look for "Response:" after the generated "Instruction:"
                                instruction_pos = full_output_decoded.find(instruction_marker)
                                if instruction_pos >= 0:
                                    # Find Response: after this Instruction:
                                    response_after_instruction = full_output_decoded.find(response_marker, instruction_pos)
                                    if response_after_instruction >= 0:
                                        after_response_pos = response_after_instruction + len(response_marker)
                                        while after_response_pos < len(full_output_decoded) and full_output_decoded[after_response_pos] in [' ', '\n', '\t']:
                                            after_response_pos += 1
                                        text_before_response = full_output_decoded[:after_response_pos]
                                        tokens_before_response = self._processing_class.encode(text_before_response, add_special_tokens=False)
                                        token_pos_after_response = len(tokens_before_response)
                            
                            # Extract from this position
                            if token_pos_after_response < generated_ids.shape[1]:
                                generated_ids = generated_ids[:, token_pos_after_response:]
                                extraction_successful = True
                    except Exception:
                        pass
                
                # Strategy 4: Try plain format ("Oppsummering:" marker) - Gemma uses this
                if not extraction_successful and (prompt_format_type == 'plain' or prompt_format_type is None):
                    try:
                        input_decoded = self._processing_class.decode(input_ids[0].cpu().tolist(), skip_special_tokens=False)
                        summary_marker = "Oppsummering:"
                        summary_pos = input_decoded.rfind(summary_marker)
                        
                        if summary_pos >= 0:
                            after_summary_pos = summary_pos + len(summary_marker)
                            while after_summary_pos < len(input_decoded) and input_decoded[after_summary_pos] in [' ', '\n', '\t', '\n\n', '###']:
                                after_summary_pos += 1
                            
                            input_before_summary = input_decoded[:after_summary_pos]
                            tokens_before_summary = self._processing_class.encode(input_before_summary, add_special_tokens=False)
                            token_pos_after_summary = len(tokens_before_summary)
                            # With left-padding, generated content starts at input_length; never include prompt tokens
                            use_pos = max(token_pos_after_summary, input_length)
                            if use_pos < generated_ids.shape[1]:
                                generated_ids = generated_ids[:, use_pos:]
                                extraction_successful = True
                    except Exception:
                        pass
                
                # Fallback: Use simple input_length slice if no format-specific extraction worked
                if not extraction_successful:
                    generated_ids = generated_ids[:, input_length:]
                    
                    # Additional check: If model seems to be copying input (includes instruction prompt),
                    # try to find where actual content starts by looking for repeated instruction text
                    if generated_ids.shape[0] > 0 and generated_ids.shape[1] > 0:
                        try:
                            decoded_output = self._processing_class.decode(generated_ids[0].cpu().tolist(), skip_special_tokens=False)
                            decoded_input = self._processing_class.decode(input_ids[0].cpu().tolist(), skip_special_tokens=False)
                            
                            # Check if output starts with instruction text (model copying input)
                            instruction_start = "Instruction:"
                            if decoded_output.strip().startswith(instruction_start):
                                # Try to find "Response:" in the output
                                response_pos = decoded_output.find("Response:")
                                if response_pos >= 0:
                                    after_response = decoded_output[response_pos + len("Response:"):].strip()
                                    if after_response:
                                        # Re-encode from Response: onwards
                                        text_from_response = "Response:" + after_response
                                        tokens_from_response = self._processing_class.encode(text_from_response, add_special_tokens=False)
                                        # Find where this starts in the generated_ids
                                        # This is approximate - we'll use the decoded approach
                                        # Update generated_ids to only include content after Response:
                                        # We'll do this by re-encoding
                                        try:
                                            # Find the token position of "Response:" in the full sequence
                                            full_decoded = self._processing_class.decode(generated_ids[0].cpu().tolist(), skip_special_tokens=False)
                                            response_token_pos = full_decoded.find("Response:")
                                            if response_token_pos >= 0:
                                                # Tokenize up to Response: to find token position
                                                text_up_to_response = full_decoded[:response_token_pos + len("Response:")]
                                                tokens_up_to_response = self._processing_class.encode(text_up_to_response, add_special_tokens=False)
                                                token_start = len(tokens_up_to_response)
                                                if token_start < generated_ids.shape[1]:
                                                    generated_ids = generated_ids[:, token_start:]
                                                    extraction_successful = True
                                        except Exception:
                                            pass
                        except Exception:
                            pass
            else:
                # No sequences or already extracted, use original slice
                generated_ids = generated_ids[:, input_length:]
        
        # Check if generated_ids is empty or all padding
        if generated_ids.numel() == 0:
            print(f'⚠ CRITICAL: generated_ids is empty after slicing! Full shape was {generated_ids.shape if hasattr(generated_ids, "shape") else "unknown"}')
        elif generated_ids.shape[1] == 0:
            print(f'⚠ CRITICAL: Generated sequence length is 0! Model may have generated nothing.')
        
        # Debug: Check what tokens are being generated
        if generated_ids.numel() > 0:
            # Get pad and eos token IDs
            pad_token_id = self._processing_class.pad_token_id
            eos_token_id = self._processing_class.eos_token_id
            
            # Check if all tokens are pad/eos (indicates model is not generating)
            non_special_mask = (generated_ids != pad_token_id) & (generated_ids != eos_token_id)
            num_non_special = non_special_mask.sum().item()
            total_tokens = generated_ids.numel()
            
            if num_non_special == 0 and total_tokens > 0:
                print(f'⚠ WARNING: Model generated only pad/eos tokens! This suggests the model may have collapsed.')
                print(f'  Sample generated token IDs (first 10): {generated_ids[0, :min(10, generated_ids.shape[1])].tolist()}')
            
        # Clear cache after generation
        torch.cuda.empty_cache()
        
        # Store predictions for JSONL output
        # Decode predictions (generated summary only, without special tokens)
        # CRITICAL: Check if model is generating only pad tokens before decoding
        pad_token_id = self._processing_class.pad_token_id
        eos_token_id = self._processing_class.eos_token_id
        
        # Check if all generated tokens are pad/eos
        if generated_ids.numel() > 0:
            non_pad_eos_mask = (generated_ids != pad_token_id) & (generated_ids != eos_token_id)
            num_valid_tokens = non_pad_eos_mask.sum().item()
            total_tokens = generated_ids.numel()
            
            if num_valid_tokens == 0:
                print(f'⚠ CRITICAL: Model generated only pad/eos tokens (model collapse or checkpoint too early). Trying permissive generation...')
                # Try with sampling and higher temperature to break out of pad token loop
                with torch.no_grad():
                    generation_kwargs_force = {
                        'input_ids': input_ids,
                        'use_cache': True,
                        'max_new_tokens': min(50, self.generation_max_length or 512),  # Shorter for early checkpoints
                        'min_new_tokens': 5,  # Force at least 5 tokens
                        'do_sample': True,
                        'temperature': 1.2,  # Higher temperature to encourage diversity
                        'top_p': 0.95,
                        'top_k': 50,
                        'pad_token_id': pad_token_id,
                        'eos_token_id': eos_token_id,
                        'repetition_penalty': 1.0,  # No penalty
                    }
                    try:
                        generated_ids_force = model.generate(**generation_kwargs_force)
                        # Check if this produced better results
                        generated_ids_force = generated_ids_force[:, input_length:]
                        non_pad_eos_mask_force = (generated_ids_force != pad_token_id) & (generated_ids_force != eos_token_id)
                        num_valid_force = non_pad_eos_mask_force.sum().item()
                        if num_valid_force > 0:
                            print(f'  ✓ Forced generation produced {num_valid_force} valid tokens! Using this instead.')
                            generated_ids = generated_ids_force
                        else:
                            print(f'  ✗ Forced generation also failed - model may have collapsed or checkpoint too early.')
                    except Exception as e:
                        print(f'  ✗ Forced generation failed: {e}')
        
        # Strip leading pad/eos and truncate at first EOS to avoid gibberish/hallucination.
        # If model doesn't emit EOS, truncate at max reasonable summary length (256 tokens).
        pad_token_id = self._processing_class.pad_token_id
        eos_token_id = self._processing_class.eos_token_id
        MAX_SUMMARY_TOKENS_BEFORE_TRUNCATE = 256  # Summaries rarely exceed this; cuts off runaway generation
        stripped_list = []
        for i in range(generated_ids.shape[0]):
            row = generated_ids[i].cpu().tolist()
            # Truncate at first EOS (model's natural stop); if no EOS, cap at MAX_SUMMARY_TOKENS_BEFORE_TRUNCATE
            first_eos = next((j for j, tid in enumerate(row) if tid == eos_token_id), None)
            if first_eos is not None:
                row = row[:first_eos]
            elif len(row) > MAX_SUMMARY_TOKENS_BEFORE_TRUNCATE:
                row = row[:MAX_SUMMARY_TOKENS_BEFORE_TRUNCATE]
            start = 0
            while start < len(row) and row[start] in (pad_token_id, eos_token_id):
                start += 1
            stripped_list.append(row[start:] if start < len(row) else row)
        max_len = max(len(r) for r in stripped_list) if stripped_list else 0
        if max_len > 0:
            padded = [r + [pad_token_id] * (max_len - len(r)) for r in stripped_list]
            generated_ids = torch.tensor(padded, device=generated_ids.device, dtype=generated_ids.dtype)
            decoded_predictions = self._processing_class.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            decoded_predictions = [""] * generated_ids.shape[0]
        
        # Clean up decoded predictions - remove special tokens and backslashes (same as in compute_metrics)
        def clean_text(text):
            """Clean decoded text by removing special tokens and unwanted characters."""
            # Truncate at [/SAK] - Normistral sometimes emits this instead of EOS and repeats it
            if '[/SAK]' in text:
                text = text.split('[/SAK]')[0].strip()
            # Plain/Gemma format: training ends output with \n\n### - truncate at first ### to drop post-summary garbage
            if '###' in text:
                text = text.split('###')[0].strip()
            # Remove common chat format tokens (Llama-2)
            text = text.replace('[/INST]', '').replace('[INST]', '')
            text = text.replace('</s>', '').replace('<s>', '')
            # Remove Llama-3 specific tokens
            text = text.replace('<|begin_of_text|>', '')
            text = text.replace('<|end_of_text|>', '')
            text = text.replace('<|eot_id|>', '')
            text = text.replace('<|start_header_id|>', '')
            text = text.replace('<|end_header_id|>', '')
            # Remove user/assistant header markers (with any content between)
            # This regex removes patterns like <|start_header_id|>assistant<|end_header_id|>
            text = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', text)
            # ChatML tokens (EuroLLM and similar)
            text = text.replace('<|im_start|>', '').replace('<|im_end|>', '')
            # Strip leading <unk> (tokenizer often decodes pad/special as <unk>)
            while text.startswith('<unk>'):
                text = text[5:].lstrip()
            # Also remove standalone "assistant" text that might appear at the start (from header)
            # Remove "assistant\n\n" or "assistant " at the beginning
            text = re.sub(r'^assistant\s*\n*\s*', '', text, flags=re.IGNORECASE)
            # Plain/Gemma format: strip echoed "Oppsummering:" or "###" at start
            if text.lstrip().startswith('Oppsummering:'):
                text = text.lstrip()[len('Oppsummering:'):].lstrip()
            while text.startswith('###'):
                text = text.lstrip()[3:].lstrip()
            # Remove backslashes (common issue with Llama-2 chat models)
            text = text.replace('\\', '')
            # Strip trailing ### or ## (Gemma/plain format end markers)
            text = re.sub(r'\s*#+\s*$', '', text)
            # Remove multiple spaces
            text = ' '.join(text.split())
            return text.strip()
        
        def truncate_repeated_paragraphs(text, min_words=12):
            """Truncate at first repeated paragraph (model sometimes repeats same content 2-4x)."""
            if not text or len(text.split()) < min_words * 2:
                return text
            words = text.split()
            # Look for a chunk of min_words that appears again later (repetition)
            for i in range(0, len(words) - min_words):
                chunk = ' '.join(words[i:i + min_words])
                # Check if this chunk appears again later
                rest = ' '.join(words[i + min_words:])
                if len(rest) < len(chunk) * 0.5:
                    continue
                if chunk in rest:
                    return ' '.join(words[:i + min_words]).strip()
            return text
        
        def fix_mid_sentence_start(text):
            """Try to fix predictions that start mid-sentence by finding the first complete sentence."""
            if not text or len(text.strip()) < 10:
                return text
            
            # If text starts with lowercase letter, comma, period, or space, it's likely mid-sentence
            first_char = text.strip()[0] if text.strip() else ''
            if first_char in [',', '.', ' ', '\n'] or (first_char and first_char.islower()):
                # Try to find the first sentence boundary (period, exclamation, question mark followed by space and capital)
                # Look for patterns like ". [A-Z]" or "! [A-Z]" or "? [A-Z]"
                sentence_end_pattern = r'[.!?]\s+[A-ZÆØÅ]'
                match = re.search(sentence_end_pattern, text)
                if match:
                    # Found a sentence boundary, start from there
                    start_pos = match.end() - 1  # Position of the capital letter
                    fixed_text = text[start_pos:].strip()
                    return fixed_text
                else:
                    # No clear sentence boundary, try to find first capital letter
                    capital_match = re.search(r'[A-ZÆØÅ]', text)
                    if capital_match:
                        start_pos = capital_match.start()
                        fixed_text = text[start_pos:].strip()
                        return fixed_text
            
            return text
        
        cleaned_predictions = []
        for i, pred in enumerate(decoded_predictions):
            cleaned = clean_text(pred)
            # Truncate at first repeated paragraph (fixes Arna-style repetition)
            cleaned = truncate_repeated_paragraphs(cleaned)
            # Try to fix mid-sentence starts
            fixed = fix_mid_sentence_start(cleaned)
            cleaned_predictions.append(fixed)
        
        # Check for empty predictions (warn once per eval run to avoid log spam; always warn if all empty)
        empty_count = sum(1 for p in cleaned_predictions if not p)
        if empty_count > 0:
            if not getattr(self, '_empty_warning_shown', False):
                print(f'⚠ WARNING: Some predictions are empty after cleaning (model may output EOS/special only for some examples). Will report total at end.')
                self._empty_warning_shown = True
            if empty_count == len(cleaned_predictions):
                print(f'⚠ CRITICAL: ALL predictions are empty! The model may have collapsed during training.')
                print(f'  This could indicate:')
                print(f'    1. Model loss went to NaN or extreme values')
                print(f'    2. Model is outputting only EOS/pad tokens immediately')
                print(f'    3. Training for too many epochs (5000+) may have caused overfitting/collapse')
                print(f'  Recommendation: Check training loss curves and consider early stopping.')
        
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
            
            # Log peak memory to wandb once per evaluation (no step= to avoid "step 0 < current step" when evaluating multiple checkpoints)
            if wandb.run is not None and not hasattr(self, '_peak_memory_logged'):
                wandb.log(peak_memory)
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
    force_recompute: bool = False,  # If True, skip "already evaluated" check and load model for re-evaluation
    examples_suffix: Optional[str] = None,  # e.g. "examples_1000" when val_data_size != 500
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
    num_gpus = torch.cuda.device_count()

    # Use model parallelism (device_map="auto") for multi-GPU - compatible with generation, unlike FSDP
    if use_multi_gpu and num_gpus > 1:
        # Try using accelerate for better device_map control
        try:
            from accelerate import infer_auto_device_map, dispatch_model
            from accelerate.utils import get_balanced_memory
            
            # First load model to CPU to get its structure
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
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
            
            # Build device map for balanced layer distribution across GPUs
            device_summary = {}
            for layer_name, device in device_map.items():
                device_str = f"cuda:{device}" if isinstance(device, int) else str(device)
                device_summary[device_str] = device_summary.get(device_str, 0) + 1
            
            # Reload with device_map
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device_map,  # Use accelerate's device_map
                token=hf_token,
                low_cpu_mem_usage=True,
            )
            
        except (ImportError, Exception) as e:
            # Fallback to simple device_map when accelerate unavailable
            device_map_strategy = "auto"
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device_map_strategy,
                token=hf_token,
                low_cpu_mem_usage=True,
            )
    else:
        device_map_strategy = "cuda:0"
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map_strategy,
            token=hf_token,
            low_cpu_mem_usage=True,
        )
    
    # Verify device_map split model across GPUs (multi-GPU only)
    if use_multi_gpu and num_gpus > 1:
        unique_devices = set(str(p.device) for p in base_model.parameters())
        if len(unique_devices) <= 1:
            print(f"Warning: Base model only on {unique_devices} - device_map may not have worked.")
    
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
        # Unified naming: checkpoint-{step} (folder regular_checkpoints/ indicates type)
        regular_ckpt_name = f"checkpoint-{checkpoint_step_int}"
        regular_ckpt_path = os.path.join(regular_ckpt_dir, regular_ckpt_name)

        # Check if backup already exists and has adapter files
        backup_adapter_file = os.path.join(regular_ckpt_path, "adapter_model.safetensors")
        if os.path.exists(regular_ckpt_path) and os.path.exists(backup_adapter_file):
            pass  # Backup exists, skip
        else:
            if os.path.exists(regular_ckpt_path):
                try:
                    shutil.rmtree(regular_ckpt_path)
                except Exception as e:
                    print(f"Warning: Failed to remove incomplete backup: {e}")
            try:
                shutil.copytree(checkpoint_dir, regular_ckpt_path)
            except Exception as e:
                print(f"Warning: Failed to copy regular checkpoint: {e}. Continuing with evaluation.")

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
        # Unified naming: checkpoint-{step} (folder major_checkpoints/ indicates type)
        major_ckpt_name = f"checkpoint-{checkpoint_step_int}"
        major_ckpt_path = os.path.join(major_ckpt_dir, major_ckpt_name)
        
        # Check if backup already exists and has adapter files
        backup_adapter_file = os.path.join(major_ckpt_path, "adapter_model.safetensors")
        if os.path.exists(major_ckpt_path) and os.path.exists(backup_adapter_file):
            pass  # Backup exists, skip
        else:
            if os.path.exists(major_ckpt_path):
                try:
                    shutil.rmtree(major_ckpt_path)
                except Exception as e:
                    print(f"Warning: Failed to remove incomplete backup: {e}")
            try:
                shutil.copytree(checkpoint_dir, major_ckpt_path)
            except Exception as e:
                print(f"Warning: Failed to copy major checkpoint: {e}. Continuing with evaluation.")
    
    # Check if this checkpoint was already evaluated using utility functions
    # Skip when force_recompute=True (monitor detected checkpoint newer than stale eval)
    # When examples_suffix (e.g. "examples_1000"), check the suffixed file so 1000-example evals don't overwrite 500
    eval_results_file = get_eval_results_path(checkpoint_dir, model_dir, examples_suffix=examples_suffix)
    old_eval_results_file = get_old_eval_results_path(checkpoint_dir)
    
    # Retrain-from-scratch: eval results older than training_started.txt = from previous run
    stale_eval_from_previous_run = False
    if not force_recompute and os.path.exists(eval_results_file):
        training_started_file = os.path.join(model_dir, "training_started.txt")
        if os.path.exists(training_started_file):
            try:
                eval_mtime = os.path.getmtime(eval_results_file)
                training_started_mtime = os.path.getmtime(training_started_file)
                if eval_mtime < training_started_mtime:
                    stale_eval_from_previous_run = True
            except OSError:
                pass
    
    if not force_recompute and not stale_eval_from_previous_run:
        # Check if results exist in new location
        if os.path.exists(eval_results_file):
            # Check if extended metrics are missing (if extended evaluation is available)
            # If extended metrics should be computed but aren't present, allow re-evaluation
            if EXTENDED_EVAL_AVAILABLE:
                try:
                    existing_results = load_eval_results(checkpoint_dir, model_dir)
                    if existing_results is None:
                        pass  # Can't parse, allow re-evaluation
                        # Don't raise AlreadyEvaluatedError - allow re-evaluation
                    else:
                        # Check if this is a major checkpoint that should have BERTScore
                        existing_results = load_eval_results(checkpoint_dir, model_dir, examples_suffix=examples_suffix)
                        
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
                            pass  # Fall through to re-evaluate
                            # Don't raise AlreadyEvaluatedError - allow re-evaluation
                        elif not has_extended_metrics:
                            pass  # Fall through to re-evaluate
                            # Don't raise AlreadyEvaluatedError - allow re-evaluation
                        # Check if NLI faithfulness is requested but missing (or details JSONL absent)
                        elif include_nli_faithfulness:
                            if should_skip_faithfulness_update(
                                existing_results, checkpoint_dir, model_dir,
                                examples_suffix=examples_suffix,
                            ):
                                raise AlreadyEvaluatedError(
                                    f"Checkpoint {checkpoint_dir} appears to be already evaluated "
                                    f"(results file exists at {eval_results_file}). "
                                    f"Skipping evaluation."
                                )
                            # Fall through: missing aggregates or missing *-faithfulness-details-*.jsonl
                        else:
                            # Extended metrics are present, skip evaluation
                            raise AlreadyEvaluatedError(
                                f"Checkpoint {checkpoint_dir} appears to be already evaluated "
                                f"(results file exists at {eval_results_file}). "
                                f"Skipping evaluation."
                            )
                except (json.JSONDecodeError, ValueError, KeyError):
                    pass  # Can't parse, allow re-evaluation
                    # Don't raise AlreadyEvaluatedError - allow re-evaluation
            else:
                # Extended evaluation not available, skip if results exist
                raise AlreadyEvaluatedError(
                    f"Checkpoint {checkpoint_dir} appears to be already evaluated "
                    f"(results file exists at {eval_results_file}). "
                    f"Skipping evaluation."
                )
        
        # Also check old location for backwards compatibility
        # When examples_suffix is set (e.g. examples_1000), the old file is from 500-example runs - do NOT skip
        if not examples_suffix and os.path.exists(old_eval_results_file):
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
                            pass  # Allow re-evaluation
                        elif not has_extended_metrics_old:
                            pass  # Allow re-evaluation
                        elif include_nli_faithfulness:
                            if should_skip_faithfulness_update(
                                existing_results_old, checkpoint_dir, model_dir,
                                examples_suffix=examples_suffix,
                            ):
                                raise AlreadyEvaluatedError(
                                    f"Checkpoint {checkpoint_dir} appears to be already evaluated "
                                    f"(old results file exists at {old_eval_results_file}). "
                                    f"Skipping evaluation."
                                )
                            # Allow re-evaluation: missing aggregates or missing details JSONL
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
        
    
    # Check if checkpoint directory is empty or only contains eval_results (no adapter files).
    # This must be outside the force_recompute block — can't re-evaluate without adapter weights.
    dir_contents = os.listdir(checkpoint_dir)
    if len(dir_contents) == 0:
        raise AlreadyEvaluatedError(
            f"Checkpoint {checkpoint_dir} is an empty directory "
            f"(adapter files may have been cleaned up). Skipping."
        )
    if len(dir_contents) == 1 and 'eval_results' in dir_contents:
        raise AlreadyEvaluatedError(
            f"Checkpoint {checkpoint_dir} only contains 'eval_results' "
            f"(adapter files have been cleaned up). Skipping."
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
            # Fallback: backup dir may be empty (e.g. major step only in major_checkpoints/).
            # Try alternate locations: main, backup dirs (both new "checkpoint-X" and legacy names).
            alt_paths = [
                os.path.join(model_dir, f"checkpoint-{checkpoint_step_int}"),
                os.path.join(model_dir, "major_checkpoints", f"checkpoint-{checkpoint_step_int}"),
                os.path.join(model_dir, "major_checkpoints", f"major-checkpoint-{checkpoint_step_int}"),
                os.path.join(model_dir, "regular_checkpoints", f"checkpoint-{checkpoint_step_int}"),
                os.path.join(model_dir, "regular_checkpoints", f"regular-checkpoint-{checkpoint_step_int}"),
            ]
            for alt in alt_paths:
                if alt != checkpoint_dir and os.path.exists(os.path.join(alt, "adapter_config.json")):
                    print(f"Warning: {checkpoint_dir} is empty or missing adapter files.")
                    print(f"Using alternate checkpoint: {alt}")
                    checkpoint_dir = alt
                    adapter_config_path = os.path.join(alt, "adapter_config.json")
                    adapter_model_path = os.path.join(alt, "adapter_model.safetensors")
                    if not os.path.exists(adapter_model_path):
                        adapter_model_path = os.path.join(alt, "adapter_model.bin")
                    dir_contents = os.listdir(checkpoint_dir)
                    break
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
    
    # Clear CUDA cache before loading adapter to avoid device-side assert errors
    # This is especially important for GPT-J models which can have CUDA state issues
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Synchronize to ensure all previous operations are complete
        torch.cuda.synchronize()
        print("Cleared CUDA cache before loading PEFT adapter")
    
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
    except (ValueError, TypeError, RuntimeError) as e:
        error_str = str(e)
        
        # Handle CUDA device-side assert errors (common with GPT-J models)
        if "CUDA error" in error_str or "device-side assert" in error_str:
            print(f"⚠ CUDA error detected when loading PEFT adapter: {e}")
            print("Attempting recovery by resetting CUDA state...")
            
            # Clear CUDA cache and reset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Try to reset CUDA state by clearing all caches
                for i in range(torch.cuda.device_count()):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            
            # Retry loading with explicit error handling
            try:
                print("Retrying PEFT adapter load after CUDA reset...")
                model = PeftModel.from_pretrained(
                    base_model,
                    checkpoint_dir,
                    is_trainable=False,
                )
                print("✓ Successfully loaded PEFT adapter after CUDA reset")
            except Exception as retry_error:
                print(f"⚠ Retry also failed: {retry_error}")
                print("This may indicate a corrupted checkpoint or model/adapter mismatch.")
                print("Try:")
                print("  1. Verify the checkpoint was saved correctly during training")
                print("  2. Check if the base model matches the checkpoint's base model")
                print("  3. Try loading with CUDA_LAUNCH_BLOCKING=1 for more detailed error info")
                raise RuntimeError(
                    f"Failed to load PEFT adapter after CUDA reset. "
                    f"Original error: {e}, Retry error: {retry_error}"
                ) from retry_error
        
        # Handle corrupted adapter_config.json files (e.g., typos like 'corda_config' instead of 'lora_config')
        elif "Can't find 'adapter_config.json'" in error_str:
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
    elif 'gemma-7b-it' in model_name_lower:
        return min(8, default_batch_size)
    elif 'normistral-11b-long' in model_name_lower:
        return min(4, default_batch_size)  # Wider context variant: keep more headroom
    elif 'normistral-11b' in model_name_lower:
        return min(6, default_batch_size)  # Increase from 4 to 6
    # Medium models (2-7B)
    elif ('gemma-2b' in model_name_lower or 'viking-7b' in model_name_lower or 
          'normistral-7b' in model_name_lower or 'norwai-mistral-7b' in model_name_lower or
          'nb-gpt-j-6b' in model_name_lower):
        return min(16, default_batch_size)
    # Medium-large models (8-9B) - similar to gemma-2-9b
    elif 'llama-3.1-8b' in model_name_lower or 'eurollm-9b' in model_name_lower or 'norskgpt-llama3-8b' in model_name_lower:
        return min(8, default_batch_size)  # Similar to gemma-7b
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
    val_data_seed: int = VAL_DATA_SEED,
    val_beam_size: int = VAL_BEAM_SIZE,
    use_greedy: bool = True,
    use_multi_gpu: bool = False,
    wandb_project: Optional[str] = "lm-evaluation",
    wandb_entity: Optional[str] = None,
    wandb_disabled: bool = False,
    wandb_run_name: Optional[str] = None,
    wandb_group: Optional[str] = None,
    major_checkpoint_interval: int = 500,  # Every Nth step is major (gets BERTScore). Default: 500 (every 500 steps = checkpoint-500, checkpoint-1000, etc.)
    include_nli_faithfulness: bool = False,  # Enable NLI faithfulness evaluation (subset: see nli_subset_size)
    nli_subset_size: int = NLI_DEFAULT_SUBSET_SIZE,  # default 100; set == val_data_size for full-set NLI
    keep_existing: bool = False,
    force_recompute: bool = False,  # If True, skip loading existing results and re-run evaluation (for rerun with corrected prompt)
    predict_only: bool = False,
    metrics_only: bool = False,
    update_metrics: Optional[Set[str]] = None,  # e.g. {"rouge", "hygiene", "bertscore", "faithfulness"}
):
    """Load a PEFT checkpoint and run evaluation with model parallelism support.

    Modes (mutually exclusive):
        default         — generate predictions + compute all metrics
        predict_only    — generate predictions, save JSONL, skip metrics
        metrics_only    — load JSONL, compute all applicable metrics (no model loading)
        update_metrics  — load JSONL + existing results, recompute selected metrics, merge
    """
    
    # Convert checkpoint_dir to absolute path
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    
    # Extract checkpoint step number early (needed for checking existing results)
    # Use shared utility so regular-/major-checkpoint names are parsed correctly.
    checkpoint_name, checkpoint_step_int = get_checkpoint_name_and_step(checkpoint_dir)
    
    # Determine if this is a "major" checkpoint for tiered evaluation
    # Major checkpoints: every Nth checkpoint (based on major_checkpoint_interval)
    # Normal checkpoints: all others
    # This allows selective computation of expensive metrics (BERTScore, NLI)
    # Tiered evaluation strategy:
    #   - Normal checkpoints: ROUGE + Hygiene only (~2 min)
    #   - Major checkpoints: ROUGE + Hygiene + BERTScore (~3-4 min)
    #   - NLI Faithfulness: Only for major checkpoints, and only if include_nli_faithfulness is enabled
    is_major_checkpoint_bool = is_major_checkpoint(checkpoint_step_int, major_checkpoint_interval)
    
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
    
    # Suffix for val_data_size variants: 1000 → "examples_1000", 500 → None (default/backward compatible)
    examples_suffix = f"examples_{val_data_size}" if val_data_size != 500 else None
    
    # Get results file path using utility function
    results_file = get_eval_results_path(checkpoint_dir, model_dir_eval, examples_suffix=examples_suffix)
    predictions_file = get_predictions_file_path(checkpoint_dir, model_dir_eval, examples_suffix=examples_suffix)

    # ---------------------------------------------------------------
    # Early-return paths for --metrics-only and --update-* modes
    # (no model loading, no prediction generation)
    # ---------------------------------------------------------------
    if metrics_only or update_metrics:
        if not os.path.exists(predictions_file):
            raise FileNotFoundError(
                f"JSONL predictions file not found: {predictions_file}\n"
                f"Run --predict_only first to generate predictions."
            )
        input_texts, prediction_texts, reference_texts = load_predictions_jsonl(predictions_file)
        print(f"Loaded {len(prediction_texts)} predictions from {predictions_file}")

        from utils.metrics import compute_metrics_from_texts

        if update_metrics:
            # Selective update: load existing results, recompute requested metrics, merge
            existing_results = {}
            if os.path.exists(results_file):
                with open(results_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)

            # Unless force_recompute, skip metrics that already exist in the results
            _METRIC_PRESENCE_KEYS = {
                "rouge": "eval_rouge1",
                "hygiene": "eval_hygiene_mean_compression_ratio",
                "bertscore": "eval_reference_bertscore_f1_mean",
                "faithfulness": "eval_faithfulness",
            }
            actually_update = set(update_metrics)
            if not force_recompute:
                for metric_name in list(actually_update):
                    if metric_name == "faithfulness":
                        details_path = get_faithfulness_details_path(
                            checkpoint_dir, model_dir_eval, examples_suffix=examples_suffix,
                        )
                        if should_skip_faithfulness_update(
                            existing_results, checkpoint_dir, model_dir_eval,
                            examples_suffix=examples_suffix,
                        ):
                            print(
                                f"⚠ Skipping faithfulness: aggregates and details file already present "
                                f"({os.path.basename(results_file)}; {os.path.basename(details_path)}) "
                                f"(use --force_recompute to overwrite)"
                            )
                            actually_update.discard(metric_name)
                        elif nli_faithfulness_aggregate_present(existing_results):
                            print(
                                f"⚠ Recomputing faithfulness: details file missing "
                                f"({os.path.basename(details_path)}); aggregates will be refreshed."
                            )
                        continue
                    presence_key = _METRIC_PRESENCE_KEYS.get(metric_name)
                    if presence_key and presence_key in existing_results and existing_results[presence_key] is not None:
                        print(f"⚠ Skipping {metric_name}: already present in {os.path.basename(results_file)} "
                              f"(use --force_recompute to overwrite)")
                        actually_update.discard(metric_name)
                if not actually_update:
                    print(f"All requested metrics already present for {checkpoint_name}. Nothing to update.")
                    return existing_results, None

            include_flags = {
                "include_rouge": "rouge" in actually_update,
                "include_hygiene": "hygiene" in actually_update,
                "include_bertscore": "bertscore" in actually_update,
                "include_faithfulness": "faithfulness" in actually_update,
            }
            # Build NLI subset texts + incremental details file path
            nli_in = nli_pred = None
            nli_indices = None
            faith_details = None
            if "faithfulness" in actually_update:
                use_first_n = val_data_size > nli_subset_size
                nli_indices = get_or_create_fixed_nli_subset(
                    total_examples=len(input_texts),
                    model_dir=model_dir_eval,
                    subset_size=nli_subset_size,
                    use_first_n_for_extended=use_first_n,
                )
                nli_in = [input_texts[i] for i in nli_indices]
                nli_pred = [prediction_texts[i] for i in nli_indices]
                faith_details = get_faithfulness_details_path(
                    checkpoint_dir, model_dir_eval, examples_suffix=examples_suffix,
                )
            result = compute_metrics_from_texts(
                input_texts, prediction_texts, reference_texts,
                nli_input_texts=nli_in, nli_prediction_texts=nli_pred,
                nli_example_indices=nli_indices,
                faithfulness_details_file=faith_details,
                **include_flags,
            )
            # Re-read the results file right before merging to minimise the
            # race window when parallel jobs update different metrics on the
            # same checkpoint (e.g. --update-rouge and --update-faithfulness).
            if os.path.exists(results_file):
                with open(results_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
            existing_results.update(result["metrics"])
            existing_results.setdefault("_timing", {}).update(result["timing"])
            existing_results["checkpoint_name"] = checkpoint_name
            existing_results["checkpoint_step"] = checkpoint_step_int
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(existing_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"Updated metrics {actually_update} in {results_file}")
            return existing_results, None

        else:
            # metrics_only: compute all applicable metrics from JSONL
            is_major = is_major_checkpoint(checkpoint_step_int, major_checkpoint_interval)
            nli_in = nli_pred = None
            nli_indices = None
            faith_details = None
            if include_nli_faithfulness and is_major:
                use_first_n = val_data_size > nli_subset_size
                nli_indices = get_or_create_fixed_nli_subset(
                    total_examples=len(input_texts),
                    model_dir=model_dir_eval,
                    subset_size=nli_subset_size,
                    use_first_n_for_extended=use_first_n,
                )
                nli_in = [input_texts[i] for i in nli_indices]
                nli_pred = [prediction_texts[i] for i in nli_indices]
                faith_details = get_faithfulness_details_path(
                    checkpoint_dir, model_dir_eval, examples_suffix=examples_suffix,
                )
            result = compute_metrics_from_texts(
                input_texts, prediction_texts, reference_texts,
                include_rouge=True,
                include_hygiene=True,
                include_bertscore=is_major,
                include_faithfulness=include_nli_faithfulness and is_major,
                nli_input_texts=nli_in, nli_prediction_texts=nli_pred,
                nli_example_indices=nli_indices,
                faithfulness_details_file=faith_details,
            )
            all_results = result["metrics"]
            all_results["_timing"] = result["timing"]
            all_results["checkpoint_name"] = checkpoint_name
            all_results["checkpoint_step"] = checkpoint_step_int
            all_results["is_major_checkpoint"] = is_major
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"Saved metrics-only results to {results_file}")
            return all_results, None
    
    # If results file exists, load and log to Wandb without re-evaluating
    # Skip loading when force_recompute=True (e.g. monitor detected checkpoint newer than stale eval from previous run)
    if force_recompute and os.path.exists(results_file):
        print(f"ℹ force_recompute: Skipping existing results, re-running evaluation (e.g. rerun with corrected prompt)")
    if os.path.exists(results_file) and not force_recompute:
        print(f"⚠ Checkpoint {checkpoint_name} already evaluated. Loading existing results...")
        if keep_existing:
            existing_results = load_eval_results(checkpoint_dir, model_dir_eval, examples_suffix=examples_suffix)
            if existing_results is None:
                print(f"✓ keep_existing enabled and results file exists. Skipping checkpoint {checkpoint_name} without overwriting.")
                return {
                    "checkpoint_name": checkpoint_name,
                    "checkpoint_step": checkpoint_step_int,
                    "status": "skipped_keep_existing_existing_file",
                    "result_file": results_file,
                }, None
            print(f"✓ keep_existing enabled. Reusing existing results for {checkpoint_name} (no overwrite).")
            return existing_results, None
        
        try:
            existing_results = load_eval_results(checkpoint_dir, model_dir_eval, examples_suffix=examples_suffix)
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
                        if not should_skip_faithfulness_update(
                            existing_results, checkpoint_dir, model_dir_eval,
                            examples_suffix=examples_suffix,
                        ):
                            missing_extended_metrics = True
                            print(
                                f"⚠ Checkpoint {checkpoint_name} missing NLI faithfulness or "
                                f"faithfulness-details file (--include_nli_faithfulness). Re-evaluating..."
                            )
                
                if all_zeros or missing_extended_metrics:
                    if all_zeros:
                        print(f"⚠ Warning: All ROUGE scores are 0.00 - this indicates a failed evaluation.")
                        print(f"   Re-evaluating checkpoint {checkpoint_name}...")
                    # Fall through to normal evaluation instead of returning
                else:
                    if keep_existing:
                        print(f"✓ keep_existing enabled and existing results complete. Skipping re-evaluation for {checkpoint_name}.")
                        return existing_results, None
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
                    
                    # Use consistent run name and ID for combining all checkpoints (separate per val_data_size)
                    # 500 vs 1000 examples → separate wandb runs so metrics don't mix
                    model_dir_eval_cached = get_model_dir_from_checkpoint(checkpoint_dir)
                    wandb_run_id_suffix_cached = f"_{examples_suffix}" if examples_suffix else ""
                    wandb_run_id_file_cached = os.path.join(model_dir_eval_cached, "all_eval_results", f".wandb_run_id{wandb_run_id_suffix_cached}")
                    
                    if wandb_run_name:
                        consistent_run_name_cached = f"{wandb_run_name}{wandb_run_id_suffix_cached}" if examples_suffix else wandb_run_name
                    else:
                        consistent_run_name_cached = f"{clean_model_name}_eval_all_checkpoints{wandb_run_id_suffix_cached}"
                    
                    # Try to load existing run ID to resume the same run
                    wandb_run_id_cached = None
                    if os.path.exists(wandb_run_id_file_cached):
                        try:
                            with open(wandb_run_id_file_cached, 'r') as f:
                                wandb_run_id_cached = f.read().strip()
                            if wandb_run_id_cached:
                                print(f">>> Found existing wandb run ID: {wandb_run_id_cached} (will resume same run)")
                        except Exception as e:
                            print(f">>> Warning: Could not load wandb run ID: {e}")
                    
                    is_major_cached = is_major_checkpoint(checkpoint_step_int, major_checkpoint_interval)
                    
                    # Initialize wandb with consistent run name and ID (separate run per val_data_size)
                    wandb_group_cached = wandb_group or f"{clean_model_name}_eval"
                    if examples_suffix:
                        wandb_group_cached = f"{wandb_group_cached}_{examples_suffix}"
                    wandb_kwargs_cached = {
                        "project": wandb_project,
                        "entity": wandb_entity,
                        "name": consistent_run_name_cached,
                        "group": wandb_group_cached,
                        "tags": [
                            "evaluation",
                            "all_checkpoints",
                            "cached",
                            f"val_{val_data_size}_examples",
                        ],
                        "config": {
                            "model": model_name,
                            "val_dataset": val_dataset_path,
                            "val_size": val_data_size,
                            "val_batch_size": val_batch_size,
                            "max_input_tokens": max_input_text_tokens,
                            "max_output_tokens": max_output_summary_tokens,
                            "num_gpus": num_gpus,
                            "major_checkpoint_interval": major_checkpoint_interval,
                        },
                        "reinit": True,
                    }
                    
                    # If we have a run ID, use it to resume the same run
                    if wandb_run_id_cached:
                        wandb_kwargs_cached["id"] = wandb_run_id_cached
                        wandb_kwargs_cached["resume"] = "allow"
                    
                    wandb.init(**wandb_kwargs_cached)
                    
                    # Save run ID for future checkpoints
                    if wandb.run is not None:
                        run_id_cached = wandb.run.id
                        os.makedirs(os.path.dirname(wandb_run_id_file_cached), exist_ok=True)
                        try:
                            with open(wandb_run_id_file_cached, 'w') as f:
                                f.write(run_id_cached)
                        except Exception as e:
                            print(f">>> Warning: Could not save wandb run ID: {e}")
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
    
    if is_main_process:
        print(f"Using batch size: {val_batch_size} for evaluation")

    def compute_metrics(eval_pred):
        """Compute ROUGE metrics using shared utility function."""
        if predict_only:
            return {}
        return compute_rouge_metrics(
            eval_pred=eval_pred,
            tokenizer=tokenizer,
            log_to_wandb=True,
            step=checkpoint_step_int,
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
    
    # Set chat template for models that use chat templates if missing (CRITICAL: must match training format)
    # Some models don't have chat_template in their tokenizer config, but we need it for consistent formatting
    # This ensures training and evaluation use the same prompt format
    model_config = get_model_config_by_hf_name(model_name)
    if model_config:
        template_type = model_config.prompt_config.template_type
        
        # Only set chat template if the model uses one and it's missing
        if template_type in ['mistral', 'llama2', 'llama3', 'llama3.1', 'chatml']:
            if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
                if is_main_process:
                    print(f"Setting chat template for {model_name} (template_type: {template_type})...")
                
                # Try to get template from official model first, then fallback to standard format
                official_model_map = {
                    'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
                    'llama2': 'meta-llama/Llama-2-7b-chat-hf',
                    'llama3': 'meta-llama/Meta-Llama-3-8B-Instruct',
                    'llama3.1': 'meta-llama/Llama-3.1-8B-Instruct',
                    'chatml': 'microsoft/DialoGPT-medium',  # ChatML format example
                }
                
                template_set = False
                if template_type in official_model_map:
                    try:
                        from transformers import AutoTokenizer as OfficialTokenizer
                        official_model = official_model_map[template_type]
                        official_tokenizer = OfficialTokenizer.from_pretrained(
                            official_model,
                            token=hf_token if hf_token else None
                        )
                        if hasattr(official_tokenizer, 'chat_template') and official_tokenizer.chat_template:
                            tokenizer.chat_template = official_tokenizer.chat_template
                            template_set = True
                            if is_main_process:
                                print(f"✓ Set {template_type} chat template from official model: {official_model}")
                    except Exception as e:
                        if is_main_process:
                            print(f"⚠ Could not load template from {official_model_map.get(template_type, 'official model')}: {e}")
                
                # Fallback to standard formats if official template not available
                if not template_set:
                    if template_type == 'mistral':
                        # Standard Mistral format: <s>[INST] {user_message} [/INST] {assistant_message}</s>
                        mistral_template = (
                            "{%- for message in messages %}"
                            "{%- if message['role'] == 'system' %}"
                            "{{ message['content'] }}"
                            "{%- elif message['role'] == 'user' %}"
                            "<s>[INST] {{ message['content'] }} [/INST]"
                            "{%- elif message['role'] == 'assistant' %}"
                            " {{ message['content'] }}</s>"
                            "{%- endif %}"
                            "{%- endfor %}"
                        )
                        tokenizer.chat_template = mistral_template
                        if is_main_process:
                            print(f"✓ Set Mistral chat template using standard format (fallback)")
                    elif template_type == 'llama2':
                        # Standard Llama-2 format: [INST] {user_message} [/INST] {assistant_message}
                        llama2_template = (
                            "{%- for message in messages %}"
                            "{%- if message['role'] == 'system' %}"
                            "<<SYS>>\n{{ message['content'] }}\n<</SYS>>\n\n"
                            "{%- elif message['role'] == 'user' %}"
                            "[INST] {{ message['content'] }} [/INST]"
                            "{%- elif message['role'] == 'assistant' %}"
                            " {{ message['content'] }}"
                            "{%- endif %}"
                            "{%- endfor %}"
                        )
                        tokenizer.chat_template = llama2_template
                        if is_main_process:
                            print(f"✓ Set Llama-2 chat template using standard format (fallback)")
                    elif template_type in ['llama3', 'llama3.1']:
                        # Standard Llama-3 format uses special tokens
                        # Note: Llama-3 models usually have chat_template, but if missing, we use manual format
                        # The manual format is already in PROMPT_LLAMA3, so we don't set chat_template here
                        # This ensures we use the manual format consistently
                        if is_main_process:
                            print(f"⚠ Llama-3/3.1 model missing chat_template - will use manual format from PROMPT_LLAMA3")
                    elif template_type == 'chatml':
                        # Standard ChatML format: <|im_start|>role\ncontent<|im_end|>\n
                        chatml_template = (
                            "{%- for message in messages %}"
                            "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
                            "{%- endfor %}"
                        )
                        tokenizer.chat_template = chatml_template
                        if is_main_process:
                            print(f"✓ Set ChatML chat template using standard format (fallback)")
                elif is_main_process:
                    print(f"✓ Chat template already set or not needed for template_type: {template_type}")

    # Create a clean model name for display
    clean_model_name = model_name.split('/')[-1].replace('-', '_')
    
    # Determine consistent run name and ID for combining all checkpoints (separate per val_data_size)
    # 500 vs 1000 examples → separate wandb runs so metrics don't mix
    model_dir_eval = get_model_dir_from_checkpoint(checkpoint_dir)
    wandb_run_id_suffix = f"_{examples_suffix}" if examples_suffix else ""
    wandb_run_id_file = os.path.join(model_dir_eval, "all_eval_results", f".wandb_run_id{wandb_run_id_suffix}")
    
    # Use provided run_name or create one based on model (without checkpoint step)
    if wandb_run_name:
        consistent_run_name = f"{wandb_run_name}{wandb_run_id_suffix}" if examples_suffix else wandb_run_name
    else:
        # Default: use model name only (no checkpoint step) so all checkpoints combine
        consistent_run_name = f"{clean_model_name}_eval_all_checkpoints{wandb_run_id_suffix}"
    
    # Try to load existing run ID to resume the same run
    wandb_run_id = None
    if os.path.exists(wandb_run_id_file):
        try:
            with open(wandb_run_id_file, 'r') as f:
                wandb_run_id = f.read().strip()
            if wandb_run_id:
                print(f">>> Found existing wandb run ID: {wandb_run_id} (will resume same run)")
        except Exception as e:
            print(f">>> Warning: Could not load wandb run ID: {e}")
    
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
        
        is_major_eval = is_major_checkpoint(checkpoint_step_int, major_checkpoint_interval)
        
        # Initialize wandb with consistent run name and ID (separate run per val_data_size)
        # If run_id exists, resume that run; otherwise create new one
        wandb_group_val = wandb_group or f"{clean_model_name}_eval"
        if examples_suffix:
            wandb_group_val = f"{wandb_group_val}_{examples_suffix}"
        wandb_kwargs = {
            "project": wandb_project,
            "entity": wandb_entity,
            "name": consistent_run_name,
            "group": wandb_group_val,  # Separate group per val_data_size (500 vs 1000)
            "tags": [
                "evaluation",
                "all_checkpoints",
                f"val_{val_data_size}_examples",
            ],
            "config": {
                "model": model_name,
                "val_dataset": val_dataset_path,
                "val_size": val_data_size,
                "nli_subset_size": nli_subset_size,
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
            "reinit": True,
        }
        
        # If we have a run ID, use it to resume the same run
        if wandb_run_id:
            wandb_kwargs["id"] = wandb_run_id
            wandb_kwargs["resume"] = "allow"  # Resume if run exists, create if not
        
        wandb.init(**wandb_kwargs)
        
        # Save run ID for future checkpoints
        if wandb.run is not None:
            run_id = wandb.run.id
            os.makedirs(os.path.dirname(wandb_run_id_file), exist_ok=True)
            try:
                with open(wandb_run_id_file, 'w') as f:
                    f.write(run_id)
                print(f">>> Saved wandb run ID to {wandb_run_id_file} for future checkpoints")
            except Exception as e:
                print(f">>> Warning: Could not save wandb run ID: {e}")
        
        print(f">>> wandb run initialized: {wandb.run.name if wandb.run else 'None'}")
        print(f">>> wandb run ID: {wandb.run.id if wandb.run else 'None'}")
        print(f">>> wandb run URL: {wandb.run.get_url() if wandb.run else 'None'}")
        print(f">>> All checkpoints will be logged to this same run for time-series plots")
        
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
            include_nli_faithfulness=include_nli_faithfulness,
            force_recompute=force_recompute,
            examples_suffix=examples_suffix,
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

    # Sample validation examples (reproducible; 1000 = canonical 500 + 500 more for comparison)
    val_data = sample_validation_data_reproducibly(
        val_data, min(val_data_size, len(val_data)), seed=val_data_seed
    )
    
    # Filter out examples with missing input or output
    val_data = [ex for ex in val_data if ex.get('input') and ex.get('output')]
    
    val_df = pd.DataFrame(val_data)
    val_dataset = Dataset.from_pandas(val_df)

    # Use shared formatting and tokenization functions
    def format_example_eval_wrapper(example):
        """Wrapper to call shared format_eval_example with model_name and tokenizer."""
        return format_eval_example(example, model_name, tokenizer=tokenizer)
    
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
        
        # Example prompts are logged to wandb config above
    
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

    # Initialize Trainer for evaluation only
    trainer = CausalLMTrainer(
        generation_max_length=max_output_summary_tokens,
        generation_num_beams=val_beam_size,
        eval_data_collator=eval_data_collator,
        use_greedy=use_greedy,
        checkpoint_dir=checkpoint_dir,  # Pass checkpoint directory to Trainer
        model_name=model_name,  # Pass model name for prompt format detection
        model=model,
        args=training_args,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Run evaluation
    if is_main_process:
        print("Running evaluation on checkpoint...")
    
    # Track total validation time
    validation_start_time = time.time()
    
    # Track prediction time (model.generate() during trainer.evaluate())
    prediction_start_time = time.time()
    eval_results = trainer.evaluate()
    prediction_time = time.time() - prediction_start_time
    
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
    
    # Save inputs, references, and predictions to JSONL file for ALL checkpoints.
    # Required for --metrics-only and --update-* modes to work on any checkpoint.
    predictions_file = None
    if is_main_process:
        predictions_file = get_predictions_file_path(checkpoint_dir, model_dir_eval, examples_suffix=examples_suffix)
        os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
        
        with open(predictions_file, 'w', encoding='utf-8') as f:
            num_examples = len(original_examples_for_jsonl)
            num_predictions = len(trainer._eval_predictions)
            num_to_save = min(num_examples, num_predictions)
            
            for i in range(num_to_save):
                entry = {
                    "input_text": original_examples_for_jsonl[i].get("input_text", ""),
                    "prompt": original_examples_for_jsonl[i].get("prompt", ""),
                    "reference": original_examples_for_jsonl[i].get("reference", ""),
                    "prediction": trainer._eval_predictions[i] if i < len(trainer._eval_predictions) else "",
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"Saved predictions to: {predictions_file}")
        print(f"  - {num_to_save} examples saved")

    if predict_only:
        print("--predict_only: skipping metrics computation.")
        return {"checkpoint_name": checkpoint_name, "checkpoint_step": checkpoint_step_int, "status": "predict_only"}, None
    
    # Run extended evaluation metrics (hygiene, BERTScore, NLI faithfulness)
    if is_main_process and not EXTENDED_EVAL_AVAILABLE:
        print("Warning: Extended evaluation not available. Only ROUGE metrics will be saved.")
    
    # Run extended evaluation metrics (hygiene, BERTScore, NLI faithfulness)
    include_faithfulness = False
    
    if is_main_process and EXTENDED_EVAL_AVAILABLE:
        try:
            # Load texts from saved JSONL (always available since we save for all checkpoints)
            input_texts, prediction_texts, reference_texts = load_predictions_jsonl(predictions_file)
            
            if len(input_texts) > 0 and len(prediction_texts) > 0 and len(reference_texts) > 0:
                # Determine which metrics to compute based on checkpoint type and user settings
                # Normal checkpoints: ROUGE + Hygiene only; Major: + BERTScore; NLI: only if --include_nli_faithfulness
                is_major_extended = is_major_checkpoint(checkpoint_step_int, major_checkpoint_interval)
                include_bertscore = is_major_extended
                include_faithfulness = include_nli_faithfulness and is_major_extended

                nli_input_texts, nli_prediction_texts, nli_reference_texts = input_texts, prediction_texts, reference_texts
                if include_faithfulness:
                    # Fixed NLI indices (saved under all_eval_results/) so all checkpoints stay comparable.
                    model_dir_eval = get_model_dir_from_checkpoint(checkpoint_dir)
                    use_first_n = val_data_size > nli_subset_size
                    nli_indices = get_or_create_fixed_nli_subset(
                        total_examples=len(input_texts),
                        model_dir=model_dir_eval,
                        subset_size=nli_subset_size,
                        use_first_n_for_extended=use_first_n,
                    )
                    if not nli_indices or len(nli_indices) == 0:
                        raise ValueError(
                            f"Fixed NLI subset is empty. Total: {len(input_texts)}, "
                            f"nli_subset_size={nli_subset_size!r}"
                        )
                    nli_input_texts, nli_prediction_texts, nli_reference_texts = apply_fixed_subset(
                        input_texts, prediction_texts, reference_texts, nli_indices
                    )
                    if len(nli_input_texts) == 0:
                        raise ValueError(f"After applying fixed subset, NLI input texts are empty.")

                # Run extended evaluation: Hygiene + optionally BERTScore (full set)
                assert extended_evaluate is not None, "extended_evaluate should be available when EXTENDED_EVAL_AVAILABLE is True"
                extended_results = extended_evaluate(
                    input_texts=input_texts,
                    prediction_texts=prediction_texts,
                    reference_texts=reference_texts,
                    print_output=False,
                    include_bertscore=include_bertscore,
                )
                extended_timing = extended_results.pop("_timing", {})
                extended_results["faithfulness"] = None

                # NLI faithfulness on subset (separate from extended_evaluate)
                if include_faithfulness:
                    assert len(nli_input_texts) <= len(input_texts), "NLI evaluation must use subset"
                    try:
                        from utils.faithfulness import NLIFaithfulnessGate
                        gate = NLIFaithfulnessGate()
                        faith_details_file = get_faithfulness_details_path(
                            checkpoint_dir, model_dir_eval, examples_suffix=examples_suffix,
                        )
                        faithfulness_out = gate.eval_faithfulness_incremental(
                            nli_input_texts, nli_prediction_texts,
                            nli_indices, faith_details_file,
                        )
                        if "_timing" in faithfulness_out:
                            extended_timing["nli_faithfulness_seconds"] = faithfulness_out.pop("_timing").get("nli_faithfulness_seconds", 0.0)
                        extended_results["faithfulness"] = faithfulness_out
                    except Exception as nli_error:
                        print(f"ERROR: NLI faithfulness evaluation failed: {nli_error}")
                        extended_results["faithfulness"] = None
                        extended_timing["nli_faithfulness_seconds"] = 0.0

                for category, metrics in extended_results.items():
                    if isinstance(metrics, dict):
                        if category == "faithfulness" and metrics is not None:
                            eval_results["eval_faithfulness"] = metrics
                        else:
                            for key, value in metrics.items():
                                eval_results[f"eval_{category}_{key}"] = value
                    else:
                        eval_results[f"eval_{category}"] = metrics

                total_validation_time = time.time() - validation_start_time
                eval_results["eval_timing"] = {
                    "total_validation_seconds": total_validation_time,
                    "prediction_seconds": prediction_time,
                    "rouge_seconds": extended_timing.get("rouge_seconds", 0.0),
                    "hygiene_seconds": extended_timing.get("hygiene_seconds", 0.0),
                    "bertscore_seconds": extended_timing.get("bertscore_seconds", 0.0),
                    "nli_faithfulness_seconds": extended_timing.get("nli_faithfulness_seconds", 0.0),
                    "extended_metrics_total_seconds": extended_timing.get("extended_metrics_total_seconds", 0.0),
                }
            else:
                print("Warning: No valid examples found in predictions file for extended evaluation")
        except Exception as e:
            print(f"ERROR: Extended evaluation failed: {e}. Continuing with ROUGE metrics only.")
            import traceback
            traceback.print_exc()
    # Note: Diagnostic messages for why extended evaluation didn't run are printed earlier
    
    if is_main_process:
        rouge_keys = [k for k in eval_results.keys() if 'rouge' in k.lower() and isinstance(eval_results[k], (int, float))]
        if rouge_keys:
            print("\nEvaluation Results:")
            for key in sorted(rouge_keys):
                v = eval_results[key]
                print(f"  {key}: {v:.4f}" if isinstance(v, (int, float)) else f"  {key}: {v}")
    
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
        
        # Call wandb.finish() so wandb marks the run as "Finished" instead of "Crashed".
        # The next checkpoint (new Python process) will resume this run via .wandb_run_id.
        print(">>> Evaluation results logged to wandb")
        print(f">>> Metrics logged with step={checkpoint_step_int} - will appear in time-series plots")
        print(f">>> View plots at: {wandb.run.get_url() if wandb.run else 'N/A'}")
        wandb.finish()
    elif wandb_disabled and is_main_process:
        print(">>> Wandb disabled - skipping wandb logging")
    
    # Save results to file (only on main process)
    if is_main_process:
        # Ensure all_eval_results dir exists (handles backup checkpoint paths)
        os.makedirs(os.path.join(model_dir_eval, "all_eval_results"), exist_ok=True)
        # Save per-checkpoint JSON so monitor and summary can read it
        saved_path = save_eval_results(
            results=eval_results,
            checkpoint_dir=checkpoint_dir,
            model_dir=model_dir_eval,
            save_to_old_location=True,  # Keep backwards compatibility (skipped when examples_suffix set)
            examples_suffix=examples_suffix,
        )
        print(f"Per-checkpoint results saved to: {saved_path}")
        
        # Update evaluation summary using utility function
        update_evaluation_summary(
            results=eval_results,
            checkpoint_dir=checkpoint_dir,
            model_name=model_name,
            val_dataset_path=val_dataset_path,
            model_dir=model_dir_eval,
            examples_suffix=examples_suffix,
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
    --model gemma-7b-it \\
    --checkpoint_dir models/gemma-7b-it_fsdp/checkpoint-100 \\
    --val_dataset data/output/processed_data_val.jsonl \\
    --hf_token YOUR_TOKEN \\
    --wandb_project lm-evaluation \\
    --use_multi_gpu

  # Single-GPU fallback:
  python evaluate_distributed_checkpoints_multigpu.py \\
    --model gemma-7b-it \\
    --checkpoint_dir models/gemma-7b-it_fsdp/checkpoint-100 \\
    --val_dataset data/output/processed_data_val.jsonl \\
    --hf_token YOUR_TOKEN

  # With NLI faithfulness (default 100 examples; use --nli_subset_size == --val_data_size for full val NLI):
  python evaluate_distributed_checkpoints_multigpu.py \\
    --model gemma-7b-it \\
    --checkpoint_dir models/gemma-7b-it_fsdp/checkpoint-100 \\
    --val_dataset data/output/processed_data_val.jsonl \\
    --hf_token YOUR_TOKEN \\
    --include_nli_faithfulness \\
    --val_data_size 500 \\
    --nli_subset_size 500
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       choices=['viking-7b', 'viking-13b', 'viking-33b',
                                'gemma-2b', 'gemma-7b-it', 'gemma-2-9b', 'gemma-2-27b',
                                'gemma-3-12b', 'gemma-3-27b',
                               'normistral-7b', 'normistral-11b', 'normistral-11b-long', 'normistral-7b-instruct',
                                'norskgpt-llama3-8b', 'llama-3.1-8b-instruct', 'llama-2-13b-chat-norwegian',
                                'eurollm-9b-instruct', 'norwai-mistral-7b-instruct', 'nb-gpt-j-6b', 'mt5'],
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
                       help=f'Number of examples to use for validation (default: {VAL_DATA_SIZE}). Use 1000 for extended eval; the 1000 contain the same 500 as used for 500-example runs.')
    parser.add_argument('--val_data_seed', type=int, default=VAL_DATA_SEED,
                       help=f'Random seed for reproducible validation sampling (default: {VAL_DATA_SEED})')
    parser.add_argument('--val_beam_size', type=int, default=VAL_BEAM_SIZE,
                       help=f'Beam size for validation generation (default: {VAL_BEAM_SIZE})')
    parser.add_argument('--use_greedy', action='store_true',
                       help='Use greedy decoding instead of beam search for faster evaluation')
    parser.add_argument('--use_multi_gpu', action='store_true',
                       help='Use model parallelism (device_map="auto") to split model across multiple GPUs. Compatible with generation.')
    parser.add_argument('--keep_existing', action='store_true',
                       help='If set, do not rerun evaluation when results already exist (skips checkpoints with saved outputs).')
    parser.add_argument('--force_recompute', action='store_true',
                       help='If set, re-run evaluation even when results exist (e.g. after prompt/config corrections). Overwrites existing results.')
    
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='lm-evaluation',
                       help='Wandb project name for evaluation runs (default: lm-evaluation)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Wandb entity/team name (default: uses your default entity)')
    parser.add_argument('--wandb_disabled', action='store_true',
                       help='Disable wandb logging for this evaluation')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name (if not provided, defaults to {model}_eval_all_checkpoints). All checkpoints will be combined into a single run automatically. The run ID is saved in model_dir/all_eval_results/.wandb_run_id for automatic resumption.')
    parser.add_argument('--wandb_group', type=str, default=None,
                       help='Wandb group name to combine multiple runs (default: model name)')
    parser.add_argument('--major_checkpoint_interval', type=int, default=500,
                       help='Every Nth step is considered "major" for BERTScore evaluation (default: 500). Major checkpoints: checkpoint-500, checkpoint-1000, checkpoint-1500, etc.')
    parser.add_argument('--include_nli_faithfulness', action='store_true',
                       help='Enable NLI-based faithfulness evaluation on a fixed subset of examples (same indices across checkpoints; see --nli_subset_size).')
    parser.add_argument(
        '--nli_subset_size',
        type=int,
        default=NLI_DEFAULT_SUBSET_SIZE,
        metavar='N',
        help='With --include_nli_faithfulness: number of examples for NLI (default: %(default)s). '
        'For NLI on the full eval set, set this equal to --val_data_size. '
        'If N is smaller than the eval set, use a reproducible random subset (seed 42), or the first N examples when '
        '--val_data_size is larger than N.',
    )

    # --- Evaluation mode flags ---
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--predict_only', action='store_true',
                           help='Generate predictions and save JSONL only; skip all metrics computation.')
    mode_group.add_argument('--metrics_only', action='store_true',
                           help='Compute all metrics from an existing JSONL predictions file; no model loading needed.')

    parser.add_argument('--update_rouge', action='store_true',
                       help='Recompute ROUGE from existing JSONL and merge into eval results.')
    parser.add_argument('--update_hygiene', action='store_true',
                       help='Recompute hygiene from existing JSONL and merge into eval results.')
    parser.add_argument('--update_bertscore', action='store_true',
                       help='Recompute BERTScore from existing JSONL and merge into eval results.')
    parser.add_argument('--update_faithfulness', action='store_true',
                       help='Recompute NLI faithfulness from existing JSONL and merge into eval results '
                            'when eval_faithfulness is missing/null or the *-faithfulness-details-*.jsonl file is absent.')

    args = parser.parse_args()

    # Build update_metrics set (empty set means not in update mode)
    update_metrics = {name for name in ("rouge", "hygiene", "bertscore", "faithfulness")
                      if getattr(args, f"update_{name}", False)}

    # Validate mode mutual exclusivity
    if update_metrics and (args.predict_only or args.metrics_only):
        parser.error("--update_* flags cannot be combined with --predict_only or --metrics_only.")

    # --metrics_only and --update_* do not need --val_dataset or model loading
    needs_prediction = not args.metrics_only and not update_metrics and not args.skip_eval

    # Validate arguments
    if needs_prediction and args.val_dataset is None:
        parser.error("--val_dataset is required for prediction (use --metrics_only, --update_*, or --skip_eval to skip)")

    if args.nli_subset_size < 1:
        parser.error("--nli_subset_size must be a positive integer")

    # Model mapping from configs
    model_mapping = get_model_name_mapping()
    try:
        model_name = model_mapping[args.model]
    except Exception as e:
        print(f"Error mapping model name: {e}")
        sys.exit(1)

    # Apply model-specific token limits (e.g. Normistral-7b has 2048 context window)
    max_input_text_tokens = args.max_input_text_tokens
    max_output_summary_tokens = args.max_output_summary_tokens
    try:
        model_config = get_model_config_by_hf_name(model_name)
        if model_config and model_config.max_input_text_tokens is not None:
            max_input_text_tokens = model_config.max_input_text_tokens
            print(f"Using model-specific max_input_text_tokens: {max_input_text_tokens}")
        if model_config and model_config.max_output_summary_tokens is not None:
            max_output_summary_tokens = model_config.max_output_summary_tokens
            print(f"Using model-specific max_output_summary_tokens: {max_output_summary_tokens}")
    except Exception:
        pass

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
                output_dir=args.output_dir,
                max_input_text_tokens=max_input_text_tokens,
                max_output_summary_tokens=max_output_summary_tokens,
                val_data_size=args.val_data_size,
                val_data_seed=args.val_data_seed,
                use_multi_gpu=args.use_multi_gpu,
                wandb_project=args.wandb_project if not args.wandb_disabled else None,
                wandb_entity=args.wandb_entity,
                wandb_disabled=args.wandb_disabled,
                wandb_run_name=args.wandb_run_name,
                wandb_group=args.wandb_group,
                major_checkpoint_interval=args.major_checkpoint_interval,
                include_nli_faithfulness=args.include_nli_faithfulness,
                nli_subset_size=args.nli_subset_size,
                keep_existing=args.keep_existing,
                force_recompute=args.force_recompute,
                predict_only=args.predict_only,
                metrics_only=args.metrics_only,
                update_metrics=update_metrics or None,
            )
        except AlreadyEvaluatedError as e:
            print(f"⚠ SKIPPING: {e}")
            print(f"Checkpoint {args.checkpoint_dir} was already evaluated. Moving to next checkpoint.")
            sys.exit(0)
        except FileNotFoundError as e:
            print(f"⚠ SKIPPING: {e}")
            sys.exit(0)
