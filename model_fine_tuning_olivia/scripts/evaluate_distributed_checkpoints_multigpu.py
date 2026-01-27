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
import sys
import time  # ADD THIS for staggered loading
from datetime import datetime
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

# Ensure scripts directory is in Python path for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Import model configurations
from model_configs import get_model_config_by_hf_name, get_model_name_mapping, get_doc_type_norwegian

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
                # Set the device attribute so Trainer knows where to put inputs
                self.args.device = torch.device(primary_device)
            else:
                # Fallback to first available CUDA device
                if torch.cuda.is_available():
                    self.args.device = torch.device("cuda:0")
            
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
                    reserved = torch.cuda.memory_reserved(i) / 1e9  # GB
                    if reserved > 80:  # >80GB used out of 102GB
                        print(f"WARNING: GPU {i} using {reserved:.1f}GB - consider reducing batch size")

        loss = None
        
        return (loss, generated_ids, labels)


def load_model_and_peft_checkpoint(
    model_name: str,
    checkpoint_dir: str,
    hf_token: Optional[str] = None,
    use_multi_gpu: bool = False,
    major_checkpoint_interval: int = 500,
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
    
    # Extract checkpoint step number early (needed for checking existing results)
    checkpoint_name = os.path.basename(checkpoint_dir.rstrip('/'))
    checkpoint_step = checkpoint_name.replace('checkpoint-', '') if 'checkpoint-' in checkpoint_name else 'unknown'
    try:
        checkpoint_step_int = int(checkpoint_step)
    except ValueError:
        checkpoint_step_int = 0
    
    # Check if this checkpoint was already evaluated
    # Look in model_dir/all_eval_results/checkpoint-nnn-eval-results.json
    model_dir = os.path.dirname(checkpoint_dir.rstrip('/'))
    all_eval_results_dir = os.path.join(model_dir, "all_eval_results")
    eval_results_file = os.path.join(all_eval_results_dir, f"{checkpoint_name}-eval-results.json")
    if os.path.exists(eval_results_file):
        # Check if extended metrics are missing (if extended evaluation is available)
        # If extended metrics should be computed but aren't present, allow re-evaluation
        if EXTENDED_EVAL_AVAILABLE:
            try:
                with open(eval_results_file, 'r') as f:
                    existing_results = json.load(f)
                
                # Check if this is a major checkpoint that should have BERTScore
                is_major_checkpoint = (checkpoint_step_int > 0 and checkpoint_step_int % major_checkpoint_interval == 0)
                
                # Check if extended metrics are missing
                has_extended_metrics = any(
                    key.startswith("eval_reference_") or 
                    key.startswith("eval_hygiene_") or 
                    key.startswith("eval_faithfulness_")
                    for key in existing_results.keys()
                )
                
                # If it's a major checkpoint and should have BERTScore but doesn't, re-evaluate
                if is_major_checkpoint and "eval_reference_bertscore_f1_mean" not in existing_results:
                    print(f"⚠ Checkpoint {checkpoint_name} was evaluated but missing BERTScore (major checkpoint).")
                    print(f"   Re-evaluating to compute extended metrics...")
                    # Don't raise AlreadyEvaluatedError - allow re-evaluation
                elif not has_extended_metrics:
                    print(f"⚠ Checkpoint {checkpoint_name} was evaluated but missing extended metrics.")
                    print(f"   Re-evaluating to compute extended metrics...")
                    # Don't raise AlreadyEvaluatedError - allow re-evaluation
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
    old_eval_results_file = os.path.join(checkpoint_dir, 'eval_results', 'eval_results.json')
    if os.path.exists(old_eval_results_file):
        # Check if extended metrics are missing (same logic as above)
        if EXTENDED_EVAL_AVAILABLE:
            try:
                with open(old_eval_results_file, 'r') as f:
                    existing_results = json.load(f)
                
                is_major_checkpoint = (checkpoint_step_int > 0 and checkpoint_step_int % major_checkpoint_interval == 0)
                has_extended_metrics = any(
                    key.startswith("eval_reference_") or 
                    key.startswith("eval_hygiene_") or 
                    key.startswith("eval_faithfulness_")
                    for key in existing_results.keys()
                )
                
                if is_major_checkpoint and "eval_reference_bertscore_f1_mean" not in existing_results:
                    print(f"⚠ Checkpoint {checkpoint_name} was evaluated but missing BERTScore (major checkpoint).")
                    print(f"   Re-evaluating to compute extended metrics...")
                    # Don't raise AlreadyEvaluatedError - allow re-evaluation
                elif not has_extended_metrics:
                    print(f"⚠ Checkpoint {checkpoint_name} was evaluated but missing extended metrics.")
                    print(f"   Re-evaluating to compute extended metrics...")
                    # Don't raise AlreadyEvaluatedError - allow re-evaluation
                else:
                    raise AlreadyEvaluatedError(
                        f"Checkpoint {checkpoint_dir} appears to be already evaluated "
                        f"(old results file exists at {old_eval_results_file}). "
                        f"Skipping evaluation."
                    )
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
    include_nli_faithfulness: bool = False,  # Enable NLI faithfulness evaluation (slow, ~37 min for 500 examples)
    nli_subset_size: Optional[int] = None,  # Subset size for NLI (None = all examples, recommended: 50-100)
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
    #   - NLI Faithfulness: Skip for all checkpoints (too slow, ~37 min - use only for final evaluation)
    is_major_checkpoint = (checkpoint_step_int > 0 and checkpoint_step_int % major_checkpoint_interval == 0)
    
    # Default training parameters for example calculation (may vary - check training config for exact values)
    DEFAULT_TRAIN_BATCH_SIZE = 4
    DEFAULT_GRADIENT_ACCUMULATION = 4
    DEFAULT_TRAIN_NUM_GPUS = 2  # Typical for multi-node training
    
    # Calculate estimated examples from checkpoint step
    estimated_examples = calculate_examples_from_steps(
        checkpoint_step_int, DEFAULT_TRAIN_BATCH_SIZE, DEFAULT_GRADIENT_ACCUMULATION, DEFAULT_TRAIN_NUM_GPUS
    ) if checkpoint_step_int > 0 else None
    
    # Determine output directory: save to model/all_eval_results/checkpoint-nnn-eval-results.json
    if output_dir is None:
        # Get model directory (parent of checkpoint_dir)
        model_dir = os.path.dirname(checkpoint_dir.rstrip('/'))
        # Create all_eval_results directory in model directory
        output_dir = os.path.join(model_dir, "all_eval_results")
    else:
        output_dir = os.path.abspath(output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as checkpoint-nnn-eval-results.json instead of eval_results.json
    results_file = os.path.join(output_dir, f"{checkpoint_name}-eval-results.json")
    
    # If results file exists, load and log to Wandb without re-evaluating
    if os.path.exists(results_file):
        print(f"⚠ Checkpoint {checkpoint_name} already evaluated. Loading existing results...")
        
        try:
            with open(results_file, 'r') as f:
                existing_results = json.load(f)
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
                    key.startswith("eval_faithfulness_")
                    for key in existing_results.keys()
                )
                
                if is_major_checkpoint and "eval_reference_bertscore_f1_mean" not in existing_results:
                    missing_extended_metrics = True
                    print(f"⚠ Checkpoint {checkpoint_name} missing BERTScore (major checkpoint). Re-evaluating...")
                elif not has_extended_metrics:
                    missing_extended_metrics = True
                    print(f"⚠ Checkpoint {checkpoint_name} missing extended metrics. Re-evaluating...")
            
            if all_zeros or missing_extended_metrics:
                if all_zeros:
                    print(f"⚠ Warning: All ROUGE scores are 0.00 - this indicates a failed evaluation.")
                    print(f"   Re-evaluating checkpoint {checkpoint_name}...")
                # Fall through to normal evaluation instead of returning
            else:
                # Initialize Wandb if needed (even for already-evaluated checkpoints)
                is_main_process = True
                if wandb_project and not wandb_disabled and wandb.run is None and is_main_process:
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
                    
                    # Use provided run_name or create one based on model
                    run_name = wandb_run_name or f"{clean_model_name}_evaluation"
                    
                    wandb.init(
                        project=wandb_project,
                        entity=wandb_entity,
                        name=run_name,
                        group=wandb_group or clean_model_name,
                        tags=[
                            "evaluation",
                            "multi-gpu" if use_multi_gpu else "single-gpu",
                            clean_model_name,
                            "already-evaluated",  # Tag to indicate this was loaded from cache
                        ],
                        config={
                            "model_name": model_name,
                            "val_dataset_path": val_dataset_path,
                            "val_data_size": val_data_size,
                            "val_batch_size": val_batch_size,
                            "val_beam_size": val_beam_size,
                            "max_input_text_tokens": max_input_text_tokens,
                            "max_output_summary_tokens": max_output_summary_tokens,
                            "num_gpus": num_gpus,
                            "use_multi_gpu": use_multi_gpu,
                            "world_size": 1,
                            "is_distributed": False,
                            "results_loaded_from_cache": True,  # Indicate these are cached results
                            **gpu_info,
                        },
                        reinit=True,
                    )
                    print(f">>> wandb run initialized: {wandb.run.name}")
                    print(f">>> wandb run URL: {wandb.run.get_url()}")
                
                # Log existing results to Wandb
                if wandb.run is not None and is_main_process and not wandb_disabled:
                    # Log with checkpoint step as the x-axis
                    wandb.log({
                        "eval/rouge1": rouge1,
                        "eval/rouge2": rouge2,
                        "eval/rougeL": rougeL,
                        "eval/rougeLsum": rougeLsum,
                        "checkpoint_step": checkpoint_step_int,
                        "from_cache": True,  # Flag to indicate this was loaded from cache
                    }, step=checkpoint_step_int)
                    
                    # Update summary
                    wandb.summary.update({
                        "latest_checkpoint": checkpoint_step_int,
                        "latest_rouge1": rouge1,
                        "latest_rouge2": rouge2,
                        "latest_rougeL": rougeL,
                        "latest_rougeLsum": rougeLsum,
                    })
                    
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
        # Use checkpoint_step as the step so all checkpoints appear in one timeline
        if wandb.run is not None and is_main_process and not wandb_disabled:
            wandb.log({
                "eval/rouge1": scores['rouge1'] * 100,
                "eval/rouge2": scores['rouge2'] * 100,
                "eval/rougeL": scores['rougeL'] * 100,
                "eval/rougeLsum": scores['rougeLsum'] * 100,
                "checkpoint_step": checkpoint_step_int,  # Track which checkpoint this is
            }, step=checkpoint_step_int)  # CRITICAL: Use checkpoint step as x-axis
        
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
        
        # Use provided run_name or create one based on model
        run_name = wandb_run_name or f"{clean_model_name}_evaluation"
        
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            group=wandb_group or clean_model_name,  # Group all checkpoints together
            tags=[
                "evaluation",
                "multi-gpu" if use_multi_gpu else "single-gpu",
                clean_model_name,
                f"checkpoint-{checkpoint_step}",
                os.path.basename(checkpoint_dir).replace('checkpoint-', 'step-'),
                "major-checkpoint" if is_major_checkpoint else "normal-checkpoint",  # Tag for tiered evaluation
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
                "world_size": 1,
                "is_distributed": False,
                # Note: Training examples calculation requires training batch_size, gradient_accumulation_steps
                # These are typically 4 and 4 respectively, but may vary. Check training config for exact values.
                "checkpoint_step_note": f"Checkpoint at step {checkpoint_step} - examples calculation requires training parameters",
                # Tiered evaluation configuration
                "checkpoint_type": "major" if is_major_checkpoint else "normal",
                "major_checkpoint_interval": major_checkpoint_interval,
                "extended_metrics": {
                    "rouge": True,  # Always computed
                    "hygiene": True,  # Always computed
                    "bertscore": is_major_checkpoint,  # Only for major checkpoints
                    "faithfulness": include_nli_faithfulness,  # User-controlled
                    "nli_subset_size": nli_subset_size if include_nli_faithfulness else None,
                },
                **gpu_info,
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
            major_checkpoint_interval=major_checkpoint_interval
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
    
    # Check if file exists
    if not os.path.exists(val_dataset_path):
        raise FileNotFoundError(f"Validation dataset file does not exist: {val_dataset_path}")
    
    # Check file size (Git LFS pointers are typically < 200 bytes)
    file_size = os.path.getsize(val_dataset_path)
    if file_size < 200:
        print(f"WARNING: Validation dataset file is very small ({file_size} bytes).")
        print(f"         This might be a Git LFS pointer file. Please ensure the actual file is downloaded.")
    
    val_data = []
    with open(val_dataset_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        # Check if it's a Git LFS pointer
        if first_line.strip().startswith('version https://git-lfs.github.com/spec/v1'):
            raise ValueError(
                f"Validation dataset file appears to be a Git LFS pointer, not actual data.\n"
                f"Please download the actual file using: git lfs pull\n"
                f"Or ensure the file at {val_dataset_path} contains actual JSONL data."
            )
        
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
                raise ValueError(
                    f"Invalid JSON on line {line_num} of validation dataset: {json_err}\n"
                    f"Line content (first 200 chars): {line[:200]}"
                )
    
    if len(val_data) == 0:
        raise ValueError(f"Validation dataset file is empty or contains no valid JSON lines: {val_dataset_path}")
    
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
        
        # Log to wandb config (lightweight - just metadata, once per evaluation)
        model_config = get_model_config_by_hf_name(model_name)
        wandb.config.update({
            "eval_prompt_examples": example_prompts,
            "eval_doc_types_seen": sorted(list(doc_types_seen)),
            "eval_prompt_template_type": model_config.prompt_config.template_type if model_config else "plain"
        })
        
        # Print to console (lightweight - just once)
        print("\n" + "=" * 70)
        print("EVALUATION PROMPT EXAMPLES (logged to wandb config):")
        print("=" * 70)
        for ex in example_prompts:
            print(f"\nExample {ex['example_num']}:")
            print(f"  Doc Type: {ex['doc_type']} -> {ex['doc_type_norwegian']}")
            print(f"  Prompt Preview: {ex['prompt_preview']}")
        print("=" * 70 + "\n")
    
    tokenized_val_dataset = formatted_val_dataset.map(tokenize_function_eval, batched=True)
    
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
    
    # Save inputs, references, and predictions to JSONL file
    predictions_file = None
    if is_main_process:
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
    
    # Run extended evaluation metrics (reference-based, hygiene, faithfulness)
    if is_main_process:
        # Debug: Print why extended evaluation might not run
        if not EXTENDED_EVAL_AVAILABLE:
            print("\n" + "=" * 70)
            print("WARNING: Extended evaluation NOT available")
            print("=" * 70)
            print("Reason: extended_evaluation.py could not be imported")
            print("Only ROUGE metrics will be saved to JSON.")
            print("=" * 70 + "\n")
        elif not predictions_file:
            print("\n" + "=" * 70)
            print("WARNING: Extended evaluation NOT running")
            print("=" * 70)
            print("Reason: predictions_file is None")
            print("Only ROUGE metrics will be saved to JSON.")
            print("=" * 70 + "\n")
        elif not os.path.exists(predictions_file):
            print("\n" + "=" * 70)
            print("WARNING: Extended evaluation NOT running")
            print("=" * 70)
            print(f"Reason: predictions_file does not exist: {predictions_file}")
            print("Only ROUGE metrics will be saved to JSON.")
            print("=" * 70 + "\n")
    
    if is_main_process and EXTENDED_EVAL_AVAILABLE and predictions_file and os.path.exists(predictions_file):
        print("\n" + "=" * 70)
        print("Running Extended Evaluation Metrics...")
        print("=" * 70)
        
        try:
            # Load texts from JSONL file
            input_texts = []
            prediction_texts = []
            reference_texts = []
            
            with open(predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    input_texts.append(entry.get("input_text", ""))
                    prediction_texts.append(entry.get("prediction", ""))
                    reference_texts.append(entry.get("reference", ""))
            
            if len(input_texts) > 0 and len(prediction_texts) > 0 and len(reference_texts) > 0:
                if not EXTENDED_EVAL_AVAILABLE:
                    print("Warning: Extended evaluation not available (extended_evaluation.py not found). Skipping extended metrics.")
                else:
                    # Determine which metrics to compute based on checkpoint type and user settings
                    # Normal checkpoints: ROUGE + Hygiene only (fast, ~2 min)
                    # Major checkpoints: ROUGE + Hygiene + BERTScore (moderate, ~3-4 min)
                    # NLI Faithfulness: Optional, can be enabled via include_nli_faithfulness parameter
                    include_bertscore = is_major_checkpoint
                    include_faithfulness = include_nli_faithfulness  # User-controlled
                    
                    # Prepare subset for NLI if requested
                    # For ROUGE, Hygiene, and BERTScore: use full dataset
                    # For NLI: use subset if specified
                    nli_input_texts = input_texts
                    nli_prediction_texts = prediction_texts
                    nli_reference_texts = reference_texts
                    
                    if include_faithfulness and nli_subset_size and nli_subset_size < len(input_texts):
                        # Sample subset for NLI
                        random.seed(42)  # Fixed seed for reproducibility
                        indices = random.sample(range(len(input_texts)), nli_subset_size)
                        nli_input_texts = [input_texts[i] for i in indices]
                        nli_prediction_texts = [prediction_texts[i] for i in indices]
                        nli_reference_texts = [reference_texts[i] for i in indices]
                        print(f"  → NLI subset: Using {nli_subset_size} examples from {len(input_texts)} total")
                    
                    checkpoint_type = "MAJOR" if is_major_checkpoint else "NORMAL"
                    print(f"Computing extended metrics on {len(input_texts)} examples...")
                    print(f"Checkpoint type: {checkpoint_type} (step {checkpoint_step_int})")
                    if is_major_checkpoint:
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
                    
                    # Second run: NLI only (on subset if specified)
                    if include_faithfulness:
                        assert extended_evaluate is not None, "extended_evaluate should be available when EXTENDED_EVAL_AVAILABLE is True"
                        nli_results = extended_evaluate(
                            input_texts=nli_input_texts,
                            prediction_texts=nli_prediction_texts,
                            reference_texts=nli_reference_texts,
                            print_output=False,
                            include_bertscore=False,  # Skip BERTScore in second run (already computed)
                            include_faithfulness=True  # Only compute NLI
                        )
                        # Merge NLI results into extended_results
                        extended_results["faithfulness"] = nli_results.get("faithfulness")
                    
                    # Note: If NLI was run on a subset, the NLI results are for that subset only
                    # The other metrics (ROUGE, Hygiene, BERTScore) are computed on the full set
                    # This is intentional - NLI is expensive, so we sample for it
                    
                    # Merge extended results into eval_results
                    # Flatten nested structure for easier access
                    for category, metrics in extended_results.items():
                        if isinstance(metrics, dict):
                            for key, value in metrics.items():
                                # Normalize key names (use eval_ prefix for consistency)
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
            print(f"Warning: Extended evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing with ROUGE metrics only...")
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
        wandb_log_dict = {
            "final/rouge1": eval_results.get("eval_rouge1", 0),
            "final/rouge2": eval_results.get("eval_rouge2", 0),
            "final/rougeL": eval_results.get("eval_rougeL", 0),
            "final/rougeLsum": eval_results.get("eval_rougeLsum", 0),
        }
        
        # Log extended metrics if available
        if EXTENDED_EVAL_AVAILABLE:
            # Reference-based metrics
            if "eval_reference_bertscore_f1_mean" in eval_results:
                wandb_log_dict["final/bertscore_f1"] = eval_results.get("eval_reference_bertscore_f1_mean", 0)
            
            # Hygiene metrics
            if "eval_hygiene_mean_compression_ratio" in eval_results:
                wandb_log_dict["final/compression_ratio"] = eval_results.get("eval_hygiene_mean_compression_ratio", 0)
            if "eval_hygiene_mean_rep_3gram" in eval_results:
                wandb_log_dict["final/rep_3gram"] = eval_results.get("eval_hygiene_mean_rep_3gram", 0)
            if "eval_hygiene_ratio_ends_with_punct" in eval_results:
                wandb_log_dict["final/ends_with_punct"] = eval_results.get("eval_hygiene_ratio_ends_with_punct", 0)
            
            # Faithfulness metrics
            if "eval_faithfulness_mean_entailment_score" in eval_results:
                wandb_log_dict["final/entailment_mean"] = eval_results.get("eval_faithfulness_mean_entailment_score", 0)
            if "eval_faithfulness_mean_outlier_rate" in eval_results:
                wandb_log_dict["final/outlier_rate"] = eval_results.get("eval_faithfulness_mean_outlier_rate", 0)
            if "eval_faithfulness_ratio_passed" in eval_results:
                wandb_log_dict["final/faithfulness_passed"] = eval_results.get("eval_faithfulness_ratio_passed", 0)
        
        wandb.log(wandb_log_dict, step=checkpoint_step_int)
        
        # Update summary with latest checkpoint results
        summary_update = {
            "latest_checkpoint": checkpoint_step_int,
            "latest_rouge1": eval_results.get("eval_rouge1", 0),
            "latest_rouge2": eval_results.get("eval_rouge2", 0),
            "latest_rougeL": eval_results.get("eval_rougeL", 0),
            "latest_rougeLsum": eval_results.get("eval_rougeLsum", 0),
        }
        
        # Add extended metrics to summary
        if EXTENDED_EVAL_AVAILABLE:
            if "eval_reference_bertscore_f1_mean" in eval_results:
                summary_update["latest_bertscore_f1"] = eval_results.get("eval_reference_bertscore_f1_mean", 0)
            if "eval_faithfulness_mean_entailment_score" in eval_results:
                summary_update["latest_entailment_mean"] = eval_results.get("eval_faithfulness_mean_entailment_score", 0)
            if "eval_faithfulness_ratio_passed" in eval_results:
                summary_update["latest_faithfulness_passed"] = eval_results.get("eval_faithfulness_ratio_passed", 0)
        
        wandb.summary.update(summary_update)
        
        # DON'T call wandb.finish() here - keep the run open for multiple checkpoints
        # Only finish if explicitly requested or at the very end
        print(">>> Evaluation results logged to wandb (run kept open for additional checkpoints)")
    elif wandb_disabled and is_main_process:
        print(">>> Wandb disabled - skipping wandb logging")
    
    # Save results to file (only on main process)
    if is_main_process:
        # results_file was already set earlier with the new naming scheme
        # output_dir was already created earlier, so just save
        with open(results_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"Results saved to: {results_file}")
        
        # Update evaluation summary JSON file
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
        
        # Create checkpoint entry with all metrics
        checkpoint_entry = {
            "checkpoint": checkpoint_name,
            "checkpoint_number": checkpoint_step_int,
            "status": "success",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "result_file": results_file
        }
        
        # Add all metrics from eval_results
        for key, value in eval_results.items():
            # Normalize key names (handle both eval_rouge* and rouge*)
            normalized_key = key.replace("eval_", "").lower()
            
            # Extract numeric values
            if isinstance(value, (int, float)):
                checkpoint_entry[normalized_key] = float(value)
            elif isinstance(value, dict):
                # Handle nested metrics (e.g., bertscore with precision/recall/f1)
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        checkpoint_entry[f"{normalized_key}_{sub_key}"] = float(sub_value)
        
        # Remove existing entry for this checkpoint if it exists, then add new one
        summary["checkpoints"] = [c for c in summary["checkpoints"] if c.get("checkpoint") != checkpoint_name]
        summary["checkpoints"].append(checkpoint_entry)
        
        # Sort checkpoints by checkpoint number
        summary["checkpoints"].sort(key=lambda x: x.get("checkpoint_number", 0))
        
        # Update statistics
        _update_summary_statistics(summary)
        
        # Update generated_at timestamp
        summary["generated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Save updated summary
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Evaluation summary updated: {summary_file}")
    
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
    --include_nli_faithfulness \\
    --nli_subset_size 100
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
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name (if not provided, uses model name). Use same name for all checkpoints to combine them.')
    parser.add_argument('--wandb_group', type=str, default=None,
                       help='Wandb group name to combine multiple runs (default: model name)')
    parser.add_argument('--major_checkpoint_interval', type=int, default=500,
                       help='Every Nth step is considered "major" for BERTScore evaluation (default: 500). Major checkpoints: checkpoint-500, checkpoint-1000, checkpoint-1500, etc.')
    parser.add_argument('--include_nli_faithfulness', action='store_true',
                       help='Enable NLI-based faithfulness evaluation (slow: ~4.5s per example, ~37 min for 500 examples)')
    parser.add_argument('--nli_subset_size', type=int, default=None,
                       help='Subset size for NLI evaluation (default: all examples if --include_nli_faithfulness is set, recommended: 50-100 for faster evaluation)')

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
                output_dir=args.output_dir,  # None will trigger default: model_dir/all_eval_results
                use_multi_gpu=args.use_multi_gpu,
                wandb_project=args.wandb_project if not args.wandb_disabled else None,
                wandb_entity=args.wandb_entity,
                wandb_disabled=args.wandb_disabled,
                wandb_run_name=args.wandb_run_name,
                wandb_group=args.wandb_group,
                major_checkpoint_interval=args.major_checkpoint_interval,
                include_nli_faithfulness=args.include_nli_faithfulness,
                nli_subset_size=args.nli_subset_size,
            )
        except AlreadyEvaluatedError as e:
            print(f"⚠ SKIPPING: {e}")
            print(f"Checkpoint {args.checkpoint_dir} was already evaluated. Moving to next checkpoint.")
            sys.exit(0)  # Exit with success code so bash loop continues
