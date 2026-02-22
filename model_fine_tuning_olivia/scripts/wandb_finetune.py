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
import shutil
import sys
import importlib
import math
import threading
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

# Import shared utilities
from utils import (
    EvalDataCollator,
    compute_rouge_metrics,
    load_jsonl_dataset,
    tokenize_train_examples,
    tokenize_eval_examples,
    format_train_example,
    format_eval_example,
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
    """Enhanced GPU memory monitoring with utilization analysis and batch size recommendations."""
    
    def __init__(self, log_interval: int = 50, warn_threshold_gb: float = 80.0):
        """Initialize memory callback.
        
        Args:
            log_interval: Log memory stats every N steps (default: 50)
            warn_threshold_gb: Warn if memory usage exceeds this (GB, default: 80)
        """
        self.log_interval = log_interval
        self.warn_threshold_gb = warn_threshold_gb
        self.peak_memory = {}  # Track peak memory per GPU
        self.step_count = 0
        
    def on_step_end(self, args, state, control, **kwargs):
        """Monitor GPU memory and provide recommendations."""
        self.step_count += 1
        
        # Only check periodically to avoid overhead
        if self.step_count % self.log_interval != 0:
            return
        
        try:
            import pynvml
            pynvml.nvmlInit()
            num_gpus = pynvml.nvmlDeviceGetCount()
            
            memory_stats = []
            for i in range(num_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_gb = mem_info.total / (1024**3)
                used_gb = mem_info.used / (1024**3)
                free_gb = mem_info.free / (1024**3)
                utilization_pct = (used_gb / total_gb * 100) if total_gb > 0 else 0
                
                # Track peak memory
                if i not in self.peak_memory or used_gb > self.peak_memory[i]:
                    self.peak_memory[i] = used_gb
                
                memory_stats.append({
                    'gpu_id': i,
                    'used_gb': used_gb,
                    'total_gb': total_gb,
                    'free_gb': free_gb,
                    'utilization_pct': utilization_pct
                })
                
                # Log to wandb periodically
                if wandb.run is not None:
                    wandb.log({
                        f"gpu_{i}_memory_gb": used_gb,
                        f"gpu_{i}_memory_pct": utilization_pct,
                    }, step=state.global_step)
                
                # Print warning if high utilization
                if used_gb > self.warn_threshold_gb:
                    print(f"⚠ Step {state.global_step} | GPU {i}: {used_gb:.1f}GB / {total_gb:.1f}GB ({utilization_pct:.1f}%) - consider reducing batch size")
            
            # Print summary every 100 steps
            if self.step_count % (self.log_interval * 2) == 0:
                avg_utilization = sum(s['utilization_pct'] for s in memory_stats) / len(memory_stats) if memory_stats else 0
                avg_free = sum(s['free_gb'] for s in memory_stats) / len(memory_stats) if memory_stats else 0
                
                if avg_utilization < 50:
                    print(f"ℹ Step {state.global_step} | Avg GPU utilization: {avg_utilization:.1f}% | Free: {avg_free:.1f}GB per GPU")
                    print(f"   → Consider increasing batch size for better GPU utilization")
                elif avg_utilization > 90:
                    print(f"⚠ Step {state.global_step} | High GPU utilization: {avg_utilization:.1f}% | Free: {avg_free:.1f}GB per GPU")
                    print(f"   → Consider decreasing batch size to avoid OOM")
                    
        except (ImportError, Exception):
            # pynvml not available or error, use PyTorch memory tracking
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1e9
                    reserved = torch.cuda.memory_reserved(i) / 1e9
                    total = torch.cuda.get_device_properties(i).total_memory / 1e9
                    utilization = (reserved / total * 100) if total > 0 else 0
                    
                    if reserved > self.warn_threshold_gb:
                        print(f"⚠ Step {state.global_step} | GPU {i}: {reserved:.1f}GB / {total:.1f}GB ({utilization:.1f}%) - consider reducing batch size")
        
    def on_train_end(self, args, state, control, **kwargs):
        """Print final memory utilization summary."""
        if self.peak_memory:
            print("\n" + "=" * 70)
            print("PEAK GPU MEMORY USAGE DURING TRAINING")
            print("=" * 70)
            for gpu_id, peak_gb in sorted(self.peak_memory.items()):
                print(f"GPU {gpu_id}: Peak memory: {peak_gb:.1f} GB")
            
            avg_peak = sum(self.peak_memory.values()) / len(self.peak_memory)
            print(f"\nAverage peak memory: {avg_peak:.1f} GB per GPU")
            
            # Recommendations
            if avg_peak < 50:
                print("→ Low memory usage - consider increasing batch size for better GPU utilization")
            elif avg_peak > 90:
                print("→ High memory usage - consider decreasing batch size to avoid OOM")
            print("=" * 70 + "\n")


class CheckpointBackupCallback(TrainerCallback):
    """Callback to backup checkpoints when they're saved (runs in background to avoid blocking training).
    
    This ensures checkpoints are backed up to regular_checkpoints/ and major_checkpoints/
    folders when saved, regardless of whether the monitor script is running.
    This prevents checkpoints from being overwritten by save_total_limit before backup.
    
    The backup runs in a background thread to avoid blocking training.
    """
    
    def __init__(self, output_dir: str, major_checkpoint_interval: int = 500, async_backup: bool = True):
        """Initialize checkpoint backup callback.
        
        This callback backs up checkpoints to regular_checkpoints/ and major_checkpoints/ folders.
        Backups always overwrite existing backups for the same checkpoint step.
        The model folder checkpoint management (save_total_limit) is handled by TrainingArguments.
        
        Args:
            output_dir: Training output directory (where checkpoints are saved)
            major_checkpoint_interval: Every Nth step is a major checkpoint (default: 500)
            async_backup: If True, backup runs in background thread (default: True). 
                         If False, backup blocks training (faster but may slow training).
        """
        self.output_dir = output_dir
        self.major_checkpoint_interval = major_checkpoint_interval
        self.async_backup = async_backup
        self.backed_up_steps = set()  # Track which steps we've already backed up
        self._backup_lock = threading.Lock()  # Thread safety for backed_up_steps
    
    def _backup_checkpoint(self, checkpoint_dir: str, checkpoint_step_int: int):
        """Internal method to perform the actual backup (runs in thread or synchronously)."""
        model_dir = self.output_dir
        regular_backup_success = False
        major_backup_success = False
        
        # Check if this is a major checkpoint
        is_major = checkpoint_step_int > 0 and checkpoint_step_int % self.major_checkpoint_interval == 0
        
        # ------------------------------------------------------------------
        # Backup: Regular checkpoints (non-major checkpoints only)
        # ------------------------------------------------------------------
        if checkpoint_step_int > 0 and not is_major:
            regular_ckpt_dir = os.path.join(model_dir, "regular_checkpoints")
            os.makedirs(regular_ckpt_dir, exist_ok=True)
            regular_ckpt_name = f"regular-checkpoint-{checkpoint_step_int}"
            regular_ckpt_path = os.path.join(regular_ckpt_dir, regular_ckpt_name)
            
            # Always overwrite existing backup (remove old one first)
            if os.path.exists(regular_ckpt_path):
                try:
                    shutil.rmtree(regular_ckpt_path)
                except Exception as e:
                    print(f"⚠ Warning: Failed to remove existing regular checkpoint backup: {e}")
            
            print(f"\n[Checkpoint Backup] Copying checkpoint-{checkpoint_step_int} to regular_checkpoints/...")
            start_time = time.time()
            try:
                shutil.copytree(checkpoint_dir, regular_ckpt_path)
                elapsed = time.time() - start_time
                print(f"✓ Successfully backed up checkpoint-{checkpoint_step_int} to {regular_ckpt_path} ({elapsed:.1f}s)")
                regular_backup_success = True
            except Exception as e:
                print(f"⚠ Warning: Failed to backup regular checkpoint: {e}")
                print("   Continuing training, but regular checkpoint backup was not created.")
                regular_backup_success = False
        
        # ------------------------------------------------------------------
        # Backup: Major checkpoints (only in major_checkpoints, not in regular_checkpoints)
        # ------------------------------------------------------------------
        if is_major:
            major_ckpt_dir = os.path.join(model_dir, "major_checkpoints")
            os.makedirs(major_ckpt_dir, exist_ok=True)
            major_ckpt_name = f"major-checkpoint-{checkpoint_step_int}"
            major_ckpt_path = os.path.join(major_ckpt_dir, major_ckpt_name)
            
            # Always overwrite existing backup (remove old one first)
            if os.path.exists(major_ckpt_path):
                try:
                    shutil.rmtree(major_ckpt_path)
                except Exception as e:
                    print(f"⚠ Warning: Failed to remove existing major checkpoint backup: {e}")
            
            print(f"[Checkpoint Backup] Copying major checkpoint-{checkpoint_step_int} to major_checkpoints/...")
            start_time = time.time()
            try:
                shutil.copytree(checkpoint_dir, major_ckpt_path)
                elapsed = time.time() - start_time
                print(f"✓ Successfully backed up major checkpoint-{checkpoint_step_int} to {major_ckpt_path} ({elapsed:.1f}s)")
                major_backup_success = True
            except Exception as e:
                print(f"⚠ Warning: Failed to backup major checkpoint: {e}")
                print("   Continuing training, but major checkpoint backup was not created.")
                major_backup_success = False
    
    def on_save(self, args, state, control, **kwargs):
        """Backup checkpoint when it's saved (non-blocking if async_backup=True).
        
        This always overwrites existing backups for the same checkpoint step.
        The model folder checkpoint management (save_total_limit) is handled by TrainingArguments.
        """
        # Only backup on main process (rank 0)
        if args.local_rank not in [-1, 0]:
            return
        
        # Get current checkpoint directory
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
        
        # Check if checkpoint directory exists and has adapter files
        if not os.path.exists(checkpoint_dir):
            return
        
        adapter_file = os.path.join(checkpoint_dir, "adapter_model.safetensors")
        if not os.path.exists(adapter_file):
            # Checkpoint might still be saving, skip this time
            return
        
        # Check if already backed up in this session to avoid duplicate backups during same run
        # Note: When resuming from a checkpoint, on_save is typically not called for that checkpoint,
        # so it won't be re-backed up. Only new checkpoints created during this run will be backed up.
        with self._backup_lock:
            if state.global_step in self.backed_up_steps:
                # Already backed up in this session - skip to avoid duplicate work
                # (This can happen if checkpoint is saved multiple times in the same run)
                return
            # Mark as being backed up immediately to prevent race conditions
            self.backed_up_steps.add(state.global_step)
        
        checkpoint_step_int = state.global_step
        
        # Run backup in background thread to avoid blocking training
        if self.async_backup:
            def backup_thread():
                try:
                    self._backup_checkpoint(checkpoint_dir, checkpoint_step_int)
                except Exception as e:
                    print(f"⚠ Error in backup thread: {e}")
                    # Remove from backed_up_steps so it can be retried
                    with self._backup_lock:
                        self.backed_up_steps.discard(checkpoint_step_int)
            
            thread = threading.Thread(target=backup_thread, daemon=True)
            thread.start()
            # Don't wait for thread - training continues immediately
        else:
            # Synchronous backup (blocks training, but ensures backup completes)
            self._backup_checkpoint(checkpoint_dir, checkpoint_step_int)


class ExamplesTrackingCallback(TrainerCallback):
    """Callback to track and log examples processed and training time during training."""
    def __init__(self, batch_size, gradient_accumulation_steps, num_gpus, resume_checkpoint: Optional[str] = None):
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_gpus = num_gpus
        self.resume_checkpoint = resume_checkpoint
        self.training_start_time = None
        self.total_training_time_before_resume = 0.0  # Time from previous training runs
        
        # Try to read total training time from checkpoint if resuming
        if resume_checkpoint:
            self._load_total_training_time_from_checkpoint(resume_checkpoint)
    
    def _load_total_training_time_from_checkpoint(self, checkpoint_path: str):
        """Load total training time from trainer_state.json if available."""
        import json
        import os
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            try:
                with open(trainer_state_path, 'r') as f:
                    trainer_state = json.load(f)
                # HuggingFace Trainer tracks log_history with timing info
                # Look for the last entry with training time
                log_history = trainer_state.get("log_history", [])
                if log_history:
                    # Find entries with training time
                    for entry in reversed(log_history):
                        if "train_runtime" in entry:
                            self.total_training_time_before_resume = entry.get("train_runtime", 0.0)
                            break
                        # Also check for cumulative time if available
                        if "total_flos" in entry and "train_runtime" in entry:
                            self.total_training_time_before_resume = entry.get("train_runtime", 0.0)
                            break
            except Exception as e:
                print(f"Warning: Could not load training time from checkpoint: {e}")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Record training start time."""
        self.training_start_time = time.time()
        # Cache division constant to avoid repeated division in hot path
        self._seconds_to_hours = 1.0 / 3600.0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log examples count and training time to wandb alongside other metrics.
        
        Optimized for minimal overhead: single time.time() call + arithmetic operations only.
        """
        if logs is not None and wandb.run is not None:
            examples_seen = state.global_step * self.batch_size * self.gradient_accumulation_steps * self.num_gpus
            logs['examples_seen'] = examples_seen
            logs['examples_seen_k'] = examples_seen / 1000.0  # Also log in thousands for readability
            
            # Calculate training time (minimal overhead: single time.time() call + arithmetic)
            if self.training_start_time is not None:
                time_since_resume = time.time() - self.training_start_time
                total_training_time = self.total_training_time_before_resume + time_since_resume
                logs['training_time_since_resume_seconds'] = time_since_resume
                logs['training_time_total_seconds'] = total_training_time
                # Use cached conversion factor for efficiency (multiplication faster than division)
                logs['training_time_since_resume_hours'] = time_since_resume * self._seconds_to_hours
                logs['training_time_total_hours'] = total_training_time * self._seconds_to_hours

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

# EvalDataCollator is now imported from utils.data_collators


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
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation with prompt masking and monitoring.
        
        This method:
        1. Computes loss using the model's forward pass (which handles label smoothing if enabled)
        2. Tracks loss on summary tokens (prompt tokens are masked with -100 and ignored)
        3. Logs metrics to wandb for monitoring
        """
        # Get labels
        labels = inputs.get("labels")
        
        # Forward pass - this will use label smoothing if enabled in TrainingArguments
        outputs = model(**inputs)
        logits = outputs.logits
        loss = outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else None
        
        # If model didn't compute loss (shouldn't happen, but safety check)
        if loss is None:
            # Fallback: compute loss manually
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # For monitoring: compute per-token loss to track summary token loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten the tokens
        loss_fct_none = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        
        # Compute per-token loss (for monitoring only)
        per_token_loss = loss_fct_none(flat_logits, flat_labels)
        
        # Identify prompt vs summary tokens
        # Prompt tokens have label = -100 (masked), summary tokens have label != -100
        is_prompt_token = (flat_labels == -100)
        is_summary_token = ~is_prompt_token
        
        # Compute summary loss (prompt tokens are ignored, so their loss is always 0)
        summary_loss = per_token_loss[is_summary_token].mean() if is_summary_token.any() else torch.tensor(0.0, device=per_token_loss.device)
        
        # Log summary loss and token ratios to wandb (only on main process, periodically)
        if self.args.local_rank in [-1, 0] and wandb.run is not None:
            # Only log every N steps to avoid spam (use logging_steps from args)
            log_interval = getattr(self.args, 'logging_steps', 10)
            if hasattr(self, 'state') and self.state.global_step % log_interval == 0:
                wandb.log({
                    "train/loss_summary": summary_loss.item() if isinstance(summary_loss, torch.Tensor) else summary_loss,
                    "train/loss_total": loss.item() if isinstance(loss, torch.Tensor) else loss,
                    "train/prompt_tokens_ratio": is_prompt_token.float().mean().item() if is_prompt_token.any() else 0.0,
                    "train/summary_tokens_ratio": is_summary_token.float().mean().item() if is_summary_token.any() else 0.0,
                }, step=self.state.global_step)
        
        if return_outputs:
            return loss, outputs
        return loss
    
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
    train_batch_size: Optional[int] = None,  # None = use model config default
    val_batch_size: Optional[int] = None,    # None = use model config default
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
    
    Note: Critical training variables (gradient_accumulation_steps, num_gpus, train_steps, train_epochs)
          are defined early in the function to prevent UnboundLocalError when used in wandb config
          and other early calculations.
    """
    
    def compute_metrics(eval_pred):
        """Compute ROUGE metrics using shared utility function."""
        is_main_process = (int(os.environ.get('RANK', 0)) == 0)
        return compute_rouge_metrics(
            eval_pred=eval_pred,
            tokenizer=tokenizer,
            log_to_wandb=True,
            step=None,  # Step will be set by Trainer
            is_main_process=is_main_process,
            verbose=True
        )

    # Set HF token via environment variable
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token

    # Determine rank for distributed training
    rank = int(os.environ.get('RANK', 0))
    is_main_process = (rank == 0)
    
    # ============================================================================
    # CRITICAL: Initialize all training variables EARLY to prevent UnboundLocalError
    # These variables are used in wandb config updates and calculations before
    # the TrainingArguments are created. They must be defined here, not later.
    # ============================================================================
    
    # Define gradient accumulation steps early (used in wandb config and calculations)
    gradient_accumulation_steps = 4  # This is set in TrainingArguments below
    
    # Resolve batch sizes from model config if not provided (must be done early)
    model_config_early = get_model_config_by_hf_name(model_name)
    if model_config_early:
        if train_batch_size is None:
            train_batch_size = model_config_early.train_batch_size if model_config_early.train_batch_size is not None else TRAIN_BATCH_SIZE
            print(f"Using model config default train_batch_size: {train_batch_size}")
        if val_batch_size is None:
            val_batch_size = model_config_early.val_batch_size if model_config_early.val_batch_size is not None else VAL_BATCH_SIZE
            print(f"Using model config default val_batch_size: {val_batch_size}")
    else:
        # Use global defaults if model config not found
        if train_batch_size is None:
            train_batch_size = TRAIN_BATCH_SIZE
        if val_batch_size is None:
            val_batch_size = VAL_BATCH_SIZE
    
    # Get number of GPUs early (used in wandb config and calculations)
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    num_gpus = world_size if world_size > 1 else (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    
    # Determine training duration early (used in wandb config and calculations)
    if max_steps is not None and max_steps > 0:
        train_epochs = 1  # Set to 1 instead of None to avoid Trainer comparison errors
        train_steps = max_steps
    else:
        train_epochs = num_train_epochs if num_train_epochs is not None else MAX_EPOCHS
        train_steps = -1  # -1 means "use epochs instead"
    
    # ============================================================================
    
    # Helper function to calculate examples from steps
    def calculate_examples_from_steps(steps, batch_size, gradient_accumulation_steps, num_gpus):
        """Calculate total number of examples processed given training parameters."""
        if steps is None or steps <= 0:
            return None
        return steps * batch_size * gradient_accumulation_steps * num_gpus
    
    # Only initialize wandb on rank 0 (unless disabled via environment)
    wandb_disabled = os.environ.get('WANDB_DISABLED', '').lower() in ('true', '1', 'yes')
    if is_main_process and not wandb_disabled:
        print("Initializing Weights & Biases...")
        # Calculate training examples info (will be updated after dataset is loaded)
        wandb_config = {
            "model_name": model_name,
            "quantization": quantization,
            "max_input_text_tokens": max_input_text_tokens,
            "max_output_summary_tokens": max_output_summary_tokens,
            "train_batch_size": train_batch_size,
            "val_batch_size": val_batch_size,
            "use_ddp": use_ddp,
            "use_fsdp": use_fsdp,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_gpus": num_gpus,
        }
        wandb.init(
            project="lm-finetuning",
            name=f"{os.path.basename(output_dir)}_{quantization}",
            config=wandb_config
        )
        print(f">>> wandb run initialized: {wandb.run.name}")
        print(f">>> wandb run URL: {wandb.run.get_url()}")
        # Get the run name for TrainingArguments
        wandb_run_name = wandb.run.name
    else:
        # Disable wandb for non-rank-0 processes or if explicitly disabled
        if is_main_process and wandb_disabled:
            print("WandB is disabled (WANDB_DISABLED=true)")
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

    # Load training dataset using shared utility
    train_data = load_jsonl_dataset(dataset_path, dataset_type="training", raise_on_error=False)
    if train_data is None:
        return  # Error already printed by load_jsonl_dataset

    # Load validation dataset using shared utility
    val_data = load_jsonl_dataset(val_dataset_path, dataset_type="validation", raise_on_error=False)
    if val_data is None:
        return  # Error already printed by load_jsonl_dataset

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

    # Use shared formatting and tokenization functions
    def format_example_train_wrapper(example):
        """Wrapper to call shared format_train_example with model_name."""
        return format_train_example(example, model_name)
    
    def format_example_eval_wrapper(example):
        """Wrapper to call shared format_eval_example with model_name."""
        return format_eval_example(example, model_name)
    
    def tokenize_function_train_wrapper(examples):
        """Wrapper to call shared tokenize_train_examples with tokenizer and config."""
        return tokenize_train_examples(
            examples=examples,
            tokenizer=tokenizer,
            max_input_text_tokens=max_input_text_tokens,
            max_extra_prompt_tokens=max_extra_prompt_tokens,
            max_output_summary_tokens=max_output_summary_tokens
        )
    
    def tokenize_function_eval_wrapper(examples):
        """Wrapper to call shared tokenize_eval_examples with tokenizer and config."""
        return tokenize_eval_examples(
            examples=examples,
            tokenizer=tokenizer,
            max_input_text_tokens=max_input_text_tokens,
            max_extra_prompt_tokens=max_extra_prompt_tokens,
            max_output_summary_tokens=max_output_summary_tokens
        )

    formatted_dataset = dataset.map(format_example_train_wrapper)
    
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
        tokenize_function_train_wrapper, 
        batched=True,
        load_from_cache_file=False,  # ADD THIS - keeps data in memory
    )
    
    # Update wandb config with final training parameters and example calculations (after dataset is loaded)
    if is_main_process and wandb.run is not None:
        # train_batch_size and val_batch_size are guaranteed to be int at this point
        # (set from model config or defaults earlier in the function, around line 610)
        assert train_batch_size is not None and isinstance(train_batch_size, int), "train_batch_size must be set to an int"
        effective_batch_size = train_batch_size * gradient_accumulation_steps * num_gpus
        examples_per_step = effective_batch_size
        
        # Initialize variables to prevent UnboundLocalError
        total_training_examples = None
        estimated_epochs = None
        estimated_steps = None
        
        # Calculate total examples based on training strategy
        if train_steps > 0:
            total_training_examples = calculate_examples_from_steps(train_steps, train_batch_size, gradient_accumulation_steps, num_gpus)
            estimated_epochs = total_training_examples / len(tokenized_dataset) if total_training_examples and len(tokenized_dataset) > 0 else None
        else:
            total_training_examples = len(tokenized_dataset) * train_epochs
            estimated_steps = total_training_examples / examples_per_step if total_training_examples and examples_per_step > 0 else None
        
        wandb.config.update({
            "num_gpus": num_gpus,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "examples_per_step": examples_per_step,
            "max_steps": train_steps if train_steps > 0 else None,
            "num_train_epochs": train_epochs if train_steps <= 0 else None,
            "total_training_examples": total_training_examples,
            "dataset_size": len(tokenized_dataset),
            "estimated_epochs": estimated_epochs if train_steps > 0 and estimated_epochs else None,
            "estimated_steps": int(estimated_steps) if train_steps <= 0 and estimated_steps else None,
        })
        
        # Print training summary
        print("\n" + "=" * 70)
        print("TRAINING CONFIGURATION:")
        print("=" * 70)
        print(f"  Batch size (per GPU): {train_batch_size}")
        print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"  Number of GPUs: {num_gpus}")
        print(f"  Effective batch size: {effective_batch_size}")
        print(f"  Examples per step: {examples_per_step}")
        print(f"  Dataset size: {len(tokenized_dataset):,} examples")
        if train_steps > 0:
            print(f"  Max steps: {train_steps:,}")
            if total_training_examples:
                print(f"  Total examples: {total_training_examples:,} ({total_training_examples/1000:.1f}k)")
                if estimated_epochs:
                    print(f"  Estimated epochs: {estimated_epochs:.2f}")
        else:
            print(f"  Epochs: {train_epochs}")
            if total_training_examples:
                print(f"  Total examples: {total_training_examples:,} ({total_training_examples/1000:.1f}k)")
                if estimated_steps:
                    print(f"  Estimated steps: {int(estimated_steps):,}")
        print("=" * 70 + "\n")
        
        # Check initial GPU memory utilization
        if is_main_process and torch.cuda.is_available():
            print("\n" + "=" * 70)
            print("INITIAL GPU MEMORY UTILIZATION")
            print("=" * 70)
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory / 1e9
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                free = total - reserved
                utilization = (reserved / total * 100) if total > 0 else 0
                
                print(f"GPU {i}: {props.name}")
                print(f"  Total: {total:.1f} GB")
                print(f"  Reserved: {reserved:.1f} GB ({utilization:.1f}%)")
                print(f"  Free: {free:.1f} GB")
            
            avg_free = sum((torch.cuda.get_device_properties(i).total_memory / 1e9 - torch.cuda.memory_reserved(i) / 1e9) for i in range(num_gpus)) / num_gpus
            if avg_free > 20:
                print(f"\n→ High free memory ({avg_free:.1f}GB per GPU) - consider increasing batch size for better GPU utilization")
            print("=" * 70 + "\n")

    # Format and tokenize the VALIDATION dataset differently
    formatted_val_dataset = val_dataset.map(format_example_eval_wrapper)
    tokenized_val_dataset = formatted_val_dataset.map(
        tokenize_function_eval_wrapper, 
        batched=True,
        load_from_cache_file=False,  # ADD THIS
    )
    
    # Data collators
    # For TRAINING: use DataCollatorForLanguageModeling which creates labels
    # IMPORTANT: Must use RIGHT padding for this to work correctly
    base_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Create a wrapper that masks prompt tokens in labels
    class PromptMaskingCollator:
        """Wrapper around DataCollatorForLanguageModeling that masks prompt tokens."""
        def __init__(self, base_collator, tokenizer):
            self.base_collator = base_collator
            self.tokenizer = tokenizer
        
        def __call__(self, features):
            # First, get the base collation
            batch = self.base_collator(features)
            
            # Get labels and input_ids
            labels = batch['labels']
            input_ids = batch['input_ids']
            
            # Mask prompt tokens in labels (set to -100)
            # Labels are shifted by 1 position in causal LM, so we mask positions 0 to prompt_length-1
            for i, feature in enumerate(features):
                prompt_length = feature.get('prompt_length', None)
                # Handle case where prompt_length might be a list (shouldn't happen, but be defensive)
                if isinstance(prompt_length, list):
                    prompt_length = prompt_length[0] if len(prompt_length) > 0 else None
                if prompt_length is not None and isinstance(prompt_length, (int, float)):
                    seq_len = labels[i].shape[0]
                    # Mask positions 0 to prompt_length-1 (accounting for shift)
                    mask_end = min(int(prompt_length), seq_len)
                    labels[i, :mask_end] = -100
            
            batch['labels'] = labels
            return batch
    
    # Wrap the collator to mask prompt tokens
    train_data_collator = PromptMaskingCollator(base_collator, tokenizer)

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
                # Special handling for prompt_length - should be a scalar, not a list
                if key == 'prompt_length':
                    if isinstance(value, list):
                        # If it's a list, take the first element (shouldn't happen, but handle it)
                        sample_dict[key] = value[0] if len(value) > 0 else None
                    elif isinstance(value, (int, np.integer, float)):
                        sample_dict[key] = int(value)
                    else:
                        sample_dict[key] = value
                    continue
                
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
                    prompt_tokens = (labels == -100).sum().item()
                    total = labels.numel()
                    print(f"Labels shape: {labels.shape}")
                    print(f"Prompt tokens (masked): {prompt_tokens}/{total} ({100*prompt_tokens/total:.1f}%)")
                    print(f"Summary tokens (active): {non_ignored}/{total} ({100*non_ignored/total:.1f}%)")
                    if non_ignored == 0:
                        print("ERROR: All labels are ignored (-100)! This will cause loss=0.0")
                    elif prompt_tokens == 0:
                        print("WARNING: No prompt tokens masked! Loss will be computed on prompt tokens too.")
                    else:
                        print("✓ Prompt masking verified: prompt tokens are masked, only summary tokens contribute to loss")
                else:
                    print(f"Labels type: {type(labels)}")
            else:
                print("WARNING: No 'labels' key in collated output")
            print("=" * 50 + "\n")
        except Exception as e:
            print(f"ERROR during data collator verification: {e}")
            sample_var = locals().get('sample', 'N/A')
            print(f"Sample structure: {sample_var}")
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
    
    # Get model config for LoRA settings (batch sizes already resolved earlier at line ~915)
    # Reuse model_config_early if available, otherwise fetch it
    if 'model_config_early' in locals():
        model_config = model_config_early
    else:
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
            candidate_path = resume_checkpoint.strip()
            
            # Try multiple resolution strategies in order of preference:
            # 1. If it's already an absolute path, use it as-is
            # 2. If it's a simple name (no path separators, e.g., "checkpoint-5000"), resolve relative to output_dir first
            # 3. If it's a relative path, try relative to current directory
            # 4. As fallback, try relative to output_dir
            
            resolved_candidates = []
            
            # Strategy 1: Absolute path (use as-is)
            if os.path.isabs(candidate_path):
                resolved_candidates.append(candidate_path)
            
            # Strategy 2: Simple checkpoint name (e.g., "checkpoint-5000") - resolve relative to output_dir first
            # This is the most common case and should be prioritized
            is_simple_name = (os.path.basename(candidate_path) == candidate_path and 
                            not os.path.dirname(candidate_path) and
                            not os.sep in candidate_path and
                            not os.altsep or (os.altsep and os.altsep not in candidate_path))
            
            if is_simple_name:
                # Simple name like "checkpoint-5000" - resolve relative to output_dir
                rel_to_output = os.path.join(output_dir, candidate_path)
                abs_from_output = os.path.abspath(rel_to_output)
                resolved_candidates.append(abs_from_output)
            
            # Strategy 3: Relative to current working directory (for full relative paths)
            abs_from_cwd = os.path.abspath(candidate_path)
            if abs_from_cwd not in resolved_candidates:
                resolved_candidates.append(abs_from_cwd)
            
            # Strategy 4: Fallback - try relative to output_dir even if it has path separators
            # (in case user provides something like "checkpoint-5000" from a different directory)
            abs_output_dir = os.path.abspath(output_dir)
            abs_candidate = os.path.abspath(candidate_path)
            
            # Only try joining if the candidate doesn't already contain the output_dir
            if not abs_candidate.startswith(abs_output_dir + os.sep) and abs_candidate != abs_output_dir:
                rel_to_output_fallback = os.path.join(output_dir, candidate_path)
                abs_from_output_fallback = os.path.abspath(rel_to_output_fallback)
                if abs_from_output_fallback not in resolved_candidates:
                    resolved_candidates.append(abs_from_output_fallback)
            
            # Try each candidate until we find one that exists
            resolved_resume_checkpoint = None
            for candidate in resolved_candidates:
                print(f"Trying checkpoint path: {candidate}")
                if os.path.isdir(candidate):
                    resolved_resume_checkpoint = candidate
                    print(f"✓ Found checkpoint at: {candidate}")
                    break
            
            if not resolved_resume_checkpoint:
                print(f"ERROR: resume checkpoint directory not found. Tried:")
                for candidate in resolved_candidates:
                    print(f"  - {candidate}")
                print(f"\nPlease provide:")
                print(f"  - Simple name (recommended): checkpoint-5000 (resolved relative to output_dir: {output_dir})")
                print(f"  - Absolute path: /full/path/to/checkpoint-5000")
                print(f"  - Relative path: models/gemma-2-9b-apptainer-fsdp/checkpoint-5000")
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
    # train_steps, train_epochs, num_gpus, and gradient_accumulation_steps already defined earlier
    
    # Calculate and print training info
    if train_steps > 0:
        total_examples = calculate_examples_from_steps(train_steps, train_batch_size, gradient_accumulation_steps, num_gpus)
        print(f"Training for {train_steps} steps (epochs ignored)")
        if total_examples:
            print(f"  → Total examples: {total_examples:,} ({total_examples/1000:.1f}k)")
    else:
        print(f"Training for {train_epochs} epochs")
        # Calculate examples per epoch
        assert train_batch_size is not None, "train_batch_size must be set"
        effective_batch = train_batch_size * gradient_accumulation_steps * num_gpus
        examples_per_epoch = len(tokenized_dataset) if 'tokenized_dataset' in locals() else 0
        if examples_per_epoch > 0:
            total_examples = examples_per_epoch * train_epochs
            print(f"  → Examples per epoch: {examples_per_epoch:,} (effective batch: {effective_batch})")
            print(f"  → Total examples: {total_examples:,} ({total_examples/1000:.1f}k)")

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
    
    # gradient_accumulation_steps already defined earlier
    training_args_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=val_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
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
        report_to="wandb" if (is_main_process and not wandb_disabled) else "none",
        run_name=wandb_run_name,  # ADD THIS - link to manually initialized wandb run
        gradient_checkpointing=not use_fsdp,  # Disable for FSDP, use activation_checkpointing instead
        label_smoothing_factor=0.1,  # Add label smoothing to improve generalization
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
    
    # Set resume_from_checkpoint in TrainingArguments if checkpoint is provided
    # This is important for FSDP to properly resume training
    if resolved_resume_checkpoint:
        training_args_kwargs['resume_from_checkpoint'] = resolved_resume_checkpoint
        print(f"Setting resume_from_checkpoint in TrainingArguments: {resolved_resume_checkpoint}")
    
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
    
    # Add examples tracking callback
    examples_tracker = ExamplesTrackingCallback(train_batch_size, gradient_accumulation_steps, num_gpus, 
                                                 resume_checkpoint=resolved_resume_checkpoint)
    callbacks.append(examples_tracker)
    print("Adding ExamplesTrackingCallback to log examples processed during training")
    
    # Add GPU memory monitoring callback
    gpu_memory_callback = GPUMemoryCallback(log_interval=50, warn_threshold_gb=80.0)
    callbacks.append(gpu_memory_callback)
    print("Adding GPUMemoryCallback to monitor GPU utilization and provide batch size recommendations")
    
    # Add checkpoint backup callback - CRITICAL: Backs up checkpoints immediately when saved
    # This ensures checkpoints are preserved even if monitor script doesn't run
    # Major checkpoint interval: every 500 steps (same as evaluation script default)
    major_checkpoint_interval = 500
    checkpoint_backup_callback = CheckpointBackupCallback(
        output_dir=output_dir,
        major_checkpoint_interval=major_checkpoint_interval
    )
    callbacks.append(checkpoint_backup_callback)
    print(f"Adding CheckpointBackupCallback to backup checkpoints immediately when saved")
    print(f"  → Regular checkpoints: all checkpoints → regular_checkpoints/")
    print(f"  → Major checkpoints: every {major_checkpoint_interval} steps → major_checkpoints/")

    # Create training started signal file for monitor script (only on main process)
    if is_main_process:
        training_started_file = os.path.join(output_dir, "training_started.txt")
        os.makedirs(output_dir, exist_ok=True)
        with open(training_started_file, 'w') as f:
            import datetime
            f.write(f"Training started at: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Output directory: {output_dir}\n")
        print(f"✓ Created training started signal file: {training_started_file}")
    
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
            # Even with FSDP, verify the checkpoint state can be read
            if resolved_resume_checkpoint:
                trainer_state_path = os.path.join(resolved_resume_checkpoint, "trainer_state.json")
                if os.path.exists(trainer_state_path):
                    try:
                        import json
                        with open(trainer_state_path, 'r') as f:
                            trainer_state_data = json.load(f)
                        checkpoint_step = trainer_state_data.get('global_step', 'unknown')
                        print(f"✓ Checkpoint trainer_state.json found - will resume from step {checkpoint_step}")
                    except Exception as e:
                        print(f"⚠ Could not read trainer_state.json: {e}")
                else:
                    print(f"⚠ trainer_state.json not found at {trainer_state_path} - training may start from step 0")

    # Start training
    manual_resume = resolved_resume_checkpoint is not None and not use_fsdp

    if manual_resume:
        print("Continuing training with manually restored checkpoint state...")
        trainer.train()
    elif resolved_resume_checkpoint is not None:
        # User explicitly provided a checkpoint - always use it (even with FSDP)
        # Note: resume_from_checkpoint is already set in TrainingArguments (line ~1692)
        print(f"Resuming training from checkpoint path: {resolved_resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resolved_resume_checkpoint)
    elif checkpoints_exist and not force_restart:
        # No explicit checkpoint, but checkpoints exist - resume from latest
        print("Resuming training from latest checkpoint in output directory...")
        trainer.train(resume_from_checkpoint=True)
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
    parser.add_argument('--train_batch_size', type=int, default=None,
                       help=f'Training batch size per device (default: use model config default, or {TRAIN_BATCH_SIZE} if model not found)')
    parser.add_argument('--val_batch_size', type=int, default=None,
                       help=f'Validation batch size per device (default: use model config default, or {VAL_BATCH_SIZE} if model not found)')
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

