"""
Parallel checkpoint evaluation monitor for FSDP training.

This script runs in parallel with FSDP training and:
1. Monitors the output directory for new checkpoints
2. Evaluates each new checkpoint as it appears
3. Logs results to wandb
4. Implements early stopping based on validation metrics

Usage:
  # Run in parallel with training (in a separate terminal/sbatch job):
  python monitor_and_evaluate_checkpoints.py \
    --output_dir models/gemma-7b-apptainer-fsdp \
    --model gemma-7b \
    --val_dataset data/output/new_processed_data_val.jsonl \
    --hf_token YOUR_TOKEN \
    --check_interval 30 \
    --early_stopping_patience 10 \
    --wandb_project lm-finetuning \
    --wandb_run_name gemma-7b-apptainer-fsdp

The training script will check for early stopping signals and stop if needed.
"""

import argparse
import json
import os
import sys
import time
import glob
from pathlib import Path
from typing import Optional, Dict, List
import wandb

# Add torch import for device count check
import torch

# Ensure scripts directory is on path when run as python scripts/monitor_... (e.g. from sbatch)
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from model_configs import get_model_config
from evaluate_distributed_checkpoints_multigpu import evaluate_checkpoint, AlreadyEvaluatedError

# Import shared utilities
from utils import (
    extract_checkpoint_step,
    get_checkpoint_name_and_step,
    is_major_checkpoint,
    get_evaluated_checkpoint_steps,
    get_model_dir_from_checkpoint,
)

# Module-level variables for logging state
_last_logged_step = None
_log_counter = 0


def get_current_training_step(output_dir: str) -> Optional[int]:
    """Get the current training step from the latest checkpoint's trainer_state.json.
    
    This helps filter out old checkpoints from previous training runs.
    
    Returns:
        Current training step (global_step) if found, None otherwise
    """
    # Find all checkpoints
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    # Sort by step number (get the latest)
    def get_step(ckpt_path: str) -> int:
        step = extract_checkpoint_step(ckpt_path)
        return step if step is not None else -1
    
    checkpoints.sort(key=get_step, reverse=True)  # Latest first
    
    # Check the latest checkpoint for trainer_state.json
    for checkpoint_path in checkpoints:
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            try:
                with open(trainer_state_path, 'r') as f:
                    trainer_state = json.load(f)
                current_step = trainer_state.get('global_step', None)
                if current_step is not None:
                    return int(current_step)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # If we can't read it, try next checkpoint
                continue
    
    # Fallback: use the step number from the latest checkpoint directory name
    latest_checkpoint = checkpoints[0]
    step = get_step(latest_checkpoint)
    return step if step >= 0 else None


def find_checkpoints(output_dir: str, max_step: Optional[int] = None) -> List[str]:
    """Find all checkpoint directories, sorted by step number.
    
    Checks both:
    1. Main checkpoint directories (checkpoint-*)
    2. Backup directories:
       - regular_checkpoints/checkpoint-* or regular-checkpoint-* (non-major checkpoints)
       - major_checkpoints/checkpoint-* or major-checkpoint-* (major checkpoints only)
    
    Note: Major checkpoints (multiples of 500) are stored ONLY in major_checkpoints/,
    not in regular_checkpoints/, to save space and avoid redundancy.
    
    Args:
        output_dir: Training output directory
        max_step: Optional maximum step number to include (filters out old checkpoints from previous runs)
    
    Returns list of checkpoint paths, prioritizing main directories if they exist.
    """
    if not os.path.exists(output_dir):
        return []
    
    checkpoints = []
    
    # 1. Check main checkpoint directories
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    main_checkpoints = glob.glob(checkpoint_pattern)
    
    # 2. Check backup directories (if main checkpoints don't exist or were deleted)
    regular_ckpt_dir = os.path.join(output_dir, "regular_checkpoints")
    major_ckpt_dir = os.path.join(output_dir, "major_checkpoints")
    
    backup_checkpoints = []
    if os.path.exists(regular_ckpt_dir):
        for p in ("checkpoint-*", "regular-checkpoint-*"):
            backup_checkpoints.extend(glob.glob(os.path.join(regular_ckpt_dir, p)))
    if os.path.exists(major_ckpt_dir):
        for p in ("checkpoint-*", "major-checkpoint-*"):
            backup_checkpoints.extend(glob.glob(os.path.join(major_ckpt_dir, p)))
    
    def get_step(ckpt_path: str) -> int:
        """Extract step number from checkpoint path using utility function."""
        step = extract_checkpoint_step(ckpt_path)
        return step if step is not None else -1
    
    # Create a map of step -> checkpoint paths (prioritize main checkpoints)
    checkpoint_map = {}
    
    # Add main checkpoints first (they take priority)
    for ckpt in main_checkpoints:
        step = get_step(ckpt)
        if step >= 0:
            checkpoint_map[step] = ckpt
    
    # Add backup checkpoints only if main checkpoint doesn't exist for that step
    for ckpt in backup_checkpoints:
        step = get_step(ckpt)
        if step >= 0 and step not in checkpoint_map:
            # Verify backup has adapter files before including it
            adapter_file = os.path.join(ckpt, "adapter_model.safetensors")
            if os.path.exists(adapter_file):
                checkpoint_map[step] = ckpt
    
    # Filter by max_step if provided (to exclude old checkpoints from previous runs)
    if max_step is not None:
        checkpoint_map = {step: path for step, path in checkpoint_map.items() if step <= max_step}
        if not checkpoint_map:
            print(f"Warning: No checkpoints found with step <= {max_step}. This may indicate training hasn't started yet.")
    
    # Sort by step number
    valid_checkpoints = [checkpoint_map[step] for step in sorted(checkpoint_map.keys())]
    return valid_checkpoints


def get_evaluated_checkpoints_from_files(output_dir: str) -> set:
    """Get set of already evaluated checkpoint steps using utility function."""
    return get_evaluated_checkpoint_steps(output_dir)


def check_early_stopping_signal(output_dir: str) -> bool:
    """Check if early stopping signal exists from evaluation monitor."""
    signal_file = os.path.join(output_dir, ".early_stop")
    return os.path.exists(signal_file)


def write_early_stopping_signal(output_dir: str):
    """Write early stopping signal file."""
    signal_file = os.path.join(output_dir, ".early_stop")
    with open(signal_file, 'w') as f:
        f.write("Early stopping triggered by evaluation monitor\n")
    print(f"✓ Early stopping signal written to {signal_file}")


def get_best_checkpoint_metric(eval_results_dir: str) -> Optional[Dict]:
    """Get the best checkpoint based on ROUGE-Lsum metric."""
    best_metric = None
    best_checkpoint = None
    
    # Check new location: all_eval_results/checkpoint-nnn-eval-results.json
    all_eval_results_dir = os.path.join(eval_results_dir, "all_eval_results")
    if os.path.exists(all_eval_results_dir):
        for eval_file in glob.glob(os.path.join(all_eval_results_dir, "checkpoint-*-eval-results.json")):
            try:
                with open(eval_file, 'r') as f:
                    results = json.load(f)
                
                # Use rougeLsum as the metric (check both with and without eval_ prefix)
                metric_value = results.get('eval_rougeLsum', results.get('rougeLsum', None))
                if metric_value is not None:
                    if best_metric is None or metric_value > best_metric:
                        best_metric = metric_value
                        # Extract checkpoint step from filename
                        filename = os.path.basename(eval_file)
                        checkpoint_step = filename.replace("checkpoint-", "").replace("-eval-results.json", "")
                        best_checkpoint = f"checkpoint-{checkpoint_step}"
            except (json.JSONDecodeError, ValueError, IndexError) as e:
                print(f"Error reading {eval_file}: {e}")
                continue
    
    # Also check old location for backwards compatibility
    for ckpt_dir in glob.glob(os.path.join(eval_results_dir, "checkpoint-*")):
        eval_file = os.path.join(ckpt_dir, "eval_results", "eval_results.json")
        if os.path.exists(eval_file):
            try:
                with open(eval_file, 'r') as f:
                    results = json.load(f)
                
                # Use rougeLsum as the metric (check both with and without eval_ prefix)
                metric_value = results.get('eval_rougeLsum', results.get('rougeLsum', None))
                if metric_value is not None:
                    if best_metric is None or metric_value > best_metric:
                        best_metric = metric_value
                        best_checkpoint = os.path.basename(ckpt_dir)
            except Exception as e:
                print(f"Error reading {eval_file}: {e}")
    
    return {
        'checkpoint': best_checkpoint,
        'rougeLsum': best_metric
    } if best_checkpoint else None


def check_training_complete(output_dir: str, checkpoints: Optional[List[str]] = None) -> bool:
    """Check if training has completed (by looking for completion signal file).
    
    Only returns True if:
    1. The .training_complete file exists AND is recent (within last hour)
    2. AND there are no checkpoints newer than the file, OR no checkpoints at all
    
    This prevents false positives from previous training runs.
    """
    completion_file = os.path.join(output_dir, ".training_complete")
    if not os.path.exists(completion_file):
        return False
    
    try:
        file_time = os.path.getmtime(completion_file)
        current_time = time.time()
        
        # If file is older than 1 hour, ignore it (probably from previous run)
        if current_time - file_time > 3600:  # 1 hour in seconds
            print(f"Warning: Found old .training_complete file (older than 1 hour). Ignoring it.")
            return False
        
        # If checkpoints are provided, check if any are newer than the completion file
        # If there are newer checkpoints, training is still ongoing - ignore the file
        if checkpoints:
            for checkpoint_path in checkpoints:
                try:
                    ckpt_time = os.path.getmtime(checkpoint_path)
                    # If checkpoint is newer than completion file, training is still running
                    if ckpt_time > file_time:
                        print(f"Warning: Found .training_complete file, but checkpoint {os.path.basename(checkpoint_path)} "
                              f"is newer. Training appears to be ongoing. Ignoring completion signal.")
                        return False
                except Exception:
                    continue
        
        # File exists, is recent, and no newer checkpoints found
        return True
    except Exception:
        # If we can't check file time, err on the side of continuing monitoring
        return False


def monitor_and_evaluate(
    output_dir: str,
    model_name: str,
    val_dataset_path: str,
    hf_token: Optional[str] = None,
    check_interval: int = 30,  # Check every 30 seconds
    early_stopping_patience: int = 10,  # Stop if no improvement for 10 evaluations
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    use_multi_gpu: bool = True,
    timeout_minutes: int = 30,  # Stop if no new checkpoints for X minutes
    major_checkpoint_interval: int = 500,  # Every Nth step is major (gets BERTScore). Default: 500 (every 500 steps = checkpoint-500, checkpoint-1000, etc.)
    include_nli_faithfulness: bool = False,  # Enable NLI faithfulness evaluation
    checkpoint_stability_seconds: int = 120,  # Wait for checkpoint to be stable (not modified) for this many seconds before evaluating
):
    """Monitor checkpoints and evaluate them as they appear.
    
    Args:
        output_dir: Training output directory (where checkpoints are saved)
        model_name: Model identifier (HF name)
        val_dataset_path: Path to validation dataset
        hf_token: HuggingFace token
        check_interval: How often to check for new checkpoints (seconds)
        early_stopping_patience: Number of evaluations without improvement before stopping
        wandb_project: Wandb project name
        wandb_entity: Wandb entity name
        wandb_run_name: Wandb run name (for linking to training run)
        use_multi_gpu: Whether to use multi-GPU evaluation
        timeout_minutes: Stop monitoring if no new checkpoints appear for this many minutes
        major_checkpoint_interval: Every Nth checkpoint is major (gets BERTScore, default: 5)
        include_nli_faithfulness: Enable NLI faithfulness evaluation (slow, default: False)
        checkpoint_stability_seconds: Wait for checkpoint to be stable (not modified) for this many seconds before evaluating (default: 120)
    """
    print(f"Monitor: {output_dir} (model={model_name}, check_interval={check_interval}s, early_stopping_patience={early_stopping_patience})")
    
    # Verify output directory exists
    if not os.path.exists(output_dir):
        print(f"WARNING: Output directory does not exist: {output_dir}")
        print("Creating directory...")
        os.makedirs(output_dir, exist_ok=True)
        print("Waiting for training to start...")
    
    # Wait for training to start (check for signal file)
    training_started_file = os.path.join(output_dir, "training_started.txt")
    max_wait_time = 3600  # Wait up to 1 hour for training to start
    wait_interval = 10  # Check every 10 seconds
    waited_time = 0
    
    if not os.path.exists(training_started_file):
        print(f"\n{'='*70}")
        print("WAITING FOR TRAINING TO START")
        print(f"{'='*70}")
        print(f"Looking for signal file: {training_started_file}")
        print(f"Will wait up to {max_wait_time // 60} minutes...")
        print(f"Checking every {wait_interval} seconds...")
        print(f"{'='*70}\n")
        
        while not os.path.exists(training_started_file) and waited_time < max_wait_time:
            time.sleep(wait_interval)
            waited_time += wait_interval
            if waited_time % 60 == 0:  # Print status every minute
                print(f"Still waiting for training to start... ({waited_time // 60} minutes elapsed)")
        
        if not os.path.exists(training_started_file):
            print(f"\n{'='*70}")
            print("ERROR: Training did not start within the timeout period")
            print(f"{'='*70}")
            print(f"Waited {waited_time // 60} minutes for training to start.")
            print(f"Expected signal file: {training_started_file}")
            print("\nPossible reasons:")
            print("  1. Training job hasn't started yet")
            print("  2. Training job failed to start")
            print("  3. Training script doesn't create the signal file")
            print("\nSuggestion: Use SLURM job dependency:")
            print("  sbatch --dependency=afterok:TRAINING_JOB_ID run_monitor_evaluation.sbatch")
            print(f"{'='*70}\n")
            return
        else:
            print(f"✓ Training started! Signal file found: {training_started_file}")
            print(f"  Waited {waited_time // 60} minutes and {waited_time % 60} seconds\n")
    
    # Initialize wandb for monitoring
    if wandb_project:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name or f"{os.path.basename(output_dir)}-monitor",
            tags=["checkpoint-monitor", "fsdp-evaluation"],
            config={
                "output_dir": output_dir,
                "model_name": model_name,
                "check_interval": check_interval,
                "early_stopping_patience": early_stopping_patience,
            },
            reinit=True,
        )
        print(f">>> wandb run initialized: {wandb.run.name}")
    
    # Load already evaluated checkpoints from disk (persists across restarts)
    evaluated_steps = get_evaluated_checkpoints_from_files(output_dir)
    if evaluated_steps:
        print(f"Found {len(evaluated_steps)} already evaluated checkpoints: {sorted(evaluated_steps)}")
    
    best_rouge_lsum = None
    consecutive_zero_rouge_count = 0  # Track consecutive checkpoints with zero ROUGE scores
    ZERO_ROUGE_THRESHOLD = 5  # Stop training if ROUGE is zero for 5 consecutive checkpoints
    best_checkpoint_step = None
    no_improvement_count = 0
    last_checkpoint_time = None  # Track when we last saw a new checkpoint
    timeout_seconds = timeout_minutes * 60
    logged_to_wandb = set()  # Track which checkpoints we've already logged to wandb
    
    # BERTScore early stopping (only for major checkpoints)
    consecutive_low_bertscore_count = 0  # Track consecutive major checkpoints with low BERTScore
    LOW_BERTSCORE_THRESHOLD = 2  # Stop if BERTScore < 0.25 for 2 consecutive major checkpoints
    BERTSCORE_LOW_THRESHOLD = 0.25  # BERTScore below this is considered "low"
    BERTSCORE_DROP_THRESHOLD = 0.10  # Drop of this magnitude is considered significant
    best_bertscore = None
    previous_bertscore = None  # Track previous BERTScore to detect drops
    
    print("\nStarting checkpoint monitoring...")
    print(f"Will stop if no new checkpoints appear for {timeout_minutes} minutes")
    print("Press Ctrl+C to stop monitoring (training will continue)\n")
    
    # Clear any stale .early_stop from a *previous* run so we don't exit immediately
    # when reusing the same output_dir (e.g. new training + monitor for same model)
    signal_file = os.path.join(output_dir, ".early_stop")
    if os.path.exists(signal_file):
        try:
            os.remove(signal_file)
            print("Cleared stale .early_stop from previous run (monitor will run for this session).\n")
        except OSError as e:
            print(f"Warning: could not remove stale .early_stop: {e}\n")
    
    try:
        while True:
            # Check if early stopping was already triggered (during this session)
            if check_early_stopping_signal(output_dir):
                print("Early stopping signal detected. Stopping monitor.")
                break
            
            # Get current training step to filter out old checkpoints from previous runs
            current_training_step = get_current_training_step(output_dir)
            
            # Find all checkpoints, filtering by current training step
            checkpoints = find_checkpoints(output_dir, max_step=current_training_step)
            
            if current_training_step is not None:
                # Log current training progress periodically (every 10 iterations to avoid spam)
                global _last_logged_step, _log_counter
                _log_counter += 1
                if (_last_logged_step != current_training_step or _log_counter % 10 == 0):
                    print(f"Step {current_training_step} | {len(checkpoints)} checkpoints to evaluate")
                    _last_logged_step = current_training_step
            
            # Check if training has completed (with time check to avoid false positives)
            # Pass checkpoints to verify no newer checkpoints exist
            if check_training_complete(output_dir, checkpoints):
                print("Training completion signal detected. Stopping monitor.")
                break
            
            if not checkpoints:
                print(f"No checkpoints found yet. Waiting {check_interval}s...")
                time.sleep(check_interval)
                continue
            
            # Find the first (oldest) checkpoint that hasn't been evaluated yet
            checkpoint_to_evaluate = None
            checkpoint_step = None
            force_recompute_checkpoint = False  # True when checkpoint is newer than stale eval (rerun scenario)
            
            # CONTINUE-RUN FIX: When resuming from checkpoint-N to 10000, skip re-evaluating 100..N.
            # Training has progressed beyond max_evaluated; old checkpoints are done. Avoids false
            # "newer than" from rsync/filesystem mtime quirks, and prevents loading stale results
            # that trigger early stopping.
            max_evaluated_step = max(evaluated_steps) if evaluated_steps else 0
            is_continue_run = (
                current_training_step is not None
                and max_evaluated_step > 0
                and current_training_step > max_evaluated_step
            )
            
            for checkpoint_path in checkpoints:
                try:
                    # Extract step and name from checkpoint path using utility function
                    checkpoint_name, step = get_checkpoint_name_and_step(checkpoint_path)
                    if step is None:
                        continue
                except (ValueError, IndexError):
                    continue
                
                # In continue runs, skip checkpoints already fully evaluated (step <= max_evaluated)
                if is_continue_run and step <= max_evaluated_step:
                    evaluated_steps.add(step)
                    continue
                
                # Check if evaluation results file exists (new location)
                # Use utility function to get model_dir (handles backup directories correctly)
                model_dir = get_model_dir_from_checkpoint(checkpoint_path)
                
                all_eval_results_dir = os.path.join(model_dir, "all_eval_results")
                eval_results_file = os.path.join(all_eval_results_dir, f"{checkpoint_name}-eval-results.json")
                
                # Also check old location for backwards compatibility
                old_eval_results_file = os.path.join(checkpoint_path, "eval_results", "eval_results.json")
                
                # Check if evaluation results exist, and if so, verify they're from current run
                eval_results_exist = os.path.exists(eval_results_file) or os.path.exists(old_eval_results_file)
                if eval_results_exist:
                    # Re-evaluate if results are stale (from previous run before retrain from scratch)
                    # Two checks: (1) checkpoint newer than eval, (2) eval older than training_started.txt
                    try:
                        eval_file_to_check = eval_results_file if os.path.exists(eval_results_file) else old_eval_results_file
                        eval_mtime = os.path.getmtime(eval_file_to_check)
                        checkpoint_mtime = os.path.getmtime(checkpoint_path)
                        
                        # Check 1: Checkpoint newer than eval (e.g. checkpoint was overwritten by new training)
                        checkpoint_newer = checkpoint_mtime > eval_mtime
                        
                        # Check 2: Eval results from before current training session (retrain-from-scratch)
                        # training_started.txt is (re)created when training starts; eval older = previous run
                        training_started_file = os.path.join(output_dir, "training_started.txt")
                        eval_from_previous_run = False
                        if os.path.exists(training_started_file):
                            training_started_mtime = os.path.getmtime(training_started_file)
                            eval_from_previous_run = eval_mtime < training_started_mtime
                        
                        if checkpoint_newer or eval_from_previous_run:
                            reason = "checkpoint newer than eval" if checkpoint_newer else "eval results from previous run (retrain from scratch)"
                            print(f"Checkpoint-{step}: {reason}. Re-evaluating...")
                            evaluated_steps.discard(step)
                            force_recompute_checkpoint = True
                        else:
                            # Already evaluated in current run - skip
                            evaluated_steps.add(step)
                            continue
                    except Exception as e:
                        # If we can't compare timestamps, assume already evaluated to be safe
                        print(f"Warning: Could not compare timestamps for checkpoint-{step}: {e}")
                        evaluated_steps.add(step)
                        continue
                elif step in evaluated_steps:
                    # No evaluation results exist, but step is in evaluated_steps (from previous run)
                    # This means checkpoint was evaluated before but results were deleted, or it's a new checkpoint
                    # Remove from evaluated_steps so it can be evaluated
                    print(f"Checkpoint-{step} was previously marked as evaluated but results are missing. Re-evaluating...")
                    evaluated_steps.discard(step)
                
                # Verify checkpoint is complete (must have adapter files)
                adapter_file = os.path.join(checkpoint_path, "adapter_model.safetensors")
                if not os.path.exists(adapter_file):
                    # Check if this checkpoint was already evaluated (only has eval_results in old location)
                    dir_contents = os.listdir(checkpoint_path) if os.path.isdir(checkpoint_path) else []
                    if len(dir_contents) == 1 and 'eval_results' in dir_contents:
                        # Already evaluated and adapter files cleaned up - mark as evaluated
                        print(f"Checkpoint-{step} appears to be already evaluated (adapter files cleaned up). Skipping.")
                        evaluated_steps.add(step)
                    # Not complete yet - skip this one, check next
                    continue
                
                # Check if checkpoint is stable (not recently modified)
                # This prevents evaluating checkpoints that are still being written
                try:
                    checkpoint_mtime = os.path.getmtime(checkpoint_path)
                    current_time = time.time()
                    time_since_modification = current_time - checkpoint_mtime
                    
                    if time_since_modification < checkpoint_stability_seconds:
                        # Checkpoint was recently modified - wait for it to stabilize
                        wait_time = checkpoint_stability_seconds - time_since_modification
                        print(f"Checkpoint-{step} was modified {time_since_modification:.0f}s ago. "
                              f"Waiting {wait_time:.0f}s for stability before evaluating...")
                        # Skip this checkpoint for now, will check again next iteration
                        continue
                except Exception as e:
                    # If we can't check mtime, be cautious and skip
                    print(f"Warning: Could not check checkpoint stability for checkpoint-{step}: {e}")
                    continue
                
                # Found an unevaluated, complete, and stable checkpoint
                checkpoint_to_evaluate = checkpoint_path
                checkpoint_step = step
                # Keep force_recompute_checkpoint as set above (for "newer than" case); reset for next iteration
                # (force_recompute applies to this checkpoint only; next checkpoint gets fresh flag)
                
                # Log if we're using a backup checkpoint
                if "regular_checkpoints" in checkpoint_path or "major_checkpoints" in checkpoint_path:
                    backup_type = "major" if "major_checkpoints" in checkpoint_path else "regular"
                    print(f"ℹ Using {backup_type} checkpoint backup for evaluation (original checkpoint was deleted)")
                break
            
            # If no checkpoint found to evaluate, wait
            if checkpoint_to_evaluate is None:
                # Check if we should update timeout
                if checkpoints:
                    latest_checkpoint = checkpoints[-1]
                    try:
                        latest_step = int(os.path.basename(latest_checkpoint).split("-")[-1])
                        print(f"All checkpoints up to {latest_step} are evaluated. Waiting for new checkpoints...")
                    except (ValueError, IndexError):
                        pass
                
                # Check timeout
                current_time = time.time()
                if last_checkpoint_time is not None:
                    time_since_last_checkpoint = current_time - last_checkpoint_time
                    if time_since_last_checkpoint > timeout_seconds:
                        print(f"\n{'='*70}")
                        print(f"Timeout: No new checkpoints for {timeout_minutes} minutes")
                        print("Assuming training has completed. Stopping monitor.")
                        print(f"{'='*70}\n")
                        break
                
                time.sleep(check_interval)
                continue
            
            # NEW CHECKPOINT TO EVALUATE - reset timeout timer
            last_checkpoint_time = time.time()
            
            # Prefer backup checkpoints if they exist (they're guaranteed to be stable)
            # Check if this checkpoint exists in backup directories
            model_dir = get_model_dir_from_checkpoint(checkpoint_to_evaluate)
            is_in_backup = "regular_checkpoints" in checkpoint_to_evaluate or "major_checkpoints" in checkpoint_to_evaluate
            
            if not is_in_backup:
                # Check if backup exists - prefer backup for stability
                # Support both unified naming (checkpoint-X) and legacy (regular-checkpoint-X / major-checkpoint-X)
                major_new = os.path.join(model_dir, "major_checkpoints", f"checkpoint-{checkpoint_step}")
                major_legacy = os.path.join(model_dir, "major_checkpoints", f"major-checkpoint-{checkpoint_step}")
                regular_new = os.path.join(model_dir, "regular_checkpoints", f"checkpoint-{checkpoint_step}")
                regular_legacy = os.path.join(model_dir, "regular_checkpoints", f"regular-checkpoint-{checkpoint_step}")
                
                backup_path = None
                for cand in (major_new, major_legacy):
                    if os.path.exists(cand):
                        backup_adapter = os.path.join(cand, "adapter_model.safetensors")
                        if os.path.exists(backup_adapter):
                            backup_path = cand
                            print(f"ℹ Found stable backup of checkpoint-{checkpoint_step} in major_checkpoints/. Using backup for evaluation.")
                            break
                if backup_path is None:
                    for cand in (regular_new, regular_legacy):
                        if os.path.exists(cand):
                            backup_adapter = os.path.join(cand, "adapter_model.safetensors")
                            if os.path.exists(backup_adapter):
                                backup_path = cand
                                print(f"ℹ Found stable backup of checkpoint-{checkpoint_step} in regular_checkpoints/. Using backup for evaluation.")
                                break
                
                if backup_path:
                    checkpoint_to_evaluate = backup_path
            
            # Final verification before evaluation (checkpoint might have been deleted)
            if not os.path.exists(checkpoint_to_evaluate):
                print(f"Warning: Checkpoint directory no longer exists: {checkpoint_to_evaluate}")
                print(f"  This checkpoint may have been backed up and removed. Skipping evaluation.")
                # Mark as evaluated to avoid infinite retries
                evaluated_steps.add(checkpoint_step)
                time.sleep(check_interval)
                continue
            
            # Verify checkpoint has required adapter files
            adapter_file = os.path.join(checkpoint_to_evaluate, "adapter_model.safetensors")
            if not os.path.exists(adapter_file):
                print(f"Warning: Checkpoint missing adapter files: {checkpoint_to_evaluate}")
                print(f"  This checkpoint may be incomplete or already cleaned up. Skipping evaluation.")
                # Mark as evaluated to avoid infinite retries
                evaluated_steps.add(checkpoint_step)
                time.sleep(check_interval)
                continue
            
            # Evaluate the checkpoint
            print(f"\n{'='*70}")
            print(f"Evaluating checkpoint-{checkpoint_step}")
            print(f"Checkpoint path: {checkpoint_to_evaluate}")
            print(f"{'='*70}")
            
            try:
                eval_results, _ = evaluate_checkpoint(
                    model_name=model_name,
                    checkpoint_dir=checkpoint_to_evaluate,
                    val_dataset_path=val_dataset_path,
                    hf_token=hf_token,
                    output_dir=None,  # None will trigger default: model_dir/all_eval_results
                    use_multi_gpu=use_multi_gpu,
                    wandb_project=None,
                    wandb_entity=None,
                    wandb_disabled=True,
                    major_checkpoint_interval=major_checkpoint_interval,
                    include_nli_faithfulness=include_nli_faithfulness,
                    force_recompute=force_recompute_checkpoint,  # Re-run when checkpoint newer than stale eval (rerun)
                )
                
                if not eval_results:
                    print(f"Warning: Evaluation returned no results for checkpoint-{checkpoint_step}")
                    evaluated_steps.add(checkpoint_step)
                    if checkpoint_step not in logged_to_wandb:
                        logged_to_wandb.add(checkpoint_step)
                    time.sleep(check_interval)
                    continue
                
                if eval_results:
                    # Check all ROUGE metrics to detect zero scores
                    rouge1 = eval_results.get('eval_rouge1', eval_results.get('rouge1', 0))
                    rouge2 = eval_results.get('eval_rouge2', eval_results.get('rouge2', 0))
                    rougeL = eval_results.get('eval_rougeL', eval_results.get('rougeL', 0))
                    rouge_lsum = eval_results.get('eval_rougeLsum', eval_results.get('rougeLsum', 0))
                    
                    # Validate metric values
                    rouge_metrics = [rouge1, rouge2, rougeL, rouge_lsum]
                    if not all(isinstance(m, (int, float)) and m >= 0 for m in rouge_metrics):
                        print(f"Warning: Invalid ROUGE values: rouge1={rouge1}, rouge2={rouge2}, rougeL={rougeL}, rougeLsum={rouge_lsum}")
                        evaluated_steps.add(checkpoint_step)
                        if checkpoint_step not in logged_to_wandb:
                            logged_to_wandb.add(checkpoint_step)
                        time.sleep(check_interval)
                        continue
                    
                    # Check if all ROUGE scores are zero (indicates model collapse)
                    all_rouge_zero = all(m == 0.0 for m in rouge_metrics)
                    
                    if all_rouge_zero:
                        consecutive_zero_rouge_count += 1
                        print(f"⚠ Warning: All ROUGE scores are 0.00 for checkpoint-{checkpoint_step}")
                        print(f"  Consecutive zero ROUGE count: {consecutive_zero_rouge_count}/{ZERO_ROUGE_THRESHOLD}")
                        
                        # Check if we've hit the threshold for early stopping
                        if consecutive_zero_rouge_count >= ZERO_ROUGE_THRESHOLD:
                            print(f"\n{'='*70}")
                            print("⚠ CRITICAL: Early stopping triggered due to zero ROUGE scores!")
                            print(f"ROUGE scores have been zero for {consecutive_zero_rouge_count} consecutive checkpoints.")
                            print(f"This indicates the model has likely collapsed (e.g., outputting only EOS tokens).")
                            print(f"Stopping training to prevent further resource waste.")
                            print(f"{'='*70}\n")
                            write_early_stopping_signal(output_dir)
                            
                            # Log to wandb
                            if wandb.run:
                                wandb.log({
                                    "monitor/early_stop_reason": "zero_rouge_scores",
                                    "monitor/consecutive_zero_rouge_count": consecutive_zero_rouge_count,
                                }, step=checkpoint_step)
                            
                            print("Early stopping signal written. Training will stop on next check.")
                            return
                        
                        # Continue monitoring but don't update best score
                        evaluated_steps.add(checkpoint_step)
                        if checkpoint_step not in logged_to_wandb:
                            logged_to_wandb.add(checkpoint_step)
                        time.sleep(check_interval)
                        continue
                    else:
                        # Reset counter if we get non-zero ROUGE scores
                        if consecutive_zero_rouge_count > 0:
                            print(f"✓ ROUGE scores recovered (non-zero). Resetting zero ROUGE counter.")
                            consecutive_zero_rouge_count = 0
                    
                    # Check BERTScore for major checkpoints (early stopping)
                    is_major = checkpoint_step is not None and is_major_checkpoint(checkpoint_step, major_checkpoint_interval)
                    bertscore_f1 = eval_results.get('eval_reference_bertscore_f1_mean', None)
                    
                    if is_major and bertscore_f1 is not None:
                        # BERTScore is available for this major checkpoint
                        print(f"  BERTScore F1: {bertscore_f1:.4f}")
                        
                        # Check for low BERTScore
                        if bertscore_f1 < BERTSCORE_LOW_THRESHOLD:
                            consecutive_low_bertscore_count += 1
                            print(f"⚠ Warning: BERTScore F1 ({bertscore_f1:.4f}) is below threshold ({BERTSCORE_LOW_THRESHOLD})")
                            print(f"  Consecutive low BERTScore count: {consecutive_low_bertscore_count}/{LOW_BERTSCORE_THRESHOLD}")
                            
                            # Check if we've hit the threshold for early stopping
                            if consecutive_low_bertscore_count >= LOW_BERTSCORE_THRESHOLD:
                                print(f"\n{'='*70}")
                                print("⚠ CRITICAL: Early stopping triggered due to low BERTScore!")
                                print(f"BERTScore F1 has been below {BERTSCORE_LOW_THRESHOLD} for {consecutive_low_bertscore_count} consecutive major checkpoints.")
                                print(f"Current BERTScore: {bertscore_f1:.4f}")
                                print(f"This indicates the model quality has degraded significantly.")
                                print(f"Stopping training to prevent further resource waste.")
                                print(f"{'='*70}\n")
                                write_early_stopping_signal(output_dir)
                                
                                # Log to wandb
                                if wandb.run:
                                    wandb.log({
                                        "monitor/early_stop_reason": "low_bertscore",
                                        "monitor/consecutive_low_bertscore_count": consecutive_low_bertscore_count,
                                        "monitor/bertscore_f1": bertscore_f1,
                                    }, step=checkpoint_step)
                                
                                print("Early stopping signal written. Training will stop on next check.")
                                return
                        else:
                            # Reset counter if BERTScore is above threshold
                            if consecutive_low_bertscore_count > 0:
                                print(f"✓ BERTScore recovered (above {BERTSCORE_LOW_THRESHOLD}). Resetting low BERTScore counter.")
                                consecutive_low_bertscore_count = 0
                        
                        # Check for significant drop in BERTScore
                        if previous_bertscore is not None:
                            bertscore_drop = previous_bertscore - bertscore_f1
                            if bertscore_drop > BERTSCORE_DROP_THRESHOLD:
                                print(f"⚠ Warning: Significant BERTScore drop detected!")
                                print(f"  Previous: {previous_bertscore:.4f}, Current: {bertscore_f1:.4f}, Drop: {bertscore_drop:.4f}")
                                print(f"  This may indicate model degradation. Monitor closely.")
                                # Log but don't stop immediately (allow recovery)
                                if wandb.run:
                                    wandb.log({
                                        "monitor/bertscore_drop": bertscore_drop,
                                        "monitor/bertscore_previous": previous_bertscore,
                                        "monitor/bertscore_current": bertscore_f1,
                                    }, step=checkpoint_step)
                        
                        # Update best and previous BERTScore
                        if best_bertscore is None or bertscore_f1 > best_bertscore:
                            was_str = f"{best_bertscore:.4f}" if best_bertscore is not None else "N/A"
                            print(f"✓ New best BERTScore F1: {bertscore_f1:.4f} (was {was_str})")
                            best_bertscore = bertscore_f1
                        previous_bertscore = bertscore_f1
                    elif is_major:
                        # Major checkpoint but BERTScore not available (shouldn't happen, but handle gracefully)
                        print(f"⚠ Warning: Major checkpoint-{checkpoint_step} but BERTScore not available in results")
                    
                    # Continue with normal evaluation flow for non-zero ROUGE scores
                    evaluated_steps.add(checkpoint_step)
                    if checkpoint_step not in logged_to_wandb:
                        logged_to_wandb.add(checkpoint_step)
                    
                    # Log to wandb
                    log_dict = {
                        "monitor/checkpoint_step": checkpoint_step,
                        "monitor/rouge1": rouge1,
                        "monitor/rouge2": rouge2,
                        "monitor/rougeL": rougeL,
                        "monitor/rougeLsum": rouge_lsum,
                        "monitor/consecutive_zero_rouge_count": consecutive_zero_rouge_count,
                    }
                    if is_major and bertscore_f1 is not None:
                        log_dict["monitor/bertscore_f1"] = bertscore_f1
                        log_dict["monitor/consecutive_low_bertscore_count"] = consecutive_low_bertscore_count
                    if wandb.run:
                        wandb.log(log_dict, step=checkpoint_step)
                    
                    # Check for improvement
                    if best_rouge_lsum is None or rouge_lsum > best_rouge_lsum:
                        was_str = f"{best_rouge_lsum:.2f}" if best_rouge_lsum is not None else "N/A"
                        print(f"✓ New best ROUGE-Lsum: {rouge_lsum:.2f} (was {was_str})")
                        best_rouge_lsum = rouge_lsum
                        best_checkpoint_step = checkpoint_step
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        print(f"  No improvement. Best: {best_rouge_lsum:.2f} at step {best_checkpoint_step}")
                        print(f"  Patience: {no_improvement_count}/{early_stopping_patience}")
                    
                    # Check early stopping
                    if no_improvement_count >= early_stopping_patience:
                        print(f"\n{'='*70}")
                        print("Early stopping triggered!")
                        print(f"No improvement for {early_stopping_patience} evaluations")
                        print(f"Best checkpoint: checkpoint-{best_checkpoint_step} (ROUGE-Lsum: {best_rouge_lsum:.2f})")
                        print(f"{'='*70}\n")
                        write_early_stopping_signal(output_dir)
                        
                        # Log best checkpoint info
                        if wandb.run:
                            wandb.log({
                                "monitor/best_checkpoint_step": best_checkpoint_step,
                                "monitor/best_rouge_lsum": best_rouge_lsum,
                                "monitor/early_stopping_triggered": True,
                            })
                        
                        break
                
            except AlreadyEvaluatedError as e:
                # Checkpoint already has eval results (e.g. from previous run) - skip gracefully
                print(f"ℹ Checkpoint-{checkpoint_step} already evaluated (results exist). Skipping.")
                evaluated_steps.add(checkpoint_step)
                if checkpoint_step not in logged_to_wandb:
                    logged_to_wandb.add(checkpoint_step)
            except ValueError as e:
                # Handle checkpoint not found errors gracefully
                error_msg = str(e)
                if "Checkpoint directory does not exist" in error_msg or "does not exist" in error_msg:
                    print(f"Warning: Checkpoint directory does not exist: {checkpoint_to_evaluate}")
                    print(f"  This checkpoint may have been backed up and removed. Skipping evaluation.")
                    # Mark as evaluated to avoid infinite retries
                    evaluated_steps.add(checkpoint_step)
                else:
                    # Other ValueError - print full traceback
                    print(f"Error evaluating checkpoint-{checkpoint_step}: {e}")
                    import traceback
                    traceback.print_exc()
                    evaluated_steps.add(checkpoint_step)
            except Exception as e:
                print(f"Error evaluating checkpoint-{checkpoint_step}: {e}")
                import traceback
                traceback.print_exc()
                # Mark as evaluated to avoid infinite retries, but don't add to logged_to_wandb
                evaluated_steps.add(checkpoint_step)
                # Continue monitoring even if evaluation fails
            
            # Wait before next check
            time.sleep(check_interval)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user. Training will continue.")
    
    finally:
        # Print summary
        print("\n" + "="*70)
        print("Evaluation Monitor Summary")
        print("="*70)
        print(f"Evaluated checkpoints: {sorted(evaluated_steps)}")
        if best_checkpoint_step:
            print(f"Best checkpoint: checkpoint-{best_checkpoint_step}")
            print(f"Best ROUGE-Lsum: {best_rouge_lsum:.2f}")
        print("="*70)
        
        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Monitor and evaluate checkpoints during FSDP training'
    )
    
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Training output directory (where checkpoints are saved)')
    parser.add_argument('--model', type=str, required=True,
                       choices=['viking-7b', 'viking-13b', 'viking-33b',
                                'gemma-2b', 'gemma-7b', 'gemma-2-9b', 'gemma-2-27b',
                                'gemma-3-12b', 'gemma-3-27b',
                                'normistral-7b', 'normistral-11b', 'normistral-7b-instruct',
                                'norskgpt-llama3-8b', 'llama-3.1-8b-instruct', 'llama-2-13b-chat-norwegian',
                                'eurollm-9b-instruct', 'norwai-mistral-7b-instruct', 'nb-gpt-j-6b', 'mt5'],
                       help='Model short name')
    parser.add_argument('--val_dataset', type=str, required=True,
                       help='Path to validation dataset (JSONL)')
    parser.add_argument('--hf_token', type=str,
                       help='HuggingFace token')
    parser.add_argument('--check_interval', type=int, default=30,
                       help='How often to check for new checkpoints (seconds)')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Number of evaluations without improvement before stopping (default: 10)')
    parser.add_argument('--wandb_project', type=str, default='lm-finetuning',
                       help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Wandb entity name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name (for linking to training run)')
    parser.add_argument('--use_multi_gpu', action='store_true',
                       help='Use multi-GPU evaluation')
    parser.add_argument('--timeout_minutes', type=int, default=30,
                       help='Stop monitoring if no new checkpoints appear for this many minutes (default: 30)')
    parser.add_argument('--major_checkpoint_interval', type=int, default=500,
                       help='Every Nth step is considered "major" for BERTScore evaluation (default: 500). Major checkpoints: checkpoint-500, checkpoint-1000, checkpoint-1500, etc.')
    parser.add_argument('--include_nli_faithfulness', action='store_true',
                       help='Enable NLI-based faithfulness evaluation (slow: ~4.5s per example, ~37 min for 500 examples)')
    parser.add_argument('--checkpoint_stability_seconds', type=int, default=120,
                       help='Wait for checkpoint to be stable (not modified) for this many seconds before evaluating (default: 120). Prevents evaluating checkpoints that are still being written.')
    
    args = parser.parse_args()
    
    # Get full model name
    model_config = get_model_config(args.model)
    model_name = model_config.hf_name
    
    monitor_and_evaluate(
        output_dir=args.output_dir,
        model_name=model_name,
        val_dataset_path=args.val_dataset,
        hf_token=args.hf_token,
        check_interval=args.check_interval,
        early_stopping_patience=args.early_stopping_patience,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        use_multi_gpu=args.use_multi_gpu,
        timeout_minutes=args.timeout_minutes,
        major_checkpoint_interval=args.major_checkpoint_interval,
        include_nli_faithfulness=args.include_nli_faithfulness,
        checkpoint_stability_seconds=args.checkpoint_stability_seconds,
    )
