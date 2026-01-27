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
    --early_stopping_patience 3 \
    --wandb_project lm-finetuning \
    --wandb_run_name gemma-7b-apptainer-fsdp

The training script will check for early stopping signals and stop if needed.
"""

import argparse
import json
import os
import time
import glob
from pathlib import Path
from typing import Optional, Dict, List
import wandb

# Add torch import for device count check
import torch

from model_configs import get_model_config
from evaluate_distributed_checkpoints_multigpu import evaluate_checkpoint


def find_checkpoints(output_dir: str) -> List[str]:
    """Find all checkpoint directories, sorted by step number."""
    if not os.path.exists(output_dir):
        return []
    
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    def get_step(ckpt_path: str) -> int:
        """Extract step number from checkpoint path."""
        try:
            return int(os.path.basename(ckpt_path).split("-")[-1])
        except (ValueError, IndexError):
            return -1
    
    # Sort by step number, filter out invalid ones
    valid_checkpoints = [ckpt for ckpt in checkpoints if get_step(ckpt) >= 0]
    valid_checkpoints.sort(key=get_step)
    return valid_checkpoints


def get_evaluated_checkpoints_from_files(output_dir: str) -> set:
    """Get set of already evaluated checkpoint steps by checking for eval results files in all_eval_results/."""
    evaluated = set()
    if not os.path.exists(output_dir):
        return evaluated
    
    # Check all_eval_results directory for checkpoint-nnn-eval-results.json files
    all_eval_results_dir = os.path.join(output_dir, "all_eval_results")
    if os.path.exists(all_eval_results_dir):
        for eval_file in glob.glob(os.path.join(all_eval_results_dir, "checkpoint-*-eval-results.json")):
            try:
                # Extract step from filename: checkpoint-123-eval-results.json -> 123
                filename = os.path.basename(eval_file)
                step = int(filename.replace("checkpoint-", "").replace("-eval-results.json", ""))
                evaluated.add(step)
            except (ValueError, IndexError):
                pass
    
    # Also check old location for backwards compatibility
    for ckpt_dir in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        old_eval_results_file = os.path.join(ckpt_dir, "eval_results", "eval_results.json")
        if os.path.exists(old_eval_results_file):
            try:
                step = int(os.path.basename(ckpt_dir).split("-")[-1])
                evaluated.add(step)
            except (ValueError, IndexError):
                pass
    
    return evaluated


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


def check_training_complete(output_dir: str, checkpoints: List[str] = None) -> bool:
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
    early_stopping_patience: int = 3,  # Stop if no improvement for 3 evaluations
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    use_multi_gpu: bool = True,
    timeout_minutes: int = 30,  # Stop if no new checkpoints for X minutes
    major_checkpoint_interval: int = 500,  # Every Nth step is major (gets BERTScore). Default: 500 (every 500 steps = checkpoint-500, checkpoint-1000, etc.)
    include_nli_faithfulness: bool = False,  # Enable NLI faithfulness evaluation
    nli_subset_size: Optional[int] = None,  # Subset size for NLI evaluation
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
        nli_subset_size: Subset size for NLI evaluation (None = all, recommended: 50-100)
    """
    print("=" * 70)
    print("Checkpoint Evaluation Monitor")
    print("=" * 70)
    print(f"Monitoring: {output_dir}")
    print(f"Model: {model_name}")
    print(f"Check interval: {check_interval} seconds")
    print(f"Early stopping patience: {early_stopping_patience}")
    print("=" * 70)
    
    # Verify output directory exists
    if not os.path.exists(output_dir):
        print(f"WARNING: Output directory does not exist: {output_dir}")
        print("Creating directory...")
        os.makedirs(output_dir, exist_ok=True)
        print("Waiting for training to start...")
    
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
    best_checkpoint_step = None
    no_improvement_count = 0
    last_checkpoint_time = None  # Track when we last saw a new checkpoint
    timeout_seconds = timeout_minutes * 60
    logged_to_wandb = set()  # Track which checkpoints we've already logged to wandb
    
    print("\nStarting checkpoint monitoring...")
    print(f"Will stop if no new checkpoints appear for {timeout_minutes} minutes")
    print("Press Ctrl+C to stop monitoring (training will continue)\n")
    
    try:
        while True:
            # Check if early stopping was already triggered
            if check_early_stopping_signal(output_dir):
                print("Early stopping signal detected. Stopping monitor.")
                break
            
            # Find all checkpoints
            checkpoints = find_checkpoints(output_dir)
            
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
            
            for checkpoint_path in checkpoints:
                try:
                    step = int(os.path.basename(checkpoint_path).split("-")[-1])
                except (ValueError, IndexError):
                    continue
                
                # Skip if already evaluated
                if step in evaluated_steps:
                    continue
                
                # Check if evaluation results file exists (new location)
                model_dir = os.path.dirname(checkpoint_path.rstrip('/'))
                all_eval_results_dir = os.path.join(model_dir, "all_eval_results")
                checkpoint_name = os.path.basename(checkpoint_path.rstrip('/'))
                eval_results_file = os.path.join(all_eval_results_dir, f"{checkpoint_name}-eval-results.json")
                
                # Also check old location for backwards compatibility
                old_eval_results_file = os.path.join(checkpoint_path, "eval_results", "eval_results.json")
                
                if os.path.exists(eval_results_file) or os.path.exists(old_eval_results_file):
                    # Already evaluated - mark it and continue
                    evaluated_steps.add(step)
                    continue
                
                # Verify checkpoint is complete
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
                
                # Found an unevaluated, complete checkpoint
                checkpoint_to_evaluate = checkpoint_path
                checkpoint_step = step
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
            
            # Evaluate the checkpoint
            print(f"\n{'='*70}")
            print(f"Evaluating checkpoint-{checkpoint_step}")
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
                    nli_subset_size=nli_subset_size,
                )
                
                if not eval_results:
                    print(f"Warning: Evaluation returned no results for checkpoint-{checkpoint_step}")
                    evaluated_steps.add(checkpoint_step)
                    if checkpoint_step not in logged_to_wandb:
                        logged_to_wandb.add(checkpoint_step)
                    time.sleep(check_interval)
                    continue
                
                if eval_results:
                    rouge_lsum = eval_results.get('eval_rougeLsum', eval_results.get('rougeLsum', 0))
                    # Validate metric value
                    if not isinstance(rouge_lsum, (int, float)) or rouge_lsum < 0:
                        print(f"Warning: Invalid ROUGE-Lsum value: {rouge_lsum}")
                        evaluated_steps.add(checkpoint_step)
                        if checkpoint_step not in logged_to_wandb:
                            logged_to_wandb.add(checkpoint_step)
                        time.sleep(check_interval)
                        continue
                    
                    # Skip if result is 0.00 (likely failed evaluation)
                    if rouge_lsum == 0:
                        print(f"Warning: ROUGE-Lsum is 0.00 for checkpoint-{checkpoint_step}. This may indicate a failed evaluation.")
                        evaluated_steps.add(checkpoint_step)
                        if checkpoint_step not in logged_to_wandb:
                            logged_to_wandb.add(checkpoint_step)
                        time.sleep(check_interval)
                        continue
                    
                    evaluated_steps.add(checkpoint_step)
                    if checkpoint_step not in logged_to_wandb:
                        logged_to_wandb.add(checkpoint_step)
                    
                    # Log to wandb
                    if wandb.run:
                        wandb.log({
                            "monitor/checkpoint_step": checkpoint_step,
                            "monitor/rouge1": eval_results.get('eval_rouge1', eval_results.get('rouge1', 0)),
                            "monitor/rouge2": eval_results.get('eval_rouge2', eval_results.get('rouge2', 0)),
                            "monitor/rougeL": eval_results.get('eval_rougeL', eval_results.get('rougeL', 0)),
                            "monitor/rougeLsum": rouge_lsum,
                        }, step=checkpoint_step)
                    
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
                                'normistral-7b', 'normistral-11b',
                                'norskgpt-llama3-8b', 'llama-2-13b-chat-norwegian'],
                       help='Model short name')
    parser.add_argument('--val_dataset', type=str, required=True,
                       help='Path to validation dataset (JSONL)')
    parser.add_argument('--hf_token', type=str,
                       help='HuggingFace token')
    parser.add_argument('--check_interval', type=int, default=30,
                       help='How often to check for new checkpoints (seconds)')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                       help='Number of evaluations without improvement before stopping')
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
    parser.add_argument('--nli_subset_size', type=int, default=None,
                       help='Subset size for NLI evaluation (default: all examples if --include_nli_faithfulness is set, recommended: 50-100 for faster evaluation)')
    
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
        nli_subset_size=args.nli_subset_size,
    )
