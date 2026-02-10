#!/usr/bin/env python3
"""
Script to check batch_size and other training arguments stored in a checkpoint.

Usage:
    python check_checkpoint_batch_size.py <checkpoint_dir>

Example:
    python check_checkpoint_batch_size.py models/gemma-2-9b-apptainer-fsdp/checkpoint-5000
"""

import argparse
import os
import sys
from transformers import TrainingArguments

def check_checkpoint_batch_size(checkpoint_dir: str):
    """Check batch_size and other training arguments from a checkpoint."""
    
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory does not exist: {checkpoint_dir}")
        return 1
    
    training_args_file = os.path.join(checkpoint_dir, "training_args.bin")
    trainer_state_file = os.path.join(checkpoint_dir, "trainer_state.json")
    
    print("=" * 70)
    print(f"Checking checkpoint: {checkpoint_dir}")
    print("=" * 70)
    
    # Check training_args.bin
    if os.path.exists(training_args_file):
        print(f"\n✓ Found training_args.bin")
        try:
            training_args = TrainingArguments.from_pretrained(checkpoint_dir)
            
            print("\nTraining Arguments from checkpoint:")
            print("-" * 70)
            print(f"  per_device_train_batch_size: {training_args.per_device_train_batch_size}")
            print(f"  per_device_eval_batch_size: {training_args.per_device_eval_batch_size}")
            print(f"  gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
            print(f"  learning_rate: {training_args.learning_rate}")
            print(f"  num_train_epochs: {training_args.num_train_epochs}")
            print(f"  max_steps: {training_args.max_steps}")
            print(f"  warmup_steps: {training_args.warmup_steps}")
            print(f"  weight_decay: {training_args.weight_decay}")
            
            # Calculate effective batch size
            effective_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
            print(f"\n  Effective batch size (per_device * grad_accum): {effective_batch}")
            
        except Exception as e:
            print(f"⚠ Error loading training_args.bin: {e}")
    else:
        print(f"\n✗ training_args.bin not found")
    
    # Check trainer_state.json for additional info
    if os.path.exists(trainer_state_file):
        print(f"\n✓ Found trainer_state.json")
        try:
            import json
            with open(trainer_state_file, 'r') as f:
                trainer_state = json.load(f)
            
            print("\nTraining State from checkpoint:")
            print("-" * 70)
            if 'global_step' in trainer_state:
                print(f"  global_step: {trainer_state['global_step']}")
            if 'epoch' in trainer_state:
                print(f"  epoch: {trainer_state['epoch']}")
            if 'max_steps' in trainer_state:
                print(f"  max_steps: {trainer_state['max_steps']}")
            if 'log_history' in trainer_state and trainer_state['log_history']:
                last_log = trainer_state['log_history'][-1]
                if 'loss' in last_log:
                    print(f"  last_loss: {last_log['loss']:.4f}")
            
        except Exception as e:
            print(f"⚠ Error loading trainer_state.json: {e}")
    else:
        print(f"\n✗ trainer_state.json not found")
    
    print("\n" + "=" * 70)
    print("What happens if you change batch_size when resuming?")
    print("=" * 70)
    print("""
When you resume training with a different batch_size:

1. The trainer will use the NEW batch_size from your TrainingArguments
2. You will see a WARNING like:
   "Warning: The following arguments do not match the ones in the 
   trainer_state.json within the checkpoint directory: 
   per_device_train_batch_size: 4 (from args) != 16 (from trainer_state.json)"

3. Training will continue with the NEW batch_size
4. The optimizer state may be affected if batch_size changes significantly
   - Smaller batch_size: May work fine, but optimizer momentum/state was 
     calibrated for larger batches
   - Larger batch_size: May cause OOM errors or require more GPU memory

5. The effective batch size (per_device * gradient_accumulation_steps) 
   determines how many examples are processed per step
   - Changing batch_size changes the effective batch size
   - This affects learning dynamics and convergence

RECOMMENDATION:
- It's generally safe to REDUCE batch_size when resuming
- INCREASING batch_size may cause OOM errors
- For best results, use the same batch_size as the original training
- If you must change it, monitor training metrics closely
    """)
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Check batch_size and training arguments from a checkpoint'
    )
    parser.add_argument('checkpoint_dir', type=str,
                       help='Path to checkpoint directory')
    
    args = parser.parse_args()
    sys.exit(check_checkpoint_batch_size(args.checkpoint_dir))
