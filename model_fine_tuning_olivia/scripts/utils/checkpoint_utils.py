"""
Checkpoint path handling utilities.

This module provides functions for extracting checkpoint step numbers,
determining checkpoint types (major vs normal), and handling checkpoint paths
consistently across the codebase.
"""

import os
from typing import Optional, Tuple


def extract_checkpoint_step(checkpoint_path: str) -> int:
    """Extract step number from checkpoint path.
    
    Handles multiple checkpoint naming formats:
    - checkpoint-123
    - regular-checkpoint-123
    - major-checkpoint-123
    
    Args:
        checkpoint_path: Path to checkpoint directory or checkpoint name
    
    Returns:
        Checkpoint step number, or -1 if extraction fails
    """
    try:
        basename = os.path.basename(checkpoint_path.rstrip('/'))
        # Handle both "checkpoint-123" and "regular-checkpoint-123" / "major-checkpoint-123"
        if basename.startswith("checkpoint-"):
            return int(basename.split("-")[-1])
        elif basename.startswith("regular-checkpoint-") or basename.startswith("major-checkpoint-"):
            return int(basename.split("-")[-1])
    except (ValueError, IndexError):
        pass
    return -1


def get_checkpoint_name_and_step(checkpoint_path: str) -> Tuple[str, int]:
    """Extract checkpoint name and step number from path.
    
    Args:
        checkpoint_path: Path to checkpoint directory
    
    Returns:
        Tuple of (checkpoint_name, step_number)
        checkpoint_name: Standard format "checkpoint-{step}"
        step_number: Integer step number, or 0 if extraction fails
    """
    checkpoint_name = os.path.basename(checkpoint_path.rstrip('/'))
    checkpoint_step = checkpoint_name.replace('checkpoint-', '').replace('regular-checkpoint-', '').replace('major-checkpoint-', '')
    
    try:
        checkpoint_step_int = int(checkpoint_step)
    except ValueError:
        checkpoint_step_int = 0
    
    # Normalize checkpoint name to standard format
    if checkpoint_step_int > 0:
        normalized_name = f"checkpoint-{checkpoint_step_int}"
    else:
        normalized_name = checkpoint_name
    
    return normalized_name, checkpoint_step_int


def is_major_checkpoint(checkpoint_step: int, major_checkpoint_interval: int = 500) -> bool:
    """Check if a checkpoint is a major checkpoint.
    
    Major checkpoints are typically evaluated with additional metrics like BERTScore.
    
    Args:
        checkpoint_step: Checkpoint step number
        major_checkpoint_interval: Every Nth step is considered major (default: 500)
    
    Returns:
        True if checkpoint is a major checkpoint, False otherwise
    """
    return checkpoint_step > 0 and checkpoint_step % major_checkpoint_interval == 0


def get_model_dir_from_checkpoint(checkpoint_dir: str) -> str:
    """Get model directory from checkpoint directory path.
    
    The model directory is the parent directory of the checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
    
    Returns:
        Path to model directory (parent of checkpoint_dir)
    """
    return os.path.dirname(checkpoint_dir.rstrip('/'))
