"""
Utility functions for managing fixed NLI faithfulness evaluation subset.

This module provides functions to create, save, and load a fixed subset of examples
for NLI faithfulness evaluation. Using a fixed subset ensures consistency across
all checkpoint evaluations and enables fair comparison.
"""

import json
import os
import random
from typing import List, Tuple, Optional


NLI_FIXED_SUBSET_SIZE = 500
NLI_FIXED_SUBSET_SEED = 42  # Fixed seed for reproducibility


def get_nli_subset_file_path(model_dir: str) -> str:
    """Get the path to the fixed NLI subset indices file.
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        Path to the subset indices JSON file
    """
    return os.path.join(model_dir, "all_eval_results", "nli_fixed_subset_indices.json")


def create_fixed_nli_subset(
    total_examples: int,
    subset_size: int = NLI_FIXED_SUBSET_SIZE,
    seed: int = NLI_FIXED_SUBSET_SEED,
    model_dir: Optional[str] = None
) -> List[int]:
    """Create a fixed subset of indices for NLI evaluation.
    
    This function creates a deterministic subset of indices that will be reused
    across all checkpoint evaluations for consistency.
    
    Args:
        total_examples: Total number of examples available
        subset_size: Number of examples to include in subset (default: 500)
        seed: Random seed for reproducibility (default: 42)
        model_dir: Optional model directory to save the subset indices
        
    Returns:
        List of indices to use for NLI evaluation
    """
    if total_examples <= subset_size:
        # If we have fewer examples than the subset size, use all
        indices = list(range(total_examples))
    else:
        # Use fixed seed for reproducibility
        random.seed(seed)
        indices = sorted(random.sample(range(total_examples), subset_size))
    
    # Save to file if model_dir is provided
    if model_dir:
        subset_file = get_nli_subset_file_path(model_dir)
        os.makedirs(os.path.dirname(subset_file), exist_ok=True)
        with open(subset_file, 'w') as f:
            json.dump({
                "subset_size": len(indices),
                "total_examples": total_examples,
                "seed": seed,
                "indices": indices
            }, f, indent=2)
        print(f"✓ Saved fixed NLI subset ({len(indices)} examples) to: {subset_file}")
    
    return indices


def load_fixed_nli_subset(model_dir: str) -> Optional[List[int]]:
    """Load the fixed NLI subset indices from file.
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        List of indices if file exists, None otherwise
    """
    subset_file = get_nli_subset_file_path(model_dir)
    if os.path.exists(subset_file):
        try:
            with open(subset_file, 'r') as f:
                data = json.load(f)
                return data.get("indices", [])
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠ Warning: Could not load NLI subset from {subset_file}: {e}")
            return None
    return None


def get_or_create_fixed_nli_subset(
    total_examples: int,
    model_dir: str,
    subset_size: int = NLI_FIXED_SUBSET_SIZE,
    seed: int = NLI_FIXED_SUBSET_SEED
) -> List[int]:
    """Get existing fixed NLI subset or create a new one.
    
    This function first tries to load an existing subset from file. If it doesn't
    exist or the total_examples has changed, it creates a new one.
    
    Args:
        total_examples: Total number of examples available
        model_dir: Path to model directory
        subset_size: Number of examples to include in subset (default: 500)
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        List of indices to use for NLI evaluation
    """
    # Try to load existing subset
    existing_indices = load_fixed_nli_subset(model_dir)
    
    if existing_indices is not None:
        # Verify the subset is still valid (all indices < total_examples)
        if all(idx < total_examples for idx in existing_indices):
            print(f"✓ Using existing fixed NLI subset ({len(existing_indices)} examples)")
            return existing_indices
        else:
            print(f"⚠ Existing NLI subset contains invalid indices (total_examples changed). Creating new subset...")
    
    # Create new subset
    return create_fixed_nli_subset(total_examples, subset_size, seed, model_dir)


def apply_fixed_subset(
    input_texts: List[str],
    prediction_texts: List[str],
    reference_texts: List[str],
    indices: List[int]
) -> Tuple[List[str], List[str], List[str]]:
    """Apply fixed subset indices to filter examples.
    
    Args:
        input_texts: List of input texts
        prediction_texts: List of prediction texts
        reference_texts: List of reference texts
        indices: List of indices to select
        
    Returns:
        Tuple of (filtered_input_texts, filtered_prediction_texts, filtered_reference_texts)
    """
    nli_input_texts = [input_texts[i] for i in indices if i < len(input_texts)]
    nli_prediction_texts = [prediction_texts[i] for i in indices if i < len(prediction_texts)]
    nli_reference_texts = [reference_texts[i] for i in indices if i < len(reference_texts)]
    return nli_input_texts, nli_prediction_texts, nli_reference_texts
