"""
Metrics computation utilities for evaluation.

This module contains functions for computing evaluation metrics including
ROUGE scores and other metrics used during model evaluation.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Any
import evaluate
from transformers import PreTrainedTokenizer
import wandb


def clean_decoded_text(text: str) -> str:
    """Clean decoded text by removing special tokens and unwanted characters.
    
    Args:
        text: Raw decoded text from tokenizer
    
    Returns:
        Cleaned text with special tokens and unwanted characters removed
    """
    # Remove common chat format tokens
    text = text.replace('[/INST]', '').replace('[INST]', '')
    text = text.replace('</s>', '').replace('<s>', '')
    # Remove backslashes (common issue with Llama-2 chat models)
    text = text.replace('\\', '')
    # Remove multiple spaces
    text = ' '.join(text.split())
    return text.strip()


def compute_rouge_metrics(
    eval_pred: Tuple[np.ndarray, np.ndarray],
    tokenizer: PreTrainedTokenizer,
    log_to_wandb: bool = False,
    step: Optional[int] = None,
    is_main_process: bool = True,
    verbose: bool = True
) -> Dict[str, float]:
    """Compute ROUGE metrics from predictions and labels.
    
    This function handles:
    - Decoding predictions and labels
    - Cleaning text (removing special tokens, normalizing)
    - Computing ROUGE scores
    - Optionally logging to WandB
    
    Args:
        eval_pred: Tuple of (predictions, labels) as numpy arrays of token IDs
        tokenizer: Tokenizer for decoding token IDs to text
        log_to_wandb: Whether to log metrics to WandB
        step: Step number for WandB logging (optional)
        is_main_process: Whether this is the main process (for distributed training)
        verbose: Whether to print debug information
    
    Returns:
        Dictionary with ROUGE metrics (rouge1, rouge2, rougeL, rougeLsum) as percentages
    """
    if is_main_process and verbose:
        print('*** evaluation: compute_metrics ***')
    
    # Load ROUGE metric (lazy loading after cache paths are set)
    rouge = evaluate.load("rouge")
    
    preds, labels = eval_pred
    if is_main_process and verbose:
        print('*** evaluation: preds ***', preds.shape)
        print('*** evaluation: labels ***', labels.shape)
    
    # Replace -100 (ignored tokens) with pad_token_id for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Fix for quantization: clip token IDs to valid vocabulary range
    # This prevents OverflowError during decoding when quantization causes out-of-range values
    vocab_size = tokenizer.vocab_size
    if is_main_process and verbose:
        print(f'*** Vocab size: {vocab_size} ***')
    
    # Clip predictions to valid token ID range [0, vocab_size)
    preds = np.clip(preds, 0, vocab_size - 1)
    labels = np.clip(labels, 0, vocab_size - 1)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Clean up decoded predictions - remove special tokens and backslashes
    decoded_preds = [clean_decoded_text(p) for p in decoded_preds]
    decoded_labels = [clean_decoded_text(l) for l in decoded_labels]
    
    # Additional strip (keep the existing strip)
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]
    
    if len(decoded_preds) > 0 and is_main_process and verbose:
        print(f'\n*** Example 1 ***')
        print(f'Prediction: {decoded_preds[0][:200]}...')
        print(f'Reference:  {decoded_labels[0][:200]}...\n')
    
    # Compute ROUGE scores
    scores = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    if is_main_process and verbose:
        print('*** evaluation: computed_metrics ***', scores)
    
    # Convert to percentages and prepare return dictionary
    result = {k: v * 100 for k, v in scores.items()}  # % values
    
    # Log to WandB if requested
    if log_to_wandb and wandb.run is not None and is_main_process:
        wandb.log({
            "eval/rouge1": result['rouge1'],
            "eval/rouge2": result['rouge2'],
            "eval/rougeL": result['rougeL'],
            "eval/rougeLsum": result['rougeLsum'],
        }, step=step)
    
    return result
