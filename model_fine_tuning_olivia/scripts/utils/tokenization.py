"""
Tokenization utilities for training and evaluation.

This module provides shared functions for tokenizing examples:
- Training examples (full text with prompt_length tracking)
- Evaluation examples (prompt + target)
"""

from typing import Dict, List, Any
from transformers import AutoTokenizer


def tokenize_train_examples(
    examples: Dict[str, List[str]],
    tokenizer: AutoTokenizer,
    max_input_text_tokens: int,
    max_extra_prompt_tokens: int,
    max_output_summary_tokens: int
) -> Dict[str, Any]:
    """
    Tokenize training examples (full text with prompt_length tracking).
    
    This function tokenizes the full formatted text (prompt + output) and
    tracks the prompt length for later masking in the data collator.
    
    Args:
        examples: Dictionary with "text" key containing formatted training examples
        tokenizer: Tokenizer instance
        max_input_text_tokens: Maximum tokens for input text
        max_extra_prompt_tokens: Maximum extra tokens for prompt
        max_output_summary_tokens: Maximum tokens for output summary
    
    Returns:
        Dictionary with tokenized data including "prompt_length" for each example
    """
    max_input_prompt_tokens = max_input_text_tokens + max_extra_prompt_tokens
    
    # Get model's maximum sequence length (respect model limits)
    model_max_length = getattr(tokenizer, 'model_max_length', None)
    if model_max_length is None or model_max_length > 100000:  # Some tokenizers have very large defaults
        # Try to get from tokenizer config
        if hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'model_max_length'):
            model_max_length = tokenizer.tokenizer.model_max_length
        else:
            # Default fallback - use a reasonable limit
            model_max_length = 2048
    
    # Calculate desired max length, but don't exceed model's limit
    desired_max_length = max_input_prompt_tokens + max_output_summary_tokens
    max_length = min(desired_max_length, model_max_length)
    
    # Tokenize full text first with proper truncation
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,  # Padding done by data collator for compatibility across tokenizer versions
        return_overflowing_tokens=False  # Don't return overflow tokens
    )
    
    # Find where summary starts by looking for summary markers in the text
    # Store the prompt length for later masking in the collator
    prompt_lengths = []
    summary_markers = [
        "Oppsummering:\n\n###\n\n",
        "Oppsummering:",
        "[/INST]",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
    ]
    
    # Get input_ids list (already tokenized)
    input_ids_list = tokenized["input_ids"]
    
    for idx, text in enumerate(examples["text"]):
        prompt_len = None
        for marker in summary_markers:
            if marker in text:
                # Tokenize up to the marker to find position
                prompt_part = text.split(marker)[0] + marker
                prompt_tokens = tokenizer.encode(prompt_part, add_special_tokens=False)
                prompt_len = len(prompt_tokens)
                break
        # If no marker found, assume first 80% is prompt (fallback)
        if prompt_len is None:
            # Use the actual tokenized input_ids length for this example
            total_tokens = len(input_ids_list[idx])
            prompt_len = int(total_tokens * 0.8)
        prompt_lengths.append(prompt_len)
    
    # Store prompt length for use in collator (as a list, one per example)
    tokenized["prompt_length"] = prompt_lengths
    
    return tokenized


def tokenize_eval_examples(
    examples: Dict[str, List[str]],
    tokenizer: AutoTokenizer,
    max_input_text_tokens: int,
    max_extra_prompt_tokens: int,
    max_output_summary_tokens: int
) -> Dict[str, Any]:
    """
    Tokenize evaluation examples (prompt + target).
    
    This function tokenizes the prompt and target separately for evaluation.
    The prompt is used for generation, and the target is stored as labels for ROUGE.
    
    Args:
        examples: Dictionary with "prompt" and "target_summary" keys
        tokenizer: Tokenizer instance
        max_input_text_tokens: Maximum tokens for input text
        max_extra_prompt_tokens: Maximum extra tokens for prompt
        max_output_summary_tokens: Maximum tokens for output summary
    
    Returns:
        Dictionary with tokenized prompts and labels (target summaries)
    """
    max_input_prompt_tokens = max_input_text_tokens + max_extra_prompt_tokens
    
    # Tokenize ONLY the prompt (without answer) for evaluation
    tokenized_prompts = tokenizer(
        examples["prompt"],
        truncation=True,
        max_length=max_input_prompt_tokens,
        padding=False
    )
    
    # Tokenize target summaries for labels
    tokenized_targets = tokenizer(
        examples["target_summary"],
        truncation=True,
        max_length=max_output_summary_tokens,
        padding=False
    )
    
    # Store target token IDs as labels
    tokenized_prompts["labels"] = tokenized_targets["input_ids"]
    
    return tokenized_prompts
