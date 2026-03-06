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
    # Many tokenizers have model_max_length set to 1e30 or similar, which is not useful
    model_max_length = getattr(tokenizer, 'model_max_length', None)
    if model_max_length is None or model_max_length > 100000:  # Some tokenizers have very large defaults
        # Try to get from tokenizer config
        if hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'model_max_length'):
            tokenizer_max = tokenizer.tokenizer.model_max_length
            if tokenizer_max is not None and tokenizer_max <= 100000:
                model_max_length = tokenizer_max
            else:
                model_max_length = None
        else:
            model_max_length = None
    
    # If still None or too large, try to infer from model config or tokenizer name
    if model_max_length is None or model_max_length > 100000:
        # Try to infer from tokenizer name or config
        tokenizer_name_lower = str(getattr(tokenizer, 'name_or_path', '')).lower()
        
        # Common model context windows:
        # - GPT-J: 2048
        # - Llama-2: 4096
        # - Llama-3: 8192
        # - Llama-3.1: 128000 (128K)
        # - Gemma: 8192
        # - Mistral: 32768
        # - Most models: 2048-8192
        
        # Check tokenizer name for hints
        # Order matters: check more specific patterns first
        if 'llama-3.1' in tokenizer_name_lower or 'llama3.1' in tokenizer_name_lower:
            model_max_length = 128000  # Llama-3.1 has 128K context
        elif 'llama-3' in tokenizer_name_lower or 'llama3' in tokenizer_name_lower or 'norskgpt-llama3' in tokenizer_name_lower:
            model_max_length = 8192  # Llama-3 has 8K context
        elif 'llama-2' in tokenizer_name_lower or 'llama2' in tokenizer_name_lower:
            model_max_length = 4096  # Llama-2 has 4K context
        elif 'gpt-j' in tokenizer_name_lower or 'gptj' in tokenizer_name_lower or 'nb-gpt-j' in tokenizer_name_lower:
            model_max_length = 2048  # GPT-J has 2K context
        elif 'mistral' in tokenizer_name_lower or 'normistral' in tokenizer_name_lower or 'viking' in tokenizer_name_lower or 'eurollm' in tokenizer_name_lower or 'norwai' in tokenizer_name_lower:
            # All Mistral-based models (Viking, Normistral, EuroLLM, NorwAI) have 32K context
            model_max_length = 32768  # Mistral has 32K context
        elif 'gemma' in tokenizer_name_lower:
            model_max_length = 8192  # Gemma has 8K context
        elif 'mt5' in tokenizer_name_lower:
            # MT5 is encoder-decoder, typically 512-1024, but we'll use 8192 as safe upper bound
            model_max_length = 8192
        else:
            # Conservative default - use 8192 for most models
            model_max_length = 8192
        
        print(f"⚠ WARNING: Tokenizer model_max_length not found or too large. Inferred {model_max_length} from model name/config.")
    
    # Calculate desired max length, but don't exceed model's limit
    # CRITICAL: Ensure we never exceed the model's actual context window
    desired_max_length = max_input_prompt_tokens + max_output_summary_tokens
    max_length = min(desired_max_length, model_max_length)
    
    # Additional safety check: if max_length is still too large, cap it
    if max_length > model_max_length:
        print(f"⚠ WARNING: Calculated max_length ({max_length}) exceeds model_max_length ({model_max_length}). Capping to {model_max_length}")
        max_length = model_max_length
    
    # Tokenize full text first with proper truncation
    # CRITICAL: Use truncation=True and max_length to prevent sequences exceeding model limits
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,  # Padding done by data collator for compatibility across tokenizer versions
        return_overflowing_tokens=False  # Don't return overflow tokens
    )
    
    # Post-processing check: Verify no sequences exceed max_length
    # This is a safety check in case truncation didn't work as expected
    input_ids_list = tokenized["input_ids"]
    for idx, input_ids in enumerate(input_ids_list):
        if len(input_ids) > max_length:
            print(f"⚠ WARNING: Example {idx} has {len(input_ids)} tokens, exceeding max_length {max_length}. Truncating manually.")
            tokenized["input_ids"][idx] = input_ids[:max_length]
    
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
    
    # Get model's maximum sequence length (same logic as training)
    model_max_length = getattr(tokenizer, 'model_max_length', None)
    if model_max_length is None or model_max_length > 100000:
        # Try to get from tokenizer config
        if hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'model_max_length'):
            tokenizer_max = tokenizer.tokenizer.model_max_length
            if tokenizer_max is not None and tokenizer_max <= 100000:
                model_max_length = tokenizer_max
            else:
                model_max_length = None
        else:
            model_max_length = None
    
    # If still None or too large, try to infer from model config or tokenizer name
    if model_max_length is None or model_max_length > 100000:
        # Try to infer from tokenizer name or config
        tokenizer_name_lower = str(getattr(tokenizer, 'name_or_path', '')).lower()
        
        # Check tokenizer name for hints (same logic as training)
        if 'llama-3.1' in tokenizer_name_lower or 'llama3.1' in tokenizer_name_lower:
            model_max_length = 128000  # Llama-3.1 has 128K context
        elif 'llama-3' in tokenizer_name_lower or 'llama3' in tokenizer_name_lower or 'norskgpt-llama3' in tokenizer_name_lower:
            model_max_length = 8192  # Llama-3 has 8K context
        elif 'llama-2' in tokenizer_name_lower or 'llama2' in tokenizer_name_lower:
            model_max_length = 4096  # Llama-2 has 4K context
        elif 'gpt-j' in tokenizer_name_lower or 'gptj' in tokenizer_name_lower or 'nb-gpt-j' in tokenizer_name_lower:
            model_max_length = 2048  # GPT-J has 2K context
        elif 'mistral' in tokenizer_name_lower or 'normistral' in tokenizer_name_lower or 'viking' in tokenizer_name_lower or 'eurollm' in tokenizer_name_lower or 'norwai' in tokenizer_name_lower:
            # All Mistral-based models (Viking, Normistral, EuroLLM, NorwAI) have 32K context
            model_max_length = 32768  # Mistral has 32K context
        elif 'gemma' in tokenizer_name_lower:
            model_max_length = 8192  # Gemma has 8K context
        elif 'mt5' in tokenizer_name_lower:
            # MT5 is encoder-decoder, typically 512-1024, but we'll use 8192 as safe upper bound
            model_max_length = 8192
        else:
            # Conservative default - use 8192 for most models
            model_max_length = 8192
        
        print(f"⚠ WARNING: Tokenizer model_max_length not found or too large. Inferred {model_max_length} from model name/config.")
    
    # Ensure we don't exceed model's context window
    max_prompt_length = min(max_input_prompt_tokens, model_max_length)
    
    # Tokenize ONLY the prompt (without answer) for evaluation
    tokenized_prompts = tokenizer(
        examples["prompt"],
        truncation=True,
        max_length=max_prompt_length,
        padding=False
    )
    
    # Safety check: Verify no sequences exceed max_length
    input_ids_list = tokenized_prompts["input_ids"]
    for idx, input_ids in enumerate(input_ids_list):
        if len(input_ids) > max_prompt_length:
            print(f"⚠ WARNING: Eval example {idx} prompt has {len(input_ids)} tokens, exceeding max_length {max_prompt_length}. Truncating manually.")
            tokenized_prompts["input_ids"][idx] = input_ids[:max_prompt_length]
    
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
