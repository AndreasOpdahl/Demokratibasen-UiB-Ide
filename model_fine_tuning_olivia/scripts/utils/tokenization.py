"""
Tokenization utilities for training and evaluation.

This module provides shared functions for tokenizing examples:
- Training examples (full text with prompt_length tracking)
- Evaluation examples (prompt + target)
"""

from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer


def _find_last_subsequence(haystack: List[int], needle: List[int]) -> Optional[int]:
    """
    Return start index of the last occurrence of `needle` in `haystack`,
    or None if not found.
    """
    if not haystack or not needle or len(needle) > len(haystack):
        return None
    for i in range(len(haystack) - len(needle), -1, -1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return None


def _compute_prompt_length_chat(input_ids: List[int], tokenizer: AutoTokenizer) -> Optional[int]:
    """
    Compute exact prompt length (tokens before assistant response) for chat templates.
    Returns None if no chat marker found (caller should use heuristic).
    """
    if not input_ids:
        return None
    # Mistral / Llama-2: ... [/INST] <space> assistant_response
    try:
        inst_id = tokenizer.convert_tokens_to_ids("[/INST]")
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if inst_id is not None and inst_id != unk_id and inst_id in input_ids:
            positions = [i for i, tid in enumerate(input_ids) if tid == inst_id]
            if positions:
                # First token of assistant is after [/INST]; often one space token follows
                last_inst = positions[-1]
                return min(last_inst + 1, len(input_ids))
    except Exception:
        pass
    # Fallback: [/INST] is often not a single token (SentencePiece split),
    # so detect it as an encoded token sequence.
    try:
        inst_marker_ids = tokenizer.encode("[/INST]", add_special_tokens=False)
        pos = _find_last_subsequence(input_ids, inst_marker_ids)
        if pos is not None:
            return min(pos + len(inst_marker_ids), len(input_ids))
    except Exception:
        pass
    # Llama-3 / 3.1: ... <|end_header_id|> \n\n assistant_response
    try:
        end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if end_header_id is not None and end_header_id != unk_id and end_header_id in input_ids:
            positions = [i for i, tid in enumerate(input_ids) if tid == end_header_id]
            if positions:
                last_end = positions[-1]
                return min(last_end + 1, len(input_ids))
    except Exception:
        pass
    try:
        end_header_ids = tokenizer.encode("<|end_header_id|>", add_special_tokens=False)
        pos = _find_last_subsequence(input_ids, end_header_ids)
        if pos is not None:
            return min(pos + len(end_header_ids), len(input_ids))
    except Exception:
        pass
    # ChatML: <|im_start|>assistant\n
    # Prefer matching the full assistant marker to avoid masking too little.
    try:
        assistant_marker_ids = tokenizer.encode("<|im_start|> assistant\n", add_special_tokens=False)
        pos = _find_last_subsequence(input_ids, assistant_marker_ids)
        if pos is not None:
            return min(pos + len(assistant_marker_ids), len(input_ids))
    except Exception:
        pass
    try:
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if im_start_id is not None and im_start_id != unk_id and im_start_id in input_ids:
            positions = [i for i, tid in enumerate(input_ids) if tid == im_start_id]
            if positions:
                # Last <|im_start|> is for assistant; content starts after \n
                last_start = positions[-1]
                return min(last_start + 1, len(input_ids))
    except Exception:
        pass
    return None


def _compute_prompt_length_plain(input_ids: List[int], tokenizer: AutoTokenizer) -> Optional[int]:
    """
    Compute exact prompt length for plain/Gemma format (ends at "Oppsummering:\\n\\n###\\n\\n").
    Returns None if marker not found (caller uses heuristic).
    """
    if not input_ids:
        return None
    try:
        marker = "Oppsummering:\n\n###\n\n"
        marker_ids = tokenizer.encode(marker, add_special_tokens=False)
        if not marker_ids:
            return None
        # Find last occurrence of marker sequence
        for i in range(len(input_ids) - len(marker_ids), -1, -1):
            if input_ids[i:i + len(marker_ids)] == marker_ids:
                return min(i + len(marker_ids), len(input_ids))
    except Exception:
        pass
    return None


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
    # OPTIMIZATION: Cap max_length at a reasonable value (e.g., 8192) even for large context models
    # This speeds up tokenization significantly while still allowing long sequences when needed
    desired_max_length = max_input_prompt_tokens + max_output_summary_tokens
    # For very large context models (128K+), cap at 8192 for tokenization speed
    # The model can still handle longer sequences, but tokenization is much faster with this cap
    effective_model_max = min(model_max_length, 8192) if model_max_length > 8192 else model_max_length
    max_length = min(desired_max_length, effective_model_max)
    
    # Additional safety check: if max_length is still too large, cap it
    if max_length > model_max_length:
        print(f"⚠ WARNING: Calculated max_length ({max_length}) exceeds model_max_length ({model_max_length}). Capping to {model_max_length}")
        max_length = model_max_length
    elif model_max_length > 8192 and max_length == 8192:
        pass  # Capped to 8192 for tokenization speed
    
    # OPTIMIZATION: Pre-truncate text at character level before tokenization
    # This is much faster than letting the tokenizer process very long texts
    # Estimate characters per token (typically 3-4 for most tokenizers)
    # Use a conservative estimate to ensure we don't truncate too aggressively
    chars_per_token_estimate = 3.5  # Conservative estimate (most tokenizers use 3-4 chars/token)
    max_chars_estimate = int(max_length * chars_per_token_estimate * 1.2)  # 20% buffer for safety
    
    # Pre-truncate texts to speed up tokenization (list comprehension is faster)
    truncated_texts = [text[:max_chars_estimate] if len(text) > max_chars_estimate else text for text in examples["text"]]
    
    # Tokenize pre-truncated text with proper truncation (as safety net)
    # CRITICAL: Use truncation=True and max_length to prevent sequences exceeding model limits
    tokenized = tokenizer(
        truncated_texts,
        truncation=True,
        max_length=max_length,
        padding=False,  # Padding done by data collator for compatibility across tokenizer versions
        return_overflowing_tokens=False  # Don't return overflow tokens
    )
    
    # Post-processing check: Verify no sequences exceed max_length
    # This is a safety check in case truncation didn't work as expected
    # OPTIMIZATION: Only check and truncate if needed (most sequences should already be truncated by tokenizer)
    input_ids_list = tokenized["input_ids"]
    sequences_exceeding_limit = 0
    
    # Fast path: Check if any sequences exceed limit (single pass)
    needs_truncation = False
    for input_ids in input_ids_list:
        if len(input_ids) > max_length:
            needs_truncation = True
            sequences_exceeding_limit += 1
    
    # Only truncate if needed (avoid creating new lists unnecessarily)
    if needs_truncation:
        for idx, input_ids in enumerate(input_ids_list):
            if len(input_ids) > max_length:
                tokenized["input_ids"][idx] = input_ids[:max_length]
        print(f"⚠ WARNING: {sequences_exceeding_limit} example(s) exceeded max_length {max_length} and were manually truncated.")
    
    # Exact prompt length: chat markers ([/INST], etc.), plain "Oppsummering:", or 80% heuristic
    prompt_lengths = []
    for input_ids in input_ids_list:
        exact = _compute_prompt_length_chat(input_ids, tokenizer)
        if exact is not None:
            prompt_lengths.append(exact)
        else:
            exact_plain = _compute_prompt_length_plain(input_ids, tokenizer)
            if exact_plain is not None:
                prompt_lengths.append(exact_plain)
            else:
                total_tokens = len(input_ids)
                prompt_lengths.append(int(total_tokens * 0.8))
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
