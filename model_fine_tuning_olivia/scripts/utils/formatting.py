"""
Formatting utilities for training and evaluation examples.

This module provides functions to format examples for training and evaluation
using model-specific prompt configurations.
"""

from typing import Dict, Any, Optional, List
import sys
import os

# Add scripts directory to path for imports
_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from model_configs import get_model_config_by_hf_name, get_doc_type_norwegian

# Cache for model config to avoid 100k+ lookups during dataset.map (major speedup)
_model_config_cache: Dict[str, Any] = {}


def _get_model_config_cached(model_name: str) -> Optional[Any]:
    """Get model config with simple cache (avoids repeated dict iteration in format_train)."""
    if model_name not in _model_config_cache:
        _model_config_cache[model_name] = get_model_config_by_hf_name(model_name)
    return _model_config_cache[model_name]


def format_train_example(example: Dict[str, Any], model_name: str, tokenizer: Optional[Any] = None) -> Dict[str, Any]:
    """Format a training example using model-specific prompt configuration.
    
    Args:
        example: Dictionary with 'input', 'output', and optionally 'metadata' keys
        model_name: HuggingFace model name (e.g., 'google/gemma-2b')
        tokenizer: Optional tokenizer instance (used for chat template formatting)
    
    Returns:
        Dictionary with 'text' key containing formatted training example
    """
    # Get model configuration (cached to avoid 135k+ lookups during map)
    model_config = _get_model_config_cached(model_name)
    
    # Extract input and output
    input_text = example.get('input', '')
    output_text = example.get('output', '')
    
    # Extract doc_type from metadata if available
    doc_type = None
    if 'metadata' in example and isinstance(example.get('metadata'), dict):
        doc_type = example['metadata'].get('doc_type')
    
    # Use model's prompt config to format
    if model_config:
        formatted_text = model_config.prompt_config.format_train(
            input_text=input_text,
            output_text=output_text,
            doc_type=doc_type,
            tokenizer=tokenizer
        )
    else:
        # Fallback to plain format if model config not found
        doc_type_nor = get_doc_type_norwegian(doc_type) if doc_type else "tekst"
        formatted_text = f"Oppgave: Oppsummer følgende {doc_type_nor}:\n\n###\n\n{input_text}\n\n###\n\nOppsummering:\n\n###\n\n{output_text}\n\n###\n"
    
    return {"text": formatted_text}


def format_train_examples_batch(
    examples: Dict[str, List[Any]],
    model_name: str,
    tokenizer: Optional[Any] = None,
    model_config: Optional[Any] = None,
    *,
    use_fast_format: bool = True,
) -> Dict[str, List[str]]:
    """Format a batch of training examples (for fast batched map with num_proc).
    
    Args:
        examples: Dict with "input", "output", and optionally "metadata" (list per key)
        model_name: HuggingFace model name
        tokenizer: Optional tokenizer (for chat templates; ignored if use_fast_format=True)
        model_config: Optional pre-fetched model config (avoids lookup per batch)
        use_fast_format: If True (default), use manual template (train_template) instead of
            tokenizer.apply_chat_template(). Much faster (~100x) for 135k examples; output
            matches the same chat format (Mistral/Llama/ChatML) used at inference.
    
    Returns:
        Dict with "text" key containing list of formatted strings
    """
    config = model_config or _get_model_config_cached(model_name)
    inputs = examples.get("input", [])
    outputs = examples.get("output", [])
    metadatas = examples.get("metadata", [None] * len(inputs))
    if len(metadatas) < len(inputs):
        metadatas = metadatas + [None] * (len(inputs) - len(metadatas))
    # Fast path: skip apply_chat_template (saves ~2 hours for 135k examples)
    skip_tokenizer = use_fast_format
    texts = []
    for i in range(len(inputs)):
        input_text = inputs[i] if i < len(inputs) else ""
        output_text = outputs[i] if i < len(outputs) else ""
        meta = metadatas[i] if i < len(metadatas) else None
        doc_type = None
        if isinstance(meta, dict):
            doc_type = meta.get("doc_type")
        if config:
            formatted_text = config.prompt_config.format_train(
                input_text=input_text,
                output_text=output_text,
                doc_type=doc_type,
                tokenizer=None if skip_tokenizer else tokenizer,
            )
        else:
            doc_type_nor = get_doc_type_norwegian(doc_type) if doc_type else "tekst"
            formatted_text = f"Oppgave: Oppsummer følgende {doc_type_nor}:\n\n###\n\n{input_text}\n\n###\n\nOppsummering:\n\n###\n\n{output_text}\n\n###\n"
        texts.append(formatted_text)
    return {"text": texts}


def format_eval_example(example: Dict[str, Any], model_name: str, tokenizer: Optional[Any] = None) -> Dict[str, Any]:
    """Format an evaluation example using model-specific prompt configuration.
    
    Args:
        example: Dictionary with 'input' and optionally 'metadata' keys
        model_name: HuggingFace model name (e.g., 'google/gemma-2b')
        tokenizer: Optional tokenizer instance (used for chat template formatting)
    
    Returns:
        Dictionary with 'prompt' and 'target_summary' keys
    """
    # Get model configuration (cached)
    model_config = _get_model_config_cached(model_name)
    
    # Extract input and output (output is the target/reference summary)
    input_text = example.get('input', '')
    output_text = example.get('output', '')
    
    # Extract doc_type from metadata if available
    doc_type = None
    if 'metadata' in example and isinstance(example.get('metadata'), dict):
        doc_type = example['metadata'].get('doc_type')
    
    # Use model's prompt config to format
    if model_config:
        formatted_prompt = model_config.prompt_config.format_eval(
            input_text=input_text,
            doc_type=doc_type,
            tokenizer=tokenizer
        )
    else:
        # Fallback to plain format if model config not found
        doc_type_nor = get_doc_type_norwegian(doc_type) if doc_type else "tekst"
        formatted_prompt = f"Oppgave: Oppsummer følgende {doc_type_nor}:\n\n###\n\n{input_text}\n\n###\n\nOppsummering:\n\n###\n\n"
    
    return {
        "prompt": formatted_prompt,
        "target_summary": output_text  # Reference summary for evaluation
    }
