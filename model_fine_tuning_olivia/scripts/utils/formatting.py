"""
Formatting utilities for training and evaluation examples.

This module provides functions to format examples for training and evaluation
using model-specific prompt configurations.
"""

from typing import Dict, Any, Optional
import sys
import os

# Add scripts directory to path for imports
_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from model_configs import get_model_config_by_hf_name, get_doc_type_norwegian


def format_train_example(example: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Format a training example using model-specific prompt configuration.
    
    Args:
        example: Dictionary with 'input', 'output', and optionally 'metadata' keys
        model_name: HuggingFace model name (e.g., 'google/gemma-2b')
    
    Returns:
        Dictionary with 'text' key containing formatted training example
    """
    # Get model configuration
    model_config = get_model_config_by_hf_name(model_name)
    
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
            doc_type=doc_type
        )
    else:
        # Fallback to plain format if model config not found
        doc_type_nor = get_doc_type_norwegian(doc_type) if doc_type else "tekst"
        formatted_text = f"Oppgave: Oppsummer følgende {doc_type_nor}:\n\n###\n\n{input_text}\n\n###\n\nOppsummering:\n\n###\n\n{output_text}\n\n###\n"
    
    return {"text": formatted_text}


def format_eval_example(example: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Format an evaluation example using model-specific prompt configuration.
    
    Args:
        example: Dictionary with 'input' and optionally 'metadata' keys
        model_name: HuggingFace model name (e.g., 'google/gemma-2b')
    
    Returns:
        Dictionary with 'prompt' and 'target_summary' keys
    """
    # Get model configuration
    model_config = get_model_config_by_hf_name(model_name)
    
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
            doc_type=doc_type
        )
    else:
        # Fallback to plain format if model config not found
        doc_type_nor = get_doc_type_norwegian(doc_type) if doc_type else "tekst"
        formatted_prompt = f"Oppgave: Oppsummer følgende {doc_type_nor}:\n\n###\n\n{input_text}\n\n###\n\nOppsummering:\n\n###\n\n"
    
    return {
        "prompt": formatted_prompt,
        "target_summary": output_text  # Reference summary for evaluation
    }
