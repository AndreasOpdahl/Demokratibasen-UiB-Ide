"""
Utility modules for fine-tuning and evaluation scripts.

This package contains shared utilities extracted from the main scripts to reduce
code duplication and improve maintainability.
"""

from .data_collators import EvalDataCollator
from .metrics import compute_rouge_metrics, clean_decoded_text
from .checkpoint_utils import (
    extract_checkpoint_step,
    get_checkpoint_name_and_step,
    is_major_checkpoint,
    get_model_dir_from_checkpoint,
)
from .eval_results import (
    get_eval_results_path,
    get_old_eval_results_path,
    load_eval_results,
    save_eval_results,
    get_evaluated_checkpoint_steps,
    update_evaluation_summary,
)
from .dataset_loading import load_jsonl_dataset
from .tokenization import tokenize_train_examples, tokenize_eval_examples
from .formatting import format_train_example, format_eval_example

__all__ = [
    # Data collators
    'EvalDataCollator',
    # Metrics
    'compute_rouge_metrics',
    'clean_decoded_text',
    # Checkpoint utilities
    'extract_checkpoint_step',
    'get_checkpoint_name_and_step',
    'is_major_checkpoint',
    'get_model_dir_from_checkpoint',
    # Evaluation results
    'get_eval_results_path',
    'get_old_eval_results_path',
    'load_eval_results',
    'save_eval_results',
    'get_evaluated_checkpoint_steps',
    'update_evaluation_summary',
    # Dataset loading
    'load_jsonl_dataset',
    # Tokenization
    'tokenize_train_examples',
    'tokenize_eval_examples',
    # Formatting
    'format_train_example',
    'format_eval_example',
]
