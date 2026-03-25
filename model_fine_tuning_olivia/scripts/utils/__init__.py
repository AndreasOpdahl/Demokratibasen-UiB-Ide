"""
Utility modules for fine-tuning and evaluation scripts.

This package contains shared utilities extracted from the main scripts to reduce
code duplication and improve maintainability.
"""

from .data_collators import EvalDataCollator
from .metrics import (
    compute_rouge_metrics,
    clean_decoded_text,
    compute_rouge,
    compute_bertscore,
    eval_hygiene,
    hygiene,
    ngram_repetition,
    extended_evaluate,
    compute_metrics_from_texts,
)
from .checkpoint_utils import (
    extract_checkpoint_step,
    get_checkpoint_name_and_step,
    is_major_checkpoint,
    get_model_dir_from_checkpoint,
)
from .eval_results import (
    get_eval_results_path,
    get_predictions_file_path,
    get_old_eval_results_path,
    load_eval_results,
    save_eval_results,
    get_evaluated_checkpoint_steps,
    update_evaluation_summary,
)
from .dataset_loading import load_jsonl_dataset
from .tokenization import tokenize_train_examples, tokenize_eval_examples
from .formatting import format_train_example, format_train_examples_batch, format_eval_example
from .nli_subset import (
    get_or_create_fixed_nli_subset,
    apply_fixed_subset,
    NLI_DEFAULT_SUBSET_SIZE,
    NLI_FIXED_SUBSET_SEED,
    NLI_FIXED_SUBSET_SIZE,
)
from .rouge_tokenizer import norwegian_tokenize, get_backend_name as get_rouge_tokenizer_backend

__all__ = [
    # Data collators
    'EvalDataCollator',
    # Metrics — ROUGE, hygiene, BERTScore, orchestrator
    'compute_rouge_metrics',
    'clean_decoded_text',
    'compute_rouge',
    'compute_bertscore',
    'eval_hygiene',
    'hygiene',
    'ngram_repetition',
    'extended_evaluate',
    'compute_metrics_from_texts',
    # Checkpoint utilities
    'extract_checkpoint_step',
    'get_checkpoint_name_and_step',
    'is_major_checkpoint',
    'get_model_dir_from_checkpoint',
    # Evaluation results
    'get_eval_results_path',
    'get_predictions_file_path',
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
    'format_train_examples_batch',
    'format_eval_example',
    # NLI subset
    'get_or_create_fixed_nli_subset',
    'apply_fixed_subset',
    'NLI_DEFAULT_SUBSET_SIZE',
    'NLI_FIXED_SUBSET_SEED',
    'NLI_FIXED_SUBSET_SIZE',
    # ROUGE tokenizer
    'norwegian_tokenize',
    'get_rouge_tokenizer_backend',
]
