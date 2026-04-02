"""
Evaluation results file handling utilities.

This module provides functions for loading, saving, and managing evaluation
results files. It handles both the new centralized location (all_eval_results/)
and the old per-checkpoint location for backwards compatibility.
"""

import json
import os
import glob
from typing import Any, Dict, List, Optional, Set
from datetime import datetime


def _get_model_dir_and_checkpoint_name(checkpoint_dir: str, model_dir: Optional[str] = None) -> tuple:
    """Resolve model dir and normalized checkpoint name (handles backup paths)."""
    if model_dir is None:
        try:
            from .checkpoint_utils import get_model_dir_from_checkpoint
            model_dir = get_model_dir_from_checkpoint(checkpoint_dir)
        except Exception:
            model_dir = os.path.dirname(checkpoint_dir.rstrip('/'))
    checkpoint_name = os.path.basename(checkpoint_dir.rstrip('/'))
    if 'regular-checkpoint-' in checkpoint_name:
        step = checkpoint_name.replace('regular-checkpoint-', '')
        checkpoint_name = f"checkpoint-{step}"
    elif 'major-checkpoint-' in checkpoint_name:
        step = checkpoint_name.replace('major-checkpoint-', '')
        checkpoint_name = f"checkpoint-{step}"
    return model_dir, checkpoint_name


def _normalize_examples_suffix(examples_suffix: Optional[str]) -> Optional[str]:
    """Normalize examples suffix to the canonical '<N>-examples' format."""
    if not examples_suffix:
        return None
    # Backward compatibility: examples_1000 -> 1000-examples
    if examples_suffix.startswith("examples_"):
        count = examples_suffix.replace("examples_", "", 1)
        if count.isdigit():
            return f"{count}-examples"
    return examples_suffix


def _legacy_examples_suffixes(examples_suffix: Optional[str]) -> List[str]:
    """Return legacy suffix variants that should be checked for reads."""
    if not examples_suffix:
        # Legacy 500-example files were often unsuffixed.
        return []
    suffix = _normalize_examples_suffix(examples_suffix)
    legacy: List[str] = []
    if suffix and suffix.endswith("-examples"):
        count = suffix.replace("-examples", "")
        if count.isdigit():
            legacy.append(f"examples_{count}")
            if count == "500":
                # Legacy default 500 often had no suffix.
                legacy.append("")
    return legacy


def get_eval_results_path(
    checkpoint_dir: str,
    model_dir: Optional[str] = None,
    examples_suffix: Optional[str] = None
) -> str:
    """Get evaluation results file path (new centralized location).
    
    The new location is: model_dir/all_eval_results/checkpoint-nnn-eval-results-<N>-examples.json
    
    Args:
        checkpoint_dir: Path to checkpoint directory (may be in regular_checkpoints/ or major_checkpoints/)
        model_dir: Model directory (parent of all checkpoints). If None, will be derived.
        examples_suffix: Optional suffix for val_data_size variants (canonical: "1000-examples").
            Legacy "examples_1000" is accepted and normalized.
    
    Returns:
        Path to evaluation results JSON file
    """
    model_dir, checkpoint_name = _get_model_dir_and_checkpoint_name(checkpoint_dir, model_dir)
    all_eval_results_dir = os.path.join(model_dir, "all_eval_results")
    base = f"{checkpoint_name}-eval-results"
    suffix = _normalize_examples_suffix(examples_suffix)
    suffix = f"-{suffix}" if suffix else ""
    return os.path.join(all_eval_results_dir, f"{base}{suffix}.json")


def get_predictions_file_path(
    checkpoint_dir: str,
    model_dir: Optional[str] = None,
    examples_suffix: Optional[str] = None
) -> str:
    """Get inputs-refs-preds JSONL file path (in all_eval_results).
    
    The location is: model_dir/all_eval_results/checkpoint-nnn-inputs-refs-preds-<N>-examples.jsonl
    
    Args:
        checkpoint_dir: Path to checkpoint directory (may be in regular_checkpoints/ or major_checkpoints/)
        model_dir: Model directory (parent of all checkpoints). If None, will be derived.
        examples_suffix: Optional suffix for val_data_size variants (canonical: "1000-examples").
            Legacy "examples_1000" is accepted and normalized.
    
    Returns:
        Path to the JSONL file
    """
    model_dir, checkpoint_name = _get_model_dir_and_checkpoint_name(checkpoint_dir, model_dir)
    all_eval_results_dir = os.path.join(model_dir, "all_eval_results")
    base = f"{checkpoint_name}-inputs-refs-preds"
    suffix = _normalize_examples_suffix(examples_suffix)
    suffix = f"-{suffix}" if suffix else ""
    return os.path.join(all_eval_results_dir, f"{base}{suffix}.jsonl")


def get_faithfulness_details_path(
    checkpoint_dir: str,
    model_dir: Optional[str] = None,
    examples_suffix: Optional[str] = None
) -> str:
    """Get per-example NLI faithfulness details JSONL path (in all_eval_results).

    The location is: model_dir/all_eval_results/checkpoint-nnn-faithfulness-details-<N>-examples.jsonl

    Each line in the JSONL contains the full score_and_gate output for one
    example, keyed by ``example_index``.  This file enables incremental
    faithfulness evaluation: when the NLI subset size grows, only the new
    examples need to be analysed.
    """
    model_dir, checkpoint_name = _get_model_dir_and_checkpoint_name(checkpoint_dir, model_dir)
    all_eval_results_dir = os.path.join(model_dir, "all_eval_results")
    base = f"{checkpoint_name}-faithfulness-details"
    suffix = _normalize_examples_suffix(examples_suffix)
    suffix = f"-{suffix}" if suffix else ""
    return os.path.join(all_eval_results_dir, f"{base}{suffix}.jsonl")


def nli_faithfulness_aggregate_present(existing_results: Dict[str, Any]) -> bool:
    """True if eval JSON already has non-null NLI faithfulness aggregates (legacy layouts included)."""
    return (
        any(key.startswith("eval_faithfulness_") for key in existing_results.keys())
        or existing_results.get("eval_faithfulness") is not None
    )


def should_skip_faithfulness_update(
    existing_results: Dict[str, Any],
    checkpoint_dir: str,
    model_dir: Optional[str] = None,
    examples_suffix: Optional[str] = None,
) -> bool:
    """Skip --update-faithfulness only when aggregates exist and the details JSONL is on disk.

    Older runs may have written eval_faithfulness without a *-faithfulness-details-*.jsonl file.
    Those aggregates remain valid, but we re-run NLI when the details file is absent so
    per-example caches exist for incremental subset expansion.
    """
    if not nli_faithfulness_aggregate_present(existing_results):
        return False
    details_path = get_faithfulness_details_path(
        checkpoint_dir, model_dir, examples_suffix=examples_suffix
    )
    return os.path.isfile(details_path)


def get_old_eval_results_path(checkpoint_dir: str) -> str:
    """Get old evaluation results file path (per-checkpoint location).
    
    The old location is: checkpoint_dir/eval_results/eval_results.json
    
    Args:
        checkpoint_dir: Path to checkpoint directory
    
    Returns:
        Path to old evaluation results JSON file
    """
    return os.path.join(checkpoint_dir, 'eval_results', 'eval_results.json')


def load_eval_results(
    checkpoint_dir: str,
    model_dir: Optional[str] = None,
    examples_suffix: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Load evaluation results from new or old location.
    
    Checks both locations for backwards compatibility:
    1. New location: model_dir/all_eval_results/checkpoint-nnn-eval-results-<N>-examples.json
       (also checks legacy suffix variants when present)
    2. Old location: checkpoint_dir/eval_results/eval_results.json
    
    Args:
        checkpoint_dir: Path to checkpoint directory (may be in regular_checkpoints/ or major_checkpoints/)
        model_dir: Model directory (parent of all checkpoints). If None, will be derived.
        examples_suffix: Optional suffix for val_data_size variants (canonical: "1000-examples").
    
    Returns:
        Evaluation results dictionary, or None if not found
    """
    examples_suffix = _normalize_examples_suffix(examples_suffix)

    # Try new location first (get_eval_results_path resolves model_dir when None)
    new_results_file = get_eval_results_path(checkpoint_dir, model_dir, examples_suffix=examples_suffix)
    if os.path.exists(new_results_file):
        try:
            with open(new_results_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {new_results_file}: {e}")

    # Try known legacy suffixed paths if canonical path is missing
    for legacy_suffix in _legacy_examples_suffixes(examples_suffix):
        legacy_results_file = get_eval_results_path(
            checkpoint_dir, model_dir, examples_suffix=legacy_suffix if legacy_suffix else None
        )
        if os.path.exists(legacy_results_file):
            try:
                with open(legacy_results_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading {legacy_results_file}: {e}")
    
    # Try old location for backwards compatibility
    # When examples_suffix is set, the old file is a legacy 500-example file - do NOT use it.
    if examples_suffix:
        return None
    old_results_file = get_old_eval_results_path(checkpoint_dir)
    if os.path.exists(old_results_file):
        try:
            with open(old_results_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {old_results_file}: {e}")
    
    return None


def save_eval_results(
    results: Dict[str, Any],
    checkpoint_dir: str,
    model_dir: Optional[str] = None,
    save_to_old_location: bool = True,
    examples_suffix: Optional[str] = None
) -> str:
    """Save evaluation results to new location (and optionally old location).
    
    Args:
        results: Evaluation results dictionary
        checkpoint_dir: Path to checkpoint directory (may be in regular_checkpoints/ or major_checkpoints/)
        model_dir: Model directory (parent of all checkpoints). If None, will be derived.
        save_to_old_location: Whether to also save to old location for backwards compatibility.
            Ignored when examples_suffix is set (suffixed variants are only saved to new location).
        examples_suffix: Optional suffix for val_data_size variants (canonical: "1000-examples").
            When set, results go to checkpoint-N-eval-results-<N>-examples.json and old location is skipped.
    
    Returns:
        Path to the new results file
    """
    if model_dir is None:
        # Use shared helper so backup paths (regular_checkpoints/, major_checkpoints/) resolve to main model dir
        try:
            from .checkpoint_utils import get_model_dir_from_checkpoint
            model_dir = get_model_dir_from_checkpoint(checkpoint_dir)
        except Exception:
            model_dir = os.path.dirname(checkpoint_dir.rstrip('/'))
    
    examples_suffix = _normalize_examples_suffix(examples_suffix)

    # Save to new location (primary): model_dir/all_eval_results/checkpoint-N-eval-results-<N>-examples.json
    new_results_file = get_eval_results_path(checkpoint_dir, model_dir, examples_suffix=examples_suffix)
    os.makedirs(os.path.dirname(new_results_file), exist_ok=True)
    
    with open(new_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save to old location for backwards compatibility (if requested and not a suffixed variant)
    if save_to_old_location and not examples_suffix:
        old_results_file = get_old_eval_results_path(checkpoint_dir)
        os.makedirs(os.path.dirname(old_results_file), exist_ok=True)
        with open(old_results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return new_results_file


def get_evaluated_checkpoint_steps(model_dir: str) -> Set[int]:
    """Get set of already evaluated checkpoint steps.
    
    Checks both new and old locations for backwards compatibility.
    
    Args:
        model_dir: Path to model directory
    
    Returns:
        Set of evaluated checkpoint step numbers
    """
    evaluated = set()
    if not os.path.exists(model_dir):
        return evaluated
    
    # Check new location:
    # - checkpoint-*-eval-results-<N>-examples.json (canonical)
    # - checkpoint-*-eval-results.json (legacy)
    all_eval_results_dir = os.path.join(model_dir, "all_eval_results")
    if os.path.exists(all_eval_results_dir):
        for eval_file in glob.glob(os.path.join(all_eval_results_dir, "checkpoint-*-eval-results*.json")):
            try:
                # Extract step from filename prefixes like:
                # checkpoint-123-eval-results.json
                # checkpoint-123-eval-results-500-examples.json
                filename = os.path.basename(eval_file)
                prefix = "checkpoint-"
                marker = "-eval-results"
                if not filename.startswith(prefix) or marker not in filename:
                    continue
                step = int(filename[len(prefix):].split(marker, 1)[0])
                evaluated.add(step)
            except (ValueError, IndexError):
                pass
    
    # Also check old location for backwards compatibility
    for ckpt_dir in glob.glob(os.path.join(model_dir, "checkpoint-*")):
        old_eval_results_file = get_old_eval_results_path(ckpt_dir)
        if os.path.exists(old_eval_results_file):
            try:
                step = int(os.path.basename(ckpt_dir).split("-")[-1])
                evaluated.add(step)
            except (ValueError, IndexError):
                pass
    
    return evaluated


def update_evaluation_summary(
    results: Dict[str, Any],
    checkpoint_dir: str,
    model_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    val_dataset_path: Optional[str] = None,
    examples_suffix: Optional[str] = None
) -> str:
    """Update evaluation_summary.json file.
    
    Args:
        results: Evaluation results dictionary
        checkpoint_dir: Path to checkpoint directory
        model_dir: Model directory (parent of checkpoint_dir). If None, will be derived.
        model_name: Model name for summary (optional)
        val_dataset_path: Validation dataset path for summary (optional)
    
    Returns:
        Path to the summary file
    """
    if model_dir is None:
        model_dir = os.path.dirname(checkpoint_dir.rstrip('/'))
    
    from .checkpoint_utils import get_checkpoint_name_and_step
    checkpoint_name, checkpoint_step = get_checkpoint_name_and_step(checkpoint_dir)
    
    summary_file = os.path.join(model_dir, "all_eval_results", "evaluation_summary.json")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    
    # Load existing summary or create new one
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
    else:
        # Initialize new summary
        summary = {
            "model": model_name or "unknown",
            "checkpoint_base_dir": model_dir,
            "val_dataset": val_dataset_path or "unknown",
            "num_gpus": 0,  # Will be updated from results if available
            "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "checkpoints": []
        }
    
    # Create checkpoint entry
    checkpoint_entry = {
        "checkpoint": checkpoint_name,
        "checkpoint_number": checkpoint_step,
        "status": "success",
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "result_file": get_eval_results_path(checkpoint_dir, model_dir, examples_suffix=examples_suffix),
    }
    
    # Add all metrics from results
    for key, value in results.items():
        normalized_key = key.replace("eval_", "").lower()
        if isinstance(value, (int, float)):
            checkpoint_entry[normalized_key] = float(value)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)):
                    checkpoint_entry[f"{normalized_key}_{sub_key}"] = float(sub_value)
    
    # Remove existing entry for this checkpoint if it exists, then add new one
    summary["checkpoints"] = [c for c in summary["checkpoints"] if c.get("checkpoint") != checkpoint_name]
    summary["checkpoints"].append(checkpoint_entry)
    
    # Sort by checkpoint number
    summary["checkpoints"].sort(key=lambda x: x.get("checkpoint_number", 0))
    
    # Update statistics (if helper function exists)
    _update_summary_statistics(summary)
    
    # Save updated summary
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary_file


def _update_summary_statistics(summary: Dict[str, Any]) -> None:
    """Update statistics in evaluation summary.
    
    Args:
        summary: Summary dictionary to update in-place
    """
    checkpoints = summary.get("checkpoints", [])
    if not checkpoints:
        summary["statistics"] = {
            "total_checkpoints": 0,
            "successful": 0,
            "failed": 0,
        }
        return
    
    successful = [c for c in checkpoints if c.get("status") == "success"]
    failed = [c for c in checkpoints if c.get("status") == "failed"]
    
    # Find best checkpoint based on ROUGE-Lsum
    best_checkpoint = None
    best_rouge_lsum = None
    for ckpt in successful:
        rouge_lsum = ckpt.get("rougelsum") or ckpt.get("rouge_lsum") or ckpt.get("eval_rougeLsum", 0)
        if best_rouge_lsum is None or rouge_lsum > best_rouge_lsum:
            best_rouge_lsum = rouge_lsum
            best_checkpoint = ckpt.get("checkpoint")
    
    summary["statistics"] = {
        "total_checkpoints": len(checkpoints),
        "successful": len(successful),
        "failed": len(failed),
    }
    
    if best_checkpoint:
        summary["best_checkpoint"] = best_checkpoint
        summary["best_rouge_lsum"] = best_rouge_lsum
