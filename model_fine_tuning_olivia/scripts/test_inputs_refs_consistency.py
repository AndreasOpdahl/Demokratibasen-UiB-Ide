"""
Test that inputs and references are identical across checkpoint prediction files.

Verifies the reproducible sampling guarantee: all checkpoint evaluations use
the same validation subset (seed=42), so input_text and reference fields must
match across JSONL files. Only predictions should differ.

Usage:
    # Compare two specific checkpoints (default: 4900 vs 5000):
    python test_inputs_refs_consistency.py

    # Compare specific checkpoints:
    python test_inputs_refs_consistency.py --checkpoints 1000 2000 3000

    # Custom results directory:
    python test_inputs_refs_consistency.py --results_dir /path/to/all_eval_results

    # Use the 1000-example gen0 variant:
    python test_inputs_refs_consistency.py --examples_suffix 1000-examples --gen 0

    # Only check input_text (skip reference check):
    python test_inputs_refs_consistency.py --inputs_only
"""

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path


DEFAULT_RESULTS_DIR = os.path.expanduser(
    "~/OneDrive/Shared/Demokratibasen-UiB-Ide/TrainingRuns/olivia/"
    "ajay_finetunes/normistral-7b-instruct-apptainer-fsdp/all_eval_results"
)


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_examples_suffix(examples_suffix: str) -> str:
    if examples_suffix.startswith("examples_"):
        count = examples_suffix.replace("examples_", "", 1)
        if count.isdigit():
            return f"{count}-examples"
    return examples_suffix


def find_predictions_file(results_dir: str, checkpoint_num: int, examples_suffix: str, generation_num: int) -> str | None:
    suffix = normalize_examples_suffix(examples_suffix)
    candidates = [
        f"checkpoint-{checkpoint_num}-gen{generation_num}-inputs-refs-preds-{suffix}.jsonl",
        f"major-checkpoint-{checkpoint_num}-gen{generation_num}-inputs-refs-preds-{suffix}.jsonl",
        f"regular-checkpoint-{checkpoint_num}-gen{generation_num}-inputs-refs-preds-{suffix}.jsonl",
    ]
    for filename in candidates:
        path = os.path.join(results_dir, filename)
        if os.path.isfile(path):
            return path
    return None


def compare_pair(
    results_dir: str,
    ckpt_a: int,
    ckpt_b: int,
    examples_suffix: str,
    generation_num: int,
    inputs_only: bool,
) -> tuple[bool, list[str]]:
    """Compare two checkpoint files. Returns (passed, list_of_issues)."""
    issues = []

    file_a = find_predictions_file(results_dir, ckpt_a, examples_suffix, generation_num)
    file_b = find_predictions_file(results_dir, ckpt_b, examples_suffix, generation_num)

    if file_a is None:
        issues.append(f"File not found for checkpoint {ckpt_a}")
        return False, issues
    if file_b is None:
        issues.append(f"File not found for checkpoint {ckpt_b}")
        return False, issues

    data_a = load_jsonl(file_a)
    data_b = load_jsonl(file_b)

    if len(data_a) != len(data_b):
        issues.append(
            f"Row count mismatch: checkpoint {ckpt_a} has {len(data_a)} rows, "
            f"checkpoint {ckpt_b} has {len(data_b)} rows"
        )
        return False, issues

    fields_to_check = ["input_text"] if inputs_only else ["input_text", "reference"]
    mismatches = {field: 0 for field in fields_to_check}
    first_mismatch_examples = {}

    for i, (row_a, row_b) in enumerate(zip(data_a, data_b)):
        for field in fields_to_check:
            val_a = row_a.get(field, "")
            val_b = row_b.get(field, "")
            if val_a != val_b:
                mismatches[field] += 1
                if field not in first_mismatch_examples:
                    first_mismatch_examples[field] = (
                        i,
                        val_a[:120] + ("..." if len(val_a) > 120 else ""),
                        val_b[:120] + ("..." if len(val_b) > 120 else ""),
                    )

    passed = True
    for field in fields_to_check:
        if mismatches[field] > 0:
            passed = False
            row_idx, snip_a, snip_b = first_mismatch_examples[field]
            issues.append(
                f"{mismatches[field]}/{len(data_a)} rows differ in '{field}'. "
                f"First mismatch at row {row_idx}:\n"
                f"  ckpt {ckpt_a}: {snip_a}\n"
                f"  ckpt {ckpt_b}: {snip_b}"
            )

    # Sanity check: predictions should actually differ (otherwise the files are just copies)
    pred_identical = sum(
        1 for a, b in zip(data_a, data_b)
        if a.get("prediction", "") == b.get("prediction", "")
    )
    if pred_identical == len(data_a) and len(data_a) > 0:
        issues.append(
            f"WARNING: All {len(data_a)} predictions are identical between "
            f"checkpoints {ckpt_a} and {ckpt_b} — files may be duplicates"
        )

    return passed, issues


def main():
    parser = argparse.ArgumentParser(
        description="Verify that inputs/references are identical across checkpoint prediction files."
    )
    parser.add_argument(
        "--results_dir", type=str, default=DEFAULT_RESULTS_DIR,
        help="Path to all_eval_results directory",
    )
    parser.add_argument(
        "--checkpoints", type=int, nargs="+", default=[4900, 5000],
        help="Checkpoint numbers to compare (all pairs are checked). Default: 4900 5000",
    )
    parser.add_argument(
        "--examples_suffix", type=str, default="1000-examples",
        help='Examples suffix, e.g. "1000-examples" (legacy "examples_1000" is normalized)',
    )
    parser.add_argument(
        "--gen", type=int, default=0,
        help="Prediction generation number to compare (default: 0)",
    )
    parser.add_argument(
        "--inputs_only", action="store_true",
        help="Only check input_text (skip reference check)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: Results directory not found: {args.results_dir}")
        sys.exit(1)

    if len(args.checkpoints) < 2:
        print("ERROR: Need at least 2 checkpoint numbers to compare")
        sys.exit(1)

    pairs = list(combinations(args.checkpoints, 2))
    fields_label = "input_text only" if args.inputs_only else "input_text + reference"
    if args.gen < 0:
        print("ERROR: --gen must be non-negative")
        sys.exit(1)

    examples_suffix = normalize_examples_suffix(args.examples_suffix)
    suffix_label = f" (gen{args.gen}, {examples_suffix})"

    print(f"Checking {fields_label} consistency across {len(args.checkpoints)} checkpoints{suffix_label}")
    print(f"Results dir: {args.results_dir}")
    print(f"Pairs to compare: {len(pairs)}")
    print()

    all_passed = True
    for ckpt_a, ckpt_b in pairs:
        passed, issues = compare_pair(
            args.results_dir, ckpt_a, ckpt_b, examples_suffix, args.gen, args.inputs_only
        )
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] checkpoint-{ckpt_a} vs checkpoint-{ckpt_b}")
        for issue in issues:
            prefix = "    WARNING: " if issue.startswith("WARNING") else "    "
            print(f"{prefix}{issue}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All checks passed: inputs and references are consistent across checkpoints.")
    else:
        print("SOME CHECKS FAILED: see details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
