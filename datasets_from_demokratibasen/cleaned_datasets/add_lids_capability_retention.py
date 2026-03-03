#!/usr/bin/env python3
"""
Add Language Identification (LID) metadata to capability-retention JSONL files.

This script is a light-weight wrapper around the LID helpers in add_lids.py,
designed for the capability_retention_data folder where examples do NOT always
use the standard {"input": ..., "output": ...} schema.

Behaviour:
  - For text-summary band files (e.g. 155452_text_summary_examples_broad.jsonl),
    it behaves like add_lids.py and uses the "input" and "output" fields.
  - For capability prediction files (3060_c2_text_prediction_*.jsonl,
    9650_c3_prompt_continuation_*.jsonl, etc.), it:
      * treats the "text" field as the input text to classify,
      * does NOT require an "output" field,
      * writes LID info into a "metadata" sub-dict:
            metadata.input_lid
            metadata.input_lid_conf
        (output_lid fields are omitted for these files).

IMPORTANT:
  - Run this from a Python environment where:
        - fasttext is installed
        - huggingface-hub is installed
        - NumPy < 2.0 (fastText is incompatible with NumPy 2.0+)
  - Example usage (from repo root):

        cd datasets_from_demokratibasen/cleaned_datasets
        python add_lids_capability_retention.py --dataset-folder text_summary_dataset_202601

"""

from __future__ import annotations

import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# We want to reuse the LID model + classifier from add_lids.py which lives in
# the same folder as this script. We ALSO need the "kommuner" package from
# prepared_datasets to be importable, because add_lids.py imports
# `from kommuner.kommune import kommunenavn`.
#
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parents[2]  # .../Demokratibasen-UiB-Ide

# Make prepared_datasets (which contains the kommuner package) importable.
sys.path.insert(0, str(repo_root / "datasets_from_demokratibasen" / "prepared_datasets"))
# Make this cleaned_datasets directory importable so we can import add_lids.
sys.path.insert(0, str(current_dir))

import add_lids  # type: ignore  # local sibling module


def _classify_text_safe(text: Optional[str], model) -> Optional[Tuple[str, float]]:
    """
    Classify a single text string using the shared _classify_lid helper.

    Unlike the default add_lids behaviour for the main dataset, this helper:
      - treats None / empty text as "skip" (returns None) instead of raising,
      - catches RuntimeError from NumPy 2.0 incompatibility and re-raises,
      - lets other exceptions fall back to a warning and skip.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    if not text.strip():
        return None

    try:
        return add_lids._classify_lid(text, model)
    except RuntimeError:
        # NumPy 2.0 vs fastText incompatibility or other fatal configuration issue.
        # Let this propagate so the user sees the same clear error message.
        raise
    except Exception as e:  # pragma: no cover - very defensive
        msg = str(e)
        # Mirror the behaviour in add_lids: only warn for "unexpected" errors.
        if "predict processes one line at a time" not in msg and "Unable to avoid copy" not in msg:
            print(f"Warning: Failed to classify text with LID in capability file: {e}", file=sys.stderr)
        return None


def _add_lid_to_capability_file(path: Path, model) -> Dict[str, Any]:
    """
    Add LID metadata to a capability-retention JSONL file.

    For each line:
      - If it has an "input" field, we treat it as a standard text-summary example and
        behave like add_lids._add_lid_to_jsonl_file (for input only).
      - Else if it has a "text" field, we classify that and store the result in
        metadata.input_lid / metadata.input_lid_conf.
      - Lines where neither "input" nor "text" is present (or both are None/empty)
        are left unchanged.

    Returns simple statistics:
      {
        "file": <filename>,
        "count_classified": <int>,
        "input_lid_distribution": {lid: count, ...},
      }
    """
    print(f"  Processing capability file: {path.name}", file=sys.stderr)

    from collections import Counter

    input_lid_counter: Counter[str] = Counter()
    classified = 0

    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                # Preserve malformed lines as-is
                examples.append(line)
                continue

            # Choose which field to classify
            text_to_use: Optional[str]
            if "input" in ex:
                text_to_use = ex.get("input")
            elif "text" in ex:
                text_to_use = ex.get("text")
            else:
                text_to_use = None

            result = _classify_text_safe(text_to_use, model)

            if result is not None:
                lid, conf = result
                if "metadata" not in ex or not isinstance(ex["metadata"], dict):
                    ex["metadata"] = {}
                ex["metadata"]["input_lid"] = lid
                ex["metadata"]["input_lid_conf"] = conf
                input_lid_counter[lid] += 1
                classified += 1

            examples.append(ex)

    # Write updated file in-place
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            if isinstance(ex, str):
                f.write(ex + "\n")
            else:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    stats: Dict[str, Any] = {
        "file": path.name,
        "count_classified": classified,
        "input_lid_distribution": dict(input_lid_counter),
    }
    return stats


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add LID metadata to capability_retention_data JSONL files.",
    )
    parser.add_argument(
        "--dataset-folder",
        type=str,
        required=True,
        help=(
            "Path to the text_summary_dataset_XXXX folder containing the "
            "capability_retention_data subfolder "
            "(e.g. 'text_summary_dataset_202601')."
        ),
    )

    args = parser.parse_args(argv)

    # Resolve base folder and capability_retention_data
    base_folder = Path(args.dataset_folder).resolve()
    cap_dir = base_folder / "capability_retention_data"

    if not cap_dir.exists():
        print(f"Error: capability_retention_data folder not found at {cap_dir}", file=sys.stderr)
        return 1

    if not getattr(add_lids, "LID_AVAILABLE", False):
        print(
            "Error: fasttext and huggingface_hub are required for LID analysis.\n"
            "Install them with: pip install fasttext huggingface-hub",
            file=sys.stderr,
        )
        return 1

    # Load LID model once
    print("Loading LID model for capability_retention_data ...", file=sys.stderr)
    lid_model = add_lids._get_lid_model()
    print("LID model loaded successfully.", file=sys.stderr)

    # All example JSONLs in capability_retention_data (exclude *_ids.jsonl)
    jsonl_files = [
        p for p in sorted(cap_dir.glob("*.jsonl"), key=lambda p: p.name)
        if not p.name.endswith("_ids.jsonl")
    ]

    if not jsonl_files:
        print(f"No example JSONL files found in {cap_dir}", file=sys.stderr)
        return 1

    all_stats: Dict[str, Any] = {}

    for path in jsonl_files:
        stats = _add_lid_to_capability_file(path, model=lid_model)
        all_stats[path.name] = stats
        dist = stats["input_lid_distribution"]
        total = sum(dist.values())
        print(
            f"{path.name}: classified={stats['count_classified']}, "
            f"input_lids={dist} (total {total})"
        )

    print("\nDONE: LID metadata added in-place to capability_retention_data JSONL files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

