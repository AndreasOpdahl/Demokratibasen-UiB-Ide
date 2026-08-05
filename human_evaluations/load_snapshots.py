"""Load LabelStudio pairwise-comparison snapshot exports into a tidy Pandas frame.

Each snapshot task = one document shown to one rater (A-F), who judges 3
summary pairs ("pair_1"/"pair_2"/"pair_3"), each pair being LEFT vs RIGHT.
This module reads the flattened "*onlyids*.json" exports (one row per task,
with `pair_N_choice` already extracted by LabelStudio) and produces:

    comparisons     pd.DataFrame, one row per rated pair
    documents       dict[document_id -> document text]
    summaries       dict[summary_id -> summary text]
    summary_model   dict[summary_id -> model_id]
    rater_email     dict[rater_id -> email]   (kept out of the frame: privacy)

Every id->value mapping is checked for consistency across the whole
snapshot set (same id must always carry the same value); a mismatch raises
ConsistencyError rather than being silently overwritten.

Which snapshot files get loaded is controlled by `pattern`, a regex matched
against each *.json filename in SNAPSHOT_FOLDER (default: every flattened
"*onlyids*.json" export).

Usage:
    python load_snapshots.py
    from load_snapshots import load_snapshots
    comparisons, documents, summaries, summary_model, rater_email = load_snapshots()
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

SNAPSHOT_FOLDER = os.path.expandvars(
    "${ONEDRIVE}/Shared/Demokratibasen-UiB-Ide/EvaluationDatasets/LabelStudioSnapshots"
)

# Default file selector: every flattened ("onlyids") JSON export. Pass a
# different regex to load_snapshots(pattern=...) to select a subset, e.g.
# r"batch0[1-8].*onlyids.*\.json$" for just the first batch file.
DEFAULT_SNAPSHOT_PATTERN = r".*onlyids.*\.json$"

# Identifies which human-evaluation round this loader belongs to (this
# folder covers July-August 2026); stamped onto every comparison row.
EVALUATION_ROUND = "202607_08"

# LabelStudio choice label -> agreed -5..5 scale. Positive = left preferred.
CHOICE_TO_SCORE = {
    "Flag right": 5,
    "Left clearly better": 3,
    "Left slightly better": 1,
    "Right slightly better": -1,
    "Right clearly better": -3,
    "Flag left": -5,
}

N_PAIRS_PER_TASK = 3

COMPARISON_COLUMNS = [
    "evaluation_round",
    "document_id", "summary_left_id", "summary_right_id",
    "model_left_id", "model_right_id",
    "rater_id", "score", "winner", "winner_model_id", "raw_choice",
    "pair_position", "block_order_position", "selection_bucket",
    "lead_time_sec", "rated_at",
    "task_id", "annotation_id", "pair_id", "batch_id", "source_doc_id",
]


class ConsistencyError(ValueError):
    """Raised when an id maps to two different values across the snapshot(s)."""


def _check_consistent(mapping: dict[str, Any], key: str, value: Any, what: str) -> None:
    existing = mapping.get(key)
    if existing is not None and existing != value:
        raise ConsistencyError(f"{what} mismatch for id {key!r}: {existing!r} != {value!r}")
    mapping[key] = value


def _find_snapshot_files(folder: str, pattern: str) -> list[Path]:
    regex = re.compile(pattern)
    files = sorted(f for f in Path(folder).glob("*.json") if regex.search(f.name))
    if not files:
        raise FileNotFoundError(
            f"No *.json snapshot files matching {pattern!r} found in {folder!r}. "
            "Check that SNAPSHOT_FOLDER is set/synced correctly, and that the pattern is right."
        )
    return files


def _dedupe_by_annotation(raw_rows: list[dict]) -> list[dict]:
    """Keep only annotated tasks; if the same annotation appears in more than
    one snapshot export (overlapping re-exports), keep the most recent copy."""
    by_annotation_id: dict[Any, dict] = {}
    for row in raw_rows:
        if not row.get("pair_1_choice"):
            continue  # task not yet annotated
        ann_id = row["annotation_id"]
        prev = by_annotation_id.get(ann_id)
        if prev is None or row["updated_at"] > prev["updated_at"]:
            by_annotation_id[ann_id] = row
    return list(by_annotation_id.values())


def load_snapshots(
    folder: str = SNAPSHOT_FOLDER,
    pattern: str = DEFAULT_SNAPSHOT_PATTERN,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    files = _find_snapshot_files(folder, pattern)

    raw_rows: list[dict] = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            raw_rows.extend(json.load(f))

    rows = _dedupe_by_annotation(raw_rows)

    documents: dict[str, str] = {}
    summaries: dict[str, str] = {}
    summary_model: dict[str, str] = {}
    rater_email: dict[str, str] = {}
    comparison_records: list[dict] = []

    for row in rows:
        document_id = row["document_id"]
        _check_consistent(documents, document_id, row["source_text"], "document text")

        rater_id = row["annotator_id"]
        _check_consistent(rater_email, rater_id, row["annotator"], "rater email")

        for n in range(1, N_PAIRS_PER_TASK + 1):
            choice = row.get(f"pair_{n}_choice")
            if not choice:
                continue  # this particular pair wasn't reached/submitted
            if choice not in CHOICE_TO_SCORE:
                raise ValueError(f"Unrecognised choice label: {choice!r}")

            summary_left_id = row[f"pair_{n}_left_summary_id"]
            summary_right_id = row[f"pair_{n}_right_summary_id"]
            model_left_id = row[f"pair_{n}_left_model_id"]
            model_right_id = row[f"pair_{n}_right_model_id"]

            _check_consistent(summaries, summary_left_id, row[f"pair_{n}_left"], "summary text")
            _check_consistent(summaries, summary_right_id, row[f"pair_{n}_right"], "summary text")
            _check_consistent(summary_model, summary_left_id, model_left_id, "summary->model")
            _check_consistent(summary_model, summary_right_id, model_right_id, "summary->model")

            score = CHOICE_TO_SCORE[choice]
            # score is never 0 (CHOICE_TO_SCORE has no tie value): positive -> left won.
            winner = 1 if score > 0 else -1
            winner_model_id = model_left_id if winner == 1 else model_right_id

            comparison_records.append({
                "evaluation_round": EVALUATION_ROUND,
                "document_id": document_id,
                "summary_left_id": summary_left_id,
                "summary_right_id": summary_right_id,
                "model_left_id": model_left_id,
                "model_right_id": model_right_id,
                "rater_id": rater_id,
                "score": score,
                "winner": winner,
                "winner_model_id": winner_model_id,
                "raw_choice": choice,
                "pair_position": n,
                "block_order_position": row.get("block_order_position"),
                "selection_bucket": row.get("selection_bucket"),
                "lead_time_sec": row.get("lead_time"),
                "rated_at": row.get("created_at"),
                "task_id": row.get("id"),
                "annotation_id": row.get("annotation_id"),
                "pair_id": row[f"pair_{n}_id"],
                "batch_id": row.get("batch_id"),
                "source_doc_id": row.get("source_doc_id"),
            })

    comparisons = pd.DataFrame.from_records(comparison_records, columns=COMPARISON_COLUMNS)
    comparisons["rated_at"] = pd.to_datetime(comparisons["rated_at"])

    return comparisons, documents, summaries, summary_model, rater_email


if __name__ == "__main__":
    comparisons, documents, summaries, summary_model, rater_email = load_snapshots()
    print(f"comparisons: {len(comparisons)} rows")
    print(f"documents:   {len(documents)} unique")
    print(f"summaries:   {len(summaries)} unique (models: {len(set(summary_model.values()))})")
    print(f"raters:      {sorted(rater_email)}")
    print()
    print(comparisons.head())
    print()
    print(comparisons["score"].value_counts().sort_index())
