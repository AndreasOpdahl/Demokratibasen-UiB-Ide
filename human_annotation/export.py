"""Export annotation datasets to CSV and JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from human_annotation.config import DEFAULT_JUDGES, DIMENSION_COLORS, DIMENSION_ID_PREFIX
from human_annotation.prompts import load_annotation_prompt


def judge_column_name(judge_id: str) -> str:
    """Stable CSV column suffix for a judge id."""
    return (
        judge_id.replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def llm_vote_column(judge_id: str) -> str:
    return f"llm_vote__{judge_column_name(judge_id)}"


def llm_vote_columns(judge_ids: tuple[str, ...] = DEFAULT_JUDGES) -> list[str]:
    return [llm_vote_column(j) for j in judge_ids]


def is_llm_vote_column(col: str) -> bool:
    return col.startswith("llm_vote__")


def judge_id_from_column(col: str) -> str:
    """Best-effort reverse map for display; column suffixes are unique."""
    return col.removeprefix("llm_vote__").replace("_", "-")


def flatten_llm_votes(record: dict[str, Any]) -> dict[str, str]:
    """Judge id -> choice_side (left/right/tie)."""
    out: dict[str, str] = {}
    for judge_id, payload in record.get("llm_judges", {}).items():
        side = payload.get("choice_side")
        if side in {"left", "right", "tie"}:
            out[judge_id] = side
    return out


def dimension_color(dimension: str) -> str:
    return DIMENSION_COLORS.get(dimension, "#757575")


def item_id(dimension: str, index: int) -> str:
    """Compact id, e.g. rel-01, con-25."""
    prefix = DIMENSION_ID_PREFIX.get(dimension, dimension[:3])
    return f"{prefix}-{index:02d}"


def build_export_rows(
    records: list[dict[str, Any]],
    *,
    dimension: str,
    judge_ids: tuple[str, ...] = DEFAULT_JUDGES,
) -> list[dict[str, Any]]:
    """CSV rows: annotation content + human_choice + flat LLM vote columns."""
    annotation_prompt = load_annotation_prompt(dimension)
    dim_color = dimension_color(dimension)
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        votes = flatten_llm_votes(record)
        vote_counts = record.get("llm_vote_counts", {})

        row: dict[str, Any] = {
            # --- identifiers ---
            "item_id": item_id(dimension, idx),
            "doc_id": record["doc_id"],
            "dimension": dimension,
            "dimension_color": dim_color,
            "pair_key": record.get("pair_key", ""),
            "selection_bucket": record.get("selection_bucket", ""),
            # --- pair content (for annotators) ---
            "left_model": record["left_model"],
            "right_model": record["right_model"],
            "summary_left": record.get("summary_left", ""),
            "summary_right": record.get("summary_right", ""),
            "source_text": record.get("source_text", ""),
            "reference_summary": record.get("reference_summary", ""),
            "annotation_prompt": annotation_prompt,
            # --- human (empty until annotated) ---
            "human_choice": "",
            # --- LLM panel summary (for agreement analysis) ---
            "llm_majority": record.get("llm_majority", ""),
            "llm_votes_left": vote_counts.get("left", 0),
            "llm_votes_right": vote_counts.get("right", 0),
            "llm_votes_tie": vote_counts.get("tie", 0),
            "llm_vote_entropy": record.get("llm_vote_entropy", ""),
        }
        for judge_id in judge_ids:
            row[llm_vote_column(judge_id)] = votes.get(judge_id, "")
        rows.append(row)
    return rows


def build_export_items(
    records: list[dict[str, Any]],
    *,
    dimension: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    annotation_prompt = load_annotation_prompt(dimension)
    dim_color = dimension_color(dimension)
    for idx, record in enumerate(records, start=1):
        votes = flatten_llm_votes(record)
        items.append(
            {
                "item_id": item_id(dimension, idx),
                "doc_id": record["doc_id"],
                "dimension": dimension,
                "dimension_color": dim_color,
                "pair_key": record.get("pair_key", ""),
                "selection_bucket": record.get("selection_bucket", ""),
                "left_model": record["left_model"],
                "right_model": record["right_model"],
                "summary_left": record.get("summary_left", ""),
                "summary_right": record.get("summary_right", ""),
                "source_text": record.get("source_text", ""),
                "reference_summary": record.get("reference_summary", ""),
                "annotation_prompt": annotation_prompt,
                "human_choice": None,
                # Compact LLM block for agreement scripts / notebooks
                "llm_votes": votes,
                "llm_majority": record.get("llm_majority", ""),
                "llm_vote_counts": record.get("llm_vote_counts", {}),
                "llm_vote_entropy": record.get("llm_vote_entropy", ""),
                # Full detail (rationales) kept in JSON only
                "llm_judges": record.get("llm_judges", {}),
            }
        )
    return items


def build_combined_export(
    csv_rows_by_dimension: dict[str, list[dict[str, Any]]],
    json_items_by_dimension: dict[str, list[dict[str, Any]]],
    *,
    dimensions: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Concatenate per-dimension exports in ``dimensions`` order."""
    combined_rows: list[dict[str, Any]] = []
    combined_items: list[dict[str, Any]] = []
    for dimension in dimensions:
        combined_rows.extend(csv_rows_by_dimension.get(dimension, []))
        combined_items.extend(json_items_by_dimension.get(dimension, []))
    return combined_rows, combined_items


LABEL_STUDIO_CSV_COLUMNS: tuple[str, ...] = (
    "dimension",
    "dimension_color",
    "annotation_prompt",
    "source_text",
    "summary_left",
    "summary_right",
)


def to_label_studio_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Subset of CSV columns for Label Studio import."""
    return [{col: row.get(col, "") for col in LABEL_STUDIO_CSV_COLUMNS} for row in rows]


def write_label_studio_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, to_label_studio_rows(rows))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
