"""Load human-annotation records from G-Eval judgment checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from human_annotation.config import DEFAULT_JUDGES, DIMENSIONS
from human_annotation.geval_io import _JUDGE_FROM_FILENAME, _pair_key, _sanitize_judge_col

__all__ = ["load_from_checkpoints", "_sanitize_judge_col"]


def _parse_checkpoint_filename(name: str) -> tuple[str, str] | None:
    if not name.endswith(".jsonl"):
        return None
    core = name[: -len(".jsonl")]
    judge_raw, dimension = core.rsplit("__", 1)
    judge = _JUDGE_FROM_FILENAME.get(judge_raw, judge_raw)
    return judge, dimension


def _parse_key(key: str) -> dict[str, str]:
    meta = json.loads(key)
    return {
        "doc_id": str(meta["doc"]),
        "left": str(meta["L"]),
        "right": str(meta["R"]),
        "dimension": str(meta["d"]),
        "judge": str(meta["j"]),
    }


def _doc_sort_key(doc_id: str) -> tuple[int, str]:
    if doc_id.startswith("doc_") and doc_id[4:].isdigit():
        return (int(doc_id[4:]), doc_id)
    return (10**9, doc_id)


def _load_pair_context(export_dir: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    """Load source text and summaries keyed by (doc_id, left, right)."""
    json_dir = export_dir / "json"
    context: dict[tuple[str, str, str], dict[str, str]] = {}

    pairs_path = json_dir / "pairs_table.json"
    if pairs_path.is_file():
        rows = json.loads(pairs_path.read_text(encoding="utf-8"))
        for row in rows:
            key = _pair_key(row["doc_id"], row["left"], row["right"])
            context[key] = {
                "summary_left": row.get("sumleft", ""),
                "summary_right": row.get("sumright", ""),
                "source_text": "",
                "reference_summary": "",
            }

    # Doc-level fields from summarization_long.json (one row per doc×model).
    long_path = json_dir / "summarization_long.json"
    if long_path.is_file():
        rows = json.loads(long_path.read_text(encoding="utf-8"))
        doc_fields: dict[str, dict[str, str]] = {}
        for row in rows:
            doc_id = row["doc_id"]
            if doc_id not in doc_fields:
                doc_fields[doc_id] = {
                    "source_text": row.get("source_text", ""),
                    "reference_summary": row.get("reference_summary", ""),
                }
        for key in context:
            doc_id = key[0]
            if doc_id in doc_fields:
                context[key]["source_text"] = doc_fields[doc_id]["source_text"]
                context[key]["reference_summary"] = doc_fields[doc_id]["reference_summary"]

    # Fallback / fill gaps from any G-Eval export table (has full row context).
    for path in sorted(json_dir.glob("geval__*.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            key = _pair_key(row["doc_id"], row["left"], row["right"])
            if key not in context:
                context[key] = {
                    "summary_left": row.get("sumleft", ""),
                    "summary_right": row.get("sumright", ""),
                    "source_text": row.get("source_text", ""),
                    "reference_summary": row.get("reference_summary", ""),
                }
            else:
                slot = context[key]
                if not slot.get("source_text"):
                    slot["source_text"] = row.get("source_text", "")
                if not slot.get("reference_summary"):
                    slot["reference_summary"] = row.get("reference_summary", "")
                if not slot.get("summary_left"):
                    slot["summary_left"] = row.get("sumleft", "")
                if not slot.get("summary_right"):
                    slot["summary_right"] = row.get("sumright", "")

    return context


def load_from_checkpoints(
    checkpoint_dir: Path,
    *,
    context_export_dir: Path | None = None,
    dimensions: tuple[str, ...] = DIMENSIONS,
    judges: tuple[str, ...] = DEFAULT_JUDGES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build pair×dimension records from checkpoint JSONL files.

    Judgments come from ``checkpoint_dir`` (source of truth for the 150-doc LLM subset).
    Document and summary text are joined from ``context_export_dir/json`` when provided.
    """
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoint_files = sorted(checkpoint_dir.glob("*.jsonl"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No *.jsonl checkpoint files in {checkpoint_dir}")

    nested: dict[tuple[str, str, str], dict[str, dict[str, dict[str, Any]]]] = {}
    doc_ids: set[str] = set()

    for path in checkpoint_files:
        parsed = _parse_checkpoint_filename(path.name)
        if parsed is None:
            continue
        judge, dimension = parsed
        if dimension not in dimensions:
            continue

        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key_str = rec.get("key")
                if not isinstance(key_str, str):
                    continue
                meta = _parse_key(key_str)
                doc_id = meta["doc_id"]
                left = meta["left"]
                right = meta["right"]
                doc_ids.add(doc_id)
                pair = _pair_key(doc_id, left, right)
                nested.setdefault(pair, {}).setdefault(dimension, {})[judge] = {
                    "choice_side": rec.get("choice_side"),
                    "chosen": rec.get("chosen"),
                    "rationale": rec.get("rationale", ""),
                }

    pair_context: dict[tuple[str, str, str], dict[str, str]] = {}
    if context_export_dir is not None:
        pair_context = _load_pair_context(context_export_dir)

    records: list[dict[str, Any]] = []
    missing_context = 0
    for pair in sorted(nested, key=lambda k: (_doc_sort_key(k[0]), k[1], k[2])):
        doc_id, left, right = pair
        ctx = pair_context.get(
            pair,
            {
                "summary_left": "",
                "summary_right": "",
                "source_text": "",
                "reference_summary": "",
            },
        )
        if not ctx.get("source_text") or not ctx.get("summary_left"):
            missing_context += 1

        for dimension in dimensions:
            judge_map = nested[pair].get(dimension, {})
            if not judge_map:
                continue
            llm_judges: dict[str, dict[str, Any]] = {}
            for judge in judges:
                if judge in judge_map:
                    llm_judges[judge] = judge_map[judge]
            for judge, payload in judge_map.items():
                if judge not in llm_judges:
                    llm_judges[judge] = payload

            records.append(
                {
                    "doc_id": doc_id,
                    "left_model": left,
                    "right_model": right,
                    "summary_left": ctx.get("summary_left", ""),
                    "summary_right": ctx.get("summary_right", ""),
                    "source_text": ctx.get("source_text", ""),
                    "reference_summary": ctx.get("reference_summary", ""),
                    "dimension": dimension,
                    "llm_judges": llm_judges,
                    "pair_key": "|".join(pair),
                }
            )

    stats = {
        "checkpoint_dir": str(checkpoint_dir),
        "context_export_dir": str(context_export_dir) if context_export_dir else None,
        "distinct_doc_ids": len(doc_ids),
        "distinct_pairs": len(nested),
        "pair_dimension_items": len(records),
        "missing_context_pairs": missing_context,
    }
    return records, stats
