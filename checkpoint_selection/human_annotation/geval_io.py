"""Load and merge G-Eval pairwise judgments from an export directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from human_annotation.config import DEFAULT_JUDGES, DIMENSIONS

_JUDGE_FROM_FILENAME: dict[str, str] = {
    "gpt-5-mini": "gpt-5-mini",
    "mistral-medium-latest": "mistral-medium-latest",
    "google__gemini-2.5-flash-preview-05-20": "google/gemini-2.5-flash-preview-05-20",
    "anthropic__claude-3-5-haiku-20241022": "anthropic/claude-3-5-haiku-20241022",
}


def _pair_key(doc_id: str, left: str, right: str) -> tuple[str, str, str]:
    return (doc_id, left, right)


def _sanitize_judge_col(judge_id: str) -> str:
    return (
        judge_id.replace("/", "__")
        .replace("-", "_")
        .replace(".", "_")
    )


def _parse_geval_filename(name: str) -> tuple[str, str] | None:
    if not name.startswith("geval__") or not name.endswith(".json"):
        return None
    core = name[len("geval__") : -len(".json")]
    judge_raw, dimension = core.rsplit("__", 1)
    judge = _JUDGE_FROM_FILENAME.get(judge_raw, judge_raw)
    return judge, dimension


def discover_geval_files(json_dir: Path) -> list[tuple[str, str, Path]]:
    out: list[tuple[str, str, Path]] = []
    for path in sorted(json_dir.glob("geval__*.json")):
        parsed = _parse_geval_filename(path.name)
        if parsed is None:
            continue
        judge, dimension = parsed
        out.append((judge, dimension, path))
    return out


def load_geval_export(
    export_dir: Path,
    *,
    dimensions: tuple[str, ...] = DIMENSIONS,
    judges: tuple[str, ...] = DEFAULT_JUDGES,
) -> list[dict[str, Any]]:
    """Return one record per (pair, dimension) with all judge responses merged."""
    json_dir = export_dir / "json"
    if not json_dir.is_dir():
        raise FileNotFoundError(f"Missing json/ under export dir: {export_dir}")

    discovered = discover_geval_files(json_dir)
    if not discovered:
        raise FileNotFoundError(f"No geval__*.json files in {json_dir}")

    # pair_key -> dimension -> judge -> judgment row
    nested: dict[tuple[str, str, str], dict[str, dict[str, dict[str, Any]]]] = {}
    base_fields: dict[tuple[str, str, str], dict[str, Any]] = {}

    for judge, dimension, path in discovered:
        if dimension not in dimensions:
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            doc_id = row["doc_id"]
            left = row["left"]
            right = row["right"]
            key = _pair_key(doc_id, left, right)
            if key not in base_fields:
                base_fields[key] = {
                    "doc_id": doc_id,
                    "left_model": left,
                    "right_model": right,
                    "summary_left": row.get("sumleft", ""),
                    "summary_right": row.get("sumright", ""),
                    "source_text": row.get("source_text", ""),
                    "reference_summary": row.get("reference_summary", ""),
                }
            nested.setdefault(key, {}).setdefault(dimension, {})[judge] = {
                "choice_side": row.get("choice_side"),
                "chosen": row.get("chosen"),
                "rationale": row.get("rationale", ""),
            }

    records: list[dict[str, Any]] = []
    for key in sorted(base_fields, key=lambda k: (int(k[0].split("_")[-1]), k[1], k[2])):
        base = base_fields[key]
        for dimension in dimensions:
            judge_map = nested.get(key, {}).get(dimension, {})
            if not judge_map:
                continue
            llm_judges: dict[str, dict[str, Any]] = {}
            for judge in judges:
                if judge in judge_map:
                    llm_judges[judge] = judge_map[judge]
                else:
                    # Include any extra judges found in export.
                    pass
            for judge, payload in judge_map.items():
                if judge not in llm_judges:
                    llm_judges[judge] = payload

            records.append(
                {
                    **base,
                    "dimension": dimension,
                    "llm_judges": llm_judges,
                    "pair_key": "|".join(key),
                }
            )

    return records
