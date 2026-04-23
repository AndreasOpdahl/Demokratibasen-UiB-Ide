"""Append-only JSONL checkpoints so G-Eval can resume after crashes (one file per judge × dimension)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd


def _safe_checkpoint_stem(judge_id: str) -> str:
    """Filename fragment: no slashes in judge ids."""
    return judge_id.replace("\\", "__").replace("/", "__")


def judgment_stable_key(judge_id: str, dimension: str, row: Mapping[str, Any]) -> str:
    """Stable id for one comparison; includes a hash of both summary texts so edited data rematches.

    Input: judge, dimension, context row (needs ``doc_id``, ``left``, ``right``, ``sumleft``, ``sumright``).
    Output: canonical JSON string used as ``key`` in checkpoint lines.
    """
    blob = (str(row.get("sumleft", "")) + "\n---\n" + str(row.get("sumright", ""))).encode(
        "utf-8", errors="replace"
    )
    h = hashlib.sha256(blob).hexdigest()[:24]
    payload = {
        "d": str(dimension),
        "doc": str(row["doc_id"]),
        "h": h,
        "j": str(judge_id),
        "L": str(row["left"]),
        "R": str(row["right"]),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def checkpoint_file_path(checkpoint_dir: Path, judge_id: str, dimension: str) -> Path:
    """Path to the JSONL file for one (judge, dimension) stream."""
    return checkpoint_dir / f"{_safe_checkpoint_stem(judge_id)}__{dimension}.jsonl"


def _judgment_jsonable(judgment: Dict[str, object]) -> Dict[str, Any]:
    """Convert judgment dict to JSON-serializable form (``chosen`` NA → null)."""
    out: Dict[str, Any] = {}
    for k, v in judgment.items():
        if k == "chosen" and pd.isna(v):
            out[k] = None
        else:
            out[k] = v
    return out


def judgment_from_checkpoint_record(rec: Mapping[str, Any]) -> Dict[str, object]:
    """Rebuild judgment fields from a loaded JSON object (null ``chosen`` → ``pd.NA``)."""
    out: Dict[str, object] = dict(rec)
    out.pop("key", None)
    if out.get("chosen") is None:
        out["chosen"] = pd.NA
    return out


def load_checkpoint_index(path: Path) -> Dict[str, Dict[str, object]]:
    """Read all complete lines from a JSONL checkpoint; map stable key → judgment dict.

    Input: path to ``.jsonl`` (may be missing → empty dict). Output: key → judgment.
    """
    if not path.is_file():
        return {}
    out: Dict[str, Dict[str, object]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = rec.get("key")
            if not isinstance(k, str):
                continue
            out[k] = judgment_from_checkpoint_record(rec)
    return out


def discover_checkpoint_leaf_dirs(checkpoint_root: Path) -> list[Path]:
    """Return directories that each contain their own ``*.jsonl`` judgment streams.

    If ``checkpoint_root`` has ``*.jsonl`` files directly (legacy flat layout), returns
    ``[checkpoint_root]``. Otherwise returns each immediate subdirectory that contains at
    least one ``*.jsonl`` (per-model layout under ``geval_judgment_checkpoints/<model>/``).
    """
    if not checkpoint_root.is_dir():
        return []
    if any(checkpoint_root.glob("*.jsonl")):
        return [checkpoint_root]
    out: list[Path] = []
    for child in sorted(checkpoint_root.iterdir()):
        if child.is_dir() and any(child.glob("*.jsonl")):
            out.append(child)
    return out


def append_judgment_line(path: Path, key: str, judgment: Dict[str, object]) -> None:
    """Append one judgment and sync to disk so only the in-flight call is at risk on crash.

    Input: checkpoint file path, stable key, judgment dict. Output: none (creates parent dirs).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"key": key, **_judgment_jsonable(judgment)}
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
