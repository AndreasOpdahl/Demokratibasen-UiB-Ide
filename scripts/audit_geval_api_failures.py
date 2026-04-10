#!/usr/bin/env python3
"""Scan G-Eval judgment checkpoints for ``[api_error]`` rationales and print grouped stats.

Run from the repo root (or anywhere, with ``--checkpoint-dir``)::

    python scripts/audit_geval_api_failures.py
    python scripts/audit_geval_api_failures.py --checkpoint-dir /path/to/jsonl/dir

Default checkpoint directory is :data:`pairwise_eval.config.GEVAL_CHECKPOINT_DIR` when set,
otherwise ``<repo>/.deepeval/geval_judgment_checkpoints``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_checkpoint_dir() -> Path:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from pairwise_eval.config import GEVAL_CHECKPOINT_DIR, REPO_ROOT

    if GEVAL_CHECKPOINT_DIR is not None:
        return Path(GEVAL_CHECKPOINT_DIR)
    return REPO_ROOT / ".deepeval" / "geval_judgment_checkpoints"


def _extract_http_status_code(body: str) -> Optional[str]:
    """Pull a 3-digit HTTP status from provider / ``requests``-style text (avoid bare digits in model ids)."""
    patterns = [
        r"\bHTTP\s*/?\s*1\.[01]\s+(\d{3})\b",
        r"\bHTTP\s+(\d{3})\b",
        r"\b(\d{3})\s+Server Error\b",
        r"\b(\d{3})\s+Client Error\b",
        r"\b(\d{3})\s+Unknown Error\b",
        r'(?:"status"|status)\s*[:=]\s*"?(\d{3})\b',
        r'"code"\s*:\s*(\d{3})\b',
        r"'code'\s*:\s*(\d{3})\b",
        r"\bstatusCode[\"']?\s*[:=]\s*(\d{3})\b",
    ]
    for pat in patterns:
        m = re.search(pat, body, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _exception_label(body: str) -> Optional[str]:
    m = re.search(r"\b([A-Za-z_][A-Za-z0-9_.]*Error)\s*:", body)
    if m:
        return f"exception_{m.group(1)}"
    return None


def _errno_label(body: str) -> Optional[str]:
    m = re.search(r"\[Errno\s+(-?\d+)\]", body, re.IGNORECASE)
    if m:
        return f"errno_{m.group(1)}"
    return None


def _categorize_api_error(rationale: str) -> str:
    """Map ``[api_error] ...`` text to a short bucket for reporting (prefer ``http_NNN``, not ``other``)."""
    if "[api_error]" not in rationale:
        return "not_api_error"
    body = rationale.split("[api_error]", 1)[-1].strip()
    bl = body.lower()

    code = _extract_http_status_code(body)

    if code == "400" and (
        "context" in bl
        or "maximum context" in bl
        or ("token" in bl and ("maximum" in bl or "limit" in bl or "long" in bl))
        or "too many tokens" in bl
    ):
        return "http_400_context_window"
    if code is not None:
        return f"http_{code}"

    if "timeout" in bl or "timed out" in bl:
        return "timeout"
    if "connection" in bl or "econnrefused" in bl:
        return "connection_error"
    if "json" in bl and ("decode" in bl or "parse" in bl or "not json" in bl):
        return "json_parse"

    el = _exception_label(body)
    if el:
        return el
    en = _errno_label(body)
    if en:
        return en

    # Short stable fingerprint when there is no HTTP code (not the vague "other")
    slug = re.sub(r"[^\w]+", "_", body[:72].strip())[:48].strip("_").lower()
    if slug:
        return f"msg_{slug}"
    return "msg_empty"


def _meta_from_record(rec: dict[str, Any], path: Path) -> tuple[str, str]:
    """Return (judge_id, dimension) from ``key`` JSON or from filename."""
    key = rec.get("key")
    if isinstance(key, str):
        try:
            meta = json.loads(key)
            j, d = meta.get("j"), meta.get("d")
            if isinstance(j, str) and isinstance(d, str) and j and d:
                return j, d
        except (json.JSONDecodeError, TypeError):
            pass
    stem = path.stem
    if "__" not in stem:
        return "(unknown)", "(unknown)"
    judge_stem, dim = stem.rsplit("__", 1)
    judge_id = judge_stem.replace("__", "/")
    return judge_id, dim


def audit_directory(checkpoint_dir: Path) -> tuple[Counter[tuple[str, str, str]], int, int, int]:
    """Return (counter for (dimension, judge, category), files_read, lines_read, api_error_lines)."""
    counts: Counter[tuple[str, str, str]] = Counter()
    files_read = 0
    lines_read = 0
    api_error_lines = 0

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {checkpoint_dir}")

    for path in sorted(checkpoint_dir.glob("*.jsonl")):
        files_read += 1
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                lines_read += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rat = rec.get("rationale")
                if not isinstance(rat, str) or "[api_error]" not in rat:
                    continue
                api_error_lines += 1
                judge_id, dim = _meta_from_record(rec, path)
                cat = _categorize_api_error(rat)
                counts[(dim, judge_id, cat)] += 1

    return counts, files_read, lines_read, api_error_lines


def _print_report(
    counts: Counter[tuple[str, str, str]],
    *,
    files_read: int,
    lines_read: int,
    api_error_lines: int,
    checkpoint_dir: Path,
) -> None:
    print(f"Checkpoint directory: {checkpoint_dir.resolve()}")
    print(f"JSONL files scanned: {files_read}")
    print(f"Non-empty lines read: {lines_read}")
    print(f"Lines with [api_error]: {api_error_lines}")
    print()

    if api_error_lines == 0:
        print("No API errors found.")
        return

    by_cat: Counter[str] = Counter()
    for (dim, judge, cat), n in counts.items():
        by_cat[cat] += n

    print("Totals by failure category")
    print("-" * 40)
    for cat, n in sorted(by_cat.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {cat}: {n}")
    print()

    print("By dimension × judge × category")
    print("-" * 40)
    w_dim = max(len(d) for d, _, _ in counts) if counts else 12
    w_judge = max(len(j) for _, j, _ in counts) if counts else 12
    w_judge = min(max(w_judge, 20), 72)
    w_dim = min(max(w_dim, 12), 24)
    w_cat = max(len(c) for c in by_cat) if by_cat else 12
    w_cat = min(max(w_cat, 14), 56)
    header = f"{'dimension':<{w_dim}}  {'judge':<{w_judge}}  {'category':<{w_cat}}  count"
    print(header)
    print("-" * len(header))
    for (dim, judge, cat), n in sorted(counts.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        jshort = judge if len(judge) <= w_judge else judge[: w_judge - 3] + "..."
        cshort = cat if len(cat) <= w_cat else cat[: w_cat - 3] + "..."
        print(f"{dim:<{w_dim}}  {jshort:<{w_judge}}  {cshort:<{w_cat}}  {n}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List statistics of [api_error] failures in G-Eval checkpoint JSONL files."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory containing *.jsonl checkpoints (default: from pairwise_eval.config)",
    )
    args = parser.parse_args()
    ck = args.checkpoint_dir
    if ck is None:
        ck = _default_checkpoint_dir()
    else:
        ck = ck.expanduser().resolve()

    try:
        counts, files_read, lines_read, api_err = audit_directory(ck)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    _print_report(
        counts,
        files_read=files_read,
        lines_read=lines_read,
        api_error_lines=api_err,
        checkpoint_dir=ck,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
