from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

DEFAULT_DIMENSIONS = (
    "faithfulness",
    "correctness",
    "completeness",
    "newsworthiness",
    "hygiene",
)

_GEVAL_RE = re.compile(r"^geval__(.+?)__([^_]+)\.json$")
_CK_RE = re.compile(r"checkpoint-(\d+)-", re.IGNORECASE)


def parse_dimension_weights(spec: str | None) -> dict[str, float]:
    weights = {d: 1.0 for d in DEFAULT_DIMENSIONS}
    if not spec:
        return weights
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip().lower()
        try:
            weights[k] = float(v.strip())
        except ValueError:
            continue
    return weights


def parse_target_dimension_weights(spec: str) -> dict[str, float]:
    """Parse ``dim=value`` for reference-judge target aggregation in weight learning.

    Each value must lie in ``[0, 1]``. Dimensions not listed get weight ``0``. At least
    one dimension must be strictly positive. The returned dict is **not** normalized;
    callers (e.g. ``learn_weights_from_geval_rows``) divide by the sum so coefficients
    sum to 1.

    Raises ``ValueError`` on empty spec, unknown dimension names, values outside
    ``[0, 1]``, or all-zero weights.
    """
    if not (spec or "").strip():
        raise ValueError("target dimension weights spec is empty")
    dim_set = set(DEFAULT_DIMENSIONS)
    weights = {d: 0.0 for d in DEFAULT_DIMENSIONS}
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if "=" not in part:
            raise ValueError(f"expected dim=value, got {part!r}")
        k, v = part.split("=", 1)
        k = k.strip().lower()
        if k not in dim_set:
            raise ValueError(
                f"unknown dimension {k!r}; expected one of {sorted(dim_set)}"
            )
        try:
            fv = float(v.strip())
        except ValueError as e:
            raise ValueError(f"invalid number in {part!r}") from e
        if fv < 0.0 or fv > 1.0:
            raise ValueError(f"weight for {k} must be in [0, 1], got {fv}")
        weights[k] = fv
    if sum(weights.values()) <= 0.0:
        raise ValueError("at least one dimension must have a weight > 0")
    return weights


def checkpoint_step(model_id: str) -> int | None:
    m = _CK_RE.search(model_id or "")
    return int(m.group(1)) if m else None


def load_geval_rows(
    json_dir: Path,
    dimensions: tuple[str, ...] = DEFAULT_DIMENSIONS,
    max_files: int | None = None,
) -> list[dict]:
    rows: list[dict] = []
    files = sorted(json_dir.glob("geval__*__*.json"))
    if max_files is not None and max_files > 0:
        files = files[:max_files]
    dim_set = set(dimensions)

    for path in files:
        m = _GEVAL_RE.match(path.name)
        if not m:
            continue
        judge_id, dim = m.group(1), m.group(2)
        if dim not in dim_set:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for r in data:
            if not isinstance(r, dict):
                continue
            rows.append(
                {
                    "judge_id": judge_id,
                    "dimension": dim,
                    "doc_id": r.get("doc_id"),
                    "left": r.get("left"),
                    "right": r.get("right"),
                    "choice_side": r.get("choice_side"),
                }
            )
    return rows


def compute_checkpoint_weighted_means(
    rows: list[dict],
    dimension_weights: dict[str, float] | None = None,
) -> dict[int, float]:
    if dimension_weights is None:
        dimension_weights = {d: 1.0 for d in DEFAULT_DIMENSIONS}

    per_ck_dim_sum = defaultdict(lambda: defaultdict(float))
    per_ck_dim_cnt = defaultdict(lambda: defaultdict(int))

    for r in rows:
        left = str(r.get("left", ""))
        right = str(r.get("right", ""))
        dim = str(r.get("dimension", ""))
        side = str(r.get("choice_side", "tie")).lower().strip()
        lck = checkpoint_step(left)
        rck = checkpoint_step(right)
        if lck is None and rck is None:
            continue

        if lck is not None:
            o = 0.5 if side == "tie" else (1.0 if side == "left" else 0.0)
            per_ck_dim_sum[lck][dim] += o
            per_ck_dim_cnt[lck][dim] += 1
        if rck is not None:
            o = 0.5 if side == "tie" else (1.0 if side == "right" else 0.0)
            per_ck_dim_sum[rck][dim] += o
            per_ck_dim_cnt[rck][dim] += 1

    out: dict[int, float] = {}
    for ck, dim_sums in per_ck_dim_sum.items():
        num = 0.0
        den = 0.0
        for dim, s in dim_sums.items():
            c = per_ck_dim_cnt[ck][dim]
            w = float(dimension_weights.get(dim, 0.0))
            if c <= 0 or w <= 0:
                continue
            num += (s / c) * w
            den += w
        if den > 0:
            out[ck] = num / den
    return out


def compute_checkpoint_dimension_means(
    rows: list[dict],
    *,
    judge_substring: str | None = None,
    dimensions: tuple[str, ...] = DEFAULT_DIMENSIONS,
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, int]]]:
    """Per-checkpoint mean win outcome (0/0.5/1) per dimension.

    If ``judge_substring`` is set, only rows whose ``judge_id`` contains that
    substring are used (case-sensitive substring match).
    """
    dim_set = set(dimensions)
    per_ck_dim_sum = defaultdict(lambda: defaultdict(float))
    per_ck_dim_cnt = defaultdict(lambda: defaultdict(int))

    for r in rows:
        jid = str(r.get("judge_id", ""))
        if judge_substring is not None and judge_substring not in jid:
            continue
        left = str(r.get("left", ""))
        right = str(r.get("right", ""))
        dim = str(r.get("dimension", ""))
        if dim not in dim_set:
            continue
        side = str(r.get("choice_side", "tie")).lower().strip()
        lck = checkpoint_step(left)
        rck = checkpoint_step(right)
        if lck is None and rck is None:
            continue

        if lck is not None:
            o = 0.5 if side == "tie" else (1.0 if side == "left" else 0.0)
            per_ck_dim_sum[lck][dim] += o
            per_ck_dim_cnt[lck][dim] += 1
        if rck is not None:
            o = 0.5 if side == "tie" else (1.0 if side == "right" else 0.0)
            per_ck_dim_sum[rck][dim] += o
            per_ck_dim_cnt[rck][dim] += 1

    means: dict[int, dict[str, float]] = {}
    for ck, dim_sums in per_ck_dim_sum.items():
        m: dict[str, float] = {}
        for dim, s in dim_sums.items():
            c = per_ck_dim_cnt[ck][dim]
            if c > 0:
                m[dim] = s / c
        if m:
            means[ck] = m

    counts: dict[int, dict[str, int]] = {
        ck: {d: per_ck_dim_cnt[ck][d] for d in per_ck_dim_sum[ck]} for ck in per_ck_dim_sum
    }
    return means, counts

