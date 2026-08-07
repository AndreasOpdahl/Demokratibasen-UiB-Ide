"""Per-dimension stratified selection for human judge validation."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from typing import Any

from human_annotation.config import (
    DEFAULT_SELECTION_RATIOS,
    DEFAULT_SEED,
    DIMENSIONS,
    REFERENCE_MODEL_ID,
    SELECTION_BUCKET_ORDER,
)


def _vote_counts(llm_judges: dict[str, dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for payload in llm_judges.values():
        side = payload.get("choice_side")
        if side in {"left", "right", "tie"}:
            counts[side] += 1
    return counts


def _vote_entropy(llm_judges: dict[str, dict[str, Any]]) -> float:
    counts = _vote_counts(llm_judges)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _majority_side(llm_judges: dict[str, dict[str, Any]]) -> str | None:
    counts = _vote_counts(llm_judges)
    if not counts:
        return None
    top = counts.most_common()
    if len(top) >= 2 and top[0][1] == top[1][1]:
        return "split"
    return top[0][0]


def _is_tie_majority(llm_judges: dict[str, dict[str, Any]]) -> bool:
    counts = _vote_counts(llm_judges)
    n_judges = sum(counts.values())
    if n_judges == 0:
        return False
    return counts.get("tie", 0) >= n_judges / 2


def _is_high_agreement(llm_judges: dict[str, dict[str, Any]]) -> bool:
    """Unanimous left/right — excludes tie-heavy cases (see tie_majority)."""
    counts = _vote_counts(llm_judges)
    n_judges = sum(counts.values())
    if n_judges == 0 or len(counts) != 1:
        return False
    return next(iter(counts)) in {"left", "right"}


def _is_reference_challenged(record: dict[str, Any]) -> bool:
    left = record["left_model"]
    right = record["right_model"]
    if REFERENCE_MODEL_ID not in {left, right}:
        return False
    counts = _vote_counts(record["llm_judges"])
    ref_side = "left" if left == REFERENCE_MODEL_ID else "right"
    other_side = "right" if ref_side == "left" else "left"
    return counts.get(other_side, 0) >= 2


def _identical_summaries(record: dict[str, Any]) -> bool:
    left = (record.get("summary_left") or "").strip()
    right = (record.get("summary_right") or "").strip()
    return bool(left) and left == right


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    """Score one pair×dimension row for selection buckets."""
    entropy = _vote_entropy(record["llm_judges"])
    counts = _vote_counts(record["llm_judges"])
    n_judges = sum(counts.values())

    buckets: list[str] = []
    if n_judges >= 2 and entropy >= 1.0:
        buckets.append("low_agreement")
    if _is_tie_majority(record["llm_judges"]):
        buckets.append("tie_majority")
    if _is_high_agreement(record["llm_judges"]):
        buckets.append("high_agreement")
    if _is_reference_challenged(record):
        buckets.append("reference_challenged")
    buckets.append("representative")

    return {
        **record,
        "llm_vote_entropy": round(entropy, 4),
        "llm_vote_counts": dict(counts),
        "llm_majority": _majority_side(record["llm_judges"]),
        "selection_buckets": buckets,
    }


def score_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [score_record(r) for r in records]


def _seed_for_dimension(base_seed: int, dimension: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{dimension}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _bucket_targets(total: int, ratios: dict[str, float]) -> dict[str, int]:
    ratio_sum = sum(ratios.values())
    norm = {k: v / ratio_sum for k, v in ratios.items()}
    targets = {bucket: int(round(total * share)) for bucket, share in norm.items()}
    while sum(targets.values()) < total:
        targets[max(norm, key=norm.get)] += 1
    while sum(targets.values()) > total:
        targets[min(targets, key=targets.get)] -= 1
    return targets


def select_for_dimension(
    records: list[dict[str, Any]],
    dimension: str,
    *,
    total: int,
    ratios: dict[str, float] | None = None,
    seed: int = DEFAULT_SEED,
    exclude_identical_summaries: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pick ``total`` pair×dimension rows for one dimension (independent of other dims)."""
    ratios = ratios or DEFAULT_SELECTION_RATIOS
    rng = random.Random(_seed_for_dimension(seed, dimension))

    pool = [
        r
        for r in records
        if r["dimension"] == dimension
        and not (exclude_identical_summaries and _identical_summaries(r))
    ]
    if not pool:
        raise ValueError(f"No eligible records for dimension {dimension!r}.")

    targets = _bucket_targets(min(total, len(pool)), ratios)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def record_id(rec: dict[str, Any]) -> str:
        return rec["pair_key"]

    def pick_from_bucket(bucket: str, n: int) -> None:
        nonlocal selected, selected_ids
        if n <= 0:
            return
        candidates = [
            r
            for r in pool
            if bucket in r.get("selection_buckets", []) and record_id(r) not in selected_ids
        ]
        if bucket == "low_agreement":
            candidates.sort(key=lambda r: (-r["llm_vote_entropy"], r["pair_key"]))
        elif bucket == "tie_majority":
            candidates.sort(
                key=lambda r: (
                    -_vote_counts(r["llm_judges"]).get("tie", 0),
                    -r["llm_vote_entropy"],
                    r["pair_key"],
                )
            )
        elif bucket == "high_agreement":
            candidates.sort(key=lambda r: (r["llm_vote_entropy"], r["pair_key"]))
        elif bucket == "reference_challenged":
            candidates.sort(
                key=lambda r: (
                    -_vote_counts(r["llm_judges"]).get(
                        "right" if r["left_model"] == REFERENCE_MODEL_ID else "left",
                        0,
                    ),
                    r["pair_key"],
                )
            )
        else:
            rng.shuffle(candidates)

        for rec in candidates[:n]:
            rid = record_id(rec)
            if rid in selected_ids:
                continue
            out = dict(rec)
            out["selection_bucket"] = bucket
            selected.append(out)
            selected_ids.add(rid)

    n_select = min(total, len(pool))
    targets = _bucket_targets(n_select, ratios)

    for bucket in SELECTION_BUCKET_ORDER:
        pick_from_bucket(bucket, targets.get(bucket, 0))

    if len(selected) < n_select:
        remaining = [r for r in pool if record_id(r) not in selected_ids]
        rng.shuffle(remaining)
        for rec in remaining:
            if len(selected) >= n_select:
                break
            rid = record_id(rec)
            out = dict(rec)
            out["selection_bucket"] = "representative"
            selected.append(out)
            selected_ids.add(rid)

    selected.sort(key=lambda r: (r["doc_id"], r["pair_key"]))

    metadata = {
        "dimension": dimension,
        "seed": _seed_for_dimension(seed, dimension),
        "total_requested": total,
        "total_selected": len(selected),
        "pool_size": len(pool),
        "excluded_identical_summary_items": sum(
            1 for r in records if r["dimension"] == dimension and _identical_summaries(r)
        ),
        "ratios": ratios,
        "counts_by_bucket": dict(Counter(r["selection_bucket"] for r in selected)),
    }
    return selected, metadata


def select_all_dimensions(
    scored_records: list[dict[str, Any]],
    *,
    per_dimension: int,
    dimensions: tuple[str, ...] = DIMENSIONS,
    ratios: dict[str, float] | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Return independent stratified samples — one list per dimension."""
    by_dim: dict[str, list[dict[str, Any]]] = {}
    dim_meta: dict[str, dict[str, Any]] = {}

    for dimension in dimensions:
        selected, meta = select_for_dimension(
            scored_records,
            dimension,
            total=per_dimension,
            ratios=ratios,
            seed=seed,
        )
        by_dim[dimension] = selected
        dim_meta[dimension] = meta

    pair_keys_by_dim = {d: {r["pair_key"] for r in rows} for d, rows in by_dim.items()}
    overlap: dict[str, int] = {}
    dims = list(dimensions)
    for i, d1 in enumerate(dims):
        for d2 in dims[i + 1 :]:
            overlap[f"{d1}__{d2}"] = len(pair_keys_by_dim[d1] & pair_keys_by_dim[d2])

    metadata = {
        "seed": seed,
        "per_dimension_requested": per_dimension,
        "per_dimension_selected": {d: len(by_dim[d]) for d in dimensions},
        "total_rows_exported": sum(len(by_dim[d]) for d in dimensions),
        "selection_unit": "pair_per_dimension",
        "dimensions": list(dimensions),
        "pair_overlap_between_dimensions": overlap,
        "by_dimension": dim_meta,
        "ratios": ratios or DEFAULT_SELECTION_RATIOS,
    }
    return by_dim, metadata
