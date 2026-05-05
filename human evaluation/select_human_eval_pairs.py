#!/usr/bin/env python3
"""Build a human-evaluation dataset from existing G-Eval pair exports.

Selection mix (default):
- 30% low-agreement pairs (hard / ambiguous)
- 20% high-agreement pairs (controls / sanity)
- 30% top-checkpoint matchups
- 20% GPT4o-challenged pairs

Outputs in the chosen output directory:
- human_eval_dataset.jsonl
- human_eval_answer_key.csv
- selection_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation_suite.core.geval_mean import (
    compute_checkpoint_weighted_means as compute_checkpoint_weighted_means_raw,
    load_geval_rows as load_geval_rows_raw,
)


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)", re.IGNORECASE)
GEVAL_FILENAME_RE = re.compile(r"^geval__(.+?)__([^_]+)\.json$")
DEFAULT_DIMENSIONS = (
    "faithfulness",
    "correctness",
    "completeness",
    "newsworthiness",
    "hygiene",
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    left = str(row.get("left", ""))
    right = str(row.get("right", ""))
    doc_id = str(row.get("doc_id", ""))
    if left <= right:
        return doc_id, left, right
    return doc_id, right, left


def _pair_key_exact(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row.get("doc_id", "")), str(row.get("left", "")), str(row.get("right", ""))


def _normalized_summary_text(text: str) -> str:
    return " ".join((text or "").split()).strip().lower()


def _stage_for_pair(left: str, right: str) -> str:
    values = []
    for model_name in (left, right):
        m = CHECKPOINT_RE.search(model_name)
        if m:
            values.append(int(m.group(1)))
    if not values:
        return "mixed_or_baseline"
    avg = sum(values) / len(values)
    if avg <= 3500:
        return "early"
    if avg <= 7000:
        return "mid"
    return "late"


def _length_bucket(source_text: str) -> str:
    n = len(source_text or "")
    if n < 1200:
        return "short"
    if n < 5000:
        return "medium"
    return "long"


def _edge_score(row: dict[str, Any]) -> float:
    sumleft = str(row.get("sumleft", ""))
    sumright = str(row.get("sumright", ""))
    source_text = str(row.get("source_text", ""))
    ref = str(row.get("reference_summary", ""))

    left_len = len(sumleft)
    right_len = len(sumright)
    min_len = min(left_len, right_len)
    max_len = max(left_len, right_len) if max(left_len, right_len) > 0 else 1
    ratio = min_len / max_len

    truncation_markers = ("...", "…")
    truncated = int(
        sumleft.endswith(truncation_markers) or sumright.endswith(truncation_markers)
    )
    very_short = int(min_len < 80)
    source_very_long = int(len(source_text) > 7000)
    large_imbalance = 1.0 - ratio
    baseline_overlap = int("gpt" in row.get("left", "").lower() or "gpt" in row.get("right", "").lower())
    has_reference = int(bool(ref))

    # Weighted toward likely problematic cases.
    score = (
        1.5 * truncated
        + 1.4 * very_short
        + 1.0 * source_very_long
        + 1.0 * large_imbalance
        + 0.4 * baseline_overlap
        + 0.2 * has_reference
    )
    return float(score)


def _load_judgments(
    json_dir: Path, max_judgment_files: int | None = None
) -> dict[tuple[str, str, str], Counter]:
    pattern = "geval__*__*.json"
    vote_map: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    paths = sorted(json_dir.glob(pattern))
    if max_judgment_files is not None and max_judgment_files > 0:
        paths = paths[:max_judgment_files]
    for path in paths:
        rows = _read_json(path)
        for row in rows:
            key = _pair_key(row)
            side = str(row.get("choice_side", "tie")).lower().strip()
            if side not in {"left", "right", "tie"}:
                side = "tie"
            vote_map[key][side] += 1
    return vote_map


def _load_judgment_details(
    json_dir: Path, max_judgment_files: int | None = None
) -> dict[tuple[str, str, str], dict[str, dict[str, str]]]:
    """Return per-pair LLM decisions as dimension -> judge -> side."""
    detail_map: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    paths = sorted(json_dir.glob("geval__*__*.json"))
    if max_judgment_files is not None and max_judgment_files > 0:
        paths = paths[:max_judgment_files]

    for path in paths:
        m = GEVAL_FILENAME_RE.match(path.name)
        if not m:
            continue
        judge_id = m.group(1)
        dimension = m.group(2)
        rows = _read_json(path)
        for row in rows:
            key = _pair_key_exact(row)
            side = str(row.get("choice_side", "tie")).lower().strip()
            if side not in {"left", "right", "tie"}:
                side = "tie"
            detail_map[key][dimension][judge_id] = side
    return detail_map


def _uncertainty_score(votes: Counter) -> float:
    total = votes["left"] + votes["right"] + votes["tie"]
    if total == 0:
        return 1.0
    p_left = votes["left"] / total
    p_right = votes["right"] / total
    p_tie = votes["tie"] / total
    margin = abs(p_left - p_right)
    tie_rate = p_tie
    # High uncertainty: small left/right margin + many ties.
    return (1.0 - margin) * 0.7 + tie_rate * 0.3


def _easy_score(votes: Counter) -> float:
    total = votes["left"] + votes["right"] + votes["tie"]
    if total == 0:
        return 0.0
    p_left = votes["left"] / total
    p_right = votes["right"] / total
    p_tie = votes["tie"] / total
    return abs(p_left - p_right) * (1.0 - p_tie)


def _dominant_label(votes: Counter) -> str:
    ranked = sorted([("left", votes["left"]), ("right", votes["right"]), ("tie", votes["tie"])], key=lambda x: x[1], reverse=True)
    if ranked[0][1] == ranked[1][1]:
        return "tie"
    return ranked[0][0]


def _agreement_from_dimension_votes(
    dim_votes: dict[str, dict[str, str]]
) -> tuple[float, float]:
    """Return (mean agreement, min agreement) over dimensions.

    Agreement per dimension is majority fraction among judges.
    Example L,L,L,L,R -> 4/5 = 0.8.
    """
    if not dim_votes:
        return 0.0, 0.0
    agreements = []
    for _dim, judge_map in dim_votes.items():
        if not judge_map:
            continue
        c = Counter(side for side in judge_map.values())
        n = sum(c.values())
        if n == 0:
            continue
        agreements.append(max(c.values()) / n)
    if not agreements:
        return 0.0, 0.0
    return float(sum(agreements) / len(agreements)), float(min(agreements))


def _dimension_vote_counts(
    dim_votes: dict[str, dict[str, str]]
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for dim, judge_map in dim_votes.items():
        c = Counter(side for side in judge_map.values())
        out[dim] = {
            "left": int(c.get("left", 0)),
            "right": int(c.get("right", 0)),
            "tie": int(c.get("tie", 0)),
        }
    return out


def _flip_side(side: str) -> str:
    if side == "left":
        return "right"
    if side == "right":
        return "left"
    return "tie"


def _dim_votes_for_row(
    row: dict[str, Any], details_map: dict[tuple[str, str, str], dict[str, dict[str, str]]]
) -> dict[str, dict[str, str]]:
    exact = _pair_key_exact(row)
    if exact in details_map:
        return details_map[exact]
    rev = (exact[0], exact[2], exact[1])
    if rev not in details_map:
        return {}
    # Re-orient sides to match row's left/right.
    flipped: dict[str, dict[str, str]] = {}
    for dim, judge_map in details_map[rev].items():
        flipped[dim] = {j: _flip_side(side) for j, side in judge_map.items()}
    return flipped


def _checkpoint_num(model_id: str) -> int | None:
    m = CHECKPOINT_RE.search(model_id or "")
    return int(m.group(1)) if m else None


def _top_checkpoint_match_score(left: str, right: str, top_steps: set[int]) -> float:
    """Higher is better for top-checkpoint matchup bucket."""
    a = _checkpoint_num(left)
    b = _checkpoint_num(right)
    if a is None or b is None:
        return 0.0
    in_top_a = a in top_steps
    in_top_b = b in top_steps
    nearby = abs(a - b) <= 2000
    both_top = in_top_a and in_top_b
    one_top = in_top_a or in_top_b
    if both_top:
        return 3.0
    if one_top and nearby:
        return 2.0
    if one_top:
        return 1.0
    return 0.0


def _gpt4o_challenge_score(
    left: str, right: str, dim_votes: dict[str, dict[str, str]]
) -> float:
    """Score pairs where non-GPT4o side is preferred by judges."""
    left_is_gpt4o = "gpt4o" in (left or "").lower()
    right_is_gpt4o = "gpt4o" in (right or "").lower()
    if left_is_gpt4o == right_is_gpt4o:
        return 0.0

    challenged_dims = 0
    total_margin = 0.0
    for _dim, judge_map in dim_votes.items():
        if not judge_map:
            continue
        c = Counter(side for side in judge_map.values())
        left_votes = c.get("left", 0)
        right_votes = c.get("right", 0)
        n = left_votes + right_votes + c.get("tie", 0)
        if n == 0:
            continue
        # Non-GPT4o majority over GPT4o counts as challenge.
        if left_is_gpt4o and right_votes > left_votes:
            challenged_dims += 1
            total_margin += (right_votes - left_votes) / n
        elif right_is_gpt4o and left_votes > right_votes:
            challenged_dims += 1
            total_margin += (left_votes - right_votes) / n
    if challenged_dims == 0:
        return 0.0
    return float(challenged_dims + total_margin)


def _is_gpt4o_pair(left: str, right: str) -> bool:
    l = (left or "").lower()
    r = (right or "").lower()
    return ("gpt4o" in l) or ("gpt4o" in r)


def _compute_checkpoint_weighted_scores(
    rows: list[dict[str, Any]],
    details_map: dict[tuple[str, str, str], dict[str, dict[str, str]]],
    dim_weights: dict[str, float],
) -> dict[int, float]:
    """Same core logic as view_geval_prefix_interactive.py (full data, not prefix):
    per dimension mean outcome (1 win, 0 loss, 0.5 tie), then weighted mean across dimensions.
    """
    per_ck_dim_sum: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    per_ck_dim_cnt: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in rows:
        left = str(row.get("left", ""))
        right = str(row.get("right", ""))
        a = _checkpoint_num(left)
        b = _checkpoint_num(right)
        if a is None and b is None:
            continue
        dim_votes = _dim_votes_for_row(row, details_map)
        if not dim_votes:
            continue
        for dim, judge_map in dim_votes.items():
            for side in judge_map.values():
                if a is not None:
                    o = 0.5 if side == "tie" else (1.0 if side == "left" else 0.0)
                    per_ck_dim_sum[a][dim] += o
                    per_ck_dim_cnt[a][dim] += 1
                if b is not None:
                    o = 0.5 if side == "tie" else (1.0 if side == "right" else 0.0)
                    per_ck_dim_sum[b][dim] += o
                    per_ck_dim_cnt[b][dim] += 1

    out: dict[int, float] = {}
    for ck, dim_sum in per_ck_dim_sum.items():
        num = 0.0
        den = 0.0
        for dim, s in dim_sum.items():
            c = per_ck_dim_cnt[ck][dim]
            if c <= 0:
                continue
            w = float(dim_weights.get(dim, 0.0))
            if w <= 0:
                continue
            num += (s / c) * w
            den += w
        if den > 0:
            out[ck] = num / den
    return out


def _parse_dimension_weights(spec: str | None) -> dict[str, float]:
    if not spec or not spec.strip():
        return {d: 1.0 for d in DEFAULT_DIMENSIONS}
    out = {d: 1.0 for d in DEFAULT_DIMENSIONS}
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip().lower()
        try:
            out[k] = float(v.strip())
        except ValueError:
            continue
    return out


def _stratified_pick(
    rows: list[dict[str, Any]],
    n: int,
    rng: random.Random,
    group_keys: tuple[str, str] = ("stage_bucket", "length_bucket"),
) -> list[dict[str, Any]]:
    if n <= 0 or not rows:
        return []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get(group_keys[0], "")), str(row.get(group_keys[1], "")))].append(row)

    picks: list[dict[str, Any]] = []
    keys = list(grouped.keys())
    rng.shuffle(keys)
    # Round-robin sampling across strata for better coverage.
    while len(picks) < n and keys:
        remaining_keys = []
        for k in keys:
            bucket = grouped[k]
            if bucket and len(picks) < n:
                idx = rng.randrange(len(bucket))
                picks.append(bucket.pop(idx))
            if bucket:
                remaining_keys.append(k)
        keys = remaining_keys
    return picks


def build_dataset(
    pairs_path: Path,
    output_dir: Path,
    total_n: int,
    seed: int,
    low_agreement_ratio: float,
    high_agreement_ratio: float,
    top_checkpoint_ratio: float,
    gpt4o_challenged_ratio: float,
    max_judgment_files: int | None = None,
    summarization_long_path: Path | None = None,
    top_k_checkpoints: int = 3,
    dimension_weights: dict[str, float] | None = None,
) -> None:
    rng = random.Random(seed)
    json_dir = pairs_path.parent
    pairs = _read_json(pairs_path)
    votes_map = _load_judgments(json_dir, max_judgment_files=max_judgment_files)

    # Some exported pairs_table.json variants do not contain source/reference text.
    # Backfill per doc_id from summarization_long.json when available.
    doc_context: dict[str, dict[str, str]] = {}
    if summarization_long_path is None:
        auto = json_dir / "summarization_long.json"
        if auto.is_file():
            summarization_long_path = auto
    if summarization_long_path is not None and summarization_long_path.is_file():
        try:
            long_rows = _read_json(summarization_long_path)
            for r in long_rows:
                doc_id = str(r.get("doc_id", ""))
                if not doc_id or doc_id in doc_context:
                    continue
                doc_context[doc_id] = {
                    "source_text": str(r.get("source_text", "")),
                    "reference_summary": str(r.get("reference_summary", "")),
                }
        except Exception:
            # Keep selector robust; fallback will remain empty strings if file is unreadable.
            doc_context = {}
    details_map = _load_judgment_details(json_dir, max_judgment_files=max_judgment_files)
    if dimension_weights is None:
        dimension_weights = {d: 1.0 for d in DEFAULT_DIMENSIONS}

    candidates: list[dict[str, Any]] = []
    skipped_identical_summaries = 0
    for row in pairs:
        if _normalized_summary_text(str(row.get("sumleft", ""))) == _normalized_summary_text(
            str(row.get("sumright", ""))
        ):
            skipped_identical_summaries += 1
            continue
        key = _pair_key(row)
        votes = votes_map.get(key, Counter())
        item = dict(row)
        item["pair_key"] = "|".join(key)
        item["votes_left"] = votes["left"]
        item["votes_right"] = votes["right"]
        item["votes_tie"] = votes["tie"]
        item["uncertainty_score"] = _uncertainty_score(votes)
        item["easy_score"] = _easy_score(votes)
        item["edge_score"] = _edge_score(item)
        item["stage_bucket"] = _stage_for_pair(str(item.get("left", "")), str(item.get("right", "")))
        item["length_bucket"] = _length_bucket(str(item.get("source_text", "")))
        item["majority_label"] = _dominant_label(votes)
        dim_votes = _dim_votes_for_row(row, details_map)
        mean_agree, min_agree = _agreement_from_dimension_votes(dim_votes)
        item["agreement_mean"] = mean_agree
        item["agreement_min"] = min_agree
        item["top_checkpoint_match_score"] = 0.0
        item["gpt4o_challenge_score"] = _gpt4o_challenge_score(
            str(item.get("left", "")), str(item.get("right", "")), dim_votes
        )
        if not str(item.get("source_text", "")).strip():
            item["source_text"] = doc_context.get(str(item.get("doc_id", "")), {}).get("source_text", "")
        if not str(item.get("reference_summary", "")).strip():
            item["reference_summary"] = doc_context.get(str(item.get("doc_id", "")), {}).get("reference_summary", "")
        candidates.append(item)

    # Deduplicate strict pair keys.
    dedup = {}
    for row in candidates:
        dedup[row["pair_key"]] = row
    pool = list(dedup.values())
    rng.shuffle(pool)

    # IMPORTANT: use the same raw-row source as the interactive prefix viewer for ranking.
    # This avoids discrepancies caused by pair-level filtering/dedup in the sampling pool.
    raw_rows = load_geval_rows_raw(
        json_dir,
        max_files=max_judgment_files,
    )
    checkpoint_scores = compute_checkpoint_weighted_means_raw(
        raw_rows,
        dimension_weights,
    )
    top_steps = {
        ck for ck, _ in sorted(checkpoint_scores.items(), key=lambda kv: kv[1], reverse=True)[: max(1, top_k_checkpoints)]
    }
    for item in pool:
        item["top_checkpoint_match_score"] = _top_checkpoint_match_score(
            str(item.get("left", "")), str(item.get("right", "")), top_steps
        )

    low_n = int(round(total_n * low_agreement_ratio))
    high_n = int(round(total_n * high_agreement_ratio))
    top_n = int(round(total_n * top_checkpoint_ratio))
    gpt4o_n = total_n - low_n - high_n - top_n
    # Keep explicit ratio target while ensuring exact total_n.
    gpt4o_n = max(0, gpt4o_n)

    selected: list[dict[str, Any]] = []
    used = set()

    def take_rows(rows: list[dict[str, Any]], k: int, tag: str) -> list[dict[str, Any]]:
        out = []
        for row in rows:
            if len(out) >= k:
                break
            key = row["pair_key"]
            if key in used:
                continue
            row["selection_bucket"] = tag
            used.add(key)
            out.append(row)
        return out

    non_gpt4o_pool = [
        r
        for r in pool
        if not _is_gpt4o_pair(str(r.get("left", "")), str(r.get("right", "")))
    ]
    gpt4o_pool = [
        r
        for r in pool
        if _is_gpt4o_pair(str(r.get("left", "")), str(r.get("right", "")))
    ]

    low_sorted = sorted(non_gpt4o_pool, key=lambda r: (r["agreement_mean"], r["agreement_min"]))
    high_sorted = sorted(non_gpt4o_pool, key=lambda r: (r["agreement_mean"], r["agreement_min"]), reverse=True)
    top_sorted = sorted(non_gpt4o_pool, key=lambda r: r["top_checkpoint_match_score"], reverse=True)
    gpt4o_sorted = sorted(gpt4o_pool, key=lambda r: r["gpt4o_challenge_score"], reverse=True)

    selected.extend(take_rows(low_sorted, low_n, "low_agreement"))
    selected.extend(take_rows(high_sorted, high_n, "high_agreement"))
    selected.extend(
        take_rows([r for r in top_sorted if r["top_checkpoint_match_score"] > 0], top_n, "top_checkpoint_matchup")
    )
    selected.extend(
        take_rows([r for r in gpt4o_sorted if r["gpt4o_challenge_score"] > 0], gpt4o_n, "gpt4o_challenged")
    )

    # Backfill any shortfall with stratified remaining pairs.
    if len(selected) < total_n:
        remaining = [
            r
            for r in non_gpt4o_pool
            if r["pair_key"] not in used
        ]
        fill_n = total_n - len(selected)
        selected.extend(
            take_rows(_stratified_pick(remaining, fill_n, rng), fill_n, "fallback_representative")
        )

    # Final trim if rounding/dup logic overshoots.
    selected = selected[:total_n]

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "human_eval_dataset.jsonl"
    answer_key_path = output_dir / "human_eval_answer_key.csv"
    metadata_path = output_dir / "selection_metadata.json"

    answer_rows = []
    with dataset_path.open("w", encoding="utf-8") as out:
        for i, row in enumerate(selected, start=1):
            side_a_is_left = rng.random() < 0.5
            summary_a = row.get("sumleft", "") if side_a_is_left else row.get("sumright", "")
            summary_b = row.get("sumright", "") if side_a_is_left else row.get("sumleft", "")

            item_id = f"human_eval_{i:04d}"
            record = {
                "item_id": item_id,
                "doc_id": row.get("doc_id"),
                "source_text": row.get("source_text", ""),
                "reference_summary": row.get("reference_summary", ""),
                "summary_a": summary_a,
                "summary_b": summary_b,
                "selection_bucket": row.get("selection_bucket"),
                "stage_bucket": row.get("stage_bucket"),
                "length_bucket": row.get("length_bucket"),
                "pair_hash": hashlib.md5(str(row.get("pair_key")).encode("utf-8")).hexdigest()[:12],
                "llm_decisions": _dim_votes_for_row(row, details_map),
                "annotation_instructions": "Choose: A better / B better / Tie / Both bad",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

            majority = row.get("majority_label", "tie")
            if majority == "left":
                expected = "A" if side_a_is_left else "B"
            elif majority == "right":
                expected = "B" if side_a_is_left else "A"
            else:
                expected = "Tie"

            per_dim_counts = _dimension_vote_counts(_dim_votes_for_row(row, details_map))
            answer_rows.append(
                {
                    "item_id": item_id,
                    "doc_id": row.get("doc_id"),
                    "left_model": row.get("left"),
                    "right_model": row.get("right"),
                    "votes_left": row.get("votes_left"),
                    "votes_right": row.get("votes_right"),
                    "votes_tie": row.get("votes_tie"),
                    "votes_completeness_left": per_dim_counts.get("completeness", {}).get("left", 0),
                    "votes_completeness_right": per_dim_counts.get("completeness", {}).get("right", 0),
                    "votes_completeness_tie": per_dim_counts.get("completeness", {}).get("tie", 0),
                    "votes_correctness_left": per_dim_counts.get("correctness", {}).get("left", 0),
                    "votes_correctness_right": per_dim_counts.get("correctness", {}).get("right", 0),
                    "votes_correctness_tie": per_dim_counts.get("correctness", {}).get("tie", 0),
                    "votes_faithfulness_left": per_dim_counts.get("faithfulness", {}).get("left", 0),
                    "votes_faithfulness_right": per_dim_counts.get("faithfulness", {}).get("right", 0),
                    "votes_faithfulness_tie": per_dim_counts.get("faithfulness", {}).get("tie", 0),
                    "votes_hygiene_left": per_dim_counts.get("hygiene", {}).get("left", 0),
                    "votes_hygiene_right": per_dim_counts.get("hygiene", {}).get("right", 0),
                    "votes_hygiene_tie": per_dim_counts.get("hygiene", {}).get("tie", 0),
                    "votes_newsworthiness_left": per_dim_counts.get("newsworthiness", {}).get("left", 0),
                    "votes_newsworthiness_right": per_dim_counts.get("newsworthiness", {}).get("right", 0),
                    "votes_newsworthiness_tie": per_dim_counts.get("newsworthiness", {}).get("tie", 0),
                    "majority_label_raw": majority,
                    "expected_label_on_blinded_sides": expected,
                    "selection_bucket": row.get("selection_bucket"),
                }
            )

    with answer_key_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(answer_rows[0].keys()) if answer_rows else [])
        if answer_rows:
            writer.writeheader()
            writer.writerows(answer_rows)

    bucket_counts = Counter(r.get("selection_bucket", "unknown") for r in selected)
    stage_counts = Counter(r.get("stage_bucket", "unknown") for r in selected)
    length_counts = Counter(r.get("length_bucket", "unknown") for r in selected)
    metadata = {
        "seed": seed,
        "total_requested": total_n,
        "total_selected": len(selected),
        "excluded_identical_summary_pairs": skipped_identical_summaries,
        "ratios": {
            "low_agreement": low_agreement_ratio,
            "high_agreement": high_agreement_ratio,
            "top_checkpoint_matchup": top_checkpoint_ratio,
            "gpt4o_challenged": gpt4o_challenged_ratio,
        },
        "counts_by_bucket": dict(bucket_counts),
        "counts_by_stage": dict(stage_counts),
        "counts_by_length": dict(length_counts),
        "inputs": {
            "pairs_table_json": str(pairs_path),
            "geval_json_dir": str(json_dir),
            "dimension_weights": dimension_weights,
            "top_k_checkpoints": top_k_checkpoints,
            "auto_top_checkpoints": sorted(top_steps),
            "auto_top_checkpoint_scores": {
                str(k): v for k, v in sorted(checkpoint_scores.items(), key=lambda kv: kv[1], reverse=True)
            },
            "non_gpt4o_only_buckets": [
                "low_agreement",
                "high_agreement",
                "top_checkpoint_matchup",
                "fallback_representative",
            ],
        },
        "outputs": {
            "dataset_jsonl": str(dataset_path),
            "answer_key_csv": str(answer_key_path),
            "selection_metadata_json": str(metadata_path),
        },
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    best_ck = sorted(checkpoint_scores.items(), key=lambda kv: kv[1], reverse=True)
    if best_ck:
        top_ck = best_ck[0]
        print(f"Best selected checkpoint (auto): checkpoint-{top_ck[0]} score={top_ck[1]:.6f}")
        print(
            "Top checkpoints (auto):",
            ", ".join(f"checkpoint-{k} ({v:.4f})" for k, v in best_ck[: max(1, top_k_checkpoints)]),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select pairwise items for human evaluation.")
    parser.add_argument(
        "--pairs-table",
        type=Path,
        default=Path(".deepeval/geval_exports/llama-2-13b/json/pairs_table.json"),
        help="Path to pairs_table.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("human evaluation"),
        help="Output directory for human eval artifacts",
    )
    parser.add_argument("--total", type=int, default=100, help="Total number of human-eval items.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--low-agreement-ratio", type=float, default=0.30)
    parser.add_argument("--high-agreement-ratio", type=float, default=0.20)
    parser.add_argument("--top-checkpoint-ratio", type=float, default=0.30)
    parser.add_argument("--gpt4o-challenged-ratio", type=float, default=0.20)
    parser.add_argument(
        "--max-judgment-files",
        type=int,
        default=None,
        help="Optional cap on number of geval__*__*.json files to load (for faster runs).",
    )
    parser.add_argument(
        "--summarization-long",
        type=Path,
        default=None,
        help="Optional path to summarization_long.json for backfilling source/reference text by doc_id.",
    )
    parser.add_argument(
        "--top-k-checkpoints",
        type=int,
        default=3,
        help="Number of top checkpoints to discover automatically for top-checkpoint-matchup bucket.",
    )
    parser.add_argument(
        "--dimension-weights",
        type=str,
        default="",
        help=(
            "Comma-separated weights, e.g. "
            "'faithfulness=1,correctness=1,completeness=1,newsworthiness=1,hygiene=1'. "
            "Missing dimensions default to 1."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratios = [
        args.low_agreement_ratio,
        args.high_agreement_ratio,
        args.top_checkpoint_ratio,
        args.gpt4o_challenged_ratio,
    ]
    if args.total <= 0:
        raise ValueError("--total must be > 0")
    if any(r < 0 for r in ratios):
        raise ValueError("All ratios must be >= 0")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    if not args.pairs_table.is_file():
        raise FileNotFoundError(f"pairs_table.json not found: {args.pairs_table}")

    build_dataset(
        pairs_path=args.pairs_table,
        output_dir=args.output_dir,
        total_n=args.total,
        seed=args.seed,
        low_agreement_ratio=args.low_agreement_ratio,
        high_agreement_ratio=args.high_agreement_ratio,
        top_checkpoint_ratio=args.top_checkpoint_ratio,
        gpt4o_challenged_ratio=args.gpt4o_challenged_ratio,
        max_judgment_files=args.max_judgment_files,
        summarization_long_path=args.summarization_long,
        top_k_checkpoints=args.top_k_checkpoints,
        dimension_weights=_parse_dimension_weights(args.dimension_weights),
    )
    print(f"Done. Dataset written to: {args.output_dir}")


if __name__ == "__main__":
    main()
