#!/usr/bin/env python3
"""Build human-evaluation batches without modifying the pairwise-eval pipeline.

The generated design treats one document as one annotation block. Each block
contains three pairwise summary comparisons among four summaries:

* GPT4o-mini reference summary
* Gemma checkpoint summary
* Viking checkpoint summary
* GPT4o-mini elaborate/newsworthiness-prompt summary

Outputs are written under ``human_eval_batching/outputs`` by default.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
# Data moved out of the repo (2026-06) into the shared OneDrive folder. Override with
# CHECKPOINT_SELECTION_DATA_DIR if your OneDrive root or the dataset snapshot name differs.
DATA_ROOT = Path(
    os.environ.get("CHECKPOINT_SELECTION_DATA_DIR")
    or (
        Path(os.environ.get("ONEDRIVE", str(Path.home() / "OneDrive")))
        / "Shared"
        / "Demokratibasen-UiB-Ide"
        / "EvaluationDatasets"
        / "CheckpointSelection"
        / "Data_202606"
    )
)
DEFAULT_EVAL_DIR = DATA_ROOT / "eval" / "2500-human-cadidates"
DEFAULT_EXPORT_JSON_DIR = REPO_ROOT / ".deepeval" / "geval_exports" / "2500-human-cadidates" / "json"
DEFAULT_TEST_JSONL = DATA_ROOT / "human" / "149978_text_summary_examples_test.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

REFERENCE_MODEL_ID = "GPT4o-mini"
ANNOTATORS = ("A", "B", "C", "D", "E", "F")
BLOCKS_PER_BATCH = 24
PAIRS_PER_BLOCK = 3
SUMMARY_POSITIONS_PER_BLOCK = PAIRS_PER_BLOCK * 2
PAIR_HIGH_RANK_CUTOFF = 0.80

PAIR_SELECTION_CRITERIA: tuple[str, ...] = (
    "low_agreement",
    "high_agreement",
    "gpt4o_challenged",
    "elaborate_interesting",
)


MODEL_ALIASES: dict[str, str] = {
    REFERENCE_MODEL_ID: "gpt4o_baseline",
    "gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples": "gemma_cp2500",
    "viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples": "viking_cp3500",
    "gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples": "gpt4o_elaborate",
}


TRIPLE_CYCLE: tuple[tuple[str, str, str], ...] = (
    ("A", "B", "C"),
    ("D", "E", "F"),
    ("A", "B", "D"),
    ("C", "E", "F"),
    ("A", "B", "E"),
    ("C", "D", "F"),
    ("A", "B", "F"),
    ("C", "D", "E"),
    ("A", "C", "D"),
    ("B", "E", "F"),
    ("A", "C", "E"),
    ("B", "D", "F"),
    ("A", "C", "F"),
    ("B", "D", "E"),
    ("A", "D", "E"),
    ("B", "C", "F"),
    ("A", "D", "F"),
    ("B", "C", "E"),
    ("A", "E", "F"),
    ("B", "C", "D"),
)


@dataclass(frozen=True)
class SummaryRecord:
    document_id: str
    summary_id: str
    model_id: str
    model_alias: str
    summary_text: str


@dataclass(frozen=True)
class PairSignal:
    doc_id: str
    pair_key: tuple[str, str]
    n_votes: int
    vote_entropy: float
    top_vote_share: float
    tie_rate: float
    gpt4o_loss_rate: float
    elaborate_win_rate: float
    criterion_scores: dict[str, float]
    criterion_ranks: dict[str, float]
    high_ranked_criteria: tuple[str, ...]
    priority_score: float


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sanitize_for_labelstudio(value: Any) -> Any:
    """Remove characters PostgreSQL/Label Studio cannot store in text fields."""
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, list):
        return [sanitize_for_labelstudio(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_for_labelstudio(item) for key, item in value.items()}
    return value


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(sanitize_for_labelstudio(row), ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_for_labelstudio(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_eval_candidates(eval_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    files = sorted(eval_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL candidate files found under {eval_dir}")

    first_rows = read_jsonl(files[0])
    input_order = [str(r["input_text"]) for r in first_rows]
    prompt_order = [str(r["prompt"]) for r in first_rows]
    reference_order = [str(r["reference"]) for r in first_rows]

    candidate_predictions: dict[str, list[str]] = {}
    for path in files:
        model_id = path.stem
        rows = read_jsonl(path)
        if len(rows) != len(first_rows):
            raise ValueError(f"{path.name}: row count {len(rows)} != {len(first_rows)}")
        if [str(r["input_text"]) for r in rows] != input_order:
            raise ValueError(f"{path.name}: input_text order differs from {files[0].name}")
        candidate_predictions[model_id] = [str(r.get("prediction", "")) for r in rows]

    docs: list[dict[str, Any]] = []
    for i, input_text in enumerate(input_order, start=1):
        doc = {
            "document_id": f"doc_{i}",
            "source_text": input_text,
            "prompt": prompt_order[i - 1],
            "reference_summary": reference_order[i - 1],
            "summaries": {
                REFERENCE_MODEL_ID: reference_order[i - 1],
                **{model_id: preds[i - 1] for model_id, preds in candidate_predictions.items()},
            },
        }
        docs.append(doc)

    model_ids = [REFERENCE_MODEL_ID] + list(candidate_predictions)
    missing_alias = [m for m in model_ids if m not in MODEL_ALIASES]
    if missing_alias:
        raise ValueError(f"Missing MODEL_ALIASES for: {missing_alias}")
    return docs, model_ids


def load_test_metadata(test_jsonl: Path) -> dict[str, dict[str, Any]]:
    if not test_jsonl.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(test_jsonl):
        input_text = str(row.get("input", "")).strip()
        meta = dict(row.get("metadata") or {})
        out[input_text] = meta
    return out


def parse_geval_filename(path: Path) -> tuple[str, str] | None:
    stem = path.stem
    if not stem.startswith("geval__"):
        return None
    body = stem[len("geval__") :]
    for dim in ("relevance", "consistency", "newsworthiness", "hygiene"):
        suffix = "__" + dim
        if body.endswith(suffix):
            return body[: -len(suffix)].replace("__", "/"), dim
    return None


def entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for n in counts.values():
        if n:
            p = n / total
            h -= p * math.log2(p)
    return h


def load_doc_selection_features(export_json_dir: Path, model_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Compute document-level features from available G-Eval rows.

    These features are selection signals only. They are not hard constraints.
    """
    features: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "judge_votes": 0,
            "pair_rows": 0,
            "elaborate_wins": 0,
            "gpt4o_losses": 0,
            "tie_votes": 0,
            "vote_counts": Counter(),
            "dimension_counts": Counter(),
        }
    )
    model_set = set(model_ids)
    elaborate_model = next((m for m in model_ids if MODEL_ALIASES[m] == "gpt4o_elaborate"), None)

    if not export_json_dir.is_dir():
        return {}

    for path in sorted(export_json_dir.glob("geval__*.json")):
        parsed = parse_geval_filename(path)
        if parsed is None:
            continue
        _, dimension = parsed
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if row.get("left") not in model_set or row.get("right") not in model_set:
                continue
            doc_id = str(row.get("doc_id", ""))
            if not doc_id:
                continue
            f = features[doc_id]
            f["judge_votes"] += 1
            f["pair_rows"] += 1
            f["dimension_counts"][dimension] += 1
            chosen = row.get("chosen")
            if chosen is None:
                f["tie_votes"] += 1
                f["vote_counts"]["tie"] += 1
            else:
                f["vote_counts"][str(chosen)] += 1
                if elaborate_model is not None and chosen == elaborate_model:
                    f["elaborate_wins"] += 1
            if REFERENCE_MODEL_ID in (row.get("left"), row.get("right")) and chosen not in (None, REFERENCE_MODEL_ID):
                f["gpt4o_losses"] += 1

    out: dict[str, dict[str, Any]] = {}
    for doc_id, f in features.items():
        vote_counts = f["vote_counts"]
        pair_rows = int(f["pair_rows"])
        e = entropy(vote_counts)
        out[doc_id] = {
            "pair_rows": pair_rows,
            "judge_votes": int(f["judge_votes"]),
            "vote_entropy": round(e, 6),
            "tie_rate": round(f["tie_votes"] / pair_rows, 6) if pair_rows else 0.0,
            "gpt4o_loss_rate": round(f["gpt4o_losses"] / pair_rows, 6) if pair_rows else 0.0,
            "elaborate_win_rate": round(f["elaborate_wins"] / pair_rows, 6) if pair_rows else 0.0,
            "dimension_counts": dict(f["dimension_counts"]),
        }
    return out


def load_pair_selection_signals(export_json_dir: Path, model_ids: list[str]) -> dict[str, dict[tuple[str, str], PairSignal]]:
    """Rank individual document/model-pair comparisons by selection criteria.

    This is the main selection signal for the block design. Documents inherit
    priority from the pairwise comparisons that made them interesting, and the
    block builder later tries to include those exact high-ranked pairs.
    """
    model_set = set(model_ids)
    elaborate_model = next((m for m in model_ids if MODEL_ALIASES[m] == "gpt4o_elaborate"), None)
    raw: dict[tuple[str, tuple[str, str]], dict[str, Any]] = defaultdict(
        lambda: {
            "vote_counts": Counter(),
            "tie_votes": 0,
            "gpt4o_losses": 0,
            "elaborate_wins": 0,
            "dimension_counts": Counter(),
        }
    )

    if not export_json_dir.is_dir():
        return {}

    for path in sorted(export_json_dir.glob("geval__*.json")):
        parsed = parse_geval_filename(path)
        if parsed is None:
            continue
        _, dimension = parsed
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            left = row.get("left")
            right = row.get("right")
            if left not in model_set or right not in model_set:
                continue
            doc_id = str(row.get("doc_id", ""))
            if not doc_id:
                continue
            pair_key = tuple(sorted((str(left), str(right))))
            rec = raw[(doc_id, pair_key)]
            rec["dimension_counts"][dimension] += 1
            chosen = row.get("chosen")
            if chosen is None:
                rec["tie_votes"] += 1
                rec["vote_counts"]["tie"] += 1
            else:
                rec["vote_counts"][str(chosen)] += 1
                if elaborate_model is not None and chosen == elaborate_model:
                    rec["elaborate_wins"] += 1
            if REFERENCE_MODEL_ID in pair_key and chosen not in (None, REFERENCE_MODEL_ID):
                rec["gpt4o_losses"] += 1

    scored: dict[tuple[str, tuple[str, str]], dict[str, Any]] = {}
    for key, rec in raw.items():
        vote_counts = rec["vote_counts"]
        n_votes = sum(vote_counts.values())
        if n_votes == 0:
            continue
        ent = entropy(vote_counts)
        top_share = max(vote_counts.values()) / n_votes
        tie_rate = rec["tie_votes"] / n_votes
        gpt4o_loss_rate = rec["gpt4o_losses"] / n_votes
        elaborate_win_rate = rec["elaborate_wins"] / n_votes
        scored[key] = {
            "n_votes": n_votes,
            "vote_entropy": ent,
            "top_vote_share": top_share,
            "tie_rate": tie_rate,
            "gpt4o_loss_rate": gpt4o_loss_rate,
            "elaborate_win_rate": elaborate_win_rate,
            "dimension_counts": dict(rec["dimension_counts"]),
            "criterion_scores": {
                "low_agreement": ent,
                "high_agreement": top_share * (1.0 - tie_rate),
                "gpt4o_challenged": gpt4o_loss_rate,
                "elaborate_interesting": elaborate_win_rate,
            },
        }

    # Convert raw criterion scores to percentile-like ranks per criterion.
    ranks: dict[tuple[str, tuple[str, str]], dict[str, float]] = {k: {} for k in scored}
    all_keys = list(scored)
    denom = max(1, len(all_keys) - 1)
    for criterion in PAIR_SELECTION_CRITERIA:
        ordered = sorted(
            all_keys,
            key=lambda k: (
                float(scored[k]["criterion_scores"].get(criterion, 0.0)),
                float(scored[k]["vote_entropy"]),
                k[0],
                k[1],
            ),
            reverse=True,
        )
        for rank_idx, key in enumerate(ordered):
            ranks[key][criterion] = 1.0 - (rank_idx / denom)

    out: dict[str, dict[tuple[str, str], PairSignal]] = defaultdict(dict)
    for (doc_id, pair_key), rec in scored.items():
        criterion_ranks = ranks[(doc_id, pair_key)]
        high = tuple(c for c in PAIR_SELECTION_CRITERIA if criterion_ranks.get(c, 0.0) >= PAIR_HIGH_RANK_CUTOFF)
        priority = max(criterion_ranks.values(), default=0.0) + 0.15 * max(0, len(high) - 1)
        out[doc_id][pair_key] = PairSignal(
            doc_id=doc_id,
            pair_key=pair_key,
            n_votes=int(rec["n_votes"]),
            vote_entropy=round(float(rec["vote_entropy"]), 6),
            top_vote_share=round(float(rec["top_vote_share"]), 6),
            tie_rate=round(float(rec["tie_rate"]), 6),
            gpt4o_loss_rate=round(float(rec["gpt4o_loss_rate"]), 6),
            elaborate_win_rate=round(float(rec["elaborate_win_rate"]), 6),
            criterion_scores={k: round(float(v), 6) for k, v in rec["criterion_scores"].items()},
            criterion_ranks={k: round(float(v), 6) for k, v in criterion_ranks.items()},
            high_ranked_criteria=high,
            priority_score=round(float(priority), 6),
        )
    return dict(out)


def pair_signal_to_json(sig: PairSignal) -> dict[str, Any]:
    return {
        "models": list(sig.pair_key),
        "model_aliases": [MODEL_ALIASES[m] for m in sig.pair_key],
        "n_votes": sig.n_votes,
        "vote_entropy": sig.vote_entropy,
        "top_vote_share": sig.top_vote_share,
        "tie_rate": sig.tie_rate,
        "gpt4o_loss_rate": sig.gpt4o_loss_rate,
        "elaborate_win_rate": sig.elaborate_win_rate,
        "criterion_scores": sig.criterion_scores,
        "criterion_ranks": sig.criterion_ranks,
        "high_ranked_criteria": list(sig.high_ranked_criteria),
        "priority_score": sig.priority_score,
    }


def build_doc_features_from_pair_signals(pair_signals: dict[tuple[str, str], PairSignal]) -> dict[str, Any]:
    if not pair_signals:
        return {}
    signals = sorted(pair_signals.values(), key=lambda s: s.priority_score, reverse=True)
    high_signals = [s for s in signals if s.high_ranked_criteria]
    criteria_counter: Counter[str] = Counter()
    for sig in high_signals:
        criteria_counter.update(sig.high_ranked_criteria)
    return {
        "pair_count": len(signals),
        "high_ranked_pair_count": len(high_signals),
        "document_priority_score": round(
            sum(sig.priority_score for sig in high_signals)
            + (signals[0].priority_score if signals else 0.0),
            6,
        ),
        "top_pair_priority_score": signals[0].priority_score,
        "top_pair_models": list(signals[0].pair_key),
        "top_pair_criteria": list(signals[0].high_ranked_criteria),
        "criteria_hit_counts": dict(criteria_counter),
        "candidate_pairs": [pair_signal_to_json(sig) for sig in signals],
    }


def bucket_for_features(features: dict[str, Any] | None) -> str:
    if not features:
        return "not_evaluated"
    hit_counts = Counter(features.get("criteria_hit_counts") or {})
    if hit_counts.get("low_agreement", 0) > 0:
        return "high_llm_disagreement"
    if hit_counts.get("gpt4o_challenged", 0) > 0:
        return "gpt4o_challenged"
    if hit_counts.get("elaborate_interesting", 0) > 0:
        return "elaborate_interesting"
    if hit_counts.get("high_agreement", 0) > 0:
        return "high_agreement"
    return "representative"


def assign_ranked_selection_buckets(selected_docs: list[dict[str, Any]]) -> None:
    """Assign exclusive document-level buckets by rank within the selected set.

    Absolute entropy thresholds are brittle here because four-model comparisons
    with many judges often produce high entropy for most documents. Quota-based
    labels make the generated design easier to audit while still preserving the
    intended sampling signals.
    """
    n = len(selected_docs)
    quotas = {
        "high_llm_disagreement": round(n * 0.40),
        "gpt4o_challenged": round(n * 0.25),
        "elaborate_interesting": round(n * 0.20),
    }
    assigned: set[str] = set()

    def take(bucket: str, key: str, limit: int) -> None:
        candidates = [d for d in selected_docs if d["document_id"] not in assigned and d.get("selection_features")]
        candidates.sort(
            key=lambda d: (
                int(((d.get("selection_features") or {}).get("criteria_hit_counts") or {}).get(key, 0)),
                float((d.get("selection_features") or {}).get("document_priority_score", 0.0)),
                -int(d["document_index"]),
            ),
            reverse=True,
        )
        for doc in candidates[:limit]:
            doc["selection_bucket"] = bucket
            assigned.add(doc["document_id"])

    take("high_llm_disagreement", "low_agreement", quotas["high_llm_disagreement"])
    take("gpt4o_challenged", "gpt4o_challenged", quotas["gpt4o_challenged"])
    take("elaborate_interesting", "elaborate_interesting", quotas["elaborate_interesting"])

    for doc in selected_docs:
        if doc["document_id"] not in assigned:
            doc["selection_bucket"] = "representative"


def selection_score(doc: dict[str, Any]) -> tuple[float, float, float, int]:
    features = doc.get("selection_features") or {}
    hit_counts = features.get("criteria_hit_counts") or {}
    return (
        float(features.get("document_priority_score", 0.0)),
        float(features.get("high_ranked_pair_count", 0.0)),
        float(sum(hit_counts.values())),
        -int(doc["document_index"]),
    )


def select_documents(docs: list[dict[str, Any]], n_blocks: int, seed: int) -> list[dict[str, Any]]:
    """Select documents deterministically.

    Documents with available LLM-judge signals are prioritized by disagreement
    and challenge signals. Remaining slots are filled from the original order.
    """
    evaluated = [d for d in docs if d.get("selection_features")]
    unevaluated = [d for d in docs if not d.get("selection_features")]
    rng = random.Random(seed)
    rng.shuffle(evaluated)
    evaluated.sort(key=selection_score, reverse=True)

    selected = evaluated[:n_blocks]
    if len(selected) < n_blocks:
        selected_ids = {d["document_id"] for d in selected}
        selected.extend(d for d in unevaluated if d["document_id"] not in selected_ids)
    selected = selected[:n_blocks]
    selected.sort(key=lambda d: d["document_index"])
    return selected


def balance_documents_across_batches(
    selected_docs: list[dict[str, Any]],
    n_batches: int,
    blocks_per_batch: int,
) -> list[dict[str, Any]]:
    """Order selected documents so every batch has a similar bucket mix."""
    if len(selected_docs) != n_batches * blocks_per_batch:
        raise ValueError("Selected document count must equal n_batches * blocks_per_batch")

    preferred_bucket_order = (
        "high_llm_disagreement",
        "gpt4o_challenged",
        "elaborate_interesting",
        "representative",
        "high_agreement",
        "not_evaluated",
    )
    docs_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in selected_docs:
        docs_by_bucket[doc["selection_bucket"]].append(doc)

    for docs in docs_by_bucket.values():
        docs.sort(key=selection_score, reverse=True)

    bucket_counts = Counter(doc["selection_bucket"] for doc in selected_docs)
    bucket_order = [b for b in preferred_bucket_order if b in bucket_counts]
    bucket_order.extend(sorted(b for b in bucket_counts if b not in set(bucket_order)))
    bucket_rank = {bucket: i for i, bucket in enumerate(bucket_order)}
    assigned_counts: Counter[str] = Counter()
    balanced: list[dict[str, Any]] = []

    for batch_idx in range(n_batches):
        batch_docs: list[dict[str, Any]] = []
        while len(batch_docs) < blocks_per_batch:
            available_buckets = [bucket for bucket in bucket_order if docs_by_bucket[bucket]]
            if not available_buckets:
                raise ValueError("Ran out of selected documents while balancing batches")

            def bucket_need(bucket: str) -> tuple[float, int, int]:
                ideal_cumulative = bucket_counts[bucket] * (batch_idx + 1) / n_batches
                deficit = ideal_cumulative - assigned_counts[bucket]
                remaining = bucket_counts[bucket] - assigned_counts[bucket]
                return (deficit, remaining, -bucket_rank[bucket])

            bucket = max(available_buckets, key=bucket_need)
            assigned_counts[bucket] += 1
            batch_docs.append(docs_by_bucket[bucket].pop(0))

        batch_docs.sort(key=lambda d: d["document_index"])
        balanced.extend(batch_docs)

    return balanced


def all_valid_pair_templates(model_ids: list[str]) -> list[tuple[tuple[str, str], tuple[str, str], tuple[str, str]]]:
    all_pairs = list(itertools.combinations(model_ids, 2))
    templates = []
    for combo in itertools.combinations(all_pairs, PAIRS_PER_BLOCK):
        counts = Counter(itertools.chain.from_iterable(combo))
        if set(counts) != set(model_ids):
            continue
        if any(v > 2 for v in counts.values()):
            continue
        templates.append(tuple(combo))
    return templates


def choose_oriented_pairs(
    model_ids: list[str],
    rng: random.Random,
    pair_counts: Counter[tuple[str, str]],
    left_counts: Counter[str],
    summary_position_counts: Counter[str],
    priority_signals: dict[tuple[str, str], PairSignal] | None = None,
) -> list[tuple[str, str]]:
    templates = all_valid_pair_templates(model_ids)
    if not templates:
        raise ValueError("No valid pair templates for model set")
    priority_signals = priority_signals or {}

    def template_cost(template: tuple[tuple[str, str], ...]) -> tuple[float, int, int, int, float]:
        template_keys = {tuple(sorted(pair)) for pair in template}
        included_priority = sum(
            priority_signals[k].priority_score
            for k in template_keys
            if k in priority_signals and priority_signals[k].high_ranked_criteria
        )
        included_high_pairs = sum(1 for k in template_keys if k in priority_signals and priority_signals[k].high_ranked_criteria)
        pair_cost = sum(pair_counts[tuple(sorted(pair))] for pair in template)
        exposure = Counter(itertools.chain.from_iterable(template))
        exposure_cost = sum((summary_position_counts[m] + exposure[m]) ** 2 for m in model_ids)
        jitter = rng.random()
        return (-included_priority, -included_high_pairs, pair_cost, exposure_cost, jitter)

    template = min(templates, key=template_cost)
    oriented: list[tuple[str, str]] = []
    for a, b in template:
        if left_counts[a] < left_counts[b]:
            left, right = a, b
        elif left_counts[b] < left_counts[a]:
            left, right = b, a
        else:
            left, right = (a, b) if rng.random() < 0.5 else (b, a)
        oriented.append((left, right))
        pair_counts[tuple(sorted((a, b)))] += 1
        left_counts[left] += 1
        summary_position_counts[left] += 1
        summary_position_counts[right] += 1
    rng.shuffle(oriented)
    return oriented


def make_summary_id(document_id: str, model_id: str) -> str:
    return f"{document_id}__{MODEL_ALIASES[model_id]}"


def build_blocks(
    selected_docs: list[dict[str, Any]],
    model_ids: list[str],
    seed: int,
    start_block_number: int = 1,
) -> tuple[list[dict[str, Any]], Counter, Counter]:
    rng = random.Random(seed)
    pair_counts: Counter[tuple[str, str]] = Counter()
    left_counts: Counter[str] = Counter()
    summary_position_counts: Counter[str] = Counter()
    blocks = []

    for i, doc in enumerate(selected_docs, start=start_block_number):
        block_id = f"block_{i:03d}"
        oriented_pairs = choose_oriented_pairs(
            model_ids,
            rng,
            pair_counts=pair_counts,
            left_counts=left_counts,
            summary_position_counts=summary_position_counts,
            priority_signals=doc.get("pair_selection_signals") or {},
        )
        pairs = []
        for j, (left_model, right_model) in enumerate(oriented_pairs, start=1):
            pair_key = tuple(sorted((left_model, right_model)))
            signal = (doc.get("pair_selection_signals") or {}).get(pair_key)
            pairs.append(
                {
                    "pair_id": f"{block_id}_pair_{j}",
                    "left_summary_id": make_summary_id(doc["document_id"], left_model),
                    "right_summary_id": make_summary_id(doc["document_id"], right_model),
                    "left_model_id": left_model,
                    "right_model_id": right_model,
                    "left_model_alias": MODEL_ALIASES[left_model],
                    "right_model_alias": MODEL_ALIASES[right_model],
                    "selection_signal": pair_signal_to_json(signal) if signal is not None else None,
                    "included_due_to_high_rank": bool(signal and signal.high_ranked_criteria),
                }
            )
        blocks.append(
            {
                "block_id": block_id,
                "document_id": doc["document_id"],
                "selection_bucket": doc["selection_bucket"],
                "pairs": pairs,
            }
        )
    return blocks, pair_counts, left_counts


def build_batches(blocks: list[dict[str, Any]], blocks_per_batch: int, start_batch_number: int = 1) -> list[dict[str, Any]]:
    batches = []
    for i in range(0, len(blocks), blocks_per_batch):
        chunk = blocks[i : i + blocks_per_batch]
        batch_number = start_batch_number + len(batches)
        batches.append(
            {
                "batch_id": f"batch_{batch_number:02d}",
                "phase": "pilot_192" if batch_number <= 8 else "expansion_576",
                "block_ids": [b["block_id"] for b in chunk],
            }
        )
    return batches


def build_assignments(batches: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    assignments = []
    for batch in batches:
        i = int(batch["batch_id"].removeprefix("batch_")) - 1
        annotators = TRIPLE_CYCLE[i % len(TRIPLE_CYCLE)]
        for annotator in annotators:
            block_order = list(batch["block_ids"])
            random.Random(f"{seed}:{batch['batch_id']}:{annotator}").shuffle(block_order)
            assignments.append(
                {
                    "assignment_id": f"{batch['batch_id']}__ann_{annotator}",
                    "batch_id": batch["batch_id"],
                    "annotator_id": annotator,
                    "block_order": block_order,
                }
            )
    return assignments


def count_block_exposures(blocks: list[dict[str, Any]]) -> tuple[Counter[tuple[str, str]], Counter[str]]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    left_counts: Counter[str] = Counter()
    for block in blocks:
        for pair in block["pairs"]:
            pair_counts[tuple(sorted((pair["left_model_id"], pair["right_model_id"])))] += 1
            left_counts[pair["left_model_id"]] += 1
    return pair_counts, left_counts


def validate_unique_documents(blocks: list[dict[str, Any]]) -> None:
    doc_counts = Counter(block["document_id"] for block in blocks)
    duplicates = sorted(doc_id for doc_id, count in doc_counts.items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:10])
        raise ValueError(f"Documents may only appear in one block. Duplicate document ids: {preview}")


def load_excluded_document_ids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        documents_path = path / "documents.jsonl" if path.is_dir() else path
        if not documents_path.is_file():
            raise FileNotFoundError(f"Could not find documents.jsonl for exclusion path: {path}")
        for row in read_jsonl(documents_path):
            doc_id = row.get("document_id")
            if doc_id:
                excluded.add(str(doc_id))
    return excluded


def build_labelstudio_tasks(
    assignments: list[dict[str, Any]],
    block_by_id: dict[str, dict[str, Any]],
    doc_by_id: dict[str, dict[str, Any]],
    summary_by_id: dict[str, SummaryRecord],
) -> list[dict[str, Any]]:
    tasks = []
    for assignment in assignments:
        for position, block_id in enumerate(assignment["block_order"], start=1):
            block = block_by_id[block_id]
            doc = doc_by_id[block["document_id"]]
            task: dict[str, Any] = {
                "assignment_id": assignment["assignment_id"],
                "annotator_id": assignment["annotator_id"],
                "batch_id": assignment["batch_id"],
                "block_order_position": position,
                "block_id": block_id,
                "document_id": doc["document_id"],
                "source_doc_id": doc.get("source_doc_id"),
                "source_text": doc["source_text"],
                "selection_bucket": doc["selection_bucket"],
            }
            for pair_idx, pair in enumerate(block["pairs"], start=1):
                left = summary_by_id[pair["left_summary_id"]]
                right = summary_by_id[pair["right_summary_id"]]
                task[f"pair_{pair_idx}_id"] = pair["pair_id"]
                task[f"pair_{pair_idx}_left"] = left.summary_text
                task[f"pair_{pair_idx}_right"] = right.summary_text
                # Hidden metadata for analysis; do not show these fields in the
                # labeling UI if model identity must remain blinded.
                task[f"pair_{pair_idx}_left_summary_id"] = left.summary_id
                task[f"pair_{pair_idx}_right_summary_id"] = right.summary_id
                task[f"pair_{pair_idx}_left_model_id"] = left.model_id
                task[f"pair_{pair_idx}_right_model_id"] = right.model_id
                task[f"pair_{pair_idx}_selection_signal"] = pair.get("selection_signal")
                task[f"pair_{pair_idx}_high_ranked_criteria"] = (
                    (pair.get("selection_signal") or {}).get("high_ranked_criteria") or []
                )
            tasks.append(task)
    return tasks


def build_labelstudio_tasks_by_block(
    batches: list[dict[str, Any]],
    block_by_id: dict[str, dict[str, Any]],
    doc_by_id: dict[str, dict[str, Any]],
    summary_by_id: dict[str, SummaryRecord],
) -> list[dict[str, Any]]:
    """Build one Label Studio task per block.

    This is the preferred Label Studio import shape when the same task should be
    annotated by multiple annotators in Label Studio, because all annotations for
    a block stay attached to the same Label Studio task id.
    """
    block_to_batch: dict[str, tuple[str, int]] = {}
    for batch in batches:
        for position, block_id in enumerate(batch["block_ids"], start=1):
            block_to_batch[block_id] = (batch["batch_id"], position)

    tasks = []
    for block_id in sorted(block_by_id):
        block = block_by_id[block_id]
        doc = doc_by_id[block["document_id"]]
        batch_id, block_order_position = block_to_batch.get(block_id, (None, None))
        task: dict[str, Any] = {
            "batch_id": batch_id,
            "block_order_position": block_order_position,
            "block_id": block_id,
            "document_id": doc["document_id"],
            "source_doc_id": doc.get("source_doc_id"),
            "source_text": doc["source_text"],
            "selection_bucket": doc["selection_bucket"],
        }
        for pair_idx, pair in enumerate(block["pairs"], start=1):
            left = summary_by_id[pair["left_summary_id"]]
            right = summary_by_id[pair["right_summary_id"]]
            task[f"pair_{pair_idx}_id"] = pair["pair_id"]
            task[f"pair_{pair_idx}_left"] = left.summary_text
            task[f"pair_{pair_idx}_right"] = right.summary_text
            # Hidden metadata for analysis; do not show these fields in the
            # labeling UI if model identity must remain blinded.
            task[f"pair_{pair_idx}_left_summary_id"] = left.summary_id
            task[f"pair_{pair_idx}_right_summary_id"] = right.summary_id
            task[f"pair_{pair_idx}_left_model_id"] = left.model_id
            task[f"pair_{pair_idx}_right_model_id"] = right.model_id
            task[f"pair_{pair_idx}_selection_signal"] = pair.get("selection_signal")
            task[f"pair_{pair_idx}_high_ranked_criteria"] = (
                (pair.get("selection_signal") or {}).get("high_ranked_criteria") or []
            )
        tasks.append(task)
    return tasks


def write_labelstudio_tasks_by_batch(output_dir: Path, block_tasks: list[dict[str, Any]]) -> dict[str, str]:
    batch_dir = output_dir / "labelstudio_tasks_by_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    for old_file in batch_dir.glob("*.json"):
        old_file.unlink()

    tasks_by_batch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in block_tasks:
        batch_id = task["batch_id"]
        tasks_by_batch[batch_id].append(task)

    files: dict[str, str] = {}
    for batch_id in sorted(tasks_by_batch):
        tasks = sorted(tasks_by_batch[batch_id], key=lambda t: int(t["block_order_position"]))
        filename = f"{batch_id}_labelstudio_tasks_by_block.json"
        write_json(batch_dir / filename, tasks)
        files[batch_id] = str(Path("labelstudio_tasks_by_batch") / filename)
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    ap.add_argument("--export-json-dir", type=Path, default=DEFAULT_EXPORT_JSON_DIR)
    ap.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST_JSONL)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument(
        "--batches",
        type=int,
        default=24,
        help="Number of batches to create. Documents = batches * blocks-per-batch.",
    )
    ap.add_argument(
        "--documents",
        type=int,
        default=None,
        help="Deprecated override for number of documents/blocks. Prefer --batches.",
    )
    ap.add_argument("--blocks-per-batch", type=int, default=BLOCKS_PER_BATCH)
    ap.add_argument(
        "--start-batch-number",
        type=int,
        default=1,
        help="First batch id number to write. Use 9 for a later project containing batches 9-20.",
    )
    ap.add_argument(
        "--exclude-documents-from",
        type=Path,
        action="append",
        default=[],
        help=(
            "Output directory or documents.jsonl file whose document_ids must be excluded. "
            "Can be passed multiple times."
        ),
    )
    ap.add_argument(
        "--design-batches",
        type=int,
        default=24,
        help=(
            "Stable master design size. The script builds this many batches first, "
            "then writes the requested prefix so earlier batches remain unchanged."
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n_documents = args.documents if args.documents is not None else args.batches * args.blocks_per_batch
    if n_documents % args.blocks_per_batch != 0:
        raise ValueError("--documents must be divisible by --blocks-per-batch")
    requested_batches = n_documents // args.blocks_per_batch
    design_batches = max(args.design_batches, requested_batches)
    design_documents = design_batches * args.blocks_per_batch

    docs, model_ids = load_eval_candidates(args.eval_dir)
    excluded_document_ids = load_excluded_document_ids(args.exclude_documents_from)
    available_docs = [doc for doc in docs if doc["document_id"] not in excluded_document_ids]
    if design_documents > len(available_docs):
        raise ValueError(
            f"Requested stable design needs {design_documents} documents after exclusions, "
            f"but only {len(available_docs)} are available"
        )
    metadata_by_input = load_test_metadata(args.test_jsonl)
    pair_signals_by_doc = load_pair_selection_signals(args.export_json_dir, model_ids)

    enriched_docs = []
    for doc in available_docs:
        meta = metadata_by_input.get(str(doc["source_text"]).strip(), {})
        pair_signals = pair_signals_by_doc.get(doc["document_id"], {})
        features = build_doc_features_from_pair_signals(pair_signals)
        enriched = {
            **doc,
            "document_index": int(str(doc["document_id"]).removeprefix("doc_")),
            "source_doc_id": meta.get("dokument_id"),
            "doc_type": meta.get("doc_type"),
            "kommune": meta.get("kommune"),
            "selection_features": features,
            "pair_selection_signals": pair_signals,
            "selection_bucket": bucket_for_features(features),
        }
        enriched_docs.append(enriched)

    design_selected_docs = select_documents(enriched_docs, design_documents, args.seed)
    assign_ranked_selection_buckets(design_selected_docs)
    design_selected_docs = balance_documents_across_batches(
        design_selected_docs,
        n_batches=design_batches,
        blocks_per_batch=args.blocks_per_batch,
    )
    start_block_number = (args.start_batch_number - 1) * args.blocks_per_batch + 1
    design_blocks, _, _ = build_blocks(
        design_selected_docs,
        model_ids,
        args.seed,
        start_block_number=start_block_number,
    )
    design_batches_all = build_batches(
        design_blocks,
        args.blocks_per_batch,
        start_batch_number=args.start_batch_number,
    )

    batches = design_batches_all[:requested_batches]
    included_block_ids = {block_id for batch in batches for block_id in batch["block_ids"]}
    blocks = [block for block in design_blocks if block["block_id"] in included_block_ids]
    validate_unique_documents(blocks)
    included_doc_ids = {block["document_id"] for block in blocks}
    selected_docs = [doc for doc in design_selected_docs if doc["document_id"] in included_doc_ids]
    pair_counts, left_counts = count_block_exposures(blocks)
    assignments = build_assignments(batches, args.seed)

    summary_records: list[SummaryRecord] = []
    for doc in selected_docs:
        for model_id in model_ids:
            summary_records.append(
                SummaryRecord(
                    document_id=doc["document_id"],
                    summary_id=make_summary_id(doc["document_id"], model_id),
                    model_id=model_id,
                    model_alias=MODEL_ALIASES[model_id],
                    summary_text=doc["summaries"][model_id],
                )
            )
    summary_by_id = {s.summary_id: s for s in summary_records}
    doc_by_id = {d["document_id"]: d for d in selected_docs}
    block_by_id = {b["block_id"]: b for b in blocks}
    tasks = build_labelstudio_tasks(assignments, block_by_id, doc_by_id, summary_by_id)
    block_tasks = build_labelstudio_tasks_by_block(batches, block_by_id, doc_by_id, summary_by_id)

    documents_rows = [
        {
            "document_id": d["document_id"],
            "document_index": d["document_index"],
            "source_doc_id": d.get("source_doc_id"),
            "doc_type": d.get("doc_type"),
            "kommune": d.get("kommune"),
            "selection_bucket": d["selection_bucket"],
            "selection_features": d["selection_features"],
            "source_text": d["source_text"],
        }
        for d in selected_docs
    ]
    summaries_rows = [
        {
            "document_id": s.document_id,
            "summary_id": s.summary_id,
            "model_id": s.model_id,
            "model_alias": s.model_alias,
            "summary_text": s.summary_text,
        }
        for s in summary_records
    ]

    output_dir = args.output_dir
    write_jsonl(output_dir / "documents.jsonl", documents_rows)
    write_jsonl(output_dir / "summaries.jsonl", summaries_rows)
    write_jsonl(output_dir / "blocks.jsonl", blocks)
    write_jsonl(output_dir / "batches.jsonl", batches)
    write_jsonl(output_dir / "assignments.jsonl", assignments)
    write_json(output_dir / "labelstudio_tasks.json", tasks)
    write_json(output_dir / "labelstudio_tasks_by_block.json", block_tasks)
    labelstudio_tasks_by_batch_files = write_labelstudio_tasks_by_batch(output_dir, block_tasks)

    bucket_counts = Counter(d["selection_bucket"] for d in selected_docs)
    selected_high_pair_count = sum(
        1
        for block in blocks
        for pair in block["pairs"]
        if pair.get("included_due_to_high_rank")
    )
    selected_pair_reason_counts: Counter[str] = Counter()
    for block in blocks:
        for pair in block["pairs"]:
            signal = pair.get("selection_signal") or {}
            selected_pair_reason_counts.update(signal.get("high_ranked_criteria") or [])
    assignment_counts = Counter(a["annotator_id"] for a in assignments)
    annotator_pair_counts: Counter[tuple[str, str]] = Counter()
    for batch in batches:
        assigned = sorted(a["annotator_id"] for a in assignments if a["batch_id"] == batch["batch_id"])
        for pair in itertools.combinations(assigned, 2):
            annotator_pair_counts[pair] += 1

    manifest = {
        "description": "Human pairwise annotation batch design for four 2500-human-candidates summaries.",
        "seed": args.seed,
        "eval_dir": str(args.eval_dir),
        "export_json_dir": str(args.export_json_dir),
        "n_documents": len(selected_docs),
        "n_models": len(model_ids),
        "model_ids": model_ids,
        "model_aliases": MODEL_ALIASES,
        "n_blocks": len(blocks),
        "pairs_per_block": PAIRS_PER_BLOCK,
        "blocks_per_batch": args.blocks_per_batch,
        "n_batches": len(batches),
        "pilot_batches": 8,
        "pilot_documents": 8 * args.blocks_per_batch,
        "stable_design_batches": design_batches,
        "stable_design_documents": design_documents,
        "requested_batches": requested_batches,
        "start_batch_number": args.start_batch_number,
        "excluded_document_count": len(excluded_document_ids),
        "exclude_documents_from": [str(path) for path in args.exclude_documents_from],
        "annotators": ANNOTATORS,
        "assignments_per_batch": 3,
        "n_assignments": len(assignments),
        "labelstudio_tasks": len(tasks),
        "labelstudio_tasks_by_block": len(block_tasks),
        "labelstudio_tasks_by_batch_files": len(labelstudio_tasks_by_batch_files),
        "selection_bucket_counts": dict(bucket_counts),
        "pair_selection_method": (
            "Rank document/model-pairs by pair-level criteria; select documents by accumulated "
            "priority from high-ranked pairs; choose each block's three pairs to include as many "
            "high-ranked selected-document pairs as possible while preserving within-block exposure "
            "constraints and global pair/left balance."
        ),
        "pair_selection_criteria": list(PAIR_SELECTION_CRITERIA),
        "pair_high_rank_cutoff": PAIR_HIGH_RANK_CUTOFF,
        "selected_high_rank_pair_slots": selected_high_pair_count,
        "selected_pair_reason_counts": dict(selected_pair_reason_counts),
        "model_pair_counts": {" :: ".join(k): v for k, v in sorted(pair_counts.items())},
        "left_position_counts": dict(left_counts),
        "assignment_counts_by_annotator": dict(assignment_counts),
        "annotator_pair_counts": {"".join(k): v for k, v in sorted(annotator_pair_counts.items())},
        "files": {
            "documents": "documents.jsonl",
            "summaries": "summaries.jsonl",
            "blocks": "blocks.jsonl",
            "batches": "batches.jsonl",
            "assignments": "assignments.jsonl",
            "labelstudio_tasks": "labelstudio_tasks.json",
            "labelstudio_tasks_by_block": "labelstudio_tasks_by_block.json",
            "labelstudio_tasks_by_batch": labelstudio_tasks_by_batch_files,
        },
    }
    write_json(output_dir / "manifest.json", manifest)

    print(f"Wrote outputs to {output_dir}")
    print(f"Documents: {len(selected_docs)}")
    print(f"Batches: {len(batches)} ({args.blocks_per_batch} blocks each)")
    print(f"Assignments: {len(assignments)}")
    print(f"Label Studio tasks: {len(tasks)}")
    print(f"Selection buckets: {dict(bucket_counts)}")


if __name__ == "__main__":
    main()
