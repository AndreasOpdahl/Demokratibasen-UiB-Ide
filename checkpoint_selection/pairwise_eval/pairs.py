"""Build pairwise comparison rows from long-form summarization data."""

from __future__ import annotations

import collections
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from pairwise_eval.config import DEFAULT_PAIR_SEED, N_PAIRS_PER_DOCUMENT

_PRIOR_COLUMNS = ("doc_id", "left", "right", "sumleft", "sumright")


def load_pairs_table_json(path: str | Path) -> pd.DataFrame:
    """Load ``json/pairs_table.json`` from a prior export (records layout).

    Input: path on disk. Output: DataFrame with columns required by ``prior_pairs_df`` in
    :func:`build_pairs_table`.
    """
    p = Path(path)
    df = pd.read_json(p)
    missing = [c for c in _PRIOR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{p}: missing columns {missing}; expected {_PRIOR_COLUMNS}")
    return df


def _pair_key(left: object, right: object) -> frozenset[str]:
    return frozenset((str(left), str(right)))


def _prior_kept_rows_for_doc(
    prior_pairs_df: pd.DataFrame,
    doc_id: object,
    by_model: dict,
) -> list[dict]:
    """Rows from ``prior_pairs_df`` for ``doc_id`` that are still valid, deduped by unordered pair.

    Preserves ``left``, ``right``, ``sumleft``, ``sumright`` from the prior row so judgment
    checkpoint keys still match. Order is the order rows appear in ``prior_pairs_df`` for that doc.
    """
    sub = prior_pairs_df[prior_pairs_df["doc_id"].astype(str) == str(doc_id)]
    kept: list[dict] = []
    seen_keys: set[frozenset[str]] = set()
    for _, r in sub.iterrows():
        left, right = r["left"], r["right"]
        if left not in by_model or right not in by_model or pd.isna(left) or pd.isna(right):
            continue
        if left == right:
            continue
        k = _pair_key(left, right)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        kept.append(
            {
                "doc_id": doc_id,
                "left": left,
                "right": right,
                "sumleft": r["sumleft"],
                "sumright": r["sumright"],
            }
        )
    return kept


def _greedy_balanced_pick(
    candidates: list[tuple[object, object]],
    k: int,
    c_model: collections.Counter,
    c_pair: collections.Counter,
    gen: np.random.Generator,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> list[tuple[object, object]]:
    """Pick ``k`` pairs from ``candidates`` minimizing combined model + pair load.

    For each remaining candidate ``(a, b)`` the cost is
    ``alpha * (c_model[a] + c_model[b]) + beta * c_pair[{a, b}]``. The lowest-cost pair is
    picked at each step, with uniform random tie-break. Updates ``c_model`` and ``c_pair`` in
    place so the caller can keep counters consistent across documents.
    """
    remaining = list(candidates)
    chosen: list[tuple[object, object]] = []
    while len(chosen) < k and remaining:
        best_cost: float | None = None
        best_indices: list[int] = []
        for i, (a, b) in enumerate(remaining):
            pair_key = _pair_key(a, b)
            cost = alpha * (c_model[a] + c_model[b]) + beta * c_pair[pair_key]
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_indices = [i]
            elif cost == best_cost:
                best_indices.append(i)
        pick = (
            best_indices[0]
            if len(best_indices) == 1
            else best_indices[int(gen.integers(0, len(best_indices)))]
        )
        a, b = remaining.pop(pick)
        chosen.append((a, b))
        c_model[a] += 1
        c_model[b] += 1
        c_pair[_pair_key(a, b)] += 1
    return chosen


def build_pairs_table(
    long_df: pd.DataFrame,
    n_pairs: int = N_PAIRS_PER_DOCUMENT,
    *,
    rng: np.random.Generator | None = None,
    seed: int = DEFAULT_PAIR_SEED,
    prior_pairs_df: pd.DataFrame | None = None,
    balanced: bool = False,
) -> pd.DataFrame:
    """Sample up to ``n_pairs`` random model pairs per document and shuffle A/B sides.

    Input: ``long_df`` (all models per doc); optional RNG/seed; optional ``prior_pairs_df`` from
    :func:`load_pairs_table_json` to **extend** a previous design — kept rows are reused verbatim
    (same ``left`` / ``right`` / summaries) so checkpoint resume stays aligned; only **new**
    unordered pairs are sampled until ``n_pairs`` is reached (capped by available combinations).

    When ``balanced=False`` (default), new pairs for each document are drawn uniformly at random
    from the available combinations — identical to the original behavior. When ``balanced=True``,
    a global greedy sampler instead picks the pair that minimizes the combined per-model and
    per-pair usage counters (random tie-break), which yields near-uniform per-model appearance
    counts and per-pair coverage across documents. Kept rows from ``prior_pairs_df`` pre-load the
    counters so extension runs stay consistent.

    Output: DataFrame with ``doc_id``, ``left``, ``right``, ``sumleft``, ``sumright``.
    """
    gen = rng if rng is not None else np.random.default_rng(seed)
    rows: list[dict] = []
    c_model: collections.Counter = collections.Counter()
    c_pair: collections.Counter = collections.Counter()
    for doc_id, g in long_df.groupby("doc_id", sort=False):
        by_model = g.set_index("model_id")["summary_text"].to_dict()
        models_here = list(by_model.keys())
        if len(models_here) < 2:
            continue
        all_pairs = list(itertools.combinations(models_here, 2))
        k_target = min(n_pairs, len(all_pairs))

        if prior_pairs_df is not None:
            kept = _prior_kept_rows_for_doc(prior_pairs_df, doc_id, by_model)
            if len(kept) > k_target:
                raise ValueError(
                    f"doc_id={doc_id!r}: prior pairs_table has {len(kept)} usable rows for this "
                    f"doc but n_pairs caps at {k_target}. Raise n_pairs or trim the prior export."
                )
            used = {_pair_key(d["left"], d["right"]) for d in kept}
            if balanced:
                for d in kept:
                    c_model[d["left"]] += 1
                    c_model[d["right"]] += 1
                    c_pair[_pair_key(d["left"], d["right"])] += 1
            need = k_target - len(kept)
            new_pairs: list[tuple[object, object]] = []
            if need > 0:
                candidates = [p for p in all_pairs if _pair_key(p[0], p[1]) not in used]
                if not candidates:
                    rows.extend(kept)
                    continue
                take = min(need, len(candidates))
                if balanced:
                    new_pairs = _greedy_balanced_pick(
                        candidates, take, c_model, c_pair, gen
                    )
                else:
                    idx = gen.choice(len(candidates), size=take, replace=False)
                    new_pairs = [candidates[i] for i in idx]
            doc_rows: list[dict] = list(kept)
            for a, b in new_pairs:
                if gen.random() < 0.5:
                    left, right = a, b
                else:
                    left, right = b, a
                doc_rows.append(
                    {
                        "doc_id": doc_id,
                        "left": left,
                        "right": right,
                        "sumleft": by_model[left],
                        "sumright": by_model[right],
                    }
                )
            rows.extend(doc_rows)
            continue

        if balanced:
            chosen_pairs = _greedy_balanced_pick(
                all_pairs, k_target, c_model, c_pair, gen
            )
        else:
            idx = gen.choice(len(all_pairs), size=k_target, replace=False)
            chosen_pairs = [all_pairs[i] for i in idx]

        for a, b in chosen_pairs:
            if gen.random() < 0.5:
                left, right = a, b
            else:
                left, right = b, a
            rows.append(
                {
                    "doc_id": doc_id,
                    "left": left,
                    "right": right,
                    "sumleft": by_model[left],
                    "sumright": by_model[right],
                }
            )
    return pd.DataFrame(rows)
