"""Build pairwise comparison rows from long-form summarization data."""

from __future__ import annotations

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


def build_pairs_table(
    long_df: pd.DataFrame,
    n_pairs: int = N_PAIRS_PER_DOCUMENT,
    *,
    rng: np.random.Generator | None = None,
    seed: int = DEFAULT_PAIR_SEED,
    prior_pairs_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Sample up to ``n_pairs`` random model pairs per document and shuffle A/B sides.

    Input: ``long_df`` (all models per doc); optional RNG/seed; optional ``prior_pairs_df`` from
    :func:`load_pairs_table_json` to **extend** a previous design — kept rows are reused verbatim
    (same ``left`` / ``right`` / summaries) so checkpoint resume stays aligned; only **new**
    unordered pairs are sampled until ``n_pairs`` is reached (capped by available combinations).

    Output: DataFrame with ``doc_id``, ``left``, ``right``, ``sumleft``, ``sumright``.
    """
    gen = rng if rng is not None else np.random.default_rng(seed)
    rows: list[dict] = []
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
            need = k_target - len(kept)
            new_pairs: list[tuple[object, object]] = []
            if need > 0:
                candidates = [p for p in all_pairs if _pair_key(p[0], p[1]) not in used]
                if not candidates:
                    rows.extend(kept)
                    continue
                take = min(need, len(candidates))
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
