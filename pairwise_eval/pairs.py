"""Build pairwise comparison rows from long-form summarization data."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from pairwise_eval.config import DEFAULT_PAIR_SEED, N_PAIRS_PER_DOCUMENT


def build_pairs_table(
    long_df: pd.DataFrame,
    n_pairs: int = N_PAIRS_PER_DOCUMENT,
    *,
    rng: np.random.Generator | None = None,
    seed: int = DEFAULT_PAIR_SEED,
) -> pd.DataFrame:
    """Sample up to ``n_pairs`` random model pairs per document and shuffle A/B sides.

    Input: ``long_df`` (all models per doc); optional RNG/seed. Output: DataFrame with
    ``doc_id``, ``left``, ``right``, ``sumleft``, ``sumright``.
    """
    gen = rng if rng is not None else np.random.default_rng(seed)
    rows: list[dict] = []
    for doc_id, g in long_df.groupby("doc_id", sort=False):
        by_model = g.set_index("model_id")["summary_text"].to_dict()
        models_here = list(by_model.keys())
        if len(models_here) < 2:
            continue
        all_pairs = list(itertools.combinations(models_here, 2))
        k = min(n_pairs, len(all_pairs))
        idx = gen.choice(len(all_pairs), size=k, replace=False)
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
