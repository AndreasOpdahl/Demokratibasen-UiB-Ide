"""Pairwise judgments: context merge, mock judge, and full G-Eval tables."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from pairwise_eval.config import (
    DEFAULT_GEVAL_BASE_SEED,
    EVAL_DIMENSIONS,
    JUDGES,
    MOCK_TIE_PROB,
)
from pairwise_eval.geval_checkpoint import (
    append_judgment_line,
    checkpoint_file_path,
    judgment_stable_key,
    load_checkpoint_index,
)

EvaluateFn = Callable[[Mapping, str, str, np.random.Generator], Dict[str, object]]


def is_tie_row(row: Mapping) -> bool:
    """Return True if this judgment is a tie (no strict winner).

    Input: row with ``choice_side`` / ``chosen``. Output: bool.
    """
    if row.get("choice_side") == "tie":
        return True
    return pd.isna(row.get("chosen"))


def rng_for_judge_dimension(
    judge_id: str, dimension: str, base_seed: int = DEFAULT_GEVAL_BASE_SEED
) -> np.random.Generator:
    """Stable RNG for mock judges: one stream per (judge, dimension).

    Input: judge id, dimension name, base seed. Output: ``numpy.random.Generator``.
    """
    digest = hashlib.sha256(f"{judge_id}:{dimension}:{base_seed}".encode()).digest()
    seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
    return np.random.default_rng(seed)


def mock_evaluate_pair(
    row: Mapping,
    dimension: str,
    judge_id: str,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Random baseline judge: tie with prob ``MOCK_TIE_PROB``, else pick left or right.

    Input: pair row (``left``/``right`` ids), dimension, judge id, RNG. Output: dict with
    ``choice_side``, ``chosen``, ``rationale``.
    """
    _ = (dimension, judge_id)
    if rng.random() < MOCK_TIE_PROB:
        return {"choice_side": "tie", "chosen": pd.NA, "rationale": ""}
    pick_left = rng.random() < 0.5
    return {
        "choice_side": "left" if pick_left else "right",
        "chosen": row["left"] if pick_left else row["right"],
        "rationale": "",
    }


def attach_doc_context(pairs_df: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    """Join ``source_text`` and ``reference_summary`` onto each pair row by ``doc_id``.

    Input: pairs table and long_df. Output: merged DataFrame for judging.
    """
    meta = long_df.groupby("doc_id", sort=False).first()[["source_text", "reference_summary"]]
    return pairs_df.merge(meta, on="doc_id", how="left")


def models_in_dimension(geval_tables: Dict[Tuple[str, str], pd.DataFrame], dimension: str) -> list[str]:
    """List distinct ``model_id``s appearing in pairs for ``dimension`` (via first configured judge).

    Input: completed ``geval_tables``, dimension name. Output: sorted model id strings.
    """
    for judge_id in JUDGES:
        key = (judge_id, dimension)
        if key in geval_tables:
            tbl = geval_tables[key]
            return sorted(set(tbl["left"]) | set(tbl["right"]))
    raise KeyError(f"No G-Eval table for dimension {dimension!r}; configured JUDGES={JUDGES!r}")


def build_geval_tables(
    pairs_df: pd.DataFrame,
    long_df: pd.DataFrame,
    dimensions: Tuple[str, ...] = EVAL_DIMENSIONS,
    judges: Tuple[str, ...] = JUDGES,
    evaluate_fn: EvaluateFn = mock_evaluate_pair,
    base_seed: int = DEFAULT_GEVAL_BASE_SEED,
    checkpoint_dir: Path | None = None,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Run ``evaluate_fn`` on every (judge × dimension × pair) and collect judgments.

    Input: pairs, long_df, judges/dimensions, ``evaluate_fn``, optional ``checkpoint_dir``.
    If ``checkpoint_dir`` is set, each new judgment is appended to a JSONL file immediately and
    existing keys are skipped on restart (see :mod:`pairwise_eval.geval_checkpoint`).
    Output: dict ``(judge_id, dimension) -> DataFrame`` with judgment columns added.
    """
    ctx = attach_doc_context(pairs_df, long_df)
    out: Dict[Tuple[str, str], pd.DataFrame] = {}
    for judge_id in judges:
        for dimension in dimensions:
            rng = rng_for_judge_dimension(judge_id, dimension, base_seed=base_seed)
            ck_path: Path | None = None
            done: Dict[str, Dict[str, object]] = {}
            if checkpoint_dir is not None:
                ck_path = checkpoint_file_path(checkpoint_dir, judge_id, dimension)
                done = load_checkpoint_index(ck_path)
            n_hit = 0
            n_miss = 0
            rows = []
            for _, row in ctx.iterrows():
                key = judgment_stable_key(judge_id, dimension, row)
                if key in done:
                    judgment = done[key]
                    n_hit += 1
                else:
                    judgment = evaluate_fn(row, dimension, judge_id, rng)
                    n_miss += 1
                    if ck_path is not None:
                        append_judgment_line(ck_path, key, judgment)
                rows.append({**row.to_dict(), **judgment})
            if checkpoint_dir is not None and ck_path is not None:
                print(
                    f"[checkpoint] {judge_id} × {dimension}: {n_hit} from disk, {n_miss} new → {ck_path}",
                    flush=True,
                )
            out[(judge_id, dimension)] = pd.DataFrame(rows)
    return out


def geval_by_judge(
    geval_tables: Dict[Tuple[str, str], pd.DataFrame],
    dimensions: Tuple[str, ...] = EVAL_DIMENSIONS,
    judges: Tuple[str, ...] = JUDGES,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Nest flat ``(judge, dim)`` tables as ``result[judge][dimension]``.

    Input: ``geval_tables`` from :func:`build_geval_tables`. Output: two-level dict of DataFrames.
    """
    return {j: {d: geval_tables[j, d] for d in dimensions} for j in judges}
