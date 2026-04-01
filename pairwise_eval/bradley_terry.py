"""Bradley–Terry MLE from pairwise outcomes (ties split 0.5 each direction in W)."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
except ImportError as e:  # pragma: no cover
    raise ImportError("Bradley–Terry MLE needs scipy (`pip install scipy`).") from e

from pairwise_eval.config import EVAL_DIMENSIONS, JUDGES, REFERENCE_SUMMARY_MODEL_ID
from pairwise_eval.judging import is_tie_row, models_in_dimension


def win_matrix_from_geval(tbl: pd.DataFrame, model_order: list[str]) -> np.ndarray:
    """Build Bradley–Terry count matrix W from one G-Eval table.

    Input: pairwise judgments for one (judge, dimension); fixed ``model_order`` index. Output: K×K
    float array (directed wins; ties split 0.5 each way).
    """
    idx = {m: i for i, m in enumerate(model_order)}
    k = len(model_order)
    w = np.zeros((k, k), dtype=float)
    for _, row in tbl.iterrows():
        left, right = row["left"], row["right"]
        i, j = idx[left], idx[right]
        if is_tie_row(row):
            w[i, j] += 0.5
            w[j, i] += 0.5
        else:
            win = row["chosen"]
            lose = right if win == left else left
            w[idx[win], idx[lose]] += 1.0
    return w


def neg_log_lik_bradley_terry(beta_free: np.ndarray, w: np.ndarray, ref_idx: int = 0) -> float:
    """Negative log-likelihood for BT with ``beta[ref_idx]`` fixed at 0 (internal to ``minimize``).

    Input: free β vector, count matrix W, reference index. Output: scalar NLL.
    """
    beta = np.insert(beta_free, ref_idx, 0.0)
    k = w.shape[0]
    ll = 0.0
    for i in range(k):
        for j in range(k):
            n = w[i, j]
            if n <= 0:
                continue
            ll += n * (beta[i] - np.log(np.exp(beta[i]) + np.exp(beta[j])))
    return -float(ll)


def fit_bradley_terry(w: np.ndarray, *, ref_idx: int = 0) -> tuple[np.ndarray, object]:
    """MLE β via L-BFGS-B with ``beta[ref_idx] = 0`` for identifiability.

    Input: count matrix W, reference index. Output: ``(beta, scipy OptimizeResult)``.
    """
    k = w.shape[0]
    x0 = np.zeros(k - 1)
    res = minimize(neg_log_lik_bradley_terry, x0, args=(w, ref_idx), method="L-BFGS-B")
    beta = np.insert(np.asarray(res.x, dtype=float), ref_idx, 0.0)
    return beta, res


def bradley_terry_long_table(
    geval_tables: Dict[Tuple[str, str], pd.DataFrame],
    dimensions: Tuple[str, ...] = EVAL_DIMENSIONS,
    judges: Tuple[str, ...] = JUDGES,
    model_order: list[str] | None = None,
    ref_model: str | None = None,
) -> pd.DataFrame:
    """Fit BT per (dimension, judge); return long table of β, θ (mean-centered), diagnostics.

    Input: full ``geval_tables``, optional ``model_order`` and ``ref_model`` for fitting anchor.
    Output: DataFrame rows = (dimension, judge, model) with ``beta``, ``theta``, etc.
    """
    mo = list(model_order) if model_order is not None else models_in_dimension(geval_tables, dimensions[0])
    if ref_model is not None:
        ref_idx = mo.index(ref_model)
    elif REFERENCE_SUMMARY_MODEL_ID in mo:
        ref_idx = mo.index(REFERENCE_SUMMARY_MODEL_ID)
    else:
        ref_idx = 0
    rows = []
    for dim in dimensions:
        for judge in judges:
            tbl = geval_tables[(judge, dim)]
            w = win_matrix_from_geval(tbl, mo)
            if (w + w.T).sum() == 0:
                continue
            beta, res = fit_bradley_terry(w, ref_idx=ref_idx)
            # Same win probs as raw MLE, but no row is fixed to θ=1: geometric mean of θ is 1.
            beta = beta - np.mean(beta)
            theta = np.exp(beta)
            for m, b, t in zip(mo, beta, theta):
                rows.append(
                    {
                        "dimension": dim,
                        "judge": judge,
                        "model": m,
                        "beta": b,
                        "theta": t,
                        "optimizer_success": res.success,
                        "n_comparisons": int((w + w.T).sum() / 2),
                    }
                )
    return pd.DataFrame(rows)


def bradley_terry_theta_wide(
    bt_long: pd.DataFrame,
    dimensions: Tuple[str, ...] = EVAL_DIMENSIONS,
    judges: Tuple[str, ...] = JUDGES,
) -> Dict[str, pd.DataFrame]:
    """Pivot long BT output to wide tables (one DataFrame per dimension).

    Input: ``bradley_terry_long_table`` result. Output: dict ``dimension -> wide DataFrame``.
    """
    out: Dict[str, pd.DataFrame] = {}
    for dim in dimensions:
        sub = bt_long[bt_long["dimension"] == dim].pivot(index="model", columns="judge", values="theta")
        sub = sub.reindex(columns=list(judges))
        sub.columns = [f"{j}_theta" for j in judges]
        out[dim] = sub.sort_index()
    return out


def markdown_bradley_terry_theta(bt_long: pd.DataFrame) -> str:
    """Format wide θ tables as a markdown document with a short methods note.

    Input: long BT table. Output: markdown string.
    """
    wide = bradley_terry_theta_wide(bt_long)
    parts = [
        "# Bradley–Terry strengths (θ = exp(β), mean-centered β)",
        "",
        f"Gold summaries use the label `{REFERENCE_SUMMARY_MODEL_ID}` (text from JSONL `reference`). "
        "β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. "
        "Sparse data can push some θ toward 0 (separation).",
        "",
    ]
    for dim, tab in wide.items():
        parts.append(f"### {dim.capitalize()}\n")
        cols = ["model"] + list(tab.columns)
        parts.append("| " + " | ".join(cols) + " |")
        parts.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for m, row in tab.iterrows():
            cells = [m] + [f"{v:.4f}" for v in row]
            parts.append("| " + " | ".join(cells) + " |")
        parts.append("")
    return "\n".join(parts)
