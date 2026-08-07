"""Learn nonnegative dimension weights that aggregate per-dimension checkpoint scores.

The reference judge (e.g. Claude) defines a *target* overall score per checkpoint as a
**weighted mean** of that judge's per-dimension mean win outcomes (default: equal weights).
*Features* are per-dimension means from one or more other judges (pooled). We fit weights
w >= 0, sum(w)=1, minimizing::

    sum_ck ( (w · x_ck) - y_ck )^2 + ridge * ||w||^2

where x_ck is the feature vector of dimension means for that checkpoint and y_ck is the
reference aggregate. When human overall scores exist, you can swap ``y`` for those instead.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from evaluation_suite.core.geval_mean import DEFAULT_DIMENSIONS, compute_checkpoint_dimension_means


def _project_onto_simplex(v: Sequence[float]) -> list[float]:
    """Euclidean projection of v onto {w : w_i >= 0, sum w_i = 1}."""
    n = len(v)
    if n == 0:
        return []
    u = sorted(float(x) for x in v)
    u.reverse()
    cssv = []
    s = 0.0
    for i, ui in enumerate(u):
        s += ui
        cssv.append(s)
    rho = -1
    for j in range(n):
        if u[j] * (j + 1) > cssv[j] - 1:
            rho = j
    assert rho >= 0
    theta = (cssv[rho] - 1.0) / (rho + 1)
    return [max(float(vi) - theta, 0.0) for vi in v]


def fit_weights_simplex_ridge(
    X: list[list[float]],
    y: list[float],
    *,
    ridge: float = 1e-6,
    max_iter: int = 20000,
    lr: float | None = None,
) -> tuple[list[float], float]:
    """Minimize ||X w - y||^2 + ridge ||w||^2 with w on the probability simplex.

    X: n_samples × n_features, y: length n_samples. Uses projected gradient descent.
    """
    if not X or not y:
        raise ValueError("X and y must be non-empty")
    n, d = len(X), len(X[0])
    if len(y) != n:
        raise ValueError("len(y) must equal len(X)")
    if any(len(row) != d for row in X):
        raise ValueError("All rows of X must have the same length")
    if d == 0:
        raise ValueError("Need at least one feature dimension")

    lr_eff = lr if lr is not None else 0.5 / (1e-8 + sum(sum(xij * xij for xij in row) for row in X) / n)
    w = [1.0 / d] * d
    best_w = w
    best_obj = float("inf")

    for _ in range(max_iter):
        # grad of ||Xw - y||^2 + ridge ||w||^2 = 2 X^T (Xw - y) + 2 ridge w
        resid = [sum(X[i][j] * w[j] for j in range(d)) - y[i] for i in range(n)]
        grad = [0.0] * d
        for j in range(d):
            g = 2.0 * sum(X[i][j] * resid[i] for i in range(n)) + 2.0 * ridge * w[j]
            grad[j] = g
        w_new = _project_onto_simplex([w[j] - lr_eff * grad[j] for j in range(d)])
        w = w_new

        pred = [sum(X[i][j] * w[j] for j in range(d)) for i in range(n)]
        obj = sum((pred[i] - y[i]) ** 2 for i in range(n)) + ridge * sum(wj * wj for wj in w)
        if obj < best_obj:
            best_obj = obj
            best_w = list(w)

    return best_w, best_obj


def _normalize_reference_target_weights(
    dimensions: tuple[str, ...],
    target_dimension_weights: dict[str, float],
) -> list[float]:
    """Nonnegative coefficients summing to 1 for y = sum_d alpha_d * mu_ref(k,d)."""
    raw = [max(0.0, float(target_dimension_weights.get(dim, 0.0))) for dim in dimensions]
    s = sum(raw)
    if s <= 0:
        raise ValueError(
            "target_dimension_weights must sum to a positive value over the evaluation dimensions."
        )
    return [w / s for w in raw]


def _mean_and_alignment(
    ref_means: dict[int, dict[str, float]],
    feat_means: dict[int, dict[str, float]],
    dimensions: tuple[str, ...],
    *,
    reference_target_alpha: Sequence[float],
) -> tuple[list[int], list[float], list[list[float]]]:
    """Align checkpoints where reference has all dims; build y and X rows.

    ``reference_target_alpha`` has length ``len(dimensions)``, nonnegative, sums to 1:
    y_k = sum_d alpha[d] * mu_ref(k, d).
    """
    cks: list[int] = []
    y: list[float] = []
    X: list[list[float]] = []

    for ck in sorted(ref_means.keys()):
        rm = ref_means[ck]
        if not all(dim in rm for dim in dimensions):
            continue
        fm = feat_means.get(ck)
        if fm is None or not all(dim in fm for dim in dimensions):
            continue
        yk = sum(reference_target_alpha[j] * rm[dimensions[j]] for j in range(len(dimensions)))
        y.append(yk)
        X.append([fm[dim] for dim in dimensions])
        cks.append(ck)

    return cks, y, X


def learn_weights_from_geval_rows(
    rows: list[dict],
    *,
    reference_judge_substring: str,
    feature_judge_substrings: Iterable[str] | None = None,
    dimensions: tuple[str, ...] = DEFAULT_DIMENSIONS,
    ridge: float = 1e-6,
    target_dimension_weights: dict[str, float] | None = None,
) -> dict:
    """Fit simplex weights using a reference judge as surrogate human.

    * **Target** ``y``: per checkpoint, weighted mean of reference judge's per-dimension
      means. If ``target_dimension_weights`` is None, uses equal weights. Otherwise pass
      nonnegative coefficients (typically from ``parse_target_dimension_weights``: each
      value in ``[0, 1]``, unlisted dimensions treated as 0); they are normalized to sum 1.
    * **Features** ``X``: per checkpoint, per-dimension means pooled over feature judges.

    If ``feature_judge_substrings`` is None, every row whose judge is not the reference
    (substring does not match ``reference_judge_substring``) is pooled into features.
    """
    ref_means, _ = compute_checkpoint_dimension_means(
        rows, judge_substring=reference_judge_substring, dimensions=dimensions
    )

    if feature_judge_substrings is None:
        feat_rows = [
            r
            for r in rows
            if reference_judge_substring not in str(r.get("judge_id", ""))
        ]
    else:
        subs = list(feature_judge_substrings)
        feat_rows = []
        for r in rows:
            jid = str(r.get("judge_id", ""))
            if any(s in jid for s in subs):
                feat_rows.append(r)

    if not feat_rows:
        raise ValueError(
            "No rows available for feature judges. "
            "Check --feature-judges substrings or that non-reference judges exist in the export."
        )

    feat_means, _ = compute_checkpoint_dimension_means(
        feat_rows, judge_substring=None, dimensions=dimensions
    )

    tw = target_dimension_weights
    if tw is None:
        tw = {d: 1.0 for d in dimensions}
    reference_target_alpha = _normalize_reference_target_weights(dimensions, tw)

    cks, y, X = _mean_and_alignment(
        ref_means, feat_means, dimensions, reference_target_alpha=reference_target_alpha
    )
    if len(cks) < 2:
        raise ValueError(
            f"Need at least 2 checkpoints with full dimensions for ref and features; got {len(cks)}. "
            "Check judge substrings and data coverage."
        )

    w, obj = fit_weights_simplex_ridge(X, y, ridge=ridge)
    pred = [sum(X[i][j] * w[j] for j in range(len(dimensions))) for i in range(len(cks))]
    mse = sum((pred[i] - y[i]) ** 2 for i in range(len(cks))) / len(cks)
    y_bar = sum(y) / len(y)
    ss_tot = sum((yi - y_bar) ** 2 for yi in y)
    ss_res = sum((pred[i] - y[i]) ** 2 for i in range(len(cks)))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    return {
        "weights": {dimensions[j]: w[j] for j in range(len(dimensions))},
        "checkpoint_steps": cks,
        "n_checkpoints": len(cks),
        "target": "weighted_mean_over_reference_dimensions",
        "reference_aggregate_weights": {
            dimensions[j]: reference_target_alpha[j] for j in range(len(dimensions))
        },
        "features": "pooled_feature_judges_per_dimension_means",
        "reference_judge_substring": reference_judge_substring,
        "feature_judge_substrings": list(feature_judge_substrings) if feature_judge_substrings is not None else None,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": r2,
        "objective": obj,
        "ridge": ridge,
        "y_reference_aggregate": {str(ck): y[i] for i, ck in enumerate(cks)},
        "y_predicted": {str(ck): pred[i] for i, ck in enumerate(cks)},
    }
