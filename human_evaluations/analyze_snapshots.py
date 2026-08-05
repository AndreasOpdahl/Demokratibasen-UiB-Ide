"""Analyse a loaded human-evaluation snapshot.

Computes, from the `comparisons` frame produced by load_snapshots.load_snapshots():

- basic stats (documents, summaries, pairs scored, raters, multi-scored pairs)
- time spent per task, overall and per rater
- per-model raw win rate, flag rate, and Bradley-Terry model-strength estimate
- inter-rater agreement on winner (left/right), overall and as a rater x rater
  matrix alongside how many pairs each rater-pair actually shares

This module is pure computation -- no printing. See report_snapshots.py for a
formatted console report built on top of it.

Usage:
    from load_snapshots import load_snapshots
    from analyze_snapshots import analyze

    comparisons, documents, summaries, summary_model, rater_email = load_snapshots()
    result = analyze(comparisons, documents, summaries, rater_email)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def rater_label(rater_id: str) -> str:
    """Display label for a rater, e.g. "A" -> "rater_A". Purely cosmetic --
    `rater_id` itself (the bare letter) remains the join key everywhere."""
    return f"rater_{rater_id}"


def derive_model_labels(comparisons: pd.DataFrame) -> dict[str, str]:
    """Short model labels, derived (not hardcoded) from the dataset itself:
    every summary_id already has the form "<document_id>__<alias>" (e.g.
    "doc_693__gpt4o_elaborate"), and each full model_id maps to exactly one
    such alias (verified across the whole snapshot set). Falls back to the
    full model_id, truncated, for any model_id that -- unexpectedly -- has
    no derivable alias, so labelling never crashes on new/unfamiliar data.
    """
    pairs = pd.concat([
        pd.DataFrame({"summary_id": comparisons["summary_left_id"], "model_id": comparisons["model_left_id"]}),
        pd.DataFrame({"summary_id": comparisons["summary_right_id"], "model_id": comparisons["model_right_id"]}),
    ]).drop_duplicates()

    labels: dict[str, str] = {}
    for model_id, group in pairs.groupby("model_id"):
        aliases = {sid.rsplit("__", 1)[-1] for sid in group["summary_id"] if "__" in sid}
        if len(aliases) == 1:
            labels[model_id] = aliases.pop()
        else:
            # No clean single alias (missing "__" or genuinely inconsistent) --
            # fall back rather than fail; report_snapshots' legend still shows
            # the real model_id alongside this label either way.
            labels[model_id] = model_id if len(model_id) <= 24 else model_id[:21] + "..."
    return labels


def add_pair_key(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Attach `pair_key`: a canonical id for the underlying (document, {left,
    right} summary-pair) being judged, independent of which rater/task saw it.

    Sorted defensively -- in this dataset left/right assignment is always
    stable for a given pair across raters (verified), but sorting means
    agreement is computed correctly even if that ever stops holding.
    """
    out = comparisons.copy()
    lr = np.sort(out[["summary_left_id", "summary_right_id"]].to_numpy(dtype=object), axis=1)
    out["pair_key"] = list(zip(out["document_id"], lr[:, 0], lr[:, 1]))
    return out


@dataclass
class SnapshotAnalysis:
    basic_stats: dict[str, Any]
    multi_scored_counts: dict[int, int]  # {1: n_single_scored, 2: n_bi_scored, 3: n_tri_scored, ...}
    time_overall: dict[str, float]
    time_per_rater: pd.DataFrame  # index=rater_label, columns=[n_tasks, total_sec, mean_sec, median_sec, max_sec]
    model_stats: pd.DataFrame  # index=model label (short), columns=[appearances, wins, win_rate, flags, flag_rate, bt_strength]
    model_labels: dict[str, str]  # full model_id -> short label, for a legend (labels are derived, not arbitrary)
    agreement_overall: dict[str, Any]
    agreement_matrix: pd.DataFrame  # winner-match rate, rater_label x rater_label (diagonal = NaN)
    biscored_matrix: pd.DataFrame  # shared-pair counts, rater_label x rater_label (diagonal = pairs that rater scored)


def compute_basic_stats(
    comparisons: pd.DataFrame, documents: dict, summaries: dict, rater_email: dict
) -> dict[str, Any]:
    return {
        "n_documents": len(documents),
        "n_summaries": len(summaries),
        "n_tasks_annotated": comparisons["annotation_id"].nunique(),
        "n_pairs_scored": len(comparisons),  # rows = (task, pair-position, rater)
        "n_unique_pairs_scored": comparisons["pair_key"].nunique(),  # distinct (doc, summary-pair) judged >=1x
        "n_raters": len(rater_email),
    }


def compute_multi_scored_counts(comparisons: pd.DataFrame) -> dict[int, int]:
    """How many distinct pairs were judged by exactly 1, 2, 3, ... raters."""
    n_raters_per_pair = comparisons.groupby("pair_key")["rater_id"].nunique()
    return n_raters_per_pair.value_counts().sort_index().to_dict()


def compute_time_stats(
    comparisons: pd.DataFrame, outlier_multiplier: float = 8.0
) -> tuple[dict[str, float], pd.DataFrame]:
    """lead_time_sec is recorded per LabelStudio *task* (annotation), identical
    across its 3 pair-rows -- dedupe to annotation_id before aggregating, or
    every rater's time would be triple-counted.

    Some tasks show implausibly large lead_time (LabelStudio counts time a task
    sat open/in a draft, not just active work). `outlier_multiplier` defines a
    cutoff at `outlier_multiplier * median task time`, pooled across all raters;
    tasks above it are excluded from the `*_filtered` stats only -- never from
    `mean_sec`/`median_sec` (the raw figures) or from any other analysis.
    """
    tasks = comparisons.drop_duplicates("annotation_id")
    median_all = tasks["lead_time_sec"].median()
    cutoff_sec = outlier_multiplier * median_all
    is_outlier = tasks["lead_time_sec"] > cutoff_sec
    kept = tasks.loc[~is_outlier, "lead_time_sec"]

    overall = {
        "n_tasks": len(tasks),
        "outlier_multiplier": outlier_multiplier,
        "cutoff_sec": cutoff_sec,
        "n_outliers": int(is_outlier.sum()),
        "total_sec": tasks["lead_time_sec"].sum(),
        "mean_sec": tasks["lead_time_sec"].mean(),
        "median_sec": median_all,
        "mean_sec_filtered": kept.mean(),
        "median_sec_filtered": kept.median(),
    }

    def _per_rater(g: pd.DataFrame) -> pd.Series:
        kept_g = g.loc[g["lead_time_sec"] <= cutoff_sec, "lead_time_sec"]
        return pd.Series({
            "n_tasks": len(g),
            "n_outliers": int((g["lead_time_sec"] > cutoff_sec).sum()),
            "total_sec": g["lead_time_sec"].sum(),
            "mean_sec": g["lead_time_sec"].mean(),
            "median_sec": g["lead_time_sec"].median(),
            "max_sec": g["lead_time_sec"].max(),
            "mean_sec_filtered": kept_g.mean(),
            "median_sec_filtered": kept_g.median(),
        })

    per_rater = tasks.groupby("rater_id").apply(_per_rater, include_groups=False)
    return overall, per_rater


def bradley_terry_strengths(comparisons: pd.DataFrame, models, max_iter: int = 200, tol: float = 1e-10) -> pd.Series:
    """Iterative MM/Zermelo fit of a Bradley-Terry model: strengths s_i > 0
    with P(i beats j) = s_i / (s_i + s_j), normalised to mean 1 across models.

    More robust than raw win rate to this study's non-uniform, actively
    selected pairing (see `selection_bucket`), since it accounts for opponent
    strength rather than treating every matchup as equally informative.
    Ties (score == 0) don't occur in this scale, so every judgement is a
    clean win/loss.
    """
    models = list(models)
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    wins_mat = np.zeros((n, n))
    for l, r, winner in zip(comparisons["model_left_id"], comparisons["model_right_id"], comparisons["winner"]):
        w, loser = (l, r) if winner == 1 else (r, l)
        wins_mat[idx[w], idx[loser]] += 1

    strengths = np.ones(n)
    for _ in range(max_iter):
        new = np.empty(n)
        for i in range(n):
            num = wins_mat[i, :].sum()
            denom = sum(
                (wins_mat[i, j] + wins_mat[j, i]) / (strengths[i] + strengths[j])
                for j in range(n)
                if j != i and (wins_mat[i, j] + wins_mat[j, i]) > 0
            )
            new[i] = num / denom if denom > 0 else strengths[i]
        new *= n / new.sum()
        if np.max(np.abs(new - strengths)) < tol:
            strengths = new
            break
        strengths = new
    return pd.Series(strengths, index=models)


def compute_model_stats(comparisons: pd.DataFrame) -> pd.DataFrame:
    appearances = pd.concat([comparisons["model_left_id"], comparisons["model_right_id"]]).value_counts()
    wins = comparisons["winner_model_id"].value_counts()
    flags_left = comparisons.loc[comparisons["raw_choice"] == "Flag left", "model_left_id"].value_counts()
    flags_right = comparisons.loc[comparisons["raw_choice"] == "Flag right", "model_right_id"].value_counts()
    flags = flags_left.add(flags_right, fill_value=0)

    stats = pd.DataFrame({"appearances": appearances})
    stats["wins"] = wins.reindex(stats.index).fillna(0).astype(int)
    stats["flags"] = flags.reindex(stats.index).fillna(0).astype(int)
    stats["win_rate"] = stats["wins"] / stats["appearances"]
    stats["flag_rate"] = stats["flags"] / stats["appearances"]
    stats["bt_strength"] = bradley_terry_strengths(comparisons, stats.index)
    return stats.sort_values("bt_strength", ascending=False)


def compute_agreement(comparisons: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Inter-rater agreement on *winner* (left/right), computed only over
    pairs that multiple raters actually both scored (bi- or tri-scored pairs).
    """
    raters = sorted(comparisons["rater_id"].unique())
    piv = comparisons.pivot_table(index="pair_key", columns="rater_id", values="winner", aggfunc="first")

    agreement = pd.DataFrame(index=raters, columns=raters, dtype=float)
    biscored = pd.DataFrame(index=raters, columns=raters, dtype="Int64")
    pooled = []
    for i in raters:
        for j in raters:
            if i == j:
                biscored.loc[i, j] = int(piv[i].notna().sum())
                agreement.loc[i, j] = np.nan
                continue
            both = piv[[i, j]].dropna()
            biscored.loc[i, j] = len(both)
            agreement.loc[i, j] = (both[i] == both[j]).mean() if len(both) else np.nan
            if i < j and len(both):
                pooled.append(both.rename(columns={i: "a", j: "b"}))

    if pooled:
        pooled_df = pd.concat(pooled)
        overall = {
            "n_rater_pair_observations": len(pooled_df),
            "winner_agreement_rate": (pooled_df["a"] == pooled_df["b"]).mean(),
        }
    else:
        overall = {"n_rater_pair_observations": 0, "winner_agreement_rate": np.nan}

    return overall, agreement, biscored


def analyze(
    comparisons: pd.DataFrame,
    documents: dict,
    summaries: dict,
    rater_email: dict,
    outlier_multiplier: float = 8.0,
) -> SnapshotAnalysis:
    comparisons = add_pair_key(comparisons)
    time_overall, time_per_rater = compute_time_stats(comparisons, outlier_multiplier)
    agreement_overall, agreement_matrix, biscored_matrix = compute_agreement(comparisons)
    model_labels = derive_model_labels(comparisons)
    model_stats = compute_model_stats(comparisons).rename(index=model_labels)
    model_stats.index.name = "model"

    # Cosmetic-only relabelling of rater ids for display; `comparisons["rater_id"]`
    # (the join key used everywhere above) is untouched.
    time_per_rater = time_per_rater.rename(index=rater_label)
    time_per_rater.index.name = "rater"
    agreement_matrix = agreement_matrix.rename(index=rater_label, columns=rater_label)
    biscored_matrix = biscored_matrix.rename(index=rater_label, columns=rater_label)

    return SnapshotAnalysis(
        basic_stats=compute_basic_stats(comparisons, documents, summaries, rater_email),
        multi_scored_counts=compute_multi_scored_counts(comparisons),
        time_overall=time_overall,
        time_per_rater=time_per_rater,
        model_stats=model_stats,
        model_labels=model_labels,
        agreement_overall=agreement_overall,
        agreement_matrix=agreement_matrix,
        biscored_matrix=biscored_matrix,
    )


if __name__ == "__main__":
    from load_snapshots import load_snapshots

    comparisons, documents, summaries, summary_model, rater_email = load_snapshots()
    result = analyze(comparisons, documents, summaries, rater_email)
    print(result.basic_stats)
    print(result.multi_scored_counts)
    print(result.model_stats)
