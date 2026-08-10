#!/usr/bin/env python3
"""Print a human-readable report over a loaded evaluation snapshot.

Covers: basic stats, time spent on task (overall + per rater, with an
adjustable outlier cutoff and a plot), per-model win rate / Bradley-Terry
strength / flag rate, and inter-rater agreement on winner (overall + rater x
rater matrix with shared-pair counts).

Usage:
    python report_snapshots.py
    python report_snapshots.py --outlier-multiplier 5
    python report_snapshots.py --pattern 'batch0[1-8].*onlyids.*\\.json$'
    python report_snapshots.py --folder /some/other/snapshot/folder
"""

import argparse
from pathlib import Path

import pandas as pd

from analyze_snapshots import SnapshotAnalysis, analyze, rater_label
from load_snapshots import DEFAULT_SNAPSHOT_PATTERN, SNAPSHOT_FOLDER, load_snapshots

DEFAULT_PLOT_PATH = Path(__file__).parent / "task_time_distribution.png"

# Categorical slots (fixed order, from the shared palette) assigned to raters
# by rater_id, never by rank/value. Falls back to a neutral grey past slot 8.
RATER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MEDIAN_COLOR = "#8a897f"  # muted grey: reference line, not a series
CUTOFF_COLOR = "#d03b3b"  # status "critical": marks the outlier boundary


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%" if pd.notna(x) else "n/a"


def _hms(seconds: float) -> str:
    if pd.isna(seconds):
        return "n/a"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def print_basic_stats(result: SnapshotAnalysis) -> None:
    b = result.basic_stats
    print("\n=== Basic stats ===")
    print(f"Documents:              {b['n_documents']}")
    print(f"Summaries:              {b['n_summaries']}")
    print(f"Tasks annotated:        {b['n_tasks_annotated']}")
    print(f"Pairs scored (rows):    {b['n_pairs_scored']}")
    print(f"Unique pairs scored:    {b['n_unique_pairs_scored']}")
    print(f"Raters:                 {b['n_raters']}")
    print()
    print("Pairs by number of raters who scored them:")
    labels = {1: "single-scored", 2: "bi-scored", 3: "tri-scored"}
    for k, v in sorted(result.multi_scored_counts.items()):
        print(f"  {labels.get(k, f'{k}x-scored'):<15} {v}")


def print_time_stats(result: SnapshotAnalysis, plot_path: str | None) -> None:
    o = result.time_overall
    print("\n=== Time spent ===")
    print(f"Total (all raters):     {_hms(o['total_sec'])}  ({o['n_tasks']} tasks)")
    print(f"Mean / task (raw):      {_hms(o['mean_sec'])}")
    print(f"Median / task (raw):    {_hms(o['median_sec'])}")
    print(
        f"Outlier cutoff:         {_hms(o['cutoff_sec'])}  "
        f"({o['outlier_multiplier']}x raw median) -- {o['n_outliers']} of {o['n_tasks']} "
        f"tasks ({_pct(o['n_outliers'] / o['n_tasks'])}) excluded below"
    )
    print(f"Mean / task (filtered):   {_hms(o['mean_sec_filtered'])}")
    print(f"Median / task (filtered): {_hms(o['median_sec_filtered'])}")
    print(
        "(raw mean is skewed by idle/draft time some tasks were left open for -- "
        "median barely moves once outliers are removed, confirming it was already robust)"
    )
    print()
    tbl = result.time_per_rater.rename(columns={"n_tasks": "tasks", "n_outliers": "outliers"}).copy()
    for c in ("total_sec", "mean_sec", "median_sec", "max_sec", "mean_sec_filtered", "median_sec_filtered"):
        tbl[c] = tbl[c].map(_hms)
    print(tbl.to_string())
    if plot_path:
        print(f"\nTask-time plot written to: {plot_path}")


def print_model_stats(result: SnapshotAnalysis) -> None:
    print("\n=== Model win rates, strength, and flags ===")
    tbl = result.model_stats.copy()
    tbl["win_rate"] = tbl["win_rate"].map(_pct)
    tbl["flag_rate"] = tbl["flag_rate"].map(_pct)
    tbl["bt_strength"] = tbl["bt_strength"].round(3)
    print(tbl.to_string())
    print()
    print("bt_strength: Bradley-Terry model strength (mean 1 across models); accounts for")
    print("opponent strength given the non-uniform pair selection, unlike raw win_rate.")
    print()
    print("Model label legend:")
    for full_id, label in sorted(result.model_labels.items(), key=lambda kv: kv[1]):
        print(f"  {label:<16} {full_id}")


def print_agreement(result: SnapshotAnalysis) -> None:
    o = result.agreement_overall
    print("\n=== Inter-rater agreement (winner: left vs right) ===")
    print(
        f"Overall (pooled over all rater pairs): {_pct(o['winner_agreement_rate'])}  "
        f"(n={o['n_rater_pair_observations']} shared pair-judgements)"
    )
    print()
    print("Agreement matrix (% of shared pairs where winner matched):")
    print(result.agreement_matrix.map(lambda x: _pct(x) if pd.notna(x) else "-").to_string())
    print()
    print("Shared (bi/tri-scored) pair counts (diagonal = pairs that rater scored):")
    print(result.biscored_matrix.to_string())


def plot_task_times(
    comparisons: pd.DataFrame,
    cutoff_sec: float,
    outlier_multiplier: float,
    out_path: str,
) -> str:
    """Save a two-panel PNG: pooled histogram of per-task time (log-scale
    minutes) with the outlier cutoff marked as a vertical bar, and a
    per-rater box plot on the same scale. Returns the path written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    tasks = comparisons.drop_duplicates("annotation_id").copy()
    tasks["minutes"] = tasks["lead_time_sec"] / 60
    median_min = tasks["minutes"].median()
    cutoff_min = cutoff_sec / 60
    n_outliers = int((tasks["minutes"] > cutoff_min).sum())

    raters = sorted(tasks["rater_id"].unique())
    colors = dict(zip(raters, RATER_COLORS))

    fig, (ax_hist, ax_box) = plt.subplots(
        2, 1, figsize=(9, 6.5), gridspec_kw={"height_ratios": [2, 1.2]}, sharex=True
    )

    bins = np.logspace(np.log10(tasks["minutes"].min()), np.log10(tasks["minutes"].max()), 40)
    ax_hist.hist(tasks["minutes"], bins=bins, color=RATER_COLORS[0], edgecolor="white", linewidth=0.4)
    ax_hist.axvline(
        median_min, color=MEDIAN_COLOR, linestyle=":", linewidth=1.5, label=f"median = {median_min:.1f} min"
    )
    ax_hist.axvline(
        cutoff_min,
        color=CUTOFF_COLOR,
        linestyle="--",
        linewidth=2,
        label=f"outlier cutoff = {outlier_multiplier:g}x median = {cutoff_min:.1f} min",
    )
    ax_hist.set_ylabel("tasks")
    ax_hist.set_title(f"Time spent per task (n={len(tasks)}, {n_outliers} above cutoff)")
    ax_hist.legend(loc="upper right", frameon=False)
    ax_hist.set_xscale("log")

    box_data = [tasks.loc[tasks["rater_id"] == r, "minutes"] for r in raters]
    bp = ax_box.boxplot(
        box_data, vert=False, tick_labels=[rater_label(r) for r in raters], patch_artist=True, widths=0.6,
        medianprops={"color": "#0b0b0b"},
    )
    for patch, r in zip(bp["boxes"], raters):
        patch.set_facecolor(colors[r])
        patch.set_alpha(0.75)
    ax_box.axvline(cutoff_min, color=CUTOFF_COLOR, linestyle="--", linewidth=2)
    ax_box.set_xlabel("minutes (log scale)")
    ax_box.set_ylabel("rater")
    ax_box.set_xscale("log")
    ax_box.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))

    for ax in (ax_hist, ax_box):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def print_report(result: SnapshotAnalysis, plot_path: str | None) -> None:
    print_basic_stats(result)
    print_time_stats(result, plot_path)
    print_model_stats(result)
    print_agreement(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", default=SNAPSHOT_FOLDER, help="snapshot folder (default: SNAPSHOT_FOLDER)")
    parser.add_argument(
        "--pattern", default=DEFAULT_SNAPSHOT_PATTERN, help="regex selecting which *.json snapshots to load"
    )
    parser.add_argument(
        "--outlier-multiplier", type=float, default=8.0,
        help="tasks with lead_time > this x the raw median are treated as outliers for the "
             "filtered mean/median and marked on the plot (default: 8, ~10min on current data)",
    )
    parser.add_argument(
        "--plot-out", default=str(DEFAULT_PLOT_PATH), help="where to write the task-time plot PNG"
    )
    parser.add_argument("--no-plot", action="store_true", help="skip writing the task-time plot")
    args = parser.parse_args()

    comparisons, documents, summaries, summary_model, rater_email = load_snapshots(args.folder, args.pattern)
    result = analyze(comparisons, documents, summaries, rater_email, outlier_multiplier=args.outlier_multiplier)

    plot_path = None
    if not args.no_plot:
        plot_path = plot_task_times(
            comparisons, result.time_overall["cutoff_sec"], args.outlier_multiplier, args.plot_out
        )

    print_report(result, plot_path)


if __name__ == "__main__":
    main()
