#!/usr/bin/env python3
"""Heatmap of learned dimension weights with each judge treated as surrogate human.

For every distinct judge in the G-Eval export, fits weights (other judges → pooled
features) and plots weights as rows (reference judge) × columns (dimension).

Example::

    python -m evaluation_suite.apps.visualize_learned_weights_per_reference_judge \\
        --geval-json-dir .deepeval/geval_exports/llama-2-13b/json \\
        --out evaluation_suite/outputs/learned_weights_by_reference_judge.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation_suite.core.geval_mean import (
    DEFAULT_DIMENSIONS,
    load_geval_rows,
    parse_target_dimension_weights,
)
from evaluation_suite.core.learn_dimension_weights import learn_weights_from_geval_rows


def _unique_judge_ids(rows: list[dict]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for r in rows:
        j = str(r.get("judge_id", ""))
        if j and j not in seen:
            seen.add(j)
            out.append(j)
    return sorted(out)


def _short_label(judge_id: str, max_len: int = 28) -> str:
    s = judge_id.replace("anthropic__", "").replace("google__", "").replace("mistral-", "m.")
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def _fit_all(
    rows: list[dict],
    *,
    judges: list[str],
    ridge: float,
    target_dimension_weights: dict[str, float] | None = None,
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Returns (successful results dicts with keys reference, weights, r2, ...), failures)."""
    ok: list[dict] = []
    failed: list[tuple[str, str]] = []
    for jid in judges:
        try:
            r = learn_weights_from_geval_rows(
                rows,
                reference_judge_substring=jid,
                feature_judge_substrings=None,
                ridge=ridge,
                target_dimension_weights=target_dimension_weights,
            )
            ok.append({"reference": jid, **r})
        except ValueError as e:
            failed.append((jid, str(e)))
    return ok, failed


def _plot_heatmap(
    results: list[dict],
    dimensions: tuple[str, ...],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_j = len(results)
    n_d = len(dimensions)
    mat = np.zeros((n_j, n_d))
    for i, res in enumerate(results):
        for j, dim in enumerate(dimensions):
            mat[i, j] = float(res["weights"].get(dim, 0.0))

    fig_h = max(4.0, 0.45 * n_j + 2.0)
    fig_w = max(7.0, 0.9 * n_d + 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n_d))
    ax.set_xticklabels(list(dimensions), rotation=35, ha="right")
    ax.set_yticks(range(n_j))
    ax.set_yticklabels([_short_label(r["reference"]) for r in results])
    ax.set_xlabel("dimension")
    ax.set_ylabel("reference judge (surrogate human)")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("learned weight")

    for i in range(n_j):
        for j in range(n_d):
            v = mat[i, j]
            if v >= 0.05:
                color = "white" if v > 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot learned weights heatmap with each judge as reference (surrogate human)."
    )
    p.add_argument(
        "--geval-json-dir",
        type=Path,
        default=Path(".deepeval/geval_exports/llama-2-13b/json"),
    )
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument(
        "--target-dimension-weights",
        type=str,
        default=None,
        help="Same as learn_dimension_weights: dim=value in [0,1], omitted dims=0, "
        "normalized to sum 1. Omitted = equal weights over dimensions.",
    )
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("evaluation_suite/outputs/learned_weights_by_reference_judge.png"),
        help="Output PNG path",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON path with per-judge weights and metrics",
    )
    args = p.parse_args()

    rows = load_geval_rows(args.geval_json_dir, max_files=args.max_files)
    judges = _unique_judge_ids(rows)
    target_tw: dict[str, float] | None
    if not args.target_dimension_weights:
        target_tw = None
    else:
        try:
            target_tw = parse_target_dimension_weights(args.target_dimension_weights)
        except ValueError as e:
            print(f"error: --target-dimension-weights: {e}", file=sys.stderr)
            sys.exit(2)
    results, failed = _fit_all(
        rows, judges=judges, ridge=args.ridge, target_dimension_weights=target_tw
    )

    for jid, err in failed:
        print(f"[skip] {jid}: {err}", file=sys.stderr)

    if len(results) < 1:
        print("No successful fits; nothing to plot.", file=sys.stderr)
        sys.exit(1)

    if target_tw is None:
        tgt = "ref judge equal-mean over dims"
    else:
        tgt = "ref judge weighted-mean over dims"
    title = f"Learned dimension weights (target = {tgt}; features = other judges pooled)"
    _plot_heatmap(results, DEFAULT_DIMENSIONS, args.out, title=title)
    print(f"Wrote {args.out}")
    print("R² (other judges → ref target aggregate):")
    for r in results:
        print(f"  {_short_label(r['reference'], 40):40s}  R²={r['r2']:.4f}  n_ck={r['n_checkpoints']}")

    if args.json_out is not None:
        payload = {
            "geval_json_dir": str(args.geval_json_dir),
            "dimensions": list(DEFAULT_DIMENSIONS),
            "ridge": args.ridge,
            "target_dimension_weights_spec": args.target_dimension_weights,
            "fits": [
                {
                    "reference": r["reference"],
                    "weights": r["weights"],
                    "r2": r["r2"],
                    "rmse": r["rmse"],
                    "n_checkpoints": r["n_checkpoints"],
                }
                for r in results
            ],
            "failed": [{"reference": j, "error": e} for j, e in failed],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
