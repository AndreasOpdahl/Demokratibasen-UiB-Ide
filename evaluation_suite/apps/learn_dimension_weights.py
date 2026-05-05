#!/usr/bin/env python3
"""Learn dimension weights from a reference judge (e.g. Claude) as surrogate human.

Loads G-Eval JSON exports, builds per-checkpoint per-dimension mean win rates for the
reference judge (target = weighted or equal mean across dimensions) and for pooled other
judges (features). Fits nonnegative weights summing to 1 that best predict the reference
aggregate in least-squares sense.

Example::

    python -m evaluation_suite.apps.learn_dimension_weights \\
        --geval-json-dir .deepeval/geval_exports/llama-2-13b/json \\
        --reference-judge claude-3-5-haiku \\
        --out evaluation_suite/outputs/learned_dimension_weights.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation_suite.core.geval_mean import load_geval_rows, parse_target_dimension_weights
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


def main() -> None:
    p = argparse.ArgumentParser(
        description="Learn simplex dimension weights from a reference judge (surrogate human)."
    )
    p.add_argument(
        "--geval-json-dir",
        type=Path,
        default=Path(".deepeval/geval_exports/llama-2-13b/json"),
        help="Directory with geval__*__*.json exports",
    )
    p.add_argument(
        "--reference-judge",
        type=str,
        required=True,
        help="Substring of judge_id for the reference (e.g. claude-3-5-haiku). "
        "Target per checkpoint is the mean of that judge's per-dimension means.",
    )
    p.add_argument(
        "--feature-judges",
        type=str,
        default=None,
        help="Comma-separated judge substrings to pool as features. "
        "Default: all rows whose judge is not the reference judge.",
    )
    p.add_argument(
        "--target-dimension-weights",
        type=str,
        default=None,
        help="Comma-separated dim=value for the reference judge target y. "
        "Each value must be in [0, 1]; dimensions not listed get 0; values are normalized "
        "to sum to 1. Example: faithfulness=0.35,hygiene=0.35,correctness=0.1,... "
        "Omitted = equal weight on all dimensions.",
    )
    p.add_argument("--ridge", type=float, default=1e-6, help="L2 penalty on weights")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument(
        "--list-judges",
        action="store_true",
        help="Print distinct judge_id values found in the exports and exit",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON result (weights, metrics, per-checkpoint y vs predicted)",
    )
    args = p.parse_args()

    rows = load_geval_rows(args.geval_json_dir, max_files=args.max_files)
    if args.list_judges:
        for jid in _unique_judge_ids(rows):
            print(jid)
        return

    feature_subs = None
    if args.feature_judges:
        feature_subs = [s.strip() for s in args.feature_judges.split(",") if s.strip()]

    target_tw: dict[str, float] | None
    if not args.target_dimension_weights:
        target_tw = None
    else:
        try:
            target_tw = parse_target_dimension_weights(args.target_dimension_weights)
        except ValueError as e:
            print(f"error: --target-dimension-weights: {e}", file=sys.stderr)
            sys.exit(2)

    result = learn_weights_from_geval_rows(
        rows,
        reference_judge_substring=args.reference_judge,
        feature_judge_substrings=feature_subs,
        ridge=args.ridge,
        target_dimension_weights=target_tw,
    )
    result["geval_json_dir"] = str(args.geval_json_dir)

    print("Reference target aggregate (normalized weights on proxy judge dims):")
    for dim, wt in sorted(result["reference_aggregate_weights"].items()):
        print(f"  {dim}: {wt:.6f}")
    print("Learned weights (nonnegative, sum to 1):")
    for dim, wt in sorted(result["weights"].items()):
        print(f"  {dim}: {wt:.6f}")
    print(f"n_checkpoints: {result['n_checkpoints']}")
    print(f"RMSE: {result['rmse']:.6f}  R^2: {result['r2']:.6f}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
