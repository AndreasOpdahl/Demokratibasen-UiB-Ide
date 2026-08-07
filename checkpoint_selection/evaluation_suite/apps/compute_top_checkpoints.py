#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation_suite.core.geval_mean import (
    compute_checkpoint_weighted_means,
    load_geval_rows,
    parse_dimension_weights,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Compute weighted mean win-rate checkpoints from G-Eval JSON.")
    p.add_argument(
        "--geval-json-dir",
        type=Path,
        default=Path(".deepeval/geval_exports/llama-2-13b/json"),
    )
    p.add_argument(
        "--dimension-weights",
        type=str,
        default="faithfulness=1,correctness=1,completeness=1,newsworthiness=1,hygiene=1",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-files", type=int, default=None)
    args = p.parse_args()

    rows = load_geval_rows(args.geval_json_dir, max_files=args.max_files)
    weights = parse_dimension_weights(args.dimension_weights)
    scores = compute_checkpoint_weighted_means(rows, weights)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    print("Top checkpoints:")
    for ck, score in ranked[: max(1, args.top_k)]:
        print(f"checkpoint-{ck}: {score:.6f}")


if __name__ == "__main__":
    main()

