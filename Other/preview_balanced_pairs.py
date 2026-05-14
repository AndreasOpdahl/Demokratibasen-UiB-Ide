"""Build the pairs_table for one eval folder using balanced sampling and report stats.

Run this *before* launching the LLM pipeline to verify that the balanced sampler in
:func:`pairwise_eval.pairs.build_pairs_table` distributes pair selections evenly across
models and unordered pairs. The resulting pairs table is written to the repo root as
``preview_pairs_table__<folder>.json`` so it can be inspected (and, if desired, passed
in later via ``EXTEND_PAIRS_TABLE_JSON``).

Usage::

    python Other/preview_balanced_pairs.py norwai-mistral-7b
    python Other/preview_balanced_pairs.py norwai-mistral-7b --n-pairs 8 --max-docs 1000
    python Other/preview_balanced_pairs.py norwai-mistral-7b --no-balanced  # vanilla for compare
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pairwise_eval.config import DEFAULT_PAIR_SEED, N_PAIRS_PER_DOCUMENT
from pairwise_eval.data import load_eval_jsonl_long_df, long_df_head_documents
from pairwise_eval.pairs import build_pairs_table


def _summarize(df, label: str) -> None:
    if df.empty:
        print(f"[{label}] (empty)")
        return
    c_model: collections.Counter = collections.Counter()
    c_pair: collections.Counter = collections.Counter()
    for left, right in zip(df["left"], df["right"]):
        c_model[left] += 1
        c_model[right] += 1
        c_pair[frozenset((left, right))] += 1
    vals = sorted(c_model.values())
    pv = sorted(c_pair.values())
    docs = df["doc_id"].nunique()
    n_pairs_doc = df.groupby("doc_id").size()
    print(
        f"[{label}] docs={docs} rows={len(df)} pairs/doc: "
        f"min={int(n_pairs_doc.min())} max={int(n_pairs_doc.max())} mean={n_pairs_doc.mean():.2f}"
    )
    print(
        f"           per-model appearances: min={vals[0]} max={vals[-1]} "
        f"mean={statistics.mean(vals):.1f} std={statistics.pstdev(vals):.2f} "
        f"ratio={vals[-1]/max(vals[0],1):.2f}"
    )
    print(
        f"           per-pair selections:   unique={len(pv)} min={pv[0]} max={pv[-1]} "
        f"mean={statistics.mean(pv):.2f} std={statistics.pstdev(pv):.2f}"
    )
    return c_model, c_pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "folder",
        help="Eval-data folder name under Data/eval (e.g. 'norwai-mistral-7b').",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=N_PAIRS_PER_DOCUMENT,
        help=f"Pairs per document (default from config: {N_PAIRS_PER_DOCUMENT}).",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Subset to first N docs (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_PAIR_SEED,
        help=f"RNG seed (default from config: {DEFAULT_PAIR_SEED}).",
    )
    parser.add_argument(
        "--no-balanced",
        action="store_true",
        help="Use the vanilla uniform sampler instead of the balanced one (for comparison).",
    )
    parser.add_argument(
        "--also-vanilla",
        action="store_true",
        help="In addition to the chosen mode, also build with the vanilla sampler and print its stats.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the JSON (default: repo-root/preview_pairs_table__<folder>.json).",
    )
    args = parser.parse_args()

    eval_dir = REPO_ROOT / "Data" / "eval" / args.folder
    if not eval_dir.is_dir():
        raise SystemExit(f"Eval folder not found: {eval_dir}")

    print(f"Loading: {eval_dir}", flush=True)
    long_df = load_eval_jsonl_long_df(eval_dir)
    print(f"  rows: {len(long_df)}, docs: {long_df['doc_id'].nunique()}, "
          f"models: {long_df['model_id'].nunique()}")

    if args.max_docs is not None:
        long_df = long_df_head_documents(long_df, args.max_docs)
        print(f"  subset to first {args.max_docs} docs → {len(long_df)} rows, "
              f"{long_df['doc_id'].nunique()} docs")

    balanced = not args.no_balanced
    mode_label = "balanced" if balanced else "vanilla"
    print(
        f"Building pairs (mode={mode_label}, n_pairs/doc={args.n_pairs}, seed={args.seed})",
        flush=True,
    )
    pairs = build_pairs_table(
        long_df, n_pairs=args.n_pairs, seed=args.seed, balanced=balanced
    )
    _summarize(pairs, mode_label)

    if args.also_vanilla and balanced:
        print("Also building vanilla for comparison...", flush=True)
        pairs_vanilla = build_pairs_table(
            long_df, n_pairs=args.n_pairs, seed=args.seed, balanced=False
        )
        _summarize(pairs_vanilla, "vanilla")

    out_path = args.out or (REPO_ROOT / f"preview_pairs_table__{args.folder}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = pairs.to_dict(orient="records")
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path} ({len(records)} rows)")


if __name__ == "__main__":
    main()
