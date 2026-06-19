#!/usr/bin/env python3
"""Human vs LLM agreement on per-dimension annotation CSVs (confusion matrices)."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from human_annotation.export import is_llm_vote_column, judge_id_from_column
from human_annotation.config import SELECTION_BUCKET_ORDER

LABELS = ("left", "right", "tie")
BUCKET_ORDER = SELECTION_BUCKET_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report human vs LLM judge agreement for one or more dimension CSVs.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Annotation CSV file(s), e.g. outputs/winners/relevance.csv",
    )
    return parser.parse_args()


def _confusion(human: list[str], llm: list[str]) -> dict[tuple[str, str], int]:
    table: Counter[tuple[str, str]] = Counter()
    for h, l in zip(human, llm):
        table[(h, l)] += 1
    return dict(table)


def _print_matrix(title: str, matrix: dict[tuple[str, str], int]) -> None:
    print(f"\n{title}")
    header = "human \\ llm".ljust(14) + "".join(lbl.rjust(10) for lbl in LABELS)
    print(header)
    for h in LABELS:
        row = h.ljust(14)
        for l in LABELS:
            row += str(matrix.get((h, l), 0)).rjust(10)
        print(row)


def _accuracy(human: list[str], llm: list[str]) -> float:
    if not human:
        return 0.0
    return sum(h == l for h, l in zip(human, llm)) / len(human)


def analyze_file(path: Path) -> None:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        print(f"{path}: empty")
        return

    dimension = rows[0].get("dimension", path.stem)
    judge_cols = [c for c in rows[0] if is_llm_vote_column(c)]
    annotated = [r for r in rows if (r.get("human_choice") or "").strip()]

    print(f"\n{'=' * 60}")
    print(f"{path.name}  (dimension={dimension})")
    print(f"Human labels: {len(annotated)} / {len(rows)}")

    if not annotated:
        print("No human_choice values yet.")
        return

    for judge_col in judge_cols:
        pairs_h, pairs_l = [], []
        for r in annotated:
            llm = (r.get(judge_col) or "").strip().lower()
            if llm in LABELS:
                pairs_h.append(r["human_choice"].strip().lower())
                pairs_l.append(llm)
        if not pairs_h:
            continue
        acc = _accuracy(pairs_h, pairs_l)
        matrix = _confusion(pairs_h, pairs_l)
        judge_name = judge_id_from_column(judge_col)
        print(
            f"\n{judge_name}: accuracy {acc:.1%} "
            f"({sum(1 for h, l in zip(pairs_h, pairs_l) if h == l)}/{len(pairs_h)})"
        )
        _print_matrix(f"Confusion matrix ({judge_name})", matrix)

    majority_pairs_h, majority_pairs_l = [], []
    for r in annotated:
        maj = (r.get("llm_majority") or "").strip().lower()
        if maj in LABELS:
            majority_pairs_h.append(r["human_choice"].strip().lower())
            majority_pairs_l.append(maj)
    if majority_pairs_h:
        acc = _accuracy(majority_pairs_h, majority_pairs_l)
        print(f"\nllm_majority: accuracy {acc:.1%}")
        _print_matrix("Confusion matrix (llm_majority)", _confusion(majority_pairs_h, majority_pairs_l))

    buckets = sorted({(r.get("selection_bucket") or "").strip() for r in annotated if (r.get("selection_bucket") or "").strip()})
    ordered = [b for b in BUCKET_ORDER if b in buckets] + [b for b in buckets if b not in BUCKET_ORDER]
    if ordered:
        print(f"\n--- By selection bucket ---")
        for bucket in ordered:
            bucket_rows = [r for r in annotated if (r.get("selection_bucket") or "").strip() == bucket]
            print(f"\n[{bucket}] n={len(bucket_rows)}")
            for judge_col in judge_cols:
                pairs_h, pairs_l = [], []
                for r in bucket_rows:
                    llm = (r.get(judge_col) or "").strip().lower()
                    if llm in LABELS:
                        pairs_h.append(r["human_choice"].strip().lower())
                        pairs_l.append(llm)
                if not pairs_h:
                    continue
                judge_name = judge_id_from_column(judge_col)
                acc = _accuracy(pairs_h, pairs_l)
                print(f"  {judge_name}: {acc:.1%} ({sum(h == l for h, l in zip(pairs_h, pairs_l))}/{len(pairs_h)})")
            majority_h, majority_l = [], []
            for r in bucket_rows:
                maj = (r.get("llm_majority") or "").strip().lower()
                if maj in LABELS:
                    majority_h.append(r["human_choice"].strip().lower())
                    majority_l.append(maj)
            if majority_h:
                acc = _accuracy(majority_h, majority_l)
                print(f"  llm_majority: {acc:.1%} ({sum(h == l for h, l in zip(majority_h, majority_l))}/{len(majority_h)})")


def main() -> None:
    args = parse_args()
    for path in args.paths:
        if path.is_dir():
            for csv_path in sorted(path.glob("*.csv")):
                analyze_file(csv_path)
        else:
            analyze_file(path)


if __name__ == "__main__":
    main()
