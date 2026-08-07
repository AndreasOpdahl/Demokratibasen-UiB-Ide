#!/usr/bin/env python3
"""Build per-dimension human annotation files for LLM judge validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from human_annotation.checkpoint_io import load_from_checkpoints  # noqa: E402
from human_annotation.config import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CONTEXT_EXPORT_DIR,
    DEFAULT_GEVAL_EXPORT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PER_DIMENSION,
    DEFAULT_SEED,
    DEFAULT_SELECTION_RATIOS,
    DIMENSION_COLORS,
    DIMENSIONS,
)
from human_annotation.export import (  # noqa: E402
    build_combined_export,
    build_export_items,
    build_export_rows,
    write_csv,
    write_json,
    write_label_studio_csv,
)
from human_annotation.geval_io import load_geval_export  # noqa: E402
from human_annotation.selection import score_records, select_all_dimensions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build independent human-annotation files per dimension "
            "for LLM judge validation."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
    )
    parser.add_argument(
        "--context-export-dir",
        type=Path,
        default=DEFAULT_CONTEXT_EXPORT_DIR,
    )
    parser.add_argument(
        "--from-export",
        action="store_true",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=DEFAULT_GEVAL_EXPORT_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "-n",
        "--per-dimension",
        type=int,
        default=DEFAULT_PER_DIMENSION,
        metavar="N",
        help=f"Pairs to select per dimension (default: {DEFAULT_PER_DIMENSION}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--name",
        default="winners",
        help="Output subfolder under outputs/ (default: winners).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = (args.output_dir / args.name).resolve()
    load_stats: dict = {}

    if args.from_export:
        records = load_geval_export(args.export_dir.resolve())
        load_stats = {"source": "geval_export", "export_dir": str(args.export_dir.resolve())}
    else:
        records, load_stats = load_from_checkpoints(
            args.checkpoint_dir.resolve(),
            context_export_dir=args.context_export_dir.resolve(),
        )
        load_stats["source"] = "judgment_checkpoints"

    if args.per_dimension < 1:
        raise SystemExit("--per-dimension must be at least 1")

    scored = score_records(records)
    by_dim, metadata = select_all_dimensions(
        scored,
        per_dimension=args.per_dimension,
        dimensions=DIMENSIONS,
        ratios=DEFAULT_SELECTION_RATIOS,
        seed=args.seed,
    )

    metadata.update(load_stats)
    metadata["per_dimension_requested_by_user"] = args.per_dimension
    metadata["dimension_colors"] = dict(DIMENSION_COLORS)
    metadata["outputs"] = {}

    csv_rows_by_dimension: dict[str, list] = {}
    json_items_by_dimension: dict[str, list] = {}

    for dimension in DIMENSIONS:
        rows = by_dim[dimension]
        csv_rows = build_export_rows(rows, dimension=dimension)
        json_items = build_export_items(rows, dimension=dimension)
        csv_rows_by_dimension[dimension] = csv_rows
        json_items_by_dimension[dimension] = json_items

        csv_path = output_root / f"{dimension}.csv"
        json_path = output_root / f"{dimension}.json"

        write_csv(csv_path, csv_rows)
        ls_csv_path = output_root / f"{dimension}_ls.csv"
        write_label_studio_csv(ls_csv_path, csv_rows)
        write_json(
            json_path,
            {
                "metadata": metadata["by_dimension"][dimension],
                "dimension_colors": dict(DIMENSION_COLORS),
                "annotation_instructions": "Choose one: left / right / tie",
                "items": json_items,
            },
        )
        metadata["outputs"][dimension] = {
            "csv": str(csv_path),
            "csv_ls": str(ls_csv_path),
            "json": str(json_path),
        }

    combined_csv_rows, combined_json_items = build_combined_export(
        csv_rows_by_dimension,
        json_items_by_dimension,
        dimensions=DIMENSIONS,
    )
    all_csv_path = output_root / "all.csv"
    all_json_path = output_root / "all.json"
    write_csv(all_csv_path, combined_csv_rows)
    all_ls_csv_path = output_root / "all_ls.csv"
    write_label_studio_csv(all_ls_csv_path, combined_csv_rows)
    write_json(
        all_json_path,
        {
            "metadata": {
                "total_items": len(combined_json_items),
                "per_dimension": metadata["per_dimension_selected"],
                "dimension_colors": dict(DIMENSION_COLORS),
                "pair_overlap_between_dimensions": metadata.get(
                    "pair_overlap_between_dimensions", {}
                ),
            },
            "annotation_instructions": "Choose one: left / right / tie",
            "items": combined_json_items,
        },
    )
    metadata["outputs"]["all"] = {
        "csv": str(all_csv_path),
        "csv_ls": str(all_ls_csv_path),
        "json": str(all_json_path),
    }

    meta_path = output_root / "selection_metadata.json"
    write_json(meta_path, metadata)
    metadata["outputs"]["selection_metadata"] = str(meta_path)

    print(f"Source: {load_stats.get('source')}")
    print(f"Pool: {load_stats.get('distinct_doc_ids', '?')} docs, {load_stats.get('distinct_pairs', '?')} pairs")
    for dimension in DIMENSIONS:
        n = len(by_dim[dimension])
        print(f"  {dimension}: {n} pairs -> {output_root / f'{dimension}.csv'} (+ {dimension}_ls.csv)")
    print(f"  all: {len(combined_csv_rows)} pairs -> {all_csv_path} (+ all_ls.csv)")
    print(f"Pair overlap across dims (sample): {metadata.get('pair_overlap_between_dimensions', {})}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
