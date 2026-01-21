#!/usr/bin/env python3
"""
Use sampled token-band IDs from old_new_splits_token_bands_o200k.json to:

- Create four band-specific JSONL files for the 202601 dataset:
    * 168161_text_summary_examples_ultra_narrow.jsonl
    * 168161_text_summary_examples_narrow.jsonl
    * 168161_text_summary_examples_medium.jsonl
    * 168161_text_summary_examples_broad.jsonl

- Remove all sampled IDs (across all bands) from:
    * 168161_text_summary_examples.jsonl
    * 168161_text_summary_examples_train.jsonl
    * 168161_text_summary_examples_val.jsonl
    * 168161_text_summary_examples_test.jsonl

Run this script from the cleaned_datasets directory:

    cd datasets_from_demokratibasen/cleaned_datasets
    python update_text_summary_dataset_202601_from_bands.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Set


BASE_DIR = Path(".")

BANDS = ["ULTRA_NARROW", "NARROW", "MEDIUM", "BROAD"]

DATASET_DIR = BASE_DIR / "text_summary_dataset_202601"

MASTER_FILE = DATASET_DIR / "168161_text_summary_examples.jsonl"
TRAIN_FILE = DATASET_DIR / "168161_text_summary_examples_train.jsonl"
VAL_FILE = DATASET_DIR / "168161_text_summary_examples_val.jsonl"
TEST_FILE = DATASET_DIR / "168161_text_summary_examples_test.jsonl"

OUTPUT_BAND_FILES = {
    "ULTRA_NARROW": DATASET_DIR / "168161_text_summary_examples_ultra_narrow.jsonl",
    "NARROW": DATASET_DIR / "168161_text_summary_examples_narrow.jsonl",
    "MEDIUM": DATASET_DIR / "168161_text_summary_examples_medium.jsonl",
    "BROAD": DATASET_DIR / "168161_text_summary_examples_broad.jsonl",
}

STATS_JSON = BASE_DIR / "old_new_splits_token_bands_o200k.json"


def load_sampled_ids() -> Dict[str, Set[str]]:
    """Load SAMPLED_* id sets per band from the stats JSON."""
    with open(STATS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    doc_ids = data.get("doc_ids", {})

    sampled_by_band: Dict[str, Set[str]] = {}
    for band in BANDS:
        key = f"SAMPLED_{band}"
        ids = doc_ids.get(key)
        if ids is None:
            raise KeyError(f"Expected key '{key}' in doc_ids of {STATS_JSON}")
        sampled_by_band[band] = set(ids)
    return sampled_by_band


def build_id_to_example(path: Path, wanted_ids: Set[str]) -> Dict[str, dict]:
    """
    Build a mapping from dokument_id -> example JSON for the given file,
    restricted to wanted_ids.
    """
    id_to_example: Dict[str, dict] = {}
    if not wanted_ids:
        return id_to_example

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            metadata = ex.get("metadata", {})
            doc_id = metadata.get("dokument_id")
            if not doc_id:
                continue
            if doc_id in wanted_ids and doc_id not in id_to_example:
                id_to_example[doc_id] = ex
                # Optional early-exit if we've found all
                if len(id_to_example) == len(wanted_ids):
                    break

    missing = wanted_ids - set(id_to_example.keys())
    if missing:
        print(
            f"WARNING: {len(missing)} requested IDs not found in {path.name} "
            f"(e.g. {list(sorted(missing))[:5]} ...)"
        )
    return id_to_example


def write_band_files(
    sampled_by_band: Dict[str, Set[str]],
    id_to_example: Dict[str, dict],
) -> None:
    """Write the four band-specific JSONL files."""
    for band, out_path in OUTPUT_BAND_FILES.items():
        ids = sampled_by_band.get(band, set())
        print(f"Writing {len(ids):,} examples to {out_path} ...")
        with open(out_path, "w", encoding="utf-8") as f:
            for doc_id in sorted(ids):
                ex = id_to_example.get(doc_id)
                if ex is None:
                    # Skip missing examples; already warned earlier.
                    continue
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def filter_file(path: Path, remove_ids: Set[str]) -> None:
    """
    Rewrite a JSONL file, dropping any examples whose dokument_id is in remove_ids.
    Operates in a simple temp-file + replace fashion.
    """
    if not path.exists():
        print(f"WARNING: {path} does not exist; skipping.")
        return

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    kept = 0
    removed = 0

    with open(path, "r", encoding="utf-8") as src, open(tmp_path, "w", encoding="utf-8") as dst:
        for line in src:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            try:
                ex = json.loads(line_stripped)
            except json.JSONDecodeError:
                # Preserve malformed lines just in case
                dst.write(line)
                kept += 1
                continue

            metadata = ex.get("metadata", {})
            doc_id = metadata.get("dokument_id")
            if doc_id and doc_id in remove_ids:
                removed += 1
                continue

            dst.write(json.dumps(ex, ensure_ascii=False) + "\n")
            kept += 1

    tmp_path.replace(path)
    print(f"Filtered {path.name}: kept {kept:,} docs, removed {removed:,} docs")


def main() -> int:
    if not MASTER_FILE.exists():
        raise SystemExit(f"Master file not found: {MASTER_FILE}")

    print(f"Loading sampled IDs from {STATS_JSON} ...")
    sampled_by_band = load_sampled_ids()

    # Union of all sampled IDs across bands
    all_sampled_ids: Set[str] = set().union(*sampled_by_band.values())
    print(f"Total unique sampled IDs across all bands: {len(all_sampled_ids):,}")

    # Build mapping from ID -> example using the master file
    print(f"Building ID -> example map from {MASTER_FILE.name} ...")
    id_to_example = build_id_to_example(MASTER_FILE, all_sampled_ids)
    print(f"Found {len(id_to_example):,} of {len(all_sampled_ids):,} sampled IDs in master file")

    # 1) Write band-specific JSONL files
    write_band_files(sampled_by_band, id_to_example)

    # 2) Filter sampled IDs out of master and split files
    print("Filtering sampled IDs out of master and split files ...")
    for path in [MASTER_FILE, TRAIN_FILE, VAL_FILE, TEST_FILE]:
        filter_file(path, all_sampled_ids)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

