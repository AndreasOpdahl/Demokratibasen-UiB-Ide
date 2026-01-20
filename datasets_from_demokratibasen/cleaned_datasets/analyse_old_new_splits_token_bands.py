#!/usr/bin/env python3
"""
Analyse overlap between old and new splits and token-length bands for available docs.

Run this script from the `cleaned_datasets` directory:

    python analyse_old_new_splits_token_bands.py

It will:
- Collect OLD_TRAIN / OLD_VAL from:
  - text_summary_dataset_202505_to_10/43221_text_summary_examples_train.jsonl
  - text_summary_dataset_202505_to_10/43221_text_summary_examples_val.jsonl
- Collect NEW_TRAIN / NEW_VAL / NEW_TEST from:
  - text_summary_dataset_202601/168161_text_summary_examples_train.jsonl
  - text_summary_dataset_202601/168161_text_summary_examples_val.jsonl
  - text_summary_dataset_202601/168161_text_summary_examples_test.jsonl
- Check:
  - no OLD_TRAIN IDs in NEW_VAL or NEW_TEST
  - no OLD_VAL IDs in NEW_TEST
- Define:
  - AVAIL_TRAIN = NEW_TRAIN - OLD_TRAIN
  - AVAIL_VAL   = NEW_VAL   - OLD_VAL
  - AVAIL_TEST  = NEW_TEST
- Tokenise all documents using o200k_base encoding,
  using tokens of the *input text only*.
- Band AVAIL_TRAIN / AVAIL_VAL / AVAIL_TEST by input token length with a 15% safety margin:
    ignore:       length < 192 * 1.15
    ULTRA_NARROW: length < 512 * 1.15
    NARROW:       length < 2048 * 1.15
    MEDIUM:       length < 8912 * 1.15
    BROAD:        the rest
- Print stats (including min/max tokens per band) and save IDs and stats to
  `old_new_splits_token_bands_o200k.json` in this directory.
"""

from __future__ import annotations

import json
import sys
import random
from pathlib import Path
from typing import Dict, Set, Tuple, List

try:
    import tiktoken
except ImportError:
    tiktoken = None  # type: ignore


BASE_DIR = Path(".")

OLD_DIR = BASE_DIR / "text_summary_dataset_202505_to_10"
NEW_DIR = BASE_DIR / "text_summary_dataset_202601"

OLD_TRAIN_PATH = OLD_DIR / "43221_text_summary_examples_train.jsonl"
OLD_VAL_PATH = OLD_DIR / "43221_text_summary_examples_val.jsonl"

NEW_TRAIN_PATH = NEW_DIR / "168161_text_summary_examples_train.jsonl"
NEW_VAL_PATH = NEW_DIR / "168161_text_summary_examples_val.jsonl"
NEW_TEST_PATH = NEW_DIR / "168161_text_summary_examples_test.jsonl"


def load_encoding() -> Tuple[object | None, str]:
    """Load tiktoken encoding for o200k_base."""
    if tiktoken is None:
        print("Warning: tiktoken is not installed; token counts will be 0.", file=sys.stderr)
        return None, "none"
    try:
        enc = tiktoken.get_encoding("o200k_base")
        return enc, "o200k_base"
    except Exception:
        print(
            "Warning: Failed to load o200k_base encoding; token counts will be 0.",
            file=sys.stderr,
        )
        return None, "none"


def extract_doc_ids(file_path: Path) -> Set[str]:
    """Extract dokument_id values from a JSONL file (from metadata)."""
    doc_ids: Set[str] = set()
    missing = 0
    json_errors = 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    metadata = ex.get("metadata", {})
                    doc_id = metadata.get("dokument_id")
                    if doc_id:
                        doc_ids.add(doc_id)
                    else:
                        missing += 1
                except json.JSONDecodeError:
                    json_errors += 1
                    if json_errors <= 5:
                        print(
                            f"Warning: JSON decode error on line {line_num} in {file_path.name}",
                            file=sys.stderr,
                        )
    except FileNotFoundError:
        print(f"ERROR: {file_path} not found", file=sys.stderr)
        sys.exit(1)
    if missing > 0:
        print(f"  Note: {missing} lines in {file_path.name} have no dokument_id", file=sys.stderr)
    return doc_ids


def build_token_index(
    paths: List[Path],
    enc,
) -> Dict[str, int]:
    """
    Build mapping dokument_id -> input_tokens for all given JSONL files.

    input_tokens = tokens of the `input` field only.
    """
    id_to_tokens: Dict[str, int] = {}

    for path in paths:
        print(f"Tokenising {path.name} ...", file=sys.stderr)
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ex = json.loads(line)
                    except json.JSONDecodeError:
                        if line_num <= 5:
                            print(
                                f"Warning: JSON decode error on line {line_num} in {path.name}",
                                file=sys.stderr,
                            )
                        continue

                    metadata = ex.get("metadata", {})
                    doc_id = metadata.get("dokument_id")
                    if not doc_id:
                        continue

                    input_text = str(ex.get("input", "") or "")
                    if enc is None:
                        tokens = 0
                    else:
                        try:
                            tokens = len(enc.encode(input_text))
                        except Exception:
                            tokens = 0
                    id_to_tokens[doc_id] = tokens
        except FileNotFoundError:
            print(f"ERROR: {path} not found while tokenising", file=sys.stderr)
            sys.exit(1)
    return id_to_tokens


def band_counts(doc_ids: Set[str], token_index: Dict[str, int]) -> Dict[str, dict]:
    """Compute band statistics for a given set of doc_ids."""
    # Bands with 15% safety margin on thresholds:
    # ignore:       len < 192  * 1.15
    # ULTRA_NARROW: len < 512  * 1.15
    # NARROW:       len < 2048 * 1.15
    # MEDIUM:       len < 8912 * 1.15
    # BROAD:        rest
    margin = 1.15
    ignore_thr = 192 * margin
    ultra_thr = 512 * margin
    narrow_thr = 2048 * margin
    medium_thr = 8912 * margin
    bands = {
        "ULTRA_NARROW": [],
        "NARROW": [],
        "MEDIUM": [],
        "BROAD": [],
    }
    ignored = 0

    for doc_id in doc_ids:
        tokens = token_index.get(doc_id, 0)
        if tokens < ignore_thr:
            ignored += 1
            continue
        if tokens < ultra_thr:
            bands["ULTRA_NARROW"].append(tokens)
        elif tokens < narrow_thr:
            bands["NARROW"].append(tokens)
        elif tokens < medium_thr:
            bands["MEDIUM"].append(tokens)
        else:
            bands["BROAD"].append(tokens)

    stats: Dict[str, dict] = {
        "ignored_lt_256": ignored,
        "total_considered": len(doc_ids) - ignored,
    }

    for name, values in bands.items():
        if not values:
            stats[name] = {
                "count": 0,
                "min_tokens": None,
                "max_tokens": None,
                "mean_tokens": None,
            }
        else:
            count = len(values)
            stats[name] = {
                "count": count,
                "min_tokens": min(values),
                "max_tokens": max(values),
                "mean_tokens": sum(values) / count,
            }

    return stats


def main() -> int:
    print("=" * 80)
    print("OLD vs NEW SPLITS & INPUT TOKEN BANDS (o200k_base)")
    print("=" * 80)
    print()

    # 1) Collect ID sets
    print("Collecting OLD_TRAIN / OLD_VAL IDs ...", file=sys.stderr)
    OLD_TRAIN = extract_doc_ids(OLD_TRAIN_PATH)
    OLD_VAL = extract_doc_ids(OLD_VAL_PATH)

    print("Collecting NEW_TRAIN / NEW_VAL / NEW_TEST IDs ...", file=sys.stderr)
    NEW_TRAIN = extract_doc_ids(NEW_TRAIN_PATH)
    NEW_VAL = extract_doc_ids(NEW_VAL_PATH)
    NEW_TEST = extract_doc_ids(NEW_TEST_PATH)

    # Output basic stats
    print("OLD SPLITS:")
    print(f"  OLD_TRAIN: {len(OLD_TRAIN):,} docs")
    print(f"  OLD_VAL:   {len(OLD_VAL):,} docs")
    print()
    print("NEW SPLITS:")
    print(f"  NEW_TRAIN: {len(NEW_TRAIN):,} docs")
    print(f"  NEW_VAL:   {len(NEW_VAL):,} docs")
    print(f"  NEW_TEST:  {len(NEW_TEST):,} docs")
    print()

    # 2) Checks
    print("Checking overlap constraints ...")
    train_in_new_val = OLD_TRAIN & NEW_VAL
    train_in_new_test = OLD_TRAIN & NEW_TEST
    val_in_new_test = OLD_VAL & NEW_TEST

    checks = {
        "OLD_TRAIN_in_NEW_VAL_count": len(train_in_new_val),
        "OLD_TRAIN_in_NEW_TEST_count": len(train_in_new_test),
        "OLD_VAL_in_NEW_TEST_count": len(val_in_new_test),
    }

    def report_overlap(name: str, overlap: Set[str]) -> None:
        if overlap:
            print(f"  VIOLATION: {name}: {len(overlap):,} overlapping IDs (e.g. {list(overlap)[:5]}...)")
        else:
            print(f"  OK: {name}: no overlaps")

    report_overlap("OLD_TRAIN ∩ NEW_VAL", train_in_new_val)
    report_overlap("OLD_TRAIN ∩ NEW_TEST", train_in_new_test)
    report_overlap("OLD_VAL ∩ NEW_TEST", val_in_new_test)
    print()

    # 3) AVAIL sets
    AVAIL_TRAIN = NEW_TRAIN - OLD_TRAIN
    AVAIL_VAL = NEW_VAL - OLD_VAL
    AVAIL_TEST = set(NEW_TEST)  # explicitly copy

    print("AVAIL SET SIZES:")
    print(f"  AVAIL_TRAIN: {len(AVAIL_TRAIN):,} docs")
    print(f"  AVAIL_VAL:   {len(AVAIL_VAL):,} docs")
    print(f"  AVAIL_TEST:  {len(AVAIL_TEST):,} docs")
    print()

    # Ensure AVAIL_* are disjoint
    overlap_train_val = AVAIL_TRAIN & AVAIL_VAL
    overlap_train_test = AVAIL_TRAIN & AVAIL_TEST
    overlap_val_test = AVAIL_VAL & AVAIL_TEST
    if overlap_train_val or overlap_train_test or overlap_val_test:
        print("ERROR: AVAIL_TRAIN / AVAIL_VAL / AVAIL_TEST are not disjoint:", file=sys.stderr)
        print(f"  AVAIL_TRAIN ∩ AVAIL_VAL:  {len(overlap_train_val)}", file=sys.stderr)
        print(f"  AVAIL_TRAIN ∩ AVAIL_TEST: {len(overlap_train_test)}", file=sys.stderr)
        print(f"  AVAIL_VAL ∩ AVAIL_TEST:   {len(overlap_val_test)}", file=sys.stderr)
        return 1

    # 4) Tokenisation
    enc, encoding_name = load_encoding()
    token_index = build_token_index(
        [NEW_TRAIN_PATH, NEW_VAL_PATH, NEW_TEST_PATH],
        enc,
    )

    # 5) Banding for AVAIL sets (stats only)
    print("Banding AVAIL_TRAIN by token length ...")
    avail_train_bands = band_counts(AVAIL_TRAIN, token_index)
    print("Banding AVAIL_VAL by token length ...")
    avail_val_bands = band_counts(AVAIL_VAL, token_index)
    print("Banding AVAIL_TEST by token length ...")
    avail_test_bands = band_counts(AVAIL_TEST, token_index)

    def print_band_stats(name: str, stats: Dict[str, dict]) -> None:
        print(f"{name}:")
        print(f"  Ignored (<256 tokens): {stats['ignored_lt_256']:,}")
        print(f"  Total considered:      {stats['total_considered']:,}")
        for band in ["ULTRA_NARROW", "NARROW", "MEDIUM", "BROAD"]:
            b = stats[band]
            print(
                f"  {band:13s}: "
                f"{b['count']:,} docs"
                f" (min={b['min_tokens']}, max={b['max_tokens']}, mean={b['mean_tokens']})"
            )
        print()

    print("=" * 80)
    print("TOKEN BAND STATS (based on input tokens, 15% safety margin on thresholds)")
    print("=" * 80)
    print_band_stats("AVAIL_TRAIN", avail_train_bands)
    print_band_stats("AVAIL_VAL", avail_val_bands)
    print_band_stats("AVAIL_TEST", avail_test_bands)

    # 6) Build per-band, per-split ID lists for sampling
    # Reuse thresholds from band_counts to keep logic in sync
    margin = 1.15
    ignore_thr = 192 * margin
    ultra_thr = 512 * margin
    narrow_thr = 2048 * margin
    medium_thr = 8912 * margin

    def classify_band(doc_id: str) -> str | None:
        tokens = token_index.get(doc_id, 0)
        if tokens < ignore_thr:
            return None
        if tokens < ultra_thr:
            return "ULTRA_NARROW"
        if tokens < narrow_thr:
            return "NARROW"
        if tokens < medium_thr:
            return "MEDIUM"
        return "BROAD"

    bands_by_split = {
        "TRAIN": {"ULTRA_NARROW": [], "NARROW": [], "MEDIUM": [], "BROAD": []},
        "VAL": {"ULTRA_NARROW": [], "NARROW": [], "MEDIUM": [], "BROAD": []},
        "TEST": {"ULTRA_NARROW": [], "NARROW": [], "MEDIUM": [], "BROAD": []},
    }

    for doc_id in AVAIL_TRAIN:
        band = classify_band(doc_id)
        if band:
            bands_by_split["TRAIN"][band].append(doc_id)
    for doc_id in AVAIL_VAL:
        band = classify_band(doc_id)
        if band:
            bands_by_split["VAL"][band].append(doc_id)
    for doc_id in AVAIL_TEST:
        band = classify_band(doc_id)
        if band:
            bands_by_split["TEST"][band].append(doc_id)

    # 7) Sampling from each band with TRAIN/VAL/TEST proportions
    random.seed(42)

    target_per_band = {
        "ULTRA_NARROW": 6500,
        "NARROW": 5000,
        "MEDIUM": 1000,
        "BROAD": 210,
    }
    proportions = {"TRAIN": 0.9, "VAL": 0.05, "TEST": 0.05}

    sampled_by_split_band: Dict[str, Dict[str, Set[str]]] = {
        split: {band: set() for band in target_per_band.keys()}
        for split in ["TRAIN", "VAL", "TEST"]
    }

    print("=" * 80)
    print("SAMPLING PER BAND (TRAIN/VAL/TEST = 0.9/0.05/0.05)")
    print("=" * 80)

    for band, band_target in target_per_band.items():
        # Available counts per split
        counts = {split: len(bands_by_split[split][band]) for split in ["TRAIN", "VAL", "TEST"]}
        band_total_available = sum(counts.values())
        print(f"{band}: target {band_target}, available {band_total_available} (TRAIN={counts['TRAIN']}, VAL={counts['VAL']}, TEST={counts['TEST']})")

        if band_total_available == 0:
            continue

        # Desired per split, limited by availability
        desired = {
            split: min(int(round(band_target * proportions[split])), counts[split])
            for split in ["TRAIN", "VAL", "TEST"]
        }
        # If because of rounding/availability sum(desired) < band_target, we just use what we can.

        for split in ["TRAIN", "VAL", "TEST"]:
            pool = bands_by_split[split][band]
            k = desired[split]
            if k <= 0 or not pool:
                continue
            # Deterministic sampling by sorting before random.sample
            pool_sorted = sorted(pool)
            if k >= len(pool_sorted):
                chosen = set(pool_sorted)
            else:
                chosen = set(random.sample(pool_sorted, k))
            sampled_by_split_band[split][band] = chosen
            print(f"  {band} - {split}: sampled {len(chosen)}")
        print()

    # 8) Merge sampled per band into overall band-level sampled sets
    sampled_band_merged: Dict[str, Set[str]] = {
        band: set()
        for band in target_per_band.keys()
    }
    for band in target_per_band.keys():
        for split in ["TRAIN", "VAL", "TEST"]:
            sampled_band_merged[band] |= sampled_by_split_band[split][band]

    print("=" * 80)
    print("MERGED SAMPLED SETS PER BAND")
    print("=" * 80)
    for band in target_per_band.keys():
        total_band = len(sampled_band_merged[band])
        print(f"{band}: {total_band} docs "
              f"(TRAIN={len(sampled_by_split_band['TRAIN'][band])}, "
              f"VAL={len(sampled_by_split_band['VAL'][band])}, "
              f"TEST={len(sampled_by_split_band['TEST'][band])})")
    print()

    # 9) Remaining docs in NEW_TRAIN / NEW_VAL / NEW_TEST after removing all sampled IDs
    all_sampled_ids: Set[str] = set()
    for band in target_per_band.keys():
        all_sampled_ids |= sampled_band_merged[band]

    REM_TRAIN = NEW_TRAIN - all_sampled_ids
    REM_VAL = NEW_VAL - all_sampled_ids
    REM_TEST = NEW_TEST - all_sampled_ids

    # Ensure REM_* are disjoint
    rem_train_val = REM_TRAIN & REM_VAL
    rem_train_test = REM_TRAIN & REM_TEST
    rem_val_test = REM_VAL & REM_TEST
    if rem_train_val or rem_train_test or rem_val_test:
        print("ERROR: REM_TRAIN / REM_VAL / REM_TEST are not disjoint:", file=sys.stderr)
        print(f"  REM_TRAIN ∩ REM_VAL:  {len(rem_train_val)}", file=sys.stderr)
        print(f"  REM_TRAIN ∩ REM_TEST: {len(rem_train_test)}", file=sys.stderr)
        print(f"  REM_VAL ∩ REM_TEST:   {len(rem_val_test)}", file=sys.stderr)
        return 1

    print("=" * 80)
    print("REMAINING DOCS AFTER SAMPLING")
    print("=" * 80)
    print(f"REM_TRAIN: {len(REM_TRAIN):,} docs")
    print(f"REM_VAL:   {len(REM_VAL):,} docs")
    print(f"REM_TEST:  {len(REM_TEST):,} docs")
    total_rem = len(REM_TRAIN) + len(REM_VAL) + len(REM_TEST)
    if total_rem > 0:
        ratio_train = len(REM_TRAIN) / total_rem
        ratio_val = len(REM_VAL) / total_rem
        ratio_test = len(REM_TEST) / total_rem
        print(f"Ratios (TRAIN/VAL/TEST): {ratio_train:.4f} / {ratio_val:.4f} / {ratio_test:.4f}")
    print()

    # 10) Save IDs and stats
    output_path = BASE_DIR / "old_new_splits_token_bands_o200k.json"
    output_data = {
        "encoding": encoding_name,
        "paths": {
            "OLD_TRAIN": str(OLD_TRAIN_PATH),
            "OLD_VAL": str(OLD_VAL_PATH),
            "NEW_TRAIN": str(NEW_TRAIN_PATH),
            "NEW_VAL": str(NEW_VAL_PATH),
            "NEW_TEST": str(NEW_TEST_PATH),
        },
        "sizes": {
            "OLD_TRAIN": len(OLD_TRAIN),
            "OLD_VAL": len(OLD_VAL),
            "NEW_TRAIN": len(NEW_TRAIN),
            "NEW_VAL": len(NEW_VAL),
            "NEW_TEST": len(NEW_TEST),
            "AVAIL_TRAIN": len(AVAIL_TRAIN),
            "AVAIL_VAL": len(AVAIL_VAL),
            "AVAIL_TEST": len(AVAIL_TEST),
        },
        "checks": checks,
        "bands": {
            "AVAIL_TRAIN": avail_train_bands,
            "AVAIL_VAL": avail_val_bands,
            "AVAIL_TEST": avail_test_bands,
        },
        "sampling": {
            "target_per_band": target_per_band,
            "proportions": proportions,
            "sampled_counts": {
                split: {
                    band: len(sampled_by_split_band[split][band])
                    for band in target_per_band.keys()
                }
                for split in ["TRAIN", "VAL", "TEST"]
            },
            "sampled_merged_counts": {
                band: len(sampled_band_merged[band])
                for band in target_per_band.keys()
            },
        },
        "remaining": {
            "REM_TRAIN": len(REM_TRAIN),
            "REM_VAL": len(REM_VAL),
            "REM_TEST": len(REM_TEST),
        },
        # Save IDs as sorted lists for reproducibility
        "doc_ids": {
            "OLD_TRAIN": sorted(OLD_TRAIN),
            "OLD_VAL": sorted(OLD_VAL),
            "NEW_TRAIN": sorted(NEW_TRAIN),
            "NEW_VAL": sorted(NEW_VAL),
            "NEW_TEST": sorted(NEW_TEST),
            "AVAIL_TRAIN": sorted(AVAIL_TRAIN),
            "AVAIL_VAL": sorted(AVAIL_VAL),
            "AVAIL_TEST": sorted(AVAIL_TEST),
            # Sampled IDs per split and band
            **{
                f"SAMPLED_{band}_TRAIN": sorted(sampled_by_split_band["TRAIN"][band])
                for band in target_per_band.keys()
            },
            **{
                f"SAMPLED_{band}_VAL": sorted(sampled_by_split_band["VAL"][band])
                for band in target_per_band.keys()
            },
            **{
                f"SAMPLED_{band}_TEST": sorted(sampled_by_split_band["TEST"][band])
                for band in target_per_band.keys()
            },
            # Merged sampled per band
            **{
                f"SAMPLED_{band}": sorted(sampled_band_merged[band])
                for band in target_per_band.keys()
            },
            # Remaining sets
            "REM_TRAIN": sorted(REM_TRAIN),
            "REM_VAL": sorted(REM_VAL),
            "REM_TEST": sorted(REM_TEST),
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved IDs and stats to {output_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

