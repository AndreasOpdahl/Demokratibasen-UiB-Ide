#!/usr/bin/env python3
"""
Re-tokenize analysis_results.json files using o200k_base (GPT-4o tokenizer).

Reads the JSONL dataset referenced by each analysis_results.json, recomputes
token statistics, and updates only the token-related fields in place.

Usage:
    python retokenize_analysis_results.py <analysis_results.json> <dataset.jsonl>
    python retokenize_analysis_results.py --all   # process known datasets
"""

import argparse
import json
import sys
from pathlib import Path

import tiktoken

ENCODING_NAME = "o200k_base"
enc = tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def compute_token_stats(jsonl_path: Path):
    input_tokens = []
    summary_tokens = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            inp = str(obj.get("input", ""))
            out = str(obj.get("output", ""))
            input_tokens.append(count_tokens(inp))
            summary_tokens.append(count_tokens(out))

    n = len(input_tokens)
    if n == 0:
        raise ValueError(f"No examples found in {jsonl_path}")

    above_2048 = sum(1 for t in input_tokens if t > 2048) / n * 100
    above_4096 = sum(1 for t in input_tokens if t > 4096) / n * 100
    above_8192 = sum(1 for t in input_tokens if t > 8192) / n * 100

    return {
        "input_text": {
            "average_length_tokens": sum(input_tokens) / n,
            "min_length_tokens": min(input_tokens),
            "max_length_tokens": max(input_tokens),
            "percentage_above_2048_tokens": above_2048,
            "percentage_above_4096_tokens": above_4096,
            "percentage_above_8192_tokens": above_8192,
        },
        "summary_text": {
            "average_length_tokens": sum(summary_tokens) / n,
            "min_length_tokens": min(summary_tokens),
            "max_length_tokens": max(summary_tokens),
        },
    }


def patch_analysis_file(analysis_path: Path, token_stats: dict):
    with open(analysis_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    old_encoding = data.get("token_encoding", "cl100k_base")

    for section in ["input_text", "summary_text"]:
        if section not in data:
            continue
        for key, value in token_stats[section].items():
            data[section][key] = value

    data["token_encoding"] = ENCODING_NAME

    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  Updated {analysis_path.name}  ({old_encoding} -> {ENCODING_NAME})")


def process_pair(analysis_path: Path, jsonl_path: Path):
    print(f"Processing: {jsonl_path.name}  ({jsonl_path.parent.name})")
    stats = compute_token_stats(jsonl_path)

    inp = stats["input_text"]
    print(f"  input  tokens: mean={inp['average_length_tokens']:.1f}  "
          f"min={inp['min_length_tokens']}  max={inp['max_length_tokens']}  "
          f">2048={inp['percentage_above_2048_tokens']:.1f}%  "
          f">4096={inp['percentage_above_4096_tokens']:.1f}%  "
          f">8192={inp['percentage_above_8192_tokens']:.1f}%")
    out = stats["summary_text"]
    print(f"  output tokens: mean={out['average_length_tokens']:.1f}  "
          f"min={out['min_length_tokens']}  max={out['max_length_tokens']}")

    patch_analysis_file(analysis_path, stats)


KNOWN_DATASETS = [
    {
        "analysis": "text_summary_dataset_202601/analysis_results/149978_text_summary_examples_analysis_results.json",
        "jsonl": "text_summary_dataset_202601/149978_text_summary_examples.jsonl",
    },
    {
        "analysis": "text_summary_dataset_202505_and_06/analysis_results/12811_text_summary_examples_analysis_results.json",
        "jsonl": "text_summary_dataset_202505_and_06/12811_text_summary_examples.jsonl",
    },
    {
        "analysis": "text_summary_dataset_202505_to_10/analysis_results/43221_text_summary_examples_analysis_results.json",
        "jsonl": "text_summary_dataset_202505_to_10/43221_text_summary_examples.jsonl",
    },
]


def main():
    parser = argparse.ArgumentParser(description="Re-tokenize analysis results with o200k_base")
    parser.add_argument("analysis_json", nargs="?", help="Path to analysis_results.json")
    parser.add_argument("dataset_jsonl", nargs="?", help="Path to dataset .jsonl")
    parser.add_argument("--all", action="store_true",
                        help="Process all known current datasets under DATA_DIR")
    parser.add_argument("--data-dir", type=str,
                        default=str(Path.home() / "OneDrive/Shared/Demokratibasen-UiB-Ide/TrainingDatasets"),
                        help="Base data directory")
    args = parser.parse_args()

    if args.all:
        data_dir = Path(args.data_dir)
        if not data_dir.is_dir():
            print(f"Error: DATA_DIR not found: {data_dir}", file=sys.stderr)
            return 1

        for entry in KNOWN_DATASETS:
            a = data_dir / entry["analysis"]
            j = data_dir / entry["jsonl"]
            if not a.exists():
                print(f"Skipping (analysis file missing): {a}", file=sys.stderr)
                continue
            if not j.exists():
                print(f"Skipping (jsonl missing): {j}", file=sys.stderr)
                continue
            process_pair(a, j)
            print()

        return 0

    if not args.analysis_json or not args.dataset_jsonl:
        parser.error("Provide both <analysis_json> and <dataset_jsonl>, or use --all")

    process_pair(Path(args.analysis_json), Path(args.dataset_jsonl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
