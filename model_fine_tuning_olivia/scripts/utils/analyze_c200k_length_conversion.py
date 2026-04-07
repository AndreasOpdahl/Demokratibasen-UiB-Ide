#!/usr/bin/env python3
"""
Build approximate c200k->model tokenizer length conversion multipliers.

Usage example:
  python scripts/utils/analyze_c200k_length_conversion.py \
    --dataset_jsonl data/dataset_149978_examples/149978_text_summary_examples_train.jsonl \
    --analysis_json data/dataset_149978_examples/analysis_results/149978_text_summary_examples_analysis_results.json \
    --models_root ~/OneDrive/Shared/Demokratibasen-UiB-Ide/TrainingRuns/olivia/models \
    --output_json scripts/utils/c200k_length_conversion_table.json \
    --sample_size 3000
"""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, List

from transformers import AutoTokenizer

DEFAULT_C200K_AVG = 86.1521889877182


def _load_summaries(dataset_jsonl: Path) -> List[str]:
    texts: List[str] = []
    with dataset_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            out = obj.get("output")
            if isinstance(out, str) and out.strip():
                texts.append(out)
    return texts


def _token_lengths(tokenizer: Any, texts: List[str]) -> List[int]:
    lengths = [len(tokenizer(t, add_special_tokens=False).input_ids) for t in texts]
    lengths.sort()
    return lengths


def _ratio(avg_tokens: float, c200k_avg: float) -> float:
    if c200k_avg <= 0:
        return 1.0
    return avg_tokens / c200k_avg


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate c200k length conversion table.")
    parser.add_argument("--dataset_jsonl", type=str, required=True)
    parser.add_argument("--analysis_json", type=str, default="")
    parser.add_argument("--models_root", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_jsonl = Path(args.dataset_jsonl).expanduser().resolve()
    models_root = Path(args.models_root).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()

    if not dataset_jsonl.exists():
        raise FileNotFoundError(f"dataset_jsonl not found: {dataset_jsonl}")
    if not models_root.exists():
        raise FileNotFoundError(f"models_root not found: {models_root}")

    c200k_avg = DEFAULT_C200K_AVG
    if args.analysis_json:
        analysis_json = Path(args.analysis_json).expanduser().resolve()
        if analysis_json.exists():
            with analysis_json.open("r", encoding="utf-8") as f:
                analysis = json.load(f)
            c200k_avg = float(analysis.get("summary_text", {}).get("average_length_tokens", c200k_avg))

    texts = _load_summaries(dataset_jsonl)
    if not texts:
        raise RuntimeError("No valid string summaries found in dataset_jsonl.")

    random.seed(args.seed)
    if args.sample_size > 0 and len(texts) > args.sample_size:
        texts = random.sample(texts, args.sample_size)

    by_short_name: Dict[str, float] = {}
    by_hf_name: Dict[str, float] = {}
    by_architecture_buckets: Dict[str, List[float]] = {}
    diagnostics: Dict[str, Any] = {}

    model_dirs = sorted([p for p in models_root.iterdir() if p.is_dir() and p.name.endswith("-apptainer-fsdp")])
    for model_dir in model_dirs:
        short_name = model_dir.name.replace("-apptainer-fsdp", "")
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
            lengths = _token_lengths(tokenizer, texts)
        except Exception as exc:
            diagnostics[short_name] = {"status": "error", "error": str(exc)}
            continue

        avg_tokens = sum(lengths) / len(lengths)
        ratio = _ratio(avg_tokens, c200k_avg)
        arch = tokenizer.__class__.__name__.lower()

        by_short_name[short_name] = round(ratio, 4)
        diagnostics[short_name] = {
            "status": "ok",
            "tokenizer_class": tokenizer.__class__.__name__,
            "avg_tokens": round(avg_tokens, 3),
            "p50_tokens": lengths[len(lengths) // 2],
            "p90_tokens": lengths[int(len(lengths) * 0.9)],
            "ratio_model_to_c200k": round(ratio, 4),
        }

        hf_name = None
        tokenizer_config = model_dir / "tokenizer_config.json"
        if tokenizer_config.exists():
            try:
                with tokenizer_config.open("r", encoding="utf-8") as f:
                    cfg = json.load(f)
                hf_name = cfg.get("name_or_path")
            except Exception:
                hf_name = None
        if isinstance(hf_name, str) and hf_name.strip():
            by_hf_name[hf_name] = round(ratio, 4)

        if "gemma" in arch:
            by_architecture_buckets.setdefault("gemma", []).append(ratio)
        elif "llama" in arch or "tokenizersbackend" in arch:
            by_architecture_buckets.setdefault("llama", []).append(ratio)
        elif "gpt2" in arch:
            by_architecture_buckets.setdefault("gptj", []).append(ratio)
        elif "mt5" in arch:
            by_architecture_buckets.setdefault("mt5", []).append(ratio)
        else:
            by_architecture_buckets.setdefault("mistral", []).append(ratio)

    by_architecture = {
        k: round(median(v), 4) for k, v in by_architecture_buckets.items() if v
    }

    payload: Dict[str, Any] = {
        "description": "Approximate conversion multipliers from c200k/o200k token goals to model tokenizer token goals.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(dataset_jsonl),
        "source_c200k_average_tokens": c200k_avg,
        "sample_size": len(texts),
        "sample_seed": args.seed,
        "method": "ratio = model_tokenizer_avg_tokens / c200k_avg_tokens",
        "default_ratio": 1.0,
        "by_short_name": by_short_name,
        "by_hf_name": by_hf_name,
        "by_architecture": by_architecture,
        "diagnostics": diagnostics,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote conversion table: {output_json}")
    print(f"Models with ratios: {len(by_short_name)}")


if __name__ == "__main__":
    main()
