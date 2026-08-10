#!/usr/bin/env python3
"""
Backfill token-count summary fields in checkpoint eval JSON files.

Targets:
  models/<model_name>/<results_dir>/checkpoint-N-genG-eval-results-X-examples.json
Using:
  models/<model_name>/<results_dir>/checkpoint-N-genG-inputs-refs-preds-X-examples.jsonl

Rules:
  - Only update checkpoints that already have the matching JSONL file.
  - Insert fields immediately after "eval_rougeLsum" (or append if absent).
  - Reuse input/reference stats per (model, X-examples) for speed, with
    per-checkpoint sanity checks on sampled text hashes.
  - If legacy eval_*output_tokens fields exist, compare to eval_*pred_tokens:
    warn on mismatch and then remove legacy output fields.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from transformers import AutoTokenizer

PRED_FILE_RE = re.compile(r"^checkpoint-(\d+)-gen(\d+)-inputs-refs-preds-(\d+)-examples\.jsonl$")
WORD_RE = re.compile(r"\w+", re.UNICODE)
NEW_FIELD_KEYS = [
    "eval_mean_input_tokens",
    "eval_min_input_tokens",
    "eval_max_input_tokens",
    "eval_mean_ref_tokens",
    "eval_min_ref_tokens",
    "eval_max_ref_tokens",
    "eval_mean_pred_tokens",
    "eval_min_pred_tokens",
    "eval_max_pred_tokens",
]
LEGACY_OUTPUT_KEYS = [
    "eval_mean_output_tokens",
    "eval_min_output_tokens",
    "eval_max_output_tokens",
]

# Keep this script self-contained (no dependency on model_configs.py / peft).
SHORT_TO_HF = {
    "gemma-2b": "google/gemma-2b",
    "gemma-7b-it": "google/gemma-7b-it",
    "gemma-2-9b": "google/gemma-2-9b",
    "gemma-2-27b": "google/gemma-2-27b",
    "gemma-3-12b": "google/gemma-3-12b-pt",
    "gemma-3-27b": "google/gemma-3-27b-pt",
    "viking-7b": "LumiOpen/Viking-7B",
    "viking-13b": "LumiOpen/Viking-13B",
    "viking-33b": "LumiOpen/Viking-33B",
    "eurollm-9b-instruct": "utter-project/EuroLLM-9B-Instruct-2512",
    "norwai-mistral-7b-instruct": "NorwAI/NorwAI-Mistral-7B-instruct",
    "normistral-7b": "norallm/normistral-7b-warm",
    "normistral-7b-instruct": "norallm/normistral-7b-warm-instruct",
    "normistral-11b": "norallm/normistral-11b-warm",
    "normistral-11b-long": "norallm/normistral-11b-long",
    "norskgpt-llama3-8b": "bineric/norskgpt-llama3-8b",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-2-13b-chat-norwegian": "ruternorway/llama-2-13b-chat-norwegian",
    "mt5": "google/mt5-base",
    "nb-gpt-j-6b": "NbAiLab/nb-gpt-j-6B-torgersen-alpaca",
}


def normalize_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return str(value) if value is not None else ""


def token_counts_from_texts(tokenizer, texts: List[str]) -> List[int]:
    if not texts:
        return []
    try:
        # Use the same tokenizer-based counting style as legacy output-token fields.
        encoded = tokenizer(texts, add_special_tokens=False, truncation=False)
        return [len(ids) for ids in encoded["input_ids"]]
    except Exception:
        # Fallback for rare tokenizer edge-cases: count individually.
        out: List[int] = []
        for t in texts:
            try:
                out.append(len(tokenizer.encode(t, add_special_tokens=False)))
            except Exception:
                out.append(0)
        return out


def compute_stats(values: List[int]) -> Tuple[float, int, int]:
    if not values:
        return (0.0, 0, 0)
    return (sum(values) / len(values), min(values), max(values))


def sample_indices(n: int) -> List[int]:
    if n <= 0:
        return []
    idx = {0, n - 1, n // 2, n // 3, (2 * n) // 3}
    return sorted(i for i in idx if 0 <= i < n)


def sample_hashes(entries: List[dict], key: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    idxs = sample_indices(len(entries))
    for i in idxs:
        value = entries[i].get(key, "")
        if not isinstance(value, str):
            value = str(value)
        out[i] = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return out


def load_jsonl_entries(path: Path) -> List[dict]:
    entries: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def insert_fields_after_rougelsum(existing: Dict, fields: Dict[str, object]) -> Dict:
    if "eval_rougeLsum" not in existing:
        out = dict(existing)
        out.update(fields)
        return out

    out: Dict[str, object] = {}
    for k, v in existing.items():
        out[k] = v
        if k == "eval_rougeLsum":
            for fk, fv in fields.items():
                out[fk] = fv
    return out


def iter_groups(results_dir: Path) -> Dict[str, List[Tuple[int, Path, Path]]]:
    """
    Return groups keyed by '<X>-examples'.
    Each value: list[(checkpoint_step, eval_json_path, preds_jsonl_path)].
    """
    groups: Dict[str, List[Tuple[int, Path, Path]]] = {}
    for pred_path in results_dir.glob("checkpoint-*-gen*-inputs-refs-preds-*-examples.jsonl"):
        m = PRED_FILE_RE.match(pred_path.name)
        if not m:
            continue
        step = int(m.group(1))
        gen = int(m.group(2))
        x = m.group(3)
        suffix = f"{x}-examples"
        eval_path = results_dir / f"checkpoint-{step}-gen{gen}-eval-results-{suffix}.json"
        if not eval_path.exists():
            continue
        groups.setdefault(suffix, []).append((step, eval_path, pred_path))

    for suffix in groups:
        groups[suffix].sort(key=lambda t: t[0])
    return groups


def _is_close(a: object, b: object, eps: float = 1e-9) -> bool:
    try:
        return abs(float(a) - float(b)) <= eps
    except Exception:
        return a == b


def iter_result_dirs(model_dir: Path, patterns: List[str]) -> List[Path]:
    out = []
    seen = set()
    for pat in patterns:
        for p in model_dir.glob(pat):
            if p.is_dir():
                key = str(p.resolve())
                if key not in seen:
                    out.append(p)
                    seen.add(key)
    return sorted(out)


def resolve_hf_name(model_dir: Path) -> str:
    short_name = model_dir.name
    if short_name.endswith("-apptainer-fsdp"):
        short_name = short_name[: -len("-apptainer-fsdp")]
    # winners/ model dirs often include checkpoint suffixes (e.g. "-cp9000")
    short_name = re.sub(r"-cp\d+$", "", short_name)
    if short_name in SHORT_TO_HF:
        return SHORT_TO_HF[short_name]
    # Fallback: allow direct HF-style model directory names if present.
    return short_name


def load_tokenizer_for_model(model_dir: Path, cache: Dict[str, object]):
    model_key = model_dir.name
    if model_key in cache:
        return cache[model_key]
    hf_name = resolve_hf_name(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
    cache[model_key] = tokenizer
    return tokenizer


def update_result_dir(results_dir: Path, tokenizer, warnings: List[str]) -> Dict[str, int]:
    groups = iter_groups(results_dir)
    updated = 0
    mismatch = 0
    skipped = 0
    legacy_output_mismatch = 0

    for suffix, items in groups.items():
        if not items:
            continue

        _, _, baseline_pred = items[0]
        baseline_entries = load_jsonl_entries(baseline_pred)
        baseline_in_hash = sample_hashes(baseline_entries, "input_text")
        baseline_ref_hash = sample_hashes(baseline_entries, "reference")

        baseline_input_stats = compute_stats(
            token_counts_from_texts(
                tokenizer, [normalize_text(e.get("input_text", "")) for e in baseline_entries]
            )
        )
        baseline_ref_stats = compute_stats(
            token_counts_from_texts(
                tokenizer, [normalize_text(e.get("reference", "")) for e in baseline_entries]
            )
        )

        for step, eval_path, pred_path in items:
            entries = load_jsonl_entries(pred_path)
            if not entries:
                skipped += 1
                continue

            pred_stats = compute_stats(
                token_counts_from_texts(
                    tokenizer, [normalize_text(e.get("prediction", "")) for e in entries]
                )
            )

            # quick sanity check of input/reference consistency
            same_inputs = sample_hashes(entries, "input_text") == baseline_in_hash
            same_refs = sample_hashes(entries, "reference") == baseline_ref_hash

            if same_inputs and same_refs:
                input_stats = baseline_input_stats
                ref_stats = baseline_ref_stats
            else:
                mismatch += 1
                input_stats = compute_stats(
                    token_counts_from_texts(
                        tokenizer, [normalize_text(e.get("input_text", "")) for e in entries]
                    )
                )
                ref_stats = compute_stats(
                    token_counts_from_texts(
                        tokenizer, [normalize_text(e.get("reference", "")) for e in entries]
                    )
                )

            with eval_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            new_fields = {
                "eval_mean_input_tokens": input_stats[0],
                "eval_min_input_tokens": input_stats[1],
                "eval_max_input_tokens": input_stats[2],
                "eval_mean_ref_tokens": ref_stats[0],
                "eval_min_ref_tokens": ref_stats[1],
                "eval_max_ref_tokens": ref_stats[2],
                "eval_mean_pred_tokens": pred_stats[0],
                "eval_min_pred_tokens": pred_stats[1],
                "eval_max_pred_tokens": pred_stats[2],
            }

            # Compare legacy output-token fields (if present) before removal.
            has_all_legacy = all(k in payload for k in LEGACY_OUTPUT_KEYS)
            if has_all_legacy:
                if not (
                    _is_close(payload.get("eval_mean_output_tokens"), new_fields["eval_mean_pred_tokens"])
                    and _is_close(payload.get("eval_min_output_tokens"), new_fields["eval_min_pred_tokens"])
                    and _is_close(payload.get("eval_max_output_tokens"), new_fields["eval_max_pred_tokens"])
                ):
                    legacy_output_mismatch += 1
                    warnings.append(
                        f"legacy output-token mismatch: {eval_path}"
                    )

            # Remove existing/new token fields first so insertion order is controlled.
            cleaned = {
                k: v
                for k, v in payload.items()
                if k not in NEW_FIELD_KEYS and k not in LEGACY_OUTPUT_KEYS
            }

            new_payload = insert_fields_after_rougelsum(cleaned, new_fields)
            with eval_path.open("w", encoding="utf-8") as f:
                json.dump(new_payload, f, ensure_ascii=False, indent=2)
            updated += 1

    return {
        "groups": len(groups),
        "updated": updated,
        "mismatch": mismatch,
        "skipped": skipped,
        "legacy_output_mismatch": legacy_output_mismatch,
    }


def update_model(
    model_dir: Path, results_dir_patterns: List[str], tokenizer_cache: Dict[str, object], warnings: List[str]
) -> Dict[str, int]:
    result_dirs = iter_result_dirs(model_dir, results_dir_patterns)
    if not result_dirs:
        return {"dirs": 0, "groups": 0, "updated": 0, "mismatch": 0, "skipped": 0, "legacy_output_mismatch": 0}

    agg = {"dirs": 0, "groups": 0, "updated": 0, "mismatch": 0, "skipped": 0, "legacy_output_mismatch": 0}
    tokenizer = load_tokenizer_for_model(model_dir, tokenizer_cache)
    for results_dir in result_dirs:
        stats = update_result_dir(results_dir, tokenizer, warnings)
        agg["dirs"] += 1
        for k in ("groups", "updated", "mismatch", "skipped", "legacy_output_mismatch"):
            agg[k] += stats[k]
    return agg


def iter_model_dirs(models_root: Path) -> Iterable[Path]:
    for p in sorted(models_root.glob("*")):
        if p.is_dir():
            yield p


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill eval token fields from inputs-refs-preds JSONL files.")
    parser.add_argument("--models_root", required=True, help="Path to models root directory.")
    parser.add_argument(
        "--results_dir_pattern",
        action="append",
        default=[],
        help=(
            "Glob pattern (relative to each model dir) for result folders to process. "
            "Can be provided multiple times. Defaults to: all_eval_results"
        ),
    )
    args = parser.parse_args()

    models_root = Path(args.models_root).expanduser().resolve()
    if not models_root.is_dir():
        raise SystemExit(f"models_root does not exist or is not a directory: {models_root}")

    patterns = args.results_dir_pattern or ["all_eval_results"]

    total_models = 0
    total_dirs = 0
    total_groups = 0
    total_updated = 0
    total_mismatch = 0
    total_skipped = 0
    total_legacy_output_mismatch = 0
    warnings: List[str] = []
    tokenizer_cache: Dict[str, object] = {}

    for model_dir in iter_model_dirs(models_root):
        total_models += 1
        stats = update_model(model_dir, patterns, tokenizer_cache, warnings)
        total_dirs += stats["dirs"]
        total_groups += stats["groups"]
        total_updated += stats["updated"]
        total_mismatch += stats["mismatch"]
        total_skipped += stats["skipped"]
        total_legacy_output_mismatch += stats["legacy_output_mismatch"]
        if stats["updated"] > 0 or stats["mismatch"] > 0 or stats["legacy_output_mismatch"] > 0:
            print(
                f"{model_dir.name}: dirs={stats['dirs']}, updated={stats['updated']}, "
                f"groups={stats['groups']}, mismatches={stats['mismatch']}, "
                f"legacy_output_mismatches={stats['legacy_output_mismatch']}, skipped={stats['skipped']}"
            )

    print(
        f"done: models={total_models}, dirs={total_dirs}, groups={total_groups}, updated={total_updated}, "
        f"mismatches={total_mismatch}, legacy_output_mismatches={total_legacy_output_mismatch}, skipped={total_skipped}"
    )
    if warnings:
        print(f"warnings={len(warnings)}")
        for w in warnings[:50]:
            print(f"WARNING: {w}")
        if len(warnings) > 50:
            print(f"WARNING: ... and {len(warnings) - 50} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

