#!/usr/bin/env python3
import json
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import tiktoken


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "analysis_results"
RESULTS_PATH = RESULTS_DIR / "analysis_results.json"

TASKS = ["3060_c2_text_prediction", "9650_c3_prompt_continuation"]
BANDS = ["ultra_narow", "narrow", "medium", "broad"]
STAGES = ["regular", "major", "final"]

ENCODING = tiktoken.get_encoding("o200k_base")


def _iter_example_files() -> Iterable[Path]:
    for task in TASKS:
        for band in BANDS:
            yield BASE_DIR / f"{task}_{band}_examples.jsonl"


def _is_unusual_char(ch: str) -> bool:
    if ch == "\n":
        return False
    if ch.isspace():
        return False
    cat = unicodedata.category(ch)
    if cat.startswith("C") or cat.startswith("M"):
        return True
    if ch.isascii():
        return False
    if ch in "æøåÆØÅ":
        return False
    if ch.isalpha() and unicodedata.name(ch, "").startswith("LATIN"):
        return False
    return True


def _text_stats(values: List[int]) -> Dict[str, float]:
    values_sorted = sorted(values)
    return {
        "min": float(values_sorted[0]),
        "max": float(values_sorted[-1]),
        "mean": float(sum(values_sorted) / len(values_sorted)),
        "median": float(statistics.median(values_sorted)),
    }


def main() -> None:
    counts_task = Counter()
    counts_band = Counter()
    counts_stage = Counter()
    counts_task_band = Counter()
    counts_task_stage = Counter()
    counts_band_stage = Counter()
    counts_task_band_stage = Counter()

    token_counts: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    char_counts: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)

    doc_ids: Dict[str, List[str]] = defaultdict(list)
    unusual_chars = Counter()
    token_errors: List[str] = []

    total_examples = 0

    for file_path in _iter_example_files():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing examples file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {file_path}: {exc}") from exc

                doc_id = obj.get("id")
                task = obj.get("task")
                band = obj.get("band")
                stage = obj.get("stage")
                text = obj.get("text", "")

                if not doc_id or not task or not band or not stage:
                    raise ValueError(
                        f"Missing fields in {file_path}: id/task/band/stage required"
                    )

                total_examples += 1
                doc_ids[doc_id].append(file_path.name)

                counts_task[task] += 1
                counts_band[band] += 1
                counts_stage[stage] += 1
                counts_task_band[(task, band)] += 1
                counts_task_stage[(task, stage)] += 1
                counts_band_stage[(band, stage)] += 1
                counts_task_band_stage[(task, band, stage)] += 1

                for ch in text:
                    if _is_unusual_char(ch):
                        unusual_chars[ch] += 1

                key = (task, band, stage)
                char_counts[key].append(len(text))
                try:
                    token_counts[key].append(len(ENCODING.encode(text)))
                except Exception as exc:  # noqa: BLE001
                    token_errors.append(f"{doc_id}: {exc}")

    duplicate_doc_ids = {k: v for k, v in doc_ids.items() if len(v) > 1}

    results = {
        "total_examples": total_examples,
        "disjoint_doc_ids": len(duplicate_doc_ids) == 0,
        "duplicate_doc_ids": duplicate_doc_ids,
        "counts": {
            "task": dict(counts_task),
            "band": dict(counts_band),
            "stage": dict(counts_stage),
            "task_band": {f"{t}|{b}": c for (t, b), c in counts_task_band.items()},
            "task_stage": {
                f"{t}|{s}": c for (t, s), c in counts_task_stage.items()
            },
            "band_stage": {
                f"{b}|{s}": c for (b, s), c in counts_band_stage.items()
            },
            "task_band_stage": {
                f"{t}|{b}|{s}": c
                for (t, b, s), c in counts_task_band_stage.items()
            },
        },
        "text_stats": {
            f"{t}|{b}|{s}": {
                "chars": _text_stats(char_counts[(t, b, s)]),
                "tokens": _text_stats(token_counts[(t, b, s)]),
            }
            for (t, b, s) in counts_task_band_stage.keys()
        },
        "unusual_unicode_chars": dict(unusual_chars),
        "tokenization_errors": token_errors,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Total examples: {total_examples:,}")
    print(f"Disjoint doc ids: {len(duplicate_doc_ids) == 0}")
    if duplicate_doc_ids:
        print(f"Duplicate doc ids: {len(duplicate_doc_ids):,}")
    print("\nCounts by task:")
    for task in TASKS:
        print(f"  {task}: {counts_task.get(task, 0):,}")
    print("\nCounts by band:")
    for band in BANDS:
        print(f"  {band}: {counts_band.get(band, 0):,}")
    print("\nCounts by stage:")
    for stage in STAGES:
        print(f"  {stage}: {counts_stage.get(stage, 0):,}")
    print("\nCounts by task+band:")
    for task in TASKS:
        for band in BANDS:
            print(f"  {task} {band}: {counts_task_band.get((task, band), 0):,}")
    print("\nCounts by task+stage:")
    for task in TASKS:
        for stage in STAGES:
            print(f"  {task} {stage}: {counts_task_stage.get((task, stage), 0):,}")
    print("\nCounts by band+stage:")
    for band in BANDS:
        for stage in STAGES:
            print(f"  {band} {stage}: {counts_band_stage.get((band, stage), 0):,}")
    print("\nCounts by task+band+stage and text stats:")
    for task in TASKS:
        for band in BANDS:
            for stage in STAGES:
                key = (task, band, stage)
                count = counts_task_band_stage.get(key, 0)
                if count == 0:
                    continue
                stats = results["text_stats"][f"{task}|{band}|{stage}"]
                chars = stats["chars"]
                tokens = stats["tokens"]
                print(f"  {task} {band} {stage}: {count:,}")
                print(
                    f"    chars  min:{chars['min']:.0f} "
                    f"max:{chars['max']:.0f} mean:{chars['mean']:.1f} "
                    f"median:{chars['median']:.0f}"
                )
                print(
                    f"    tokens min:{tokens['min']:.0f} "
                    f"max:{tokens['max']:.0f} mean:{tokens['mean']:.1f} "
                    f"median:{tokens['median']:.0f}"
                )

    if unusual_chars:
        print("\nUnusual Unicode characters detected:")
        for ch, count in unusual_chars.most_common(50):
            name = unicodedata.name(ch, "UNKNOWN")
            print(f"  {repr(ch)} {name}: {count:,}")
    else:
        print("\nNo unusual Unicode characters detected.")

    if token_errors:
        print("\nTokenization errors:")
        for err in token_errors[:20]:
            print(f"  {err}")

    print(f"\nSaved analysis results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
