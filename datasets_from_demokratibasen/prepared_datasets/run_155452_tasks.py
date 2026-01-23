#!/usr/bin/env python3
"""
Process tasks for 155452_text_summary_examples dataset.

TASK 1:
- Collect short inputs (<30 chars) and short outputs (<40 chars).
- Sort by length then alphabetically.
- Write to short_inputs.txt and short_outputs.txt.
- Write stats to short_input_output_stats.json.

TASK 2:
- Add embedding_distance to each embeddings JSONL line.
- Compute mean/min/max distances.
- Write top-20 (distance, input, output) pairs to large_distance_intput_output_pairs.json.

TASK 3:
- Clean excessive '|' characters in input/output for main and split files.
"""

from __future__ import annotations

import json
import math
import re
import heapq
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


BASE_DIR = Path("datasets_from_demokratibasen/cleaned_datasets/text_summary_dataset_202601")
MAIN_FILE = BASE_DIR / "155452_text_summary_examples.jsonl"
EMBEDDINGS_FILE = BASE_DIR / "155452_text_summary_examples_embeddings.jsonl"

SPLIT_FILES = [
    BASE_DIR / "155452_text_summary_examples_train.jsonl",
    BASE_DIR / "155452_text_summary_examples_val.jsonl",
    BASE_DIR / "155452_text_summary_examples_test.jsonl",
]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl(path: Path, items):
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def task1_collect_short_texts():
    short_inputs: List[str] = []
    short_outputs: List[str] = []
    total = 0
    short_input_count = 0
    short_output_count = 0

    for obj in iter_jsonl(MAIN_FILE):
        total += 1
        input_text = str(obj.get("input", ""))
        output_text = str(obj.get("output", ""))

        if len(input_text) < 30:
            short_inputs.append(input_text)
            short_input_count += 1

        if len(output_text) < 40:
            short_outputs.append(output_text)
            short_output_count += 1

    short_inputs.sort(key=lambda s: (len(s), s))
    short_outputs.sort(key=lambda s: (len(s), s))

    short_inputs_path = BASE_DIR / "short_inputs.txt"
    short_outputs_path = BASE_DIR / "short_outputs.txt"
    stats_path = BASE_DIR / "short_input_output_stats.json"

    short_inputs_path.write_text("\n".join(short_inputs) + ("\n" if short_inputs else ""), encoding="utf-8")
    short_outputs_path.write_text("\n".join(short_outputs) + ("\n" if short_outputs else ""), encoding="utf-8")

    stats = {
        "source_file": str(MAIN_FILE),
        "total_examples": total,
        "short_input_threshold_chars": 30,
        "short_output_threshold_chars": 40,
        "short_input_count": short_input_count,
        "short_output_count": short_output_count,
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cosine_distance(input_emb: List[float], output_emb: List[float]) -> Optional[float]:
    if input_emb is None or output_emb is None:
        return None
    if len(input_emb) == 0 or len(output_emb) == 0:
        return None
    if len(input_emb) != len(output_emb):
        return None
    dot = float(np.dot(input_emb, output_emb))
    return 1.0 - dot


def task2_add_embedding_distance_and_top_pairs():
    tmp_path = EMBEDDINGS_FILE.with_suffix(".jsonl.tmp")

    count = 0
    dist_sum = 0.0
    dist_min = None
    dist_max = None

    # Keep top 20 largest distances
    top_heap: List[Tuple[float, str]] = []

    with EMBEDDINGS_FILE.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            input_emb = obj.get("input_embedding")
            output_emb = obj.get("output_embedding")
            doc_id = obj.get("dokument_id")

            dist = cosine_distance(input_emb, output_emb)
            obj["embedding_distance"] = dist

            if dist is not None:
                count += 1
                dist_sum += dist
                dist_min = dist if dist_min is None else min(dist_min, dist)
                dist_max = dist if dist_max is None else max(dist_max, dist)

                if doc_id is not None:
                    if len(top_heap) < 20:
                        heapq.heappush(top_heap, (dist, str(doc_id)))
                    else:
                        if dist > top_heap[0][0]:
                            heapq.heapreplace(top_heap, (dist, str(doc_id)))

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    tmp_path.replace(EMBEDDINGS_FILE)

    # Prepare top 20 in descending order
    top_list = sorted(top_heap, key=lambda x: x[0], reverse=True)
    top_doc_ids = {doc_id for _, doc_id in top_list}

    # Map doc_id -> (input, output)
    doc_texts: Dict[str, Tuple[str, str]] = {}
    for obj in iter_jsonl(MAIN_FILE):
        metadata = obj.get("metadata", {})
        doc_id = metadata.get("dokument_id")
        if doc_id is None:
            continue
        doc_id_str = str(doc_id)
        if doc_id_str in top_doc_ids:
            doc_texts[doc_id_str] = (str(obj.get("input", "")), str(obj.get("output", "")))
            if len(doc_texts) == len(top_doc_ids):
                break

    top_pairs = []
    for dist, doc_id in top_list:
        input_text, output_text = doc_texts.get(doc_id, ("", ""))
        top_pairs.append({
            "dokument_id": doc_id,
            "embedding_distance": dist,
            "input": input_text,
            "output": output_text,
        })

    pairs_path = BASE_DIR / "large_distance_intput_output_pairs.json"
    pairs_path.write_text(json.dumps(top_pairs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    stats_path = BASE_DIR / "embedding_distance_stats.json"
    stats = {
        "source_file": str(EMBEDDINGS_FILE),
        "count": count,
        "mean": (dist_sum / count) if count else None,
        "min": dist_min,
        "max": dist_max,
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def clean_bar_text(text: str) -> str:
    if text is None:
        return text
    if text.count("|") <= 5:
        return text
    # 1) Substitute each bar with space
    text = text.replace("|", " ")
    # 2) Collapse consecutive spaces to single space
    text = re.sub(r" {2,}", " ", text)
    # 3) Remove spaces around line feeds
    text = re.sub(r" *\n *", "\n", text)
    # 4) Replace 3+ consecutive line feeds with two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def process_file_bars(path: Path):
    tmp_path = path.with_suffix(".jsonl.tmp")
    with path.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            input_text = obj.get("input")
            output_text = obj.get("output")

            if isinstance(input_text, str):
                obj["input"] = clean_bar_text(input_text)
            if isinstance(output_text, str):
                obj["output"] = clean_bar_text(output_text)

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    tmp_path.replace(path)


def task3_clean_bars():
    files_to_process = [MAIN_FILE] + SPLIT_FILES
    for path in files_to_process:
        if path.exists():
            process_file_bars(path)


def main():
    if not MAIN_FILE.exists():
        raise FileNotFoundError(f"Main file not found: {MAIN_FILE}")
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_FILE}")

    # TASK 1: Collect short texts from original file
    task1_collect_short_texts()

    # TASK 3: Clean bar characters in input/output (main + splits)
    task3_clean_bars()

    # TASK 2: Add embedding distances and output stats/top pairs
    task2_add_embedding_distance_and_top_pairs()


if __name__ == "__main__":
    main()
