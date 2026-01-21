#!/usr/bin/env python3
import json
import math
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import tiktoken


BASE_DIR = Path(__file__).resolve().parent

# NOTE: The requested band label is "ultra_narow" (spelling preserved).
BAND_SOURCES = {
    "ultra_narow": "155452_text_summary_examples_ultra_narrow.jsonl",
    "narrow": "155452_text_summary_examples_narrow.jsonl",
    "medium": "155452_text_summary_examples_medium.jsonl",
    "broad": "155452_text_summary_examples_broad.jsonl",
}

TASKS = {
    "3060_c2_text_prediction": {
        "total": {
            "ultra_narow": 1500,
            "narrow": 1250,
            "medium": 250,
            "broad": 60,
        },
        "val": {
            "ultra_narow": 500,
            "narrow": 250,
            "medium": 50,
            "broad": 10,
        },
        "test": {
            "ultra_narow": 1000,
            "narrow": 1000,
            "medium": 200,
            "broad": 50,
        },
        "regular": {
            "ultra_narow": 100,
            "narrow": 50,
            "medium": 10,
            "broad": 0,
        },
    },
    "9650_c3_prompt_continuation": {
        "total": {
            "ultra_narow": 5000,
            "narrow": 3750,
            "medium": 750,
            "broad": 149,
        },
        "val": {
            "ultra_narow": 2500,
            "narrow": 1250,
            "medium": 250,
            "broad": 50,
        },
        "test": {
            "ultra_narow": 2500,
            "narrow": 2500,
            "medium": 500,
            "broad": 99,
        },
        "regular": {
            "ultra_narow": 500,
            "narrow": 250,
            "medium": 50,
            "broad": 0,
        },
    },
}

RANDOM_SEED = 1337

TOKEN_LIMITS = {
    "ultra_narow": 192,
    "narrow": 512,
    "medium": 2048,
    "broad": 8912,
}
TOKEN_MARGIN = 1.2

ENCODING = tiktoken.get_encoding("o200k_base")


@dataclass
class SplitResult:
    val_ids: List[str]
    test_ids: List[str]
    regular_ids: Set[str]


def _extract_doc_ids(file_path: Path) -> List[str]:
    doc_ids: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            metadata = obj.get("metadata", {})
            doc_id = metadata.get("dokument_id")
            if doc_id:
                doc_ids.append(doc_id)
    return doc_ids


def _validate_counts() -> None:
    for task_name, specs in TASKS.items():
        for band in BAND_SOURCES.keys():
            total = specs["total"][band]
            val = specs["val"][band]
            test = specs["test"][band]
            regular = specs["regular"][band]
            if val + test != total:
                raise ValueError(
                    f"{task_name} {band}: val({val}) + test({test}) != total({total})"
                )
            if regular > val:
                raise ValueError(
                    f"{task_name} {band}: regular({regular}) > val({val})"
                )


def _split_task_band(
    doc_ids: List[str],
    task_name: str,
    band: str,
    rng: random.Random,
) -> SplitResult:
    specs = TASKS[task_name]
    total = specs["total"][band]
    val_count = specs["val"][band]
    test_count = specs["test"][band]
    regular_count = specs["regular"][band]

    if len(doc_ids) < total:
        raise ValueError(
            f"Not enough documents for {task_name} {band}: "
            f"need {total}, have {len(doc_ids)}"
        )

    rng.shuffle(doc_ids)
    selected = doc_ids[:total]
    val_ids = selected[:val_count]
    test_ids = selected[val_count : val_count + test_count]

    if len(test_ids) != test_count:
        raise ValueError(
            f"{task_name} {band}: expected test {test_count}, got {len(test_ids)}"
        )

    regular_ids = set()
    if regular_count > 0:
        regular_ids = set(rng.sample(val_ids, regular_count))

    return SplitResult(val_ids=val_ids, test_ids=test_ids, regular_ids=regular_ids)


def _write_output(
    output_path: Path,
    task_name: str,
    band: str,
    split: SplitResult,
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for doc_id in split.val_ids:
            stage = "regular" if doc_id in split.regular_ids else "major"
            f.write(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "task": task_name,
                        "band": band,
                        "stage": stage,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        for doc_id in split.test_ids:
            f.write(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "task": task_name,
                        "band": band,
                        "stage": "final",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _is_word_char(ch: str) -> bool:
    return ch.isalpha()


def _token_start_positions(tokens: List[int]) -> List[int]:
    positions: List[int] = []
    offset = 0
    for tok in tokens:
        positions.append(offset)
        piece = ENCODING.decode([tok])
        offset += len(piece)
    return positions


def _is_word_start(text: str, start_pos: int) -> bool:
    if start_pos >= len(text):
        return False
    if not _is_word_char(text[start_pos]):
        return False
    if start_pos == 0:
        return True
    return not _is_word_char(text[start_pos - 1])


def _is_paragraph_start(text: str, start_pos: int) -> bool:
    if start_pos == 0:
        return True
    if start_pos >= 2 and text[start_pos - 2 : start_pos] == "\n\n":
        return True
    return False


def _is_sentence_start(text: str, start_pos: int) -> bool:
    if start_pos == 0:
        return True
    idx = start_pos - 1
    while idx >= 0 and text[idx].isspace():
        idx -= 1
    if idx < 0:
        return True
    return text[idx] in ".!?"


def _sanitize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    cleaned: List[str] = []
    for ch in text:
        if ch == "\n":
            cleaned.append(ch)
            continue
        if ch.isspace():
            cleaned.append(" ")
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("C"):
            continue
        cleaned.append(ch)
    return "".join(cleaned).strip()


def process_input_text(task: str, band: str, text: str) -> str:
    base = TOKEN_LIMITS[band]
    target_tokens = int(math.ceil(base * TOKEN_MARGIN))
    sanitized_text = _sanitize_text(text)
    tokens = ENCODING.encode(sanitized_text)
    if not tokens:
        assert False, f"No tokens for {task} {band}: {sanitized_text}"
    # if len(tokens) <= target_tokens:
    #     return sanitized_text

    positions = _token_start_positions(tokens)
    max_start = len(tokens) - target_tokens

    def token_to_pos(idx: int) -> int:
        return positions[idx]

    valid_starts: List[int] = []
    if task == "3060_c2_text_prediction":
        for i in range(0, max_start + 1):
            if _is_word_start(sanitized_text, token_to_pos(i)):
                valid_starts.append(i)
    else:
        for i in range(0, max_start + 1):
            if _is_paragraph_start(sanitized_text, token_to_pos(i)):
                valid_starts.append(i)
        if not valid_starts:
            for i in range(0, max_start + 1):
                if _is_sentence_start(sanitized_text, token_to_pos(i)):
                    valid_starts.append(i)

    if not valid_starts:
        start_idx = 0
    else:
        start_idx = random.choice(valid_starts)

    selected_tokens = tokens[start_idx : start_idx + target_tokens]
    return _sanitize_text(ENCODING.decode(selected_tokens))


def _load_band_inputs(band: str) -> Dict[str, str]:
    file_path = BASE_DIR / BAND_SOURCES[band]
    inputs: Dict[str, str] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            metadata = obj.get("metadata", {})
            doc_id = metadata.get("dokument_id")
            if not doc_id:
                continue
            inputs[doc_id] = obj.get("input", "")
    return inputs


def _write_examples_from_ids(task_name: str, band: str, inputs: Dict[str, str]) -> int:
    ids_path = BASE_DIR / f"{task_name}_{band}_ids.jsonl"
    out_path = BASE_DIR / f"{task_name}_{band}_examples.jsonl"
    count = 0
    with open(ids_path, "r", encoding="utf-8") as f_in, open(
        out_path, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            doc_id = item.get("doc_id")
            task = item.get("task")
            band_value = item.get("band")
            stage = item.get("stage")
            text = inputs.get(doc_id)
            if text is None:
                raise ValueError(
                    f"Missing doc_id in band {band}: {doc_id} (task {task_name})"
                )
            processed_text = process_input_text(task_name, band, text)
            f_out.write(
                json.dumps(
                    {
                        "id": doc_id,
                        "task": task,
                        "band": band_value,
                        "stage": stage,
                        "text": processed_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


def main() -> None:
    _validate_counts()

    print("Loading band files...")
    band_doc_ids: Dict[str, List[str]] = {}
    for band, file_name in BAND_SOURCES.items():
        file_path = BASE_DIR / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Missing band file: {file_path}")
        band_doc_ids[band] = _extract_doc_ids(file_path)
        print(f"  {band}: {len(band_doc_ids[band]):,} docs")

    ids_paths = [
        BASE_DIR / f"3060_c2_text_prediction_{band}_ids.jsonl"
        for band in BAND_SOURCES.keys()
    ] + [
        BASE_DIR / f"9650_c3_prompt_continuation_{band}_ids.jsonl"
        for band in BAND_SOURCES.keys()
    ]
    ids_exist = all(path.exists() for path in ids_paths)

    if not ids_exist:
        print("\nAssigning examples to tasks and splits...")
        rng = random.Random(RANDOM_SEED)

        for band, doc_ids in band_doc_ids.items():
            required_total = (
                TASKS["3060_c2_text_prediction"]["total"][band]
                + TASKS["9650_c3_prompt_continuation"]["total"][band]
            )
            if len(doc_ids) < required_total:
                raise ValueError(
                    f"{band}: need {required_total} docs for tasks, "
                    f"have {len(doc_ids)}"
                )

            rng.shuffle(doc_ids)
            c2_total = TASKS["3060_c2_text_prediction"]["total"][band]
            c2_ids = doc_ids[:c2_total]
            c3_ids = doc_ids[
                c2_total : c2_total
                + TASKS["9650_c3_prompt_continuation"]["total"][band]
            ]

            c2_rng = random.Random(f"{RANDOM_SEED}-{band}-c2")
            c3_rng = random.Random(f"{RANDOM_SEED}-{band}-c3")

            c2_split = _split_task_band(c2_ids, "3060_c2_text_prediction", band, c2_rng)
            c3_split = _split_task_band(
                c3_ids, "9650_c3_prompt_continuation", band, c3_rng
            )

            c2_out = BASE_DIR / f"3060_c2_text_prediction_{band}_ids.jsonl"
            c3_out = BASE_DIR / f"9650_c3_prompt_continuation_{band}_ids.jsonl"

            _write_output(c2_out, "3060_c2_text_prediction", band, c2_split)
            _write_output(c3_out, "9650_c3_prompt_continuation", band, c3_split)

            print(f"\nBand: {band}")
            print("  3060_c2_text_prediction:")
            print(f"    total: {len(c2_ids):,}")
            print(
                f"    val: {len(c2_split.val_ids):,} "
                f"(regular: {len(c2_split.regular_ids):,})"
            )
            print(f"    test: {len(c2_split.test_ids):,}")
            print("  9650_c3_prompt_continuation:")
            print(f"    total: {len(c3_ids):,}")
            print(
                f"    val: {len(c3_split.val_ids):,} "
                f"(regular: {len(c3_split.regular_ids):,})"
            )
            print(f"    test: {len(c3_split.test_ids):,}")

        print("\nDone. Wrote 8 TASK_BAND_ids.jsonl files.")
    else:
        print("\nSkipping ID generation (all 8 *_ids.jsonl files already exist).")

    print("\nBuilding TASK_BAND_examples.jsonl files...")
    for band in BAND_SOURCES.keys():
        inputs = _load_band_inputs(band)
        for task_name in TASKS.keys():
            count = _write_examples_from_ids(task_name, band, inputs)
            print(f"  {task_name} {band}: {count:,} examples")

    print("\nDone. Wrote 8 TASK_BAND_examples.jsonl files.")


if __name__ == "__main__":
    main()
