#!/usr/bin/env python3
"""
Regenerate predictions for bad-hygiene examples and merge with good-hygiene lines.

Expected folder layout:
  <hygiene_folder>/all_eval_results/
    checkpoint-<N>-good-hygiene-1000-examples.jsonl
    checkpoint-<N>-bad-hygiene-1000-examples.jsonl

For each checkpoint N divisible by --major_interval, this script:
  1) Verifies good/bad files are line-wise complementary (exactly one JSON object per line pair)
  2) Generates new predictions only for non-empty lines in the bad-hygiene file
  3) Merges lines into:
       checkpoint-<N>-gen1-inputs-refs-preds-1000-examples.jsonl

Adapter checkpoint lookup (for each checkpoint):
  <model_folder>/checkpoint-<N>
  <model_folder>/major_checkpoints/checkpoint-<N>
  <model_folder>/major_checkpoints/major-checkpoint-<N>
  <model_folder>/regular_checkpoints/checkpoint-<N>
  <model_folder>/regular_checkpoints/regular-checkpoint-<N>
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_configs import get_model_config

BAD_FILE_RE = re.compile(r"^checkpoint-(\d+)-bad-hygiene-(.+)\.jsonl$")


def clean_decoded_text(text: str) -> str:
    """Match evaluation cleanup behavior for decoded predictions."""
    text = text.replace("[/INST]", "").replace("[INST]", "")
    text = text.replace("</s>", "").replace("<s>", "")
    text = text.replace("\\", "")
    text = " ".join(text.split())
    return text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate predictions for bad-hygiene lines and merge with good-hygiene lines."
    )
    parser.add_argument(
        "--hygiene_folder",
        required=True,
        help="Specific model hygiene folder containing all_eval_results (e.g. hygiene_filtering/<model>-apptainer-fsdp).",
    )
    parser.add_argument(
        "--model_folder",
        required=True,
        help="Specific model checkpoint folder (e.g. models/<model>-apptainer-fsdp).",
    )
    parser.add_argument(
        "--examples_suffix",
        default="1000-examples",
        help="Suffix to process (default: 1000-examples).",
    )
    parser.add_argument(
        "--major_interval",
        type=int,
        default=500,
        help="Only process checkpoints where step %% major_interval == 0 (default: 500).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Generation batch size for bad-hygiene lines.",
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=2048,
        help="Prompt truncation length before generation.",
    )
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=40,
        help="Generation min_new_tokens.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=180,
        help="Generation max_new_tokens.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Beam size (1 = greedy).",
    )
    parser.add_argument(
        "--hf_token",
        default=os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN"),
        help="Hugging Face token (defaults to HUGGINGFACE_TOKEN/HF_TOKEN env).",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model short name override (usually inferred from folder name).",
    )
    parser.add_argument(
        "--pred_dataset",
        action="append",
        default=[],
        help=(
            "Optional explicit bad-hygiene JSONL path. Can be repeated. "
            "When set, these files are processed instead of scanning --hygiene_folder/all_eval_results."
        ),
    )
    parser.add_argument(
        "--pred_batch_size",
        type=int,
        default=None,
        help="Alias for --batch_size (preferred from sbatch wrappers).",
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Accepted for interface compatibility; regeneration requires LoRA adapters and this flag is ignored.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing checkpoint-<N>-gen1-inputs-refs-preds-<suffix>.jsonl files.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only list work; do not load models or write files.",
    )
    return parser.parse_args()


def iter_targets(
    results_dir: Path,
    examples_suffix: str,
    major_interval: int,
) -> Iterable[Tuple[int, Path, Path, Path]]:
    """Yield (step, good_file, bad_file, out_file)."""
    for bad_file in sorted(results_dir.glob("checkpoint-*-bad-hygiene-*.jsonl")):
        m = BAD_FILE_RE.match(bad_file.name)
        if not m:
            continue
        step = int(m.group(1))
        suffix = m.group(2)
        if suffix != examples_suffix:
            continue
        if step % major_interval != 0:
            continue

        good_file = results_dir / f"checkpoint-{step}-good-hygiene-{suffix}.jsonl"
        if not good_file.exists():
            raise FileNotFoundError(f"Missing complementary file: {good_file}")
        out_file = results_dir / f"checkpoint-{step}-gen1-inputs-refs-preds-{suffix}.jsonl"
        yield step, good_file, bad_file, out_file


def iter_targets_from_pred_datasets(
    pred_datasets: List[str],
    results_dir: Path,
    examples_suffix: str,
    major_interval: int,
) -> Iterable[Tuple[int, Path, Path, Path]]:
    seen = set()
    for raw_path in pred_datasets:
        bad_file = Path(raw_path).expanduser().resolve()
        if not bad_file.exists():
            raise FileNotFoundError(f"--pred_dataset file does not exist: {bad_file}")
        if bad_file.is_dir():
            raise ValueError(f"--pred_dataset must be a file path, got directory: {bad_file}")

        m = BAD_FILE_RE.match(bad_file.name)
        if not m:
            raise ValueError(
                f"--pred_dataset must match checkpoint-<N>-bad-hygiene-<suffix>.jsonl, got: {bad_file.name}"
            )
        step = int(m.group(1))
        suffix = m.group(2)
        if suffix != examples_suffix:
            continue
        if step % major_interval != 0:
            continue

        if bad_file.parent.resolve() != results_dir.resolve():
            raise ValueError(
                f"--pred_dataset file is outside hygiene folder all_eval_results: {bad_file}"
            )
        good_file = results_dir / f"checkpoint-{step}-good-hygiene-{suffix}.jsonl"
        if not good_file.exists():
            raise FileNotFoundError(f"Missing complementary file: {good_file}")
        out_file = results_dir / f"checkpoint-{step}-gen1-inputs-refs-preds-{suffix}.jsonl"

        key = str(out_file.resolve())
        if key in seen:
            continue
        seen.add(key)
        yield step, good_file, bad_file, out_file


def resolve_checkpoint_dir(model_folder: Path, step: int) -> Path:
    base = model_folder
    candidates = [
        base / f"checkpoint-{step}",
        base / "major_checkpoints" / f"checkpoint-{step}",
        base / "major_checkpoints" / f"major-checkpoint-{step}",
        base / "regular_checkpoints" / f"checkpoint-{step}",
        base / "regular_checkpoints" / f"regular-checkpoint-{step}",
    ]
    for c in candidates:
        if (c / "adapter_config.json").exists() and (
            (c / "adapter_model.safetensors").exists() or (c / "adapter_model.bin").exists()
        ):
            return c
    raise FileNotFoundError(
        f"No adapter checkpoint found for step={step} under {base}"
    )


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def parse_json_line(line: str, path: Path, line_idx: int) -> Dict:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON at {path}:{line_idx + 1}: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object at {path}:{line_idx + 1}, got {type(obj)}")
    return obj


def validate_and_collect(
    good_lines: List[str], bad_lines: List[str], good_path: Path, bad_path: Path
) -> Tuple[List[Optional[Dict]], List[Optional[Dict]], List[int]]:
    if len(good_lines) != len(bad_lines):
        raise ValueError(
            f"Line-count mismatch: {good_path} has {len(good_lines)}, {bad_path} has {len(bad_lines)}"
        )

    good_objs: List[Optional[Dict]] = [None] * len(good_lines)
    bad_objs: List[Optional[Dict]] = [None] * len(bad_lines)
    bad_indices: List[int] = []

    for i, (g, b) in enumerate(zip(good_lines, bad_lines)):
        g_nonempty = bool(g.strip())
        b_nonempty = bool(b.strip())
        if g_nonempty == b_nonempty:
            raise ValueError(
                f"Files are not complementary at line {i + 1}: "
                f"good_nonempty={g_nonempty}, bad_nonempty={b_nonempty}"
            )

        if g_nonempty:
            good_obj = parse_json_line(g, good_path, i)
            good_objs[i] = good_obj
            # sanity for merged output contract
            if "input_text" not in good_obj or "reference" not in good_obj:
                raise ValueError(
                    f"Good-hygiene JSON at {good_path}:{i + 1} missing input_text/reference fields"
                )
        else:
            bad_obj = parse_json_line(b, bad_path, i)
            bad_objs[i] = bad_obj
            if "input_text" not in bad_obj or "reference" not in bad_obj:
                raise ValueError(
                    f"Bad-hygiene JSON at {bad_path}:{i + 1} missing input_text/reference fields"
                )
            bad_indices.append(i)

    return good_objs, bad_objs, bad_indices


def batched(items: List[int], batch_size: int) -> Iterable[List[int]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def generate_for_bad_indices(
    *,
    bad_indices: List[int],
    bad_objs: List[Optional[Dict]],
    model_short: str,
    tokenizer,
    model,
    batch_size: int,
    max_input_tokens: int,
    min_new_tokens: int,
    max_new_tokens: int,
    num_beams: int,
) -> Dict[int, str]:
    if not bad_indices:
        return {}

    config = get_model_config(model_short)
    prompt_cfg = config.prompt_config
    max_input = max_input_tokens
    if config.max_input_text_tokens is not None:
        max_input = min(max_input, int(config.max_input_text_tokens))

    predictions: Dict[int, str] = {}
    for idx_batch in batched(bad_indices, batch_size):
        prompts: List[str] = []
        for idx in idx_batch:
            obj = bad_objs[idx]
            assert obj is not None
            doc_type = obj.get("doc_type")
            if doc_type is None and isinstance(obj.get("metadata"), dict):
                doc_type = obj["metadata"].get("doc_type")
            prompt = prompt_cfg.format_eval(
                input_text=str(obj.get("input_text", "")),
                doc_type=doc_type,
                tokenizer=tokenizer,
            )
            prompts.append(prompt)

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input,
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=num_beams,
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        attn = encoded["attention_mask"]
        for j, line_idx in enumerate(idx_batch):
            prompt_len = int(attn[j].sum().item())
            gen_ids = generated[j][prompt_len:]
            pred = tokenizer.decode(gen_ids, skip_special_tokens=True)
            predictions[line_idx] = clean_decoded_text(pred)

    return predictions


def write_merged_output(
    out_file: Path,
    good_objs: List[Optional[Dict]],
    bad_objs: List[Optional[Dict]],
    regenerated_predictions: Dict[int, str],
) -> None:
    with out_file.open("w", encoding="utf-8") as f:
        for i in range(len(good_objs)):
            good_obj = good_objs[i]
            if good_obj is not None:
                out_obj = dict(good_obj)
            else:
                bad_obj = dict(bad_objs[i] or {})
                bad_obj["prediction"] = regenerated_predictions[i]
                out_obj = bad_obj
            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")


def load_tokenizer_and_model(model_short: str, checkpoint_dir: Path, hf_token: Optional[str]):
    config = get_model_config(model_short)

    tokenizer = AutoTokenizer.from_pretrained(config.hf_name, token=hf_token, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        config.hf_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        token=hf_token,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, str(checkpoint_dir), is_trainable=False)
    model.eval()
    return tokenizer, model


def main() -> int:
    args = parse_args()
    hygiene_folder = Path(args.hygiene_folder).expanduser().resolve()
    model_folder = Path(args.model_folder).expanduser().resolve()
    results_dir = hygiene_folder / "all_eval_results"
    inferred_from_hygiene = (
        hygiene_folder.name[: -len("-apptainer-fsdp")]
        if hygiene_folder.name.endswith("-apptainer-fsdp")
        else None
    )
    inferred_from_model = (
        model_folder.name[: -len("-apptainer-fsdp")]
        if model_folder.name.endswith("-apptainer-fsdp")
        else None
    )
    model_short = args.model[0] if args.model else (inferred_from_hygiene or inferred_from_model)
    effective_batch_size = args.pred_batch_size if args.pred_batch_size is not None else args.batch_size

    if args.no_lora:
        print("Note: --no_lora is ignored. This script always loads LoRA adapters from checkpoints.")

    if not hygiene_folder.is_dir():
        raise SystemExit(f"hygiene_folder does not exist: {hygiene_folder}")
    if not results_dir.is_dir():
        raise SystemExit(f"hygiene_folder missing all_eval_results: {results_dir}")
    if not model_folder.is_dir():
        raise SystemExit(f"model_folder does not exist: {model_folder}")
    if not model_short:
        raise SystemExit(
            "Could not infer model short-name from folder names. Pass --model=<short-name>."
        )

    if args.pred_dataset:
        targets = list(
            iter_targets_from_pred_datasets(
                pred_datasets=args.pred_dataset,
                results_dir=results_dir,
                examples_suffix=args.examples_suffix,
                major_interval=args.major_interval,
            )
        )
    else:
        targets = list(
            iter_targets(
                results_dir=results_dir,
                examples_suffix=args.examples_suffix,
                major_interval=args.major_interval,
            )
        )
    
    if not targets:
        print("No matching checkpoints found.")
        return 0

    print(f"Found {len(targets)} checkpoint hygiene pairs to process.")
    for step, good_file, bad_file, out_file in targets:
        print(f"  - {model_short} checkpoint-{step}: {good_file.name} + {bad_file.name}")

    if args.dry_run:
        print("--dry_run enabled; exiting without generation.")
        return 0

    for step, good_file, bad_file, out_file in targets:
        if out_file.exists() and not args.overwrite:
            print(f"Skipping existing output (use --overwrite): {out_file}")
            continue

        print(f"\nProcessing {model_short} checkpoint-{step}")
        good_lines = read_lines(good_file)
        bad_lines = read_lines(bad_file)
        good_objs, bad_objs, bad_indices = validate_and_collect(good_lines, bad_lines, good_file, bad_file)
        print(
            f"  line_count={len(good_lines)}, regenerate={len(bad_indices)}, "
            f"copy_good={len(good_lines) - len(bad_indices)}"
        )

        checkpoint_dir = resolve_checkpoint_dir(model_folder, step)
        print(f"  using adapter checkpoint: {checkpoint_dir}")
        tokenizer, model = load_tokenizer_and_model(model_short, checkpoint_dir, args.hf_token)

        regenerated = generate_for_bad_indices(
            bad_indices=bad_indices,
            bad_objs=bad_objs,
            model_short=model_short,
            tokenizer=tokenizer,
            model=model,
            batch_size=effective_batch_size,
            max_input_tokens=args.max_input_tokens,
            min_new_tokens=args.min_new_tokens,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        if set(regenerated.keys()) != set(bad_indices):
            raise RuntimeError(
                f"Internal error: regenerated indices mismatch for {bad_file}. "
                f"expected={len(bad_indices)} got={len(regenerated)}"
            )

        write_merged_output(out_file, good_objs, bad_objs, regenerated)
        print(f"  wrote: {out_file}")

        # Free VRAM between checkpoints.
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
