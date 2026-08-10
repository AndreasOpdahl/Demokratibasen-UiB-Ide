#!/usr/bin/env python3
"""
Regenerate predictions for hygiene-bad examples and merge with hygiene-good lines.

Expected folder layout:
  <results_folder>/all_eval_results/
    checkpoint-<N>-gen<M>-hygiene-good-<suffix>.jsonl
    checkpoint-<N>-gen<M>-hygiene-bad-<suffix>.jsonl

For each checkpoint N divisible by --major_interval, this script:
  1) Verifies good/bad files are line-wise complementary (exactly one JSON object per line pair)
  2) Generates new predictions only for non-empty lines in the hygiene-bad file
  3) Merges lines into:
       checkpoint-<N>-gen<M+1>-inputs-refs-preds-<suffix>.jsonl

The ``gen<M>-`` fragment is required. For example,
``checkpoint-500-gen0-hygiene-bad-1000-examples.jsonl`` produces
``checkpoint-500-gen1-inputs-refs-preds-1000-examples.jsonl``.
By default only gen0 hygiene-bad files are processed; pass ``--gen 1`` to
advance gen1 hygiene outputs to gen2, and so on.

Adapter checkpoint lookup (for each checkpoint):
  <model_folder>/checkpoint-<N>
  <model_folder>/major_checkpoints/checkpoint-<N>
  <model_folder>/major_checkpoints/major-checkpoint-<N>
  <model_folder>/regular_checkpoints/checkpoint-<N>
  <model_folder>/regular_checkpoints/regular-checkpoint-<N>
"""

import argparse
from argparse import Namespace
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import importlib.util

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_configs import get_model_config

_generation_utils_path = Path(__file__).resolve().parent / "utils" / "generation.py"
_generation_spec = importlib.util.spec_from_file_location("generation_utils", _generation_utils_path)
if _generation_spec is None or _generation_spec.loader is None:
    raise ImportError(f"Could not load generation utilities from {_generation_utils_path}")
_generation_utils = importlib.util.module_from_spec(_generation_spec)
sys.modules.setdefault("generation_utils", _generation_utils)
_generation_spec.loader.exec_module(_generation_utils)
extract_generated_continuations = _generation_utils.extract_generated_continuations
make_inputs_refs_preds_record = _generation_utils.make_inputs_refs_preds_record
postprocess_generated_summary_text = _generation_utils.postprocess_generated_summary_text
sync_model_tokenizer_special_tokens = _generation_utils.sync_model_tokenizer_special_tokens

_report_hygiene_path = Path(__file__).resolve().parent / "report_hygiene.py"
_report_hygiene_spec = importlib.util.spec_from_file_location("report_hygiene_utils", _report_hygiene_path)
if _report_hygiene_spec is None or _report_hygiene_spec.loader is None:
    raise ImportError(f"Could not load hygiene utilities from {_report_hygiene_path}")
_report_hygiene_utils = importlib.util.module_from_spec(_report_hygiene_spec)
sys.modules.setdefault("report_hygiene_utils", _report_hygiene_utils)
_report_hygiene_spec.loader.exec_module(_report_hygiene_utils)
evaluate_summary = _report_hygiene_utils.evaluate_summary
check_criteria = _report_hygiene_utils.check_criteria
NorwegianLightParser = _report_hygiene_utils.NorwegianLightParser
load_lexicon = _report_hygiene_utils.load_lexicon

BAD_FILE_RE = re.compile(r"^(.*?)checkpoint-(\d+)-gen(\d+)-hygiene-bad-(.+)\.jsonl$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate predictions for hygiene-bad lines and merge with hygiene-good lines."
    )
    parser.add_argument(
        "--results_folder",
        "--hygiene_folder",
        dest="results_folder",
        required=True,
        help=(
            "Specific model results folder containing all_eval_results "
            "(e.g. models/<model>-apptainer-fsdp). --hygiene_folder is a deprecated alias."
        ),
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
        "--generation",
        "--gen",
        dest="generation_num",
        type=int,
        default=0,
        help="Input hygiene generation to regenerate (default: 0, producing gen1 outputs).",
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
        help="Generation batch size for [gen<M>-]hygiene-bad lines.",
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
        "--num_candidates",
        type=int,
        default=8,
        help="Sample this many candidate summaries per bad input (the regeneration budget). "
        "A candidate is accepted when it passes hygiene AND meets every enabled quality "
        "tolerance; otherwise the candidate closest to the original scores is chosen "
        "(default: 8).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for regeneration (default: 0.7).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability for regeneration (default: 0.9).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling cutoff for regeneration (default: 50; use 0 to disable).",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=3,
        help="Prevent exact repeated n-grams of this size during regeneration (default: 3; use 0 to disable).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampled regeneration. Omit for non-deterministic retries.",
    )
    parser.add_argument(
        "--rougeLsum_tolerance",
        type=float,
        default=0.95,
        help="Accept a candidate only if its ROUGE-Lsum (vs reference) is at least this "
        "fraction of the original prediction's ROUGE-Lsum. 0 disables the ROUGE gate "
        "(default: 0.95).",
    )
    parser.add_argument(
        "--bertscore_tolerance",
        type=float,
        default=0.95,
        help="Accept a candidate only if its BERTScore F1 (eval_reference_bertscore_f1_mean, "
        "vs reference) is at least this fraction of the original's. 0 disables the BERTScore "
        "gate (default: 0.95).",
    )
    parser.add_argument(
        "--faithfulness_tolerance",
        type=float,
        default=0.95,
        help="Accept a candidate only if its NLI faithfulness (eval_faithfulness."
        "mean_entailment_score, vs source) is at least this fraction of the original's. "
        "0 disables the faithfulness gate (default: 0.95).",
    )
    parser.add_argument("--max_rep_3gram", type=float, default=0.2)
    parser.add_argument("--min_compression_ratio", type=float, default=1.0)
    parser.add_argument("--min_pred_chars", type=int, default=50)
    parser.add_argument("--max_pred_ref_char_ratio", type=float, default=1.5)
    parser.add_argument("--max_markup_ratio", type=float, default=0.01)
    parser.add_argument("--min_punctuation_score", type=float, default=-0.1)
    parser.add_argument("--min_complete_sentence_ratio", type=float, default=0.6)
    parser.add_argument("--min_known_word_ratio", type=float, default=0.9)
    parser.add_argument(
        "--lexicon",
        default=None,
        help="Optional external Norwegian lexicon file used for candidate hygiene scoring.",
    )
    parser.add_argument(
        "--spacy_model",
        "--spacy-model",
        dest="spacy_model",
        default="nb_core_news_md",
        help="spaCy model for candidate hygiene scoring (default: nb_core_news_md).",
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
            "Optional explicit [gen<M>-]hygiene-bad JSONL path. Can be repeated. "
            "When set, these files are processed instead of scanning --results_folder/all_eval_results."
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
    exist_group = parser.add_mutually_exclusive_group()
    exist_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    exist_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip targets whose output already exists (resume mode).",
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
    generation_num: int,
) -> Iterable[Tuple[int, Path, Path, Path]]:
    """Yield (step, good_file, bad_file, out_file)."""
    for bad_file in sorted(results_dir.glob("*checkpoint-*-hygiene-bad-*.jsonl")):
        m = BAD_FILE_RE.match(bad_file.name)
        if not m:
            continue
        prefix = m.group(1)
        step = int(m.group(2))
        gen_num = int(m.group(3))
        if gen_num != generation_num:
            continue
        gen_part = f"gen{gen_num}-"
        suffix = m.group(4)
        if suffix != examples_suffix:
            continue
        if step % major_interval != 0:
            continue

        good_file = results_dir / f"{prefix}checkpoint-{step}-{gen_part}hygiene-good-{suffix}.jsonl"
        if not good_file.exists():
            raise FileNotFoundError(f"Missing complementary file: {good_file}")
        out_file = results_dir / f"{prefix}checkpoint-{step}-gen{gen_num + 1}-inputs-refs-preds-{suffix}.jsonl"
        yield step, good_file, bad_file, out_file


def iter_targets_from_pred_datasets(
    pred_datasets: List[str],
    results_dir: Path,
    examples_suffix: str,
    major_interval: int,
    generation_num: int,
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
                f"--pred_dataset must match [<prefix>]checkpoint-<N>-gen<M>-hygiene-bad-<suffix>.jsonl, got: {bad_file.name}"
            )
        prefix = m.group(1)
        step = int(m.group(2))
        gen_num = int(m.group(3))
        if gen_num != generation_num:
            raise ValueError(
                f"--pred_dataset generation gen{gen_num} does not match --gen {generation_num}: {bad_file.name}"
            )
        gen_part = f"gen{gen_num}-"
        suffix = m.group(4)
        if suffix != examples_suffix:
            continue
        if step % major_interval != 0:
            continue

        if bad_file.parent.resolve() != results_dir.resolve():
            raise ValueError(
                f"--pred_dataset file is outside results folder all_eval_results: {bad_file}"
            )
        good_file = results_dir / f"{prefix}checkpoint-{step}-{gen_part}hygiene-good-{suffix}.jsonl"
        if not good_file.exists():
            raise FileNotFoundError(f"Missing complementary file: {good_file}")
        out_file = results_dir / f"{prefix}checkpoint-{step}-gen{gen_num + 1}-inputs-refs-preds-{suffix}.jsonl"

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


def load_original_predictions(bad_file: Path, bad_indices: List[int]) -> Dict[int, str]:
    """Recover the original (pre-regeneration) prediction text for each bad line.

    The hygiene-bad file stores ``prediction: null`` for failed lines, so the
    baseline text is read from the line-aligned ``inputs-refs-preds`` file of the
    same generation. Used to compute original ROUGE/BERTScore/faithfulness scores
    against which candidate tolerances are measured.
    """
    src_file = bad_file.with_name(bad_file.name.replace("-hygiene-bad-", "-inputs-refs-preds-"))
    if "-inputs-refs-preds-" not in src_file.name or not src_file.exists():
        raise FileNotFoundError(
            "Could not locate the source inputs-refs-preds file needed for "
            f"quality tolerances: expected {src_file}. Disable the gates with "
            "--rougeLsum_tolerance=0 --bertscore_tolerance=0 --faithfulness_tolerance=0, "
            "or provide the missing file."
        )
    lines = read_lines(src_file)
    out: Dict[int, str] = {}
    bad_set = set(bad_indices)
    for i, line in enumerate(lines):
        if i not in bad_set:
            continue
        obj = parse_json_line(line, src_file, i)
        out[i] = str(obj.get("prediction") or "")
    missing = bad_set - set(out.keys())
    if missing:
        raise ValueError(
            f"Source file {src_file} is missing {len(missing)} bad lines "
            f"(line count {len(lines)} vs bad indices up to {max(bad_indices)})."
        )
    return out


def batched(items: List[int], batch_size: int) -> Iterable[List[int]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def score_candidate(
    *,
    doc: str,
    prediction: str,
    reference: str,
    nlp_parser,
    hygiene_args: Namespace,
) -> Tuple[Tuple[int, int, float, float, float, float], Dict]:
    metrics = evaluate_summary(doc, prediction, reference, nlp_parser)
    criteria = check_criteria(metrics, hygiene_args)
    passed_count = sum(1 for ok in criteria.values() if ok)
    passed_all = 1 if all(criteria.values()) else 0
    rep_3gram = float(metrics.get("rep_3gram") or 0.0)
    known_word_ratio = float(metrics.get("known_word_ratio") or 0.0)
    complete_ratio = float(metrics.get("complete_sentence_ratio") or 0.0)
    pred_ref_ratio = metrics.get("pred_ref_char_ratio")
    ratio_penalty = abs(float(pred_ref_ratio) - 1.0) if pred_ref_ratio is not None else 99.0
    # Higher tuple is better. Hygiene pass count dominates; soft metrics break ties.
    return (
        passed_all,
        passed_count,
        -rep_3gram,
        known_word_ratio,
        complete_ratio,
        -ratio_penalty,
    ), metrics


class CandidateQualityScorer:
    """Per-candidate quality scoring for ROUGE-Lsum, BERTScore F1, and NLI faithfulness.

    Only the metrics whose tolerance is > 0 are computed, so disabling a gate
    (tolerance 0) also removes its runtime cost. Scores are on a 0-1 scale.
    """

    def __init__(self, *, use_rouge: bool, use_bertscore: bool, use_faithfulness: bool):
        self.use_rouge = use_rouge
        self.use_bertscore = use_bertscore
        self.use_faithfulness = use_faithfulness
        self._gate = None

    @property
    def enabled(self) -> bool:
        return self.use_rouge or self.use_bertscore or self.use_faithfulness

    def _nli_gate(self):
        if self._gate is None:
            from utils.faithfulness import NLIFaithfulnessGate

            self._gate = NLIFaithfulnessGate()
        return self._gate

    def score_batch(
        self, docs: List[str], preds: List[str], refs: List[str]
    ) -> List[Dict[str, float]]:
        """Return one {metric: value} dict per (doc, pred, ref) triple."""
        n = len(preds)
        out: List[Dict[str, float]] = [dict() for _ in range(n)]
        if n == 0:
            return out
        if self.use_rouge:
            from utils.metrics import compute_rouge

            for i, (p, r) in enumerate(zip(preds, refs)):
                try:
                    out[i]["rougeLsum"] = float(compute_rouge([p], [r]).get("rougeLsum", 0.0))
                except Exception:  # noqa: BLE001 - degrade gracefully
                    out[i]["rougeLsum"] = 0.0
        if self.use_bertscore:
            from utils.metrics import compute_bertscore_per_example

            f1s = compute_bertscore_per_example(preds, refs)
            for i in range(n):
                out[i]["bertscore_f1"] = float(f1s[i]) if i < len(f1s) else 0.0
        if self.use_faithfulness:
            gate = self._nli_gate()
            for i, (d, p) in enumerate(zip(docs, preds)):
                try:
                    res = gate.score_and_gate(d, p)
                    out[i]["entailment"] = float(res["faithfulness"]["entailment_mean"])
                except Exception:  # noqa: BLE001 - degrade gracefully
                    out[i]["entailment"] = 0.0
        return out


def select_candidate_index(
    *,
    hygiene_tuples: List[tuple],
    hygiene_passed: List[bool],
    candidate_quality: List[Dict[str, float]],
    original_quality: Dict[str, float],
    tolerances: Dict[str, float],
) -> int:
    """Pick the winning candidate among a single bad example's candidates.

    Rules (matching the requested semantics):
      * A candidate is "accepted" when it passes hygiene AND every enabled metric
        satisfies new >= tolerance * original (originals <= 0 impose no constraint).
      * If any candidate is accepted, return the accepted one with the highest
        combined quality ratio (tie-broken by hygiene tuple).
      * Otherwise (budget exhausted, none accepted), return the candidate closest
        to the original scores (smallest total relative shortfall), preferring
        hygiene-passing candidates.
      * With no enabled metric, fall back to the legacy best-by-hygiene choice.
    """
    n = len(hygiene_tuples)
    if n == 0:
        raise ValueError("select_candidate_index called with no candidates")

    if not tolerances:
        return max(range(n), key=lambda i: hygiene_tuples[i])

    def combined_ratio(i: int) -> float:
        total = 0.0
        for m in tolerances:
            orig = original_quality.get(m, 0.0)
            new = candidate_quality[i].get(m, 0.0)
            total += (new / orig) if orig > 0 else 1.0
        return total

    def total_shortfall(i: int) -> float:
        total = 0.0
        for m in tolerances:
            orig = original_quality.get(m, 0.0)
            if orig > 0:
                new = candidate_quality[i].get(m, 0.0)
                total += max(0.0, (orig - new) / orig)
        return total

    def meets(i: int) -> bool:
        if not hygiene_passed[i]:
            return False
        for m, tol in tolerances.items():
            orig = original_quality.get(m, 0.0)
            if orig <= 0:
                continue  # cannot regress below a non-positive baseline
            if candidate_quality[i].get(m, 0.0) < tol * orig:
                return False
        return True

    accepted = [i for i in range(n) if meets(i)]
    if accepted:
        return max(accepted, key=lambda i: (combined_ratio(i), hygiene_tuples[i]))

    # Fallback: closest to original; prefer hygiene-passing candidates.
    pool = [i for i in range(n) if hygiene_passed[i]] or list(range(n))
    return min(pool, key=lambda i: (total_shortfall(i), -combined_ratio(i)))


def generate_for_bad_indices(
    *,
    bad_indices: List[int],
    bad_objs: List[Optional[Dict]],
    original_preds: Dict[int, str],
    quality_scorer: Optional[CandidateQualityScorer],
    tolerances: Dict[str, float],
    model_short: str,
    tokenizer,
    model,
    batch_size: int,
    max_input_tokens: int,
    min_new_tokens: int,
    max_new_tokens: int,
    num_beams: int,
    num_candidates: int,
    temperature: float,
    top_p: float,
    top_k: int,
    no_repeat_ngram_size: int,
    nlp_parser,
    hygiene_args: Namespace,
) -> Dict[int, str]:
    if not bad_indices:
        return {}

    config = get_model_config(model_short)
    strip_token_type_ids = model_short.startswith("normistral")
    prompt_cfg = config.prompt_config
    max_input = max_input_tokens
    if config.max_input_text_tokens is not None:
        max_input = min(max_input, int(config.max_input_text_tokens))

    predictions: Dict[int, str] = {}
    passed_all_selected = 0
    accepted_within_tolerance = 0
    for idx_batch in batched(bad_indices, batch_size):
        prompts: List[str] = []
        source_objs: List[Dict] = []
        for idx in idx_batch:
            obj = bad_objs[idx]
            assert obj is not None
            source_objs.append(obj)
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
        if strip_token_type_ids:
            encoded.pop("token_type_ids", None)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        effective_candidates = max(1, int(num_candidates))
        effective_num_beams = num_beams
        if effective_num_beams > 1:
            effective_num_beams = max(effective_num_beams, effective_candidates)
        generation_kwargs = {
            **encoded,
            "use_cache": True,
            "do_sample": True,
            "num_beams": effective_num_beams,
            "num_return_sequences": effective_candidates,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.1,
            "temperature": temperature,
            "top_p": top_p,
        }
        if top_k > 0:
            generation_kwargs["top_k"] = top_k
        if no_repeat_ngram_size > 0:
            generation_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

        with torch.no_grad():
            generated = model.generate(**generation_kwargs)

        input_width = encoded["input_ids"].shape[1]
        continuation_ids = extract_generated_continuations(generated, input_width)

        # Decode every candidate and score it for hygiene first.
        per_line_candidates: List[List[str]] = []
        per_line_hyg_tuples: List[List[tuple]] = []
        per_line_hyg_passed: List[List[bool]] = []
        docs_batch: List[str] = []
        refs_batch: List[str] = []
        for j, _line_idx in enumerate(idx_batch):
            obj = source_objs[j]
            doc = str(obj.get("input_text", ""))
            ref = str(obj.get("reference", ""))
            docs_batch.append(doc)
            refs_batch.append(ref)
            cand_texts: List[str] = []
            hyg_tuples: List[tuple] = []
            hyg_passed: List[bool] = []
            start = j * effective_candidates
            stop = start + effective_candidates
            for candidate_ids in continuation_ids[start:stop]:
                decoded = tokenizer.decode(candidate_ids, skip_special_tokens=True)
                candidate = postprocess_generated_summary_text(decoded)
                score, _metrics = score_candidate(
                    doc=doc,
                    prediction=candidate,
                    reference=ref,
                    nlp_parser=nlp_parser,
                    hygiene_args=hygiene_args,
                )
                cand_texts.append(candidate)
                hyg_tuples.append(score)
                hyg_passed.append(bool(score[0]))
            per_line_candidates.append(cand_texts)
            per_line_hyg_tuples.append(hyg_tuples)
            per_line_hyg_passed.append(hyg_passed)

        use_quality = (
            quality_scorer is not None and quality_scorer.enabled and bool(tolerances)
        )

        # Quality-score all candidates (flattened) plus each original prediction.
        cand_quality_flat: List[Dict[str, float]] = []
        orig_quality_list: List[Dict[str, float]] = []
        if use_quality:
            assert quality_scorer is not None
            flat_docs: List[str] = []
            flat_preds: List[str] = []
            flat_refs: List[str] = []
            for j in range(len(idx_batch)):
                for cand in per_line_candidates[j]:
                    flat_docs.append(docs_batch[j])
                    flat_preds.append(cand)
                    flat_refs.append(refs_batch[j])
            cand_quality_flat = quality_scorer.score_batch(flat_docs, flat_preds, flat_refs)

            orig_docs = list(docs_batch)
            orig_preds = [original_preds.get(line_idx, "") for line_idx in idx_batch]
            orig_refs = list(refs_batch)
            orig_quality_list = quality_scorer.score_batch(orig_docs, orig_preds, orig_refs)

        flat_offset = 0
        for j, line_idx in enumerate(idx_batch):
            cand_texts = per_line_candidates[j]
            n_c = len(cand_texts)
            if use_quality:
                cand_quality = cand_quality_flat[flat_offset : flat_offset + n_c]
                original_quality = orig_quality_list[j]
            else:
                cand_quality = [dict() for _ in range(n_c)]
                original_quality = {}
            flat_offset += n_c

            sel = select_candidate_index(
                hygiene_tuples=per_line_hyg_tuples[j],
                hygiene_passed=per_line_hyg_passed[j],
                candidate_quality=cand_quality,
                original_quality=original_quality,
                tolerances=tolerances if use_quality else {},
            )
            if per_line_hyg_passed[j][sel]:
                passed_all_selected += 1
            if use_quality:
                meets_all = per_line_hyg_passed[j][sel] and all(
                    (original_quality.get(m, 0.0) <= 0)
                    or (cand_quality[sel].get(m, 0.0) >= tol * original_quality.get(m, 0.0))
                    for m, tol in tolerances.items()
                )
                if meets_all:
                    accepted_within_tolerance += 1
            predictions[line_idx] = cand_texts[sel]

    if quality_scorer is not None and quality_scorer.enabled and tolerances:
        print(
            f"  quality-gated: accepted_within_tolerance={accepted_within_tolerance}/{len(bad_indices)} "
            f"({100 * accepted_within_tolerance / max(1, len(bad_indices)):.1f}%); "
            f"tolerances={tolerances}"
        )

    print(
        f"  regenerated candidates: selected_passed_all={passed_all_selected}/{len(bad_indices)} "
        f"({100 * passed_all_selected / len(bad_indices):.1f}%)"
    )

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
                source_obj = good_obj
                prediction = good_obj.get("prediction", "")
            else:
                source_obj = bad_objs[i] or {}
                prediction = regenerated_predictions[i]
            out_obj = make_inputs_refs_preds_record(
                input_text=source_obj.get("input_text", ""),
                prompt=source_obj.get("prompt", ""),
                reference=source_obj.get("reference", ""),
                prediction=prediction,
            )
            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")


def load_tokenizer_and_model(model_short: str, checkpoint_dir: Path, hf_token: Optional[str]):
    config = get_model_config(model_short)

    tokenizer = AutoTokenizer.from_pretrained(config.hf_name, token=hf_token, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
    sync_model_tokenizer_special_tokens(model, tokenizer)
    return tokenizer, model


def main() -> int:
    args = parse_args()
    results_folder = Path(args.results_folder).expanduser().resolve()
    model_folder = Path(args.model_folder).expanduser().resolve()
    results_dir = results_folder / "all_eval_results"
    inferred_from_results = (
        results_folder.name[: -len("-apptainer-fsdp")]
        if results_folder.name.endswith("-apptainer-fsdp")
        else None
    )
    inferred_from_model = (
        model_folder.name[: -len("-apptainer-fsdp")]
        if model_folder.name.endswith("-apptainer-fsdp")
        else None
    )
    model_short = args.model[0] if args.model else (inferred_from_results or inferred_from_model)
    effective_batch_size = args.pred_batch_size if args.pred_batch_size is not None else args.batch_size

    if args.num_candidates < 1:
        raise SystemExit("--num_candidates must be >= 1")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be > 0")
    if not (0 < args.top_p <= 1):
        raise SystemExit("--top_p must be in the interval (0, 1]")
    if args.top_k < 0:
        raise SystemExit("--top_k must be >= 0")
    if args.no_repeat_ngram_size < 0:
        raise SystemExit("--no_repeat_ngram_size must be >= 0")
    for name, val in (
        ("--rougeLsum_tolerance", args.rougeLsum_tolerance),
        ("--bertscore_tolerance", args.bertscore_tolerance),
        ("--faithfulness_tolerance", args.faithfulness_tolerance),
    ):
        if not (0.0 <= val <= 1.0):
            raise SystemExit(f"{name} must be in the interval [0, 1]")
    if args.generation_num < 0:
        raise SystemExit("--generation/--gen must be non-negative")

    tolerances: Dict[str, float] = {}
    if args.rougeLsum_tolerance > 0:
        tolerances["rougeLsum"] = float(args.rougeLsum_tolerance)
    if args.bertscore_tolerance > 0:
        tolerances["bertscore_f1"] = float(args.bertscore_tolerance)
    if args.faithfulness_tolerance > 0:
        tolerances["entailment"] = float(args.faithfulness_tolerance)
    quality_scorer = CandidateQualityScorer(
        use_rouge="rougeLsum" in tolerances,
        use_bertscore="bertscore_f1" in tolerances,
        use_faithfulness="entailment" in tolerances,
    )
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.no_lora:
        print("Note: --no_lora is ignored. This script always loads LoRA adapters from checkpoints.")

    if not results_folder.is_dir():
        raise SystemExit(f"results_folder does not exist: {results_folder}")
    if not results_dir.is_dir():
        raise SystemExit(f"results_folder missing all_eval_results: {results_dir}")
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
                generation_num=args.generation_num,
            )
        )
    else:
        targets = list(
            iter_targets(
                results_dir=results_dir,
                examples_suffix=args.examples_suffix,
                major_interval=args.major_interval,
                generation_num=args.generation_num,
            )
        )
    targets.sort(key=lambda item: (item[0], item[2].name))
    
    if not targets:
        print("No matching checkpoints found.")
        return 0

    print(
        f"Found {len(targets)} checkpoint hygiene pairs to process "
        f"for gen{args.generation_num} -> gen{args.generation_num + 1}."
    )
    for step, good_file, bad_file, out_file in targets:
        print(f"  - {model_short} checkpoint-{step}: {good_file.name} + {bad_file.name}")

    # --- Pre-flight: check for existing outputs ---
    existing = [out_file for _, _, _, out_file in targets if out_file.exists()]
    if existing:
        if args.skip_existing:
            print(f"Skipping {len(existing)} targets with existing output (--skip-existing):")
            for p in existing:
                print(f"  {p}")
            targets = [(s, g, b, o) for s, g, b, o in targets if not o.exists()]
            if not targets:
                print("All targets already exist. Nothing to do.")
                return 0
            print(f"Remaining targets to process: {len(targets)}")
        elif not args.overwrite:
            print("ERROR: the following output files already exist:", file=sys.stderr)
            for p in existing:
                print(f"  {p}", file=sys.stderr)
            print(
                "Use --skip-existing to resume, or --overwrite to replace them.",
                file=sys.stderr,
            )
            return 1

    if args.dry_run:
        print("--dry_run enabled; exiting without generation.")
        return 0

    if quality_scorer.enabled:
        print(
            "Quality gates active (candidate accepted only if it passes hygiene AND "
            f"new >= tolerance*original): {tolerances}. "
            "On budget exhaustion the candidate closest to the original scores is kept."
        )
    else:
        print("Quality gates disabled (all tolerances 0); selecting best candidate by hygiene only.")

    lexicon = load_lexicon(args.lexicon) if args.lexicon else None
    nlp_parser = NorwegianLightParser(lexicon=lexicon, model=args.spacy_model)

    for step, good_file, bad_file, out_file in targets:

        print(f"\nProcessing {model_short} checkpoint-{step}")
        good_lines = read_lines(good_file)
        bad_lines = read_lines(bad_file)
        good_objs, bad_objs, bad_indices = validate_and_collect(good_lines, bad_lines, good_file, bad_file)
        print(
            f"  line_count={len(good_lines)}, regenerate={len(bad_indices)}, "
            f"copy_good={len(good_lines) - len(bad_indices)}"
        )

        original_preds: Dict[int, str] = {}
        if quality_scorer.enabled and bad_indices:
            original_preds = load_original_predictions(bad_file, bad_indices)

        checkpoint_dir = resolve_checkpoint_dir(model_folder, step)
        print(f"  using adapter checkpoint: {checkpoint_dir}")
        tokenizer, model = load_tokenizer_and_model(model_short, checkpoint_dir, args.hf_token)

        regenerated = generate_for_bad_indices(
            bad_indices=bad_indices,
            bad_objs=bad_objs,
            original_preds=original_preds,
            quality_scorer=quality_scorer,
            tolerances=tolerances,
            model_short=model_short,
            tokenizer=tokenizer,
            model=model,
            batch_size=effective_batch_size,
            max_input_tokens=args.max_input_tokens,
            min_new_tokens=args.min_new_tokens,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            num_candidates=args.num_candidates,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            nlp_parser=nlp_parser,
            hygiene_args=args,
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
