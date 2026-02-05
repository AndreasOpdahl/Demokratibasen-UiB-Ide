"""
Evaluate text-summary pairs (and optionally prediction-summary pairs) using hygiene, faithfulness,
and reference-based metrics (ROUGE, BERTScore).

Reads:
- A JSONL file with reference summaries (e.g. 149978_text_summary_examples_val.jsonl).
  Each line has "input" (document), "output" (reference summary), and metadata.dokument_id.
- Optionally: a folder of JSON files with predictions. Each file is named {dokument_id}.json
  and contains "oppsummering" (the prediction text). When provided, reference metrics
  (pred vs ref) are computed and hygiene/faithfulness use the prediction as summary.

Usage:
  # Text-summary only (hygiene + faithfulness on reference summaries):
  python evaluate_texts_summaries_predictions.py <summary_jsonl> [N]

  # With predictions (reference metrics + hygiene + faithfulness on predictions):
  python evaluate_texts_summaries_predictions.py <summary_jsonl> <predictions_folder> [N]

Outputs are saved in baseline_metrics/ under the summary file's parent folder:
  Per-document: baseline_metrics/summarisation_evaluations/<dokument_id>-evaluation.json
    Top-level fields: dokument_id, reference_metrics, hygiene_metrics, faithfulness_metrics
  Overall: baseline_metrics/evaluation_<stem>_<num_docs>.json
    stem = summary file stem (e.g. 149978_text_summary_examples_test)
"""

import json
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _summarise_reasons_failed(reasons_failed: List[List[str]]) -> str:
    """Summarise per-doc failure reasons into a short string (no full list)."""
    high_con_docs = 0
    low_ent_sent_docs = 0
    low_mean_docs = 0
    thresh_high = thresh_low_sent = thresh_low_mean = None
    for doc_reasons in reasons_failed:
        has_high = has_low_sent = has_low_mean = False
        for r in doc_reasons:
            if "high_contradiction_sentences" in r:
                has_high = True
                if thresh_high is None:
                    m = re.search(r"threshold\s+([0-9.]+)", r)
                    if m:
                        thresh_high = m.group(1)
            if "low_entailment_sentences" in r:
                has_low_sent = True
                if thresh_low_sent is None:
                    m = re.search(r"threshold\s+([0-9.]+)", r)
                    if m:
                        thresh_low_sent = m.group(1)
            if "low_mean_entailment" in r:
                has_low_mean = True
                if thresh_low_mean is None:
                    m = re.search(r"<\s*([0-9.]+)", r)
                    if m:
                        thresh_low_mean = m.group(1)
        if has_high:
            high_con_docs += 1
        if has_low_sent:
            low_ent_sent_docs += 1
        if has_low_mean:
            low_mean_docs += 1
    parts = []
    if high_con_docs:
        t = thresh_high if thresh_high is not None else "?"
        parts.append(f"{high_con_docs} docs with # high_contradiction_sentences > 0 (threshold {t})")
    if low_ent_sent_docs:
        t = thresh_low_sent if thresh_low_sent is not None else "?"
        parts.append(f"{low_ent_sent_docs} docs with # low_entailment_sentences > 0 (threshold {t})")
    if low_mean_docs:
        t = thresh_low_mean if thresh_low_mean is not None else "?"
        parts.append(f"{low_mean_docs} docs with low_mean_entailment (threshold {t})")
    return ", ".join(parts) if parts else "0 docs failed"


def _read_prediction_file(prediction_path: Path) -> Optional[str]:
    """Load a single prediction JSON file. Returns oppsummering text or None."""
    try:
        with prediction_path.open("r", encoding="utf-8") as f:
            record = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(record, dict):
        return None
    oppsummering = record.get("oppsummering")
    return oppsummering if isinstance(oppsummering, str) else ("" if oppsummering is None else str(oppsummering))


def _load_text_summary_pairs(
    summary_jsonl_path: Path,
    max_pairs: Optional[int] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Load (dokument_ids, documents, reference_summaries) from JSONL."""
    dokument_ids: List[str] = []
    docs: List[str] = []
    refs: List[str] = []
    with summary_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_pairs is not None and len(dokument_ids) >= max_pairs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            metadata = obj.get("metadata") or {}
            doc_id = metadata.get("dokument_id")
            input_text = obj.get("input")
            output_text = obj.get("output")
            if not isinstance(doc_id, str) or not isinstance(input_text, str) or not isinstance(output_text, str):
                continue
            dokument_ids.append(doc_id)
            docs.append(input_text)
            refs.append(output_text)
    return dokument_ids, docs, refs


def _load_prediction_pairs(
    summary_jsonl_path: Path,
    predictions_folder: Path,
    max_pairs: Optional[int] = None,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Load (dokument_ids, docs, predictions, references) for documents that have prediction files."""
    dokument_ids: List[str] = []
    docs: List[str] = []
    preds: List[str] = []
    refs: List[str] = []
    with summary_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_pairs is not None and len(dokument_ids) >= max_pairs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            metadata = obj.get("metadata") or {}
            doc_id = metadata.get("dokument_id")
            input_text = obj.get("input")
            output_text = obj.get("output")
            if not isinstance(doc_id, str) or not isinstance(output_text, str):
                continue
            prediction_path = predictions_folder / f"{doc_id}.json"
            if not prediction_path.exists():
                continue
            pred_text = _read_prediction_file(prediction_path)
            if pred_text is None:
                continue
            dokument_ids.append(doc_id)
            docs.append(input_text if isinstance(input_text, str) else "")
            preds.append(pred_text)
            refs.append(output_text)
    return dokument_ids, docs, preds, refs


# Suppress PyTorch/CUDA pynvml deprecation FutureWarning (must be before torch is imported)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir
while _repo_root != _repo_root.parent and not (_repo_root / "model_fine_tuning_olivia").exists():
    _repo_root = _repo_root.parent
if not (_repo_root / "model_fine_tuning_olivia").exists():
    _repo_root = _script_dir.parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from model_fine_tuning_olivia.scripts import summarisation_evaluation as se


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv or sys.argv
    if len(argv) < 2:
        print("Usage: evaluate_texts_summaries_predictions.py <summary_jsonl> [predictions_folder] [N]", file=sys.stderr)
        print("  summary_jsonl: JSONL with input, output, metadata.dokument_id.", file=sys.stderr)
        print("  predictions_folder: optional; folder of {dokument_id}.json with 'oppsummering'.", file=sys.stderr)
        print("  N: optional; first N documents only; if omitted, evaluate all.", file=sys.stderr)
        return 1

    summary_file = Path(argv[1]).resolve()
    predictions_folder: Optional[Path] = None
    first_n: Optional[int] = None
    if len(argv) >= 3:
        arg2 = Path(argv[2])
        if arg2.is_dir():
            predictions_folder = arg2.resolve()
            if len(argv) >= 4:
                try:
                    first_n = int(argv[3])
                except ValueError:
                    pass
        else:
            try:
                first_n = int(argv[2])
            except ValueError:
                pass

    if not summary_file.exists():
        print(f"Summary file not found: {summary_file}", file=sys.stderr)
        return 1
    if predictions_folder is not None and not predictions_folder.is_dir():
        print(f"Predictions folder not found or not a directory: {predictions_folder}", file=sys.stderr)
        return 1

    dataset_dir = summary_file.parent
    out_dir = dataset_dir / "baseline_metrics"
    eval_dir = out_dir / "summarisation_evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    with_predictions = predictions_folder is not None
    if with_predictions:
        print(f"Loading text-summary pairs and matching predictions from {predictions_folder} ...")
        dokument_ids, docs, summaries, refs = _load_prediction_pairs(summary_file, predictions_folder, max_pairs=first_n)
        # summaries = predictions in this mode
    else:
        print(f"Loading text-summary pairs from {summary_file} ...")
        dokument_ids, docs, refs = _load_text_summary_pairs(summary_file, max_pairs=first_n)
        summaries = refs  # use reference as "summary" for hygiene/faithfulness

    actual_n = len(dokument_ids)
    if actual_n == 0:
        print("No valid examples loaded.", file=sys.stderr)
        return 1

    print(f"Loaded {actual_n} examples.")
    if first_n is not None:
        print(f"  (Limited to first {first_n})")
    start_time = time.time()

    # Reference metrics (only when predictions provided)
    reference_per_doc: List[Dict[str, Any]] = []
    if with_predictions:
        print(f"Evaluating {actual_n} prediction-reference pairs (ROUGE + BERTScore)...")
        ref_start = time.time()
        for pred, ref in zip(summaries, refs):
            per_doc = se.eval_reference([pred], [ref])
            reference_per_doc.append({k: v for k, v in per_doc.items() if k != "reference_metrics_failed"})
        print(f"  Done in {time.time() - ref_start:.1f}s")

    # Hygiene + faithfulness per document
    gate = se.NLIFaithfulnessGate()
    hygiene_per_doc: List[Dict[str, Any]] = []
    faithfulness_per_doc: List[Dict[str, Any]] = []
    report_interval = max(1, min(10, actual_n // 20))

    print("Running hygiene and faithfulness evaluation (NLI)...")
    for i, (doc, summ) in enumerate(zip(docs, summaries)):
        hygiene_per_doc.append(se.hygiene(doc, summ))
        faithfulness_per_doc.append(gate.score_and_gate(doc, summ))
        done = i + 1
        if done % report_interval == 0 or done == actual_n:
            elapsed = time.time() - start_time
            docs_per_sec = done / elapsed if elapsed > 0 else 0.0
            pct = 100.0 * done / actual_n
            remaining_sec = (actual_n - done) / docs_per_sec if docs_per_sec > 0 else 0
            eta_sec = time.time() + remaining_sec
            eta_str = datetime.fromtimestamp(eta_sec, tz=timezone.utc).strftime("%H:%M:%S UTC") if remaining_sec < 86400 else f"{remaining_sec / 3600:.1f}h"
            print(f"  [{done}/{actual_n}] {pct:.1f}% | elapsed {elapsed:.0f}s | {docs_per_sec:.2f} docs/s | ETA {remaining_sec:.0f}s (~{eta_str})", flush=True)

    elapsed = time.time() - start_time
    print(f"Evaluation runtime for {actual_n} docs: {elapsed:.1f}s ({actual_n / elapsed:.2f} docs/s)")

    # Aggregate hygiene
    nh = len(hygiene_per_doc)
    compression_ratios = [h["compression_ratio"] for h in hygiene_per_doc if h.get("compression_ratio") is not None]
    hygiene_overall: Dict[str, Any] = {
        "mean_compression_ratio": sum(compression_ratios) / len(compression_ratios) if compression_ratios else None,
        "mean_rep_3gram": sum(h["rep_3gram"] for h in hygiene_per_doc) / nh if nh else 0,
        "ratio_ends_with_punct": sum(h["ends_with_punct"] for h in hygiene_per_doc) / nh if nh else 0,
    }

    # Aggregate faithfulness
    nf = len(faithfulness_per_doc)
    num_premise_pairs = sum(f.get("num_premise_sentence_pairs", 0) for f in faithfulness_per_doc)
    faithfulness_overall: Dict[str, Any] = {
        "mean_entailment_score": sum(f["faithfulness"]["entailment_mean"] for f in faithfulness_per_doc) / nf if nf else 0,
        "min_entailment_score": min(f["faithfulness"]["entailment_min"] for f in faithfulness_per_doc) if nf else 0,
        "mean_ratio_low_entailment_sentences": sum(f["faithfulness"]["low_entailment_sentences"] for f in faithfulness_per_doc) / nf if nf else 0,
        "max_contradiction_score": max(f["faithfulness"]["contradiction_max"] for f in faithfulness_per_doc) if nf else 0,
        "mean_ratio_high_contradiction_sentences": sum(f["faithfulness"]["high_contradiction_sentences"] for f in faithfulness_per_doc) / nf if nf else 0,
        "mean_ratio_outliers": sum(f["faithfulness"]["outlier_rate"] for f in faithfulness_per_doc) / nf if nf else 0,
        "ratio_passed_documents": sum(1 for f in faithfulness_per_doc if f["passed"]) / nf if nf else 0,
        "reasons_failed": _summarise_reasons_failed([f["reasons"] for f in faithfulness_per_doc if not f["passed"]]),
        "num_premise_sentence_pairs": num_premise_pairs,
        "premise_sentence_pairs_per_second": num_premise_pairs / elapsed if elapsed > 0 else 0.0,
    }

    # Aggregate reference metrics (when with predictions)
    reference_overall: Dict[str, Any] = {}
    if with_predictions and reference_per_doc:
        ref_keys = {"rouge1", "rouge2", "rougeL", "rougeLsum", "bertscore_f1_mean"}
        for key in ref_keys:
            vals = [r.get(key) for r in reference_per_doc if r.get(key) is not None]
            if vals:
                reference_overall[key] = sum(vals) / len(vals)

    # Make paths relative to repo root
    try:
        summary_file_rel = summary_file.relative_to(_repo_root)
    except ValueError:
        summary_file_rel = str(summary_file)
    try:
        predictions_folder_rel = predictions_folder.relative_to(_repo_root) if predictions_folder else None
    except ValueError:
        predictions_folder_rel = str(predictions_folder) if predictions_folder else None

    # Overall output
    num_docs_str = f"first{actual_n}" if first_n is not None else "all"
    input_stem = summary_file.stem

    overall = {
        "num_docs": actual_n,
        "summary_file": str(summary_file_rel),
        "runtime_seconds": elapsed,
        "docs_per_second": actual_n / elapsed if elapsed > 0 else 0.0,
        "reference_metrics": reference_overall,
        "hygiene_metrics": hygiene_overall,
        "faithfulness_metrics": faithfulness_overall,
    }
    if with_predictions and predictions_folder_rel:
        overall["predictions_folder"] = str(predictions_folder_rel)
    if first_n is not None:
        overall["dokument_ids"] = dokument_ids

    print("\nOVERALL REFERENCE METRICS:")
    for k, v in reference_overall.items():
        print(f"  {k}: {v}")
    if not reference_overall:
        print("  (none - run with predictions_folder for ROUGE/BERTScore)")
    print("\nOVERALL HYGIENE METRICS:")
    for k, v in hygiene_overall.items():
        print(f"  {k}: {v}")
    print("\nOVERALL FAITHFULNESS METRICS:")
    for k, v in faithfulness_overall.items():
        print(f"  {k}: {v}")

    overall_path = out_dir / f"evaluation_{input_stem}_{num_docs_str}.json"
    with overall_path.open("w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    print(f"\nOverall metrics saved to: {overall_path}")

    # Per-document output: dokument_id, reference_metrics, hygiene_metrics, faithfulness_metrics
    print(f"Saving per-document results to: {eval_dir}")
    ref_keys = {"rouge1", "rouge2", "rougeL", "rougeLsum", "bertscore_f1_mean"}
    for i, doc_id in enumerate(dokument_ids):
        ref_metrics: Dict[str, Any] = {}
        if with_predictions and i < len(reference_per_doc):
            ref_metrics = {k: v for k, v in reference_per_doc[i].items() if k in ref_keys}

        per_doc = {
            "dokument_id": doc_id,
            "reference_metrics": ref_metrics,
            "hygiene_metrics": hygiene_per_doc[i],
            "faithfulness_metrics": faithfulness_per_doc[i],
        }
        out_path = eval_dir / f"{doc_id}-evaluation.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(per_doc, f, ensure_ascii=False, indent=2, default=str)

    print("Per-document evaluation files written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
