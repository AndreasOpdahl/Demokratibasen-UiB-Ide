"""
Baseline evaluation of (input, output) text-summary pairs from a JSONL dataset.

Usage: python evaluate_text_summary_pairs.py <input_file> [N]
  input_file: path to JSONL with "input", "output", and metadata.dokument_id per line.
  N: optional; if given, evaluate only the first N documents; if omitted, evaluate all.

Uses hygiene and NLI-based faithfulness metrics from
model_fine_tuning_olivia.scripts.summarisation_evaluation (imported from its
current location). Outputs are saved in the parent folder of the input file: overall metrics to
evaluation_<input_stem>_<num_docs>.json (num_docs is "firstN" or "all") and
per-document results to summarisation_evaluations/<dokument_id>-evaluation.json.
"""

import json
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


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


# Suppress PyTorch/CUDA pynvml deprecation FutureWarning (must be before torch is imported)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

# Ensure repo root is on path so we can import summarisation_evaluation from its current location
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parents[2]  # cleaned_datasets -> datasets_from_demokratibasen -> repo
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Filter must be active before summarisation_evaluation (and thus torch) is imported
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

from model_fine_tuning_olivia.scripts import summarisation_evaluation as se


def main(argv: List[str] | None = None) -> int:
    argv = argv or sys.argv
    if len(argv) < 2:
        print("Usage: evaluate_text_summary_pairs.py <input_file> [N]", file=sys.stderr)
        print("  input_file: path to JSONL (input, output, metadata.dokument_id).", file=sys.stderr)
        print("  N: optional; first N docs only; if omitted, evaluate all.", file=sys.stderr)
        return 1

    data_file = Path(argv[1]).resolve()
    first_n: int | None = None
    if len(argv) >= 3:
        try:
            first_n = int(argv[2])
        except ValueError:
            pass

    # Outputs saved in the parent folder of the input file
    out_dir = data_file.parent
    eval_dir = out_dir / "summarisation_evaluations"

    if not data_file.exists():
        print(f"Data file not found: {data_file}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load first N examples
    doc_ids: List[str] = []
    docs: List[str] = []
    preds: List[str] = []

    with data_file.open("r", encoding="utf-8") as f:
        for line in f:
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
            doc_ids.append(doc_id)
            docs.append(input_text)
            preds.append(output_text)
            if first_n is not None and len(docs) >= first_n:
                break

    actual_n = len(docs)
    if actual_n == 0:
        print("No valid examples loaded.", file=sys.stderr)
        return 1

    print(f"Loaded {actual_n} examples from {data_file}")
    start_time = time.time()

    # Single loop: hygiene + faithfulness per document, with progress (docs/s, % complete, ETA)
    gate = se.NLIFaithfulnessGate()
    hygiene_per_doc: List[Dict[str, Any]] = []
    faithfulness_per_doc: List[Dict[str, Any]] = []
    report_interval = max(1, min(10, actual_n // 20))

    print("Running hygiene and faithfulness evaluation (NLI)...")
    for i, (doc, pred) in enumerate(zip(docs, preds)):
        hygiene_per_doc.append(se.hygiene(doc, pred))
        faithfulness_per_doc.append(gate.score_and_gate(doc, pred))
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

    # Aggregate hygiene overall from per-doc results (same as eval_hygiene)
    nh = len(hygiene_per_doc)
    compression_ratios = [h["compression_ratio"] for h in hygiene_per_doc if h.get("compression_ratio") is not None]
    hygiene_overall: Dict[str, Any] = {
        "mean_compression_ratio": sum(compression_ratios) / len(compression_ratios) if compression_ratios else None,
        "mean_rep_3gram": sum(h["rep_3gram"] for h in hygiene_per_doc) / nh if nh else 0,
        "ratio_ends_with_punct": sum(h["ends_with_punct"] for h in hygiene_per_doc) / nh if nh else 0,
    }

    # Aggregate faithfulness overall from per-doc results (same as eval_faithfulness)
    nf = len(faithfulness_per_doc)
    faithfulness_overall: Dict[str, Any] = {
        "mean_entailment_score": sum(f["faithfulness"]["entailment_mean"] for f in faithfulness_per_doc) / nf if nf else 0,
        "min_entailment_score": min(f["faithfulness"]["entailment_min"] for f in faithfulness_per_doc) if nf else 0,
        "mean_ratio_low_entailment_sentences": sum(f["faithfulness"]["low_entailment_sentences"] for f in faithfulness_per_doc) / nf if nf else 0,
        "max_contradiction_score": max(f["faithfulness"]["contradiction_max"] for f in faithfulness_per_doc) if nf else 0,
        "mean_ratio_high_contradiction_sentences": sum(f["faithfulness"]["high_contradiction_sentences"] for f in faithfulness_per_doc) / nf if nf else 0,
        "mean_ratio_outliers": sum(f["faithfulness"]["outlier_rate"] for f in faithfulness_per_doc) / nf if nf else 0,
        "ratio_passed_documents": sum(1 for f in faithfulness_per_doc if f["passed"]) / nf if nf else 0,
        "reasons_failed": _summarise_reasons_failed([f["reasons"] for f in faithfulness_per_doc if not f["passed"]]),
        "num_premise_sentence_pairs": sum(f["num_premise_sentence_pairs"] for f in faithfulness_per_doc),
    }

    num_pairs = faithfulness_overall.get("num_premise_sentence_pairs", 0)
    # Enrich faithfulness with per-second pair processing rate; keep only faithfulness-scoped stats
    faithfulness_overall["premise_sentence_pairs_per_second"] = num_pairs / elapsed if elapsed > 0 else 0.0
    
    # Make path relative to project root
    try:
        data_file_rel = data_file.relative_to(_repo_root)
    except ValueError:
        # If path is not under repo root, use absolute path
        data_file_rel = str(data_file)
    
    overall = {
        "num_docs": actual_n,
        "text_summary_file": str(data_file_rel),
        "runtime_seconds": elapsed,
        "docs_per_second": actual_n / elapsed if elapsed > 0 else 0.0,
        "hygiene": hygiene_overall,
        "faithfulness": faithfulness_overall,
        "reference": {},  # Will contain rouge/bertscore metrics when reference evaluation is added
    }

    print("\nOVERALL HYGIENE METRICS:")
    for k, v in hygiene_overall.items():
        print(f"  {k}: {v}")
    print("\nOVERALL FAITHFULNESS METRICS:")
    for k, v in faithfulness_overall.items():
        print(f"  {k}: {v}")

    input_stem = data_file.stem
    num_docs_str = f"first{actual_n}" if first_n is not None else "all"
    overall_path = out_dir / f"evaluation_{input_stem}_{num_docs_str}.json"
    with overall_path.open("w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    print(f"\nOverall metrics saved to: {overall_path}")

    print(f"Saving per-document results to: {eval_dir}")
    for doc_id, h_res, f_res in zip(doc_ids, hygiene_per_doc, faithfulness_per_doc):
        per_doc = {
            "dokument_id": doc_id,
            "hygiene": h_res,
            "faithfulness": f_res,
        }
        out_path = eval_dir / f"{doc_id}-evaluation.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(per_doc, f, ensure_ascii=False, indent=2)

    print("Per-document evaluation files written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
