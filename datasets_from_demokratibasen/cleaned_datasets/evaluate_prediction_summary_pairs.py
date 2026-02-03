"""
Evaluate prediction–reference summary pairs using ROUGE and BERTScore.

Reads:
- A JSONL file with reference summaries (e.g. 149978_text_summary_examples_val.jsonl).
  The reference summary is the "output" field; document id is metadata.dokument_id.
- A folder of JSON files with predictions. Each file is named {dokument_id}.json and contains
  "oppsummering" (the prediction text). The script reads the summary JSONL in order and, for
  each line, loads the matching prediction file; it stops after N pairs when N is given.

Pairs are matched by document id: for each summary line (metadata.dokument_id), the script
loads predictions_folder / "{dokument_id}.json" and uses its "oppsummering" field.

Usage: python evaluate_prediction_summary_pairs.py <summary_jsonl_file> <predictions_folder> [N]
  summary_jsonl_file: path to JSONL with "output" and metadata.dokument_id per line.
  predictions_folder: path to folder containing JSON files named {dokument_id}.json with "oppsummering".
  N: optional; if given, evaluate only the first N matching pairs; if omitted, evaluate all.

Outputs are saved in baseline_metrics/ under the summary file's parent folder:
  Overall: baseline_metrics/evaluation_<pred_summary_stem>_<num_pairs>.json
  (pred_summary_stem = summary file stem with "text_summary" replaced by "pred_summary")
  When N is given, overall file includes "dokument_ids" list; when not limited, it is omitted.
  Per-document: baseline_metrics/pred_ref_summary_evaluations/<dokument_id>-evaluation.json
  where num_pairs is "firstN" (if N was given) or "all".
  The "reference_metrics_failed" field is not included in any output.
"""

import json
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple


# Suppress PyTorch/CUDA pynvml deprecation FutureWarning (must be before torch is imported)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

# Ensure repo root is on path so we can import summarisation_evaluation
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir
while _repo_root != _repo_root.parent and not (_repo_root / "model_fine_tuning_olivia").exists():
    _repo_root = _repo_root.parent
if not (_repo_root / "model_fine_tuning_olivia").exists():
    _repo_root = _script_dir.parents[2]  # fallback: cleaned_datasets -> datasets_from_demokratibasen -> repo
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from model_fine_tuning_olivia.scripts import summarisation_evaluation as se


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


def collect_matching_pairs(
    summary_jsonl_path: Path,
    predictions_folder: Path,
    max_pairs: Optional[int] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Read summary JSONL in order; for each line, load the matching prediction file (doc_id.json).
    Stop when we have max_pairs matches (or at end of file). Returns (dokument_ids, predictions, references).
    """
    dokument_ids: List[str] = []
    predictions: List[str] = []
    references: List[str] = []
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
            predictions.append(pred_text)
            references.append(output_text)
    return dokument_ids, predictions, references


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv or sys.argv
    if len(argv) < 3:
        print("Usage: evaluate_prediction_summary_pairs.py <summary_jsonl_file> <predictions_folder> [N]", file=sys.stderr)
        print("  summary_jsonl_file: JSONL with 'output' and metadata.dokument_id.", file=sys.stderr)
        print("  predictions_folder: folder of JSON files named {dokument_id}.json with 'oppsummering'.", file=sys.stderr)
        print("  N: optional; first N matching pairs only; if omitted, evaluate all.", file=sys.stderr)
        return 1

    summary_file = Path(argv[1]).resolve()
    predictions_folder = Path(argv[2]).resolve()
    first_n: Optional[int] = None
    if len(argv) >= 4:
        try:
            first_n = int(argv[3])
        except ValueError:
            pass

    if not summary_file.exists():
        print(f"Summary file not found: {summary_file}", file=sys.stderr)
        return 1
    if not predictions_folder.is_dir():
        print(f"Predictions folder not found or not a directory: {predictions_folder}", file=sys.stderr)
        return 1

    # Output: baseline_metrics under the *dataset* folder (parent of summary file)
    dataset_dir = summary_file.parent
    out_dir = dataset_dir / "baseline_metrics"
    eval_subdir = out_dir / "pred_ref_summary_evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_subdir.mkdir(parents=True, exist_ok=True)

    print(f"Reading summary file in order and loading matching predictions from {predictions_folder} ...")
    if first_n is not None:
        print(f"  Stopping after {first_n} matching pairs.")
    common_ids, predictions, references = collect_matching_pairs(summary_file, predictions_folder, max_pairs=first_n)
    num_pairs = len(common_ids)
    if num_pairs == 0:
        print("No matching pairs (summary line with existing prediction file).", file=sys.stderr)
        return 1
    print(f"  Collected {num_pairs} prediction–reference pairs.")

    print(f"Evaluating {num_pairs} prediction–reference pairs (ROUGE + BERTScore) per document...")
    start = time.time()
    per_doc_results: List[dict] = []
    for doc_id, pred, ref in zip(common_ids, predictions, references):
        per_doc_result = se.eval_reference([pred], [ref])
        per_doc_results.append(per_doc_result)
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s ({num_pairs / elapsed:.2f} docs/s)")

    # Aggregate overall from per-document results (mean of numeric metrics); drop reference_metrics_failed
    # Separate reference metrics (rouge/bertscore) from other metrics
    n = len(per_doc_results)
    aggregate = {}
    reference_metrics = {}
    reference_keys = {"rouge1", "rouge2", "rougeL", "rougeLsum", "bertscore_f1_mean"}
    if n > 0:
        for key in per_doc_results[0]:
            if key == "reference_metrics_failed":
                continue
            vals = [r.get(key) for r in per_doc_results]
            numeric_vals = [v for v in vals if v is not None and isinstance(v, (int, float))]
            if numeric_vals:
                value = sum(numeric_vals) / len(numeric_vals)
            else:
                value = per_doc_results[0].get(key)
            
            # Move reference metrics under "reference" field
            if key in reference_keys:
                reference_metrics[key] = value
            else:
                aggregate[key] = value

    # Overall file: baseline_metrics/evaluation_<pred_summary_stem>_<num_docs>.json
    num_docs_str = f"first{num_pairs}" if first_n is not None else "all"
    pred_summary_stem = summary_file.stem.replace("text_summary", "pred_summary", 1)
    
    # Make paths relative to project root
    try:
        summary_file_rel = summary_file.relative_to(_repo_root)
        predictions_folder_rel = predictions_folder.relative_to(_repo_root)
    except ValueError:
        # If paths are not under repo root, use absolute paths
        summary_file_rel = str(summary_file)
        predictions_folder_rel = str(predictions_folder)
    
    overall_data = {
        "num_docs": num_pairs,
        "summary_file": str(summary_file_rel),
        "predictions_folder": str(predictions_folder_rel),
        "runtime_seconds": elapsed,
        "docs_per_second": num_pairs / elapsed if elapsed > 0 else 0.0,
        "reference": reference_metrics,
        **aggregate,
    }
    if first_n is not None:
        overall_data["dokument_ids"] = common_ids

    overall_path = out_dir / f"evaluation_{pred_summary_stem}_{num_docs_str}.json"
    with overall_path.open("w", encoding="utf-8") as f:
        json.dump(overall_data, f, ensure_ascii=False, indent=2)
    print(f"Overall results saved to: {overall_path}")

    # Per-document files: baseline_metrics/pred_ref_summary_evaluations/<dokument_id>-evaluation.json
    # Drop reference_metrics_failed from per-doc output and restructure under "reference" field
    print(f"Saving per-document results to: {eval_subdir}")
    reference_keys = {"rouge1", "rouge2", "rougeL", "rougeLsum", "bertscore_f1_mean"}
    for doc_id, per_doc_result in zip(common_ids, per_doc_results):
        # Separate reference metrics from other metrics
        ref_metrics = {k: v for k, v in per_doc_result.items() 
                      if k != "reference_metrics_failed" and k in reference_keys}
        other_metrics = {k: v for k, v in per_doc_result.items() 
                       if k != "reference_metrics_failed" and k not in reference_keys}
        per_doc = {
            "dokument_id": doc_id,
            "reference": ref_metrics,
            **other_metrics,
        }
        per_doc_path = eval_subdir / f"{doc_id}-evaluation.json"
        with per_doc_path.open("w", encoding="utf-8") as f:
            json.dump(per_doc, f, ensure_ascii=False, indent=2)
    print("Per-document evaluation files written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
