"""
Metrics computation for Norwegian summarisation evaluation.

Consolidates all evaluation metrics into one module:
- ROUGE with Norwegian tokenizer (Trainer callback + standalone)
- Hygiene metrics (compression ratio, n-gram repetition, punctuation)
- BERTScore with Norwegian BERT encoder
- extended_evaluate() orchestrator for post-prediction metrics

NLI-based faithfulness lives in utils.faithfulness (heavy, separate concern).
"""

import json
import re
import sys
import time
import unicodedata

import numpy as np
from typing import Tuple, Dict, Optional, Any, List

import evaluate
from transformers import PreTrainedTokenizer, AutoTokenizer

from .rouge_tokenizer import norwegian_tokenize


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_decoded_text(text: str) -> str:
    """Remove special tokens and unwanted characters from decoded text."""
    text = text.replace('[/INST]', '').replace('[INST]', '')
    text = text.replace('</s>', '').replace('<s>', '')
    text = text.replace('\\', '')
    text = ' '.join(text.split())
    return text.strip()


# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------

def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Compute ROUGE scores from text strings.

    Single implementation used by both the Trainer callback and standalone callers.
    Returns dict with rouge1, rouge2, rougeL, rougeLsum as 0–1 floats.
    """
    rouge = evaluate.load("rouge")
    scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=False,
        tokenizer=norwegian_tokenize,
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
    )
    return dict(scores)


def compute_rouge_metrics(
    eval_pred: Tuple[np.ndarray, np.ndarray],
    tokenizer: PreTrainedTokenizer,
    log_to_wandb: bool = False,
    step: Optional[int] = None,
    is_main_process: bool = True,
    verbose: bool = True,
) -> Dict[str, float]:
    """Thin Trainer.evaluate() callback: decode token IDs → compute_rouge().

    Returns dict with rouge1, rouge2, rougeL, rougeLsum as 0–100 percentages.
    """
    import wandb

    preds, labels = eval_pred

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    vocab_size = tokenizer.vocab_size
    preds = np.clip(preds, 0, vocab_size - 1)
    labels = np.clip(labels, 0, vocab_size - 1)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [clean_decoded_text(p).strip() for p in decoded_preds]
    decoded_labels = [clean_decoded_text(l).strip() for l in decoded_labels]

    if len(decoded_preds) > 0 and is_main_process and verbose:
        print(f"\n*** Example 1 ***")
        print(f"Prediction: {decoded_preds[0][:200]}...")
        print(f"Reference:  {decoded_labels[0][:200]}...\n")

    scores = compute_rouge(decoded_preds, decoded_labels)
    result = {k: v * 100 for k, v in scores.items()}

    if is_main_process and verbose:
        print("*** evaluation: computed_metrics ***", result)

    if log_to_wandb and wandb.run is not None and is_main_process:
        wandb.log(
            {
                "eval/rouge1": result["rouge1"],
                "eval/rouge2": result["rouge2"],
                "eval/rougeL": result["rougeL"],
                "eval/rougeLsum": result["rougeLsum"],
            },
            step=step,
        )
    return result


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

_bertscore_metric = None
_bert_tokenizer = None


def _get_bertscore():
    global _bertscore_metric
    if _bertscore_metric is None:
        try:
            _bertscore_metric = evaluate.load("bertscore")
        except Exception as e:
            raise ImportError(
                f"BERTScore could not be loaded. Install with: pip install bert_score. "
                f"Original error: {e}"
            )
    return _bertscore_metric


def _get_bert_tokenizer():
    global _bert_tokenizer
    if _bert_tokenizer is None:
        _bert_tokenizer = AutoTokenizer.from_pretrained("NbAiLab/nb-bert-large")
    return _bert_tokenizer


def _truncate_text_for_bert(text: str, max_tokens: int = 510) -> str:
    """Truncate text to fit within BERT's 512-token limit (510 + [CLS] + [SEP])."""
    tok = _get_bert_tokenizer()
    tokens = tok.encode(text, add_special_tokens=False, max_length=max_tokens, truncation=True)
    return tok.decode(tokens, skip_special_tokens=True)


def compute_bertscore(
    predictions: List[str],
    references: List[str],
) -> Dict[str, Any]:
    """Compute BERTScore using NbAiLab/nb-bert-large (Norwegian encoder).

    Returns dict with bertscore_f1_mean (float) and _timing.
    """
    start = time.time()
    try:
        bertscore = _get_bertscore()
        preds_trunc = [_truncate_text_for_bert(t) for t in predictions]
        refs_trunc = [_truncate_text_for_bert(t) for t in references]

        b = bertscore.compute(
            predictions=preds_trunc,
            references=refs_trunc,
            model_type="NbAiLab/nb-bert-large",
            num_layers=24,
            rescale_with_baseline=False,
        )
        return {
            "bertscore_f1_mean": sum(b["f1"]) / len(b["f1"]),
            "_timing": {"bertscore_seconds": time.time() - start},
        }
    except ImportError as e:
        print(
            f"Warning: BERTScore not available ({e}). Continuing without BERTScore.",
            file=sys.stderr,
        )
        return {"_timing": {"bertscore_seconds": time.time() - start}}


# ---------------------------------------------------------------------------
# Hygiene metrics
# ---------------------------------------------------------------------------

def ngram_repetition(doc: str, n: int = 3) -> float:
    """Proportion of repeated n-grams in *doc*."""
    tokens = re.findall(r"\d+(?:[.,]\d+)?|[\w/-]+|[^\w\s]", doc.lower())
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return (len(ngrams) - len(set(ngrams))) / len(ngrams)


def hygiene(doc: str, pred_summary: str) -> Dict[str, Any]:
    """Per-document hygiene metrics."""
    doc_words = len(re.findall(r"\w+", doc))
    pred_sum_words = len(re.findall(r"\w+", pred_summary))
    summary_stripped = pred_summary.strip()
    ends_with_punct = bool(summary_stripped) and unicodedata.category(summary_stripped[-1]).startswith("P")
    return {
        "pred_summary_words": pred_sum_words,
        "doc_words": doc_words,
        "compression_ratio": (pred_sum_words / doc_words) if doc_words else None,
        "rep_3gram": ngram_repetition(pred_summary, n=3),
        "ends_with_punct": ends_with_punct,
    }


def eval_hygiene(docs: List[str], pred_summaries: List[str]) -> Dict[str, Any]:
    """Aggregate hygiene metrics over a batch."""
    start = time.time()
    rows = [hygiene(d, p) for d, p in zip(docs, pred_summaries)]
    ratios = [h["compression_ratio"] for h in rows if h["compression_ratio"] is not None]
    return {
        "mean_compression_ratio": sum(ratios) / len(ratios) if ratios else None,
        "mean_rep_3gram": sum(h["rep_3gram"] for h in rows) / len(rows),
        "ratio_ends_with_punct": sum(h["ends_with_punct"] for h in rows) / len(rows),
        "_timing": {"hygiene_seconds": time.time() - start},
    }


# ---------------------------------------------------------------------------
# Orchestrator — extended_evaluate
# ---------------------------------------------------------------------------

def extended_evaluate(
    input_texts: List[str],
    prediction_texts: List[str],
    reference_texts: List[str],
    print_output: bool = False,
    include_bertscore: bool = True,
) -> Dict[str, Any]:
    """Compute post-prediction metrics: hygiene + optionally BERTScore.

    ROUGE is NOT computed here — it comes from compute_rouge_metrics() in the
    Trainer callback (or from compute_rouge() for standalone scripts).

    NLI faithfulness is NOT computed here — call
    utils.faithfulness.NLIFaithfulnessGate.eval_faithfulness() separately
    on the appropriate subset.

    Return format (backward-compatible with callers that flatten by category):
        {
            "reference": {"bertscore_f1_mean": ...} or {},
            "hygiene":   {"mean_compression_ratio": ..., ...},
            "_timing":   {timing_fields},
        }
    """
    extended_start = time.time()
    timing: Dict[str, float] = {}

    # --- Hygiene (always) ---
    hygiene_out = eval_hygiene(input_texts, prediction_texts)
    if print_output:
        print("HYGIENE:")
        print(json.dumps(hygiene_out, indent=2, ensure_ascii=False, default=str))
    if "_timing" in hygiene_out:
        timing.update(hygiene_out.pop("_timing"))

    # --- BERTScore (optional) ---
    reference_out: Dict[str, Any] = {}
    if include_bertscore:
        bs = compute_bertscore(prediction_texts, reference_texts)
        if "_timing" in bs:
            timing.update(bs.pop("_timing"))
        reference_out.update(bs)
        if print_output:
            print("BERTSCORE:")
            print(json.dumps(reference_out, indent=2, ensure_ascii=False, default=str))
    else:
        timing["bertscore_seconds"] = 0.0

    timing["extended_metrics_total_seconds"] = time.time() - extended_start

    return {
        "reference": reference_out,
        "hygiene": hygiene_out,
        "_timing": timing,
    }


# ---------------------------------------------------------------------------
# Unified metrics-from-text (for --metrics-only and --update-* modes)
# ---------------------------------------------------------------------------

def compute_metrics_from_texts(
    input_texts: List[str],
    prediction_texts: List[str],
    reference_texts: List[str],
    include_rouge: bool = True,
    include_hygiene: bool = True,
    include_bertscore: bool = False,
    include_faithfulness: bool = False,
    nli_input_texts: Optional[List[str]] = None,
    nli_prediction_texts: Optional[List[str]] = None,
    nli_example_indices: Optional[List[int]] = None,
    faithfulness_details_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute selected metrics from text strings.

    Returns a dict with two keys:
        "metrics" — flat dict whose keys match eval_results conventions
                    (e.g. "eval_rouge1", "eval_hygiene_mean_rep_3gram",
                     "eval_reference_bertscore_f1_mean", "eval_faithfulness")
        "timing"  — per-metric timing information

    When *faithfulness_details_file* and *nli_example_indices* are provided,
    faithfulness evaluation is **incremental**: per-example NLI results are
    loaded from / saved to the details file and only missing examples are
    computed.
    """
    metrics: Dict[str, Any] = {}
    timing: Dict[str, float] = {}
    total_start = time.time()

    if include_rouge:
        t0 = time.time()
        rouge_scores = compute_rouge(prediction_texts, reference_texts)
        timing["rouge_seconds"] = time.time() - t0
        for k, v in rouge_scores.items():
            metrics[f"eval_{k}"] = v * 100  # 0-1 → 0-100 to match Trainer

    if include_hygiene:
        hygiene_out = eval_hygiene(input_texts, prediction_texts)
        if "_timing" in hygiene_out:
            timing.update(hygiene_out.pop("_timing"))
        for k, v in hygiene_out.items():
            metrics[f"eval_hygiene_{k}"] = v

    if include_bertscore:
        bs = compute_bertscore(prediction_texts, reference_texts)
        if "_timing" in bs:
            timing.update(bs.pop("_timing"))
        for k, v in bs.items():
            metrics[f"eval_reference_{k}"] = v

    if include_faithfulness:
        from .faithfulness import NLIFaithfulnessGate

        docs = nli_input_texts if nli_input_texts is not None else input_texts
        preds = nli_prediction_texts if nli_prediction_texts is not None else prediction_texts
        gate = NLIFaithfulnessGate()

        if faithfulness_details_file and nli_example_indices is not None:
            faithfulness_out = gate.eval_faithfulness_incremental(
                docs, preds, nli_example_indices, faithfulness_details_file,
            )
        else:
            faithfulness_out = gate.eval_faithfulness(docs, preds)

        if "_timing" in faithfulness_out:
            timing.update(faithfulness_out.pop("_timing"))
        metrics["eval_faithfulness"] = faithfulness_out

    timing["total_metrics_seconds"] = time.time() - total_start
    return {"metrics": metrics, "timing": timing}
