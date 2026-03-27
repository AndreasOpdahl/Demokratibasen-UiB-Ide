"""
NLI-based faithfulness evaluation for summarisation.

For each summary sentence the model checks entailment/contradiction against
overlapping chunks of the source document, using an XNLI model.

Gate fails if:
  - too many sentences have low best-entailment
  - OR any sentence has high contradiction
  - OR mean entailment is below a threshold

Per-example NLI results are persisted to a JSONL *details file* so that
expanding the NLI subset size (e.g. 100 → 200) only requires running NLI
on the *new* examples; previously computed results are reused and the
aggregate is recalculated over the full set.
"""

import json
import os
import re
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Suppress PyTorch/CUDA pynvml deprecation FutureWarning (before torch is imported)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()


# ---------------------------------------------------------------------------
# Sentence / paragraph splitting
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


# ---------------------------------------------------------------------------
# Document chunking
# ---------------------------------------------------------------------------

MAX_NLI_TOKENS = 512
PREMISE_CHUNK_OVERLAP = 96
PREMISE_SENT_MAX_TOKENS = 96
PREMISE_SENT_PART_TOKENS = 72
PREMISE_SENT_PART_OVERLAP = 24


def _build_overlapping_chunks(token_ids: List[int], max_tokens: int, overlap: int) -> List[List[int]]:
    if max_tokens <= 0:
        return []
    if len(token_ids) <= max_tokens:
        return [token_ids]

    step = max(1, max_tokens - overlap)
    chunks: List[List[int]] = []
    for start in range(0, len(token_ids), step):
        end = min(len(token_ids), start + max_tokens)
        chunks.append(token_ids[start:end])
        if end == len(token_ids):
            break
    return chunks


def _split_long_sentence_ids(ids: List[int]) -> List[List[int]]:
    if len(ids) <= PREMISE_SENT_MAX_TOKENS:
        return [ids]
    step = max(1, PREMISE_SENT_PART_TOKENS - PREMISE_SENT_PART_OVERLAP)
    parts: List[List[int]] = []
    for start in range(0, len(ids), step):
        end = min(len(ids), start + PREMISE_SENT_PART_TOKENS)
        parts.append(ids[start:end])
        if end == len(ids):
            break
    return parts


def _build_sentence_aligned_chunks(text: str, tokenizer, max_tokens: int, overlap: int) -> List[List[int]]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    sent_ids: List[List[int]] = []
    for s in sentences:
        ids = tokenizer.encode(s, add_special_tokens=False)
        sent_ids.extend(_split_long_sentence_ids(ids))

    chunks: List[List[int]] = []
    current: List[int] = []

    for ids in sent_ids:
        if not current:
            current = ids[:]
            continue

        if len(current) + len(ids) <= max_tokens:
            current.extend(ids)
        else:
            chunks.append(current)
            if overlap > 0:
                current = current[-overlap:] + ids
            else:
                current = ids[:]

        if len(current) > max_tokens:
            overflow_chunks = _build_overlapping_chunks(current, max_tokens, overlap)
            chunks.extend(overflow_chunks[:-1])
            current = overflow_chunks[-1] if overflow_chunks else []

    if current:
        chunks.append(current)
    return chunks


def _split_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r"\n\s*\n+", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def _build_paragraph_aware_chunks(text: str, tokenizer, max_tokens: int, overlap: int) -> List[List[int]]:
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: List[List[int]] = []
    for paragraph in paragraphs:
        para_text = re.sub(r"\s+", " ", paragraph).strip()
        if not para_text:
            continue

        para_ids = tokenizer.encode(para_text, add_special_tokens=False)
        if len(para_ids) <= max_tokens:
            para_chunks = [para_ids]
        else:
            para_chunks = _build_sentence_aligned_chunks(para_text, tokenizer, max_tokens=max_tokens, overlap=overlap)
            if not para_chunks:
                para_chunks = _build_overlapping_chunks(para_ids, max_tokens=max_tokens, overlap=overlap)

        for chunk_ids in para_chunks:
            if chunks and overlap > 0:
                carry = chunks[-1][-overlap:]
                chunk_ids = carry + chunk_ids
                if len(chunk_ids) > max_tokens:
                    chunk_ids = chunk_ids[:max_tokens]
            chunks.append(chunk_ids)

    return chunks


def chunk_document(
    doc: str,
    tokenizer,
    max_tokens: int = 350,
    overlap: int = PREMISE_CHUNK_OVERLAP,
    prefer_sentence_boundaries: bool = True,
) -> List[str]:
    """Chunk document for NLI premise–hypothesis evaluation."""
    doc = doc.strip()
    if not doc:
        return [""]

    if prefer_sentence_boundaries:
        chunk_ids_list = _build_paragraph_aware_chunks(doc, tokenizer, max_tokens=max_tokens, overlap=overlap)
        if not chunk_ids_list:
            ids = tokenizer.encode(doc, add_special_tokens=False)
            chunk_ids_list = _build_overlapping_chunks(ids, max_tokens=max_tokens, overlap=overlap)
    else:
        ids = tokenizer.encode(doc, add_special_tokens=False)
        chunk_ids_list = _build_overlapping_chunks(ids, max_tokens=max_tokens, overlap=overlap)

    chunks = [tokenizer.decode(chunk_ids, skip_special_tokens=True) for chunk_ids in chunk_ids_list]
    return chunks if chunks else [""]


# ---------------------------------------------------------------------------
# NLI faithfulness scorer + gate
# ---------------------------------------------------------------------------

class NLIFaithfulnessGate:
    """
    NLI-based faithfulness scoring + CI gate.

    For each summary sentence:
      - compute entailment/contradiction probabilities vs each doc chunk
      - take best entailment over chunks (support)
      - take worst contradiction over chunks (conflict risk)

    Gate fails if:
      - too many sentences have low best-entailment
      - OR any sentence has high contradiction
      - OR mean entailment is below a threshold
    """

    def __init__(
        self,
        model_name: str = "joeddav/xlm-roberta-large-xnli",
        device: Optional[str] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        assert torch.cuda.is_available(), "CUDA is not available"
        device = "cuda"
        self.device = torch.device(device)
        self.model.to(self.device)

        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.entailment_idx = self._find_label_idx(id2label, "entail")
        self.contradiction_idx = self._find_label_idx(id2label, "contrad")

    @staticmethod
    def _find_label_idx(id2label: Dict[int, str], needle: str) -> int:
        for idx, lab in id2label.items():
            if needle in lab.lower():
                return idx
        raise ValueError(f"Could not find label containing '{needle}' in id2label: {id2label}")

    @torch.no_grad()
    def _probs(self, premise: str, hypothesis: str) -> torch.Tensor:
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=False,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**inputs).logits[0]
        return torch.softmax(logits, dim=-1)

    def score_and_gate(
        self,
        document: str,
        summary: str,
        doc_chunk_tokens: int = 350,
        batch_size: int = 4,
        entailment_mean_min: float = 0.6,
        entailment_sentence_min: float = 0.50,
        max_low_entailment_sentences: int = 1,
        contradiction_sentence_max: float = 0.50,
        max_high_contradiction_sentences: int = 1,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        sents = split_sentences(summary)
        hyp_ids_list = [self.tokenizer.encode(s, add_special_tokens=False) for s in sents]
        hyp_lengths = [len(ids) for ids in hyp_ids_list]
        max_hyp_len = max(hyp_lengths) if hyp_lengths else 0
        max_chunk_size = MAX_NLI_TOKENS - max_hyp_len - 3
        max_chunk_size = max(413, max_chunk_size)

        doc_chunks = chunk_document(
            document,
            self.tokenizer,
            max_tokens=max_chunk_size,
            overlap=PREMISE_CHUNK_OVERLAP,
            prefer_sentence_boundaries=True,
        )
        chunk_ids_list = [self.tokenizer.encode(c, add_special_tokens=False) for c in doc_chunks]

        per_sentence: List[Dict[str, Any]] = []
        best_entailments: List[float] = [-1.0 for _ in sents]
        worst_contradictions: List[float] = [-1.0 for _ in sents]
        best_ent_chunk_idx: List[int] = [-1 for _ in sents]
        worst_con_chunk_idx: List[int] = [-1 for _ in sents]

        special_tokens = self.tokenizer.num_special_tokens_to_add(pair=True)

        pair_inputs: List[Tuple[int, int, List[int], List[int]]] = []
        for s_idx, hyp_ids in enumerate(hyp_ids_list):
            for c_idx, prem_ids in enumerate(chunk_ids_list):
                prem_ids_use = prem_ids
                hyp_ids_use = hyp_ids
                total_len = len(prem_ids_use) + len(hyp_ids_use) + special_tokens
                if total_len > MAX_NLI_TOKENS:
                    to_remove = total_len - MAX_NLI_TOKENS
                    truncated = self.tokenizer.truncate_sequences(
                        prem_ids_use,
                        hyp_ids_use,
                        num_tokens_to_remove=to_remove,
                        truncation_strategy="longest_first",
                    )
                    if len(truncated) >= 2:
                        prem_ids_use, hyp_ids_use = truncated[0], truncated[1]
                    else:
                        prem_ids_use = truncated[0] if truncated else prem_ids_use
                input_ids = self.tokenizer.build_inputs_with_special_tokens(prem_ids_use, hyp_ids_use)
                token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(prem_ids_use, hyp_ids_use)
                pair_inputs.append((s_idx, c_idx, input_ids, token_type_ids))

        pair_inputs.sort(key=lambda x: len(x[2]), reverse=True)
        num_premise_sentence_pairs = len(pair_inputs)

        start = 0
        while start < len(pair_inputs):
            cur_batch_size = batch_size
            while True:
                batch = pair_inputs[start : start + cur_batch_size]
                if not batch:
                    break
                max_len = max(len(x[2]) for x in batch)
                input_ids_batch = []
                token_type_ids_batch = []
                attention_mask_batch = []
                for _, _, input_ids, token_type_ids in batch:
                    pad_len = max_len - len(input_ids)
                    input_ids_batch.append(input_ids + [self.tokenizer.pad_token_id] * pad_len)
                    token_type_ids_batch.append(token_type_ids + [0] * pad_len)
                    attention_mask_batch.append([1] * len(input_ids) + [0] * pad_len)

                inputs = {
                    "input_ids": torch.tensor(input_ids_batch, device=self.device),
                    "attention_mask": torch.tensor(attention_mask_batch, device=self.device),
                }
                if "token_type_ids" in self.tokenizer.model_input_names:
                    inputs["token_type_ids"] = torch.tensor(token_type_ids_batch, device=self.device)

                try:
                    logits = self.model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1)
                except torch.cuda.OutOfMemoryError:
                    if cur_batch_size <= 1:
                        raise
                    torch.cuda.empty_cache()
                    cur_batch_size = max(1, cur_batch_size // 2)
                    continue

                for i, (s_idx, c_idx, _, _) in enumerate(batch):
                    p_ent = float(probs[i, self.entailment_idx].cpu().item())
                    p_con = float(probs[i, self.contradiction_idx].cpu().item())
                    if p_ent > best_entailments[s_idx]:
                        best_entailments[s_idx] = p_ent
                        best_ent_chunk_idx[s_idx] = c_idx
                    if p_con > worst_contradictions[s_idx]:
                        worst_contradictions[s_idx] = p_con
                        worst_con_chunk_idx[s_idx] = c_idx
                break

            start += cur_batch_size

        for idx, sent in enumerate(sents):
            best_chunk_idx = best_ent_chunk_idx[idx]
            worst_chunk_idx = worst_con_chunk_idx[idx]
            per_sentence.append(
                {
                    "sentence": sent,
                    "best_entailment": best_entailments[idx],
                    "best_entailment_chunk_idx": best_chunk_idx,
                    "best_entailment_premise": doc_chunks[best_chunk_idx] if 0 <= best_chunk_idx < len(doc_chunks) else None,
                    "worst_contradiction": worst_contradictions[idx],
                    "worst_contradiction_chunk_idx": worst_chunk_idx,
                    "worst_contradiction_premise": doc_chunks[worst_chunk_idx] if 0 <= worst_chunk_idx < len(doc_chunks) else None,
                }
            )

        if best_entailments:
            entail_mean = sum(best_entailments) / len(best_entailments)
            entail_min = min(best_entailments)
            low_entail_count = sum(1 for x in best_entailments if x < entailment_sentence_min)
        else:
            entail_mean, entail_min, low_entail_count = 0.0, 0.0, 0

        if worst_contradictions:
            con_max = max(worst_contradictions)
            high_con_count = sum(1 for x in worst_contradictions if x > contradiction_sentence_max)
        else:
            con_max, high_con_count = 0.0, 0

        if len(best_entailments) > 0:
            outlier_count = sum(
                1
                for i in range(len(best_entailments))
                if (
                    best_entailments[i] < entailment_sentence_min
                    or worst_contradictions[i] > contradiction_sentence_max
                )
            )
            outlier_rate = outlier_count / len(best_entailments)
        else:
            outlier_rate = 0.0

        failing_pairs: List[Dict[str, Any]] = []
        for idx, sent in enumerate(sents):
            prem_low = doc_chunks[best_ent_chunk_idx[idx]] if 0 <= best_ent_chunk_idx[idx] < len(doc_chunks) else None
            prem_high = doc_chunks[worst_con_chunk_idx[idx]] if 0 <= worst_con_chunk_idx[idx] < len(doc_chunks) else None
            if best_entailments[idx] < entailment_sentence_min:
                failing_pairs.append(
                    {
                        "premise": prem_low or prem_high,
                        "hypothesis": sent,
                        "entailment": best_entailments[idx],
                        "contradiction": worst_contradictions[idx],
                        "reason": "low_entailment",
                    }
                )
            if worst_contradictions[idx] > contradiction_sentence_max:
                failing_pairs.append(
                    {
                        "premise": prem_high or prem_low,
                        "hypothesis": sent,
                        "entailment": best_entailments[idx],
                        "contradiction": worst_contradictions[idx],
                        "reason": "high_contradiction",
                    }
                )

        reasons = []
        if entail_mean < entailment_mean_min:
            reasons.append(f"low_mean_entailment {entail_mean:.3f} < {entailment_mean_min:.3f}")
        if low_entail_count > max_low_entailment_sentences:
            reasons.append(
                f"low_entailment_sentences {low_entail_count} > {max_low_entailment_sentences} "
                f"(threshold {entailment_sentence_min:.2f})"
            )
        if high_con_count > max_high_contradiction_sentences:
            reasons.append(
                f"high_contradiction_sentences {high_con_count} > {max_high_contradiction_sentences} "
                f"(threshold {contradiction_sentence_max:.2f})"
            )

        passed = len(reasons) == 0

        runtime_seconds = time.perf_counter() - start_time
        premise_sentence_pairs_per_second = num_premise_sentence_pairs / runtime_seconds if runtime_seconds > 0 else 0.0

        return {
            "faithfulness": {
                "entailment_mean": entail_mean,
                "entailment_min": entail_min,
                "low_entailment_sentences": low_entail_count / len(sents) if len(sents) > 0 else 0.0,
                "contradiction_max": con_max,
                "high_contradiction_sentences": high_con_count / len(sents) if len(sents) > 0 else 0.0,
                "outlier_rate": outlier_rate,
                "num_sentences": len(sents),
            },
            "passed": passed,
            "reasons": reasons,
            "per_sentence": per_sentence,
            "failing_pairs": failing_pairs,
            "runtime_seconds": runtime_seconds,
            "num_premise_sentence_pairs": num_premise_sentence_pairs,
            "premise_sentence_pairs_per_second": premise_sentence_pairs_per_second,
        }

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_faithfulness_results(
        per_example_results: List[Dict[str, Any]],
        total_time: Optional[float] = None,
        num_reused: int = 0,
        num_computed: int = 0,
    ) -> Dict[str, Any]:
        """Compute aggregate faithfulness metrics from per-example ``score_and_gate`` results."""
        n = len(per_example_results)
        if n == 0:
            return {"error": "no examples to aggregate"}

        ratio_passed = sum(1 for f in per_example_results if f["passed"]) / n
        reasons_failed = [f["reasons"] for f in per_example_results if not f["passed"]]
        total_pairs = sum(f["num_premise_sentence_pairs"] for f in per_example_results)
        mean_ent = sum(f["faithfulness"]["entailment_mean"] for f in per_example_results) / n
        min_ent = min(f["faithfulness"]["entailment_min"] for f in per_example_results)
        mean_low_ent = sum(f["faithfulness"]["low_entailment_sentences"] for f in per_example_results) / n
        max_con = max(f["faithfulness"]["contradiction_max"] for f in per_example_results)
        mean_high_con = sum(f["faithfulness"]["high_contradiction_sentences"] for f in per_example_results) / n
        mean_outliers = sum(f["faithfulness"]["outlier_rate"] for f in per_example_results) / n

        result: Dict[str, Any] = {
            "mean_entailment_score": mean_ent,
            "min_entailment_score": min_ent,
            "mean_ratio_low_entailment_sentences": mean_low_ent,
            "max_contradiction_score": max_con,
            "mean_ratio_high_contradiction_sentences": mean_high_con,
            "mean_ratio_outliers": mean_outliers,
            "ratio_passed_documents": ratio_passed,
            "reasons_failed": reasons_failed,
            "num_premise_sentence_pairs": total_pairs,
            "nli_subset_size": n,
        }
        if num_reused or num_computed:
            result["num_reused"] = num_reused
            result["num_computed"] = num_computed
        if total_time is not None:
            result["_timing"] = {"nli_faithfulness_seconds": total_time}
        return result

    # ------------------------------------------------------------------
    # Batch evaluation (original non-incremental API — kept for compat)
    # ------------------------------------------------------------------

    def eval_faithfulness(self, docs: List[str], pred_summaries: List[str]) -> Dict[str, Any]:
        """Aggregate NLI faithfulness over a batch of documents."""
        t0 = time.time()
        per_example = [self.score_and_gate(doc, pred) for doc, pred in zip(docs, pred_summaries)]
        return self.aggregate_faithfulness_results(
            per_example,
            total_time=time.time() - t0,
            num_computed=len(per_example),
        )

    # ------------------------------------------------------------------
    # Incremental evaluation with per-example caching
    # ------------------------------------------------------------------

    def eval_faithfulness_incremental(
        self,
        docs: List[str],
        pred_summaries: List[str],
        example_indices: List[int],
        details_file: str,
    ) -> Dict[str, Any]:
        """Incremental NLI faithfulness with per-example result caching.

        Previously computed results (keyed by ``example_index``) are loaded from
        *details_file*.  Only examples absent from the cache are evaluated via
        ``score_and_gate``.  All results (old + new) are then saved back, and
        aggregate metrics are computed over the requested subset.

        Returns the same dict shape as ``eval_faithfulness`` so callers need not
        distinguish between the two APIs.
        """
        t0 = time.time()
        cached = load_faithfulness_details(details_file)

        to_compute: List[int] = []  # positions in docs/pred_summaries
        for pos, idx in enumerate(example_indices):
            if idx not in cached:
                to_compute.append(pos)

        num_reused = len(example_indices) - len(to_compute)
        num_computed = len(to_compute)

        if to_compute:
            print(f"  NLI faithfulness: {num_reused} cached, {num_computed} to compute …")
            for pos in to_compute:
                idx = example_indices[pos]
                result = self.score_and_gate(docs[pos], pred_summaries[pos])
                cached[idx] = {"example_index": idx, **result}
            save_faithfulness_details(details_file, cached)
        else:
            print(f"  NLI faithfulness: all {num_reused} examples cached — skipping NLI inference")

        ordered = [cached[idx] for idx in example_indices if idx in cached]
        return self.aggregate_faithfulness_results(
            ordered,
            total_time=time.time() - t0,
            num_reused=num_reused,
            num_computed=num_computed,
        )


# ---------------------------------------------------------------------------
# Faithfulness detail file I/O  (module-level so they can be imported directly)
# ---------------------------------------------------------------------------

def load_faithfulness_details(details_file: str) -> Dict[int, Dict[str, Any]]:
    """Load per-example NLI results from a JSONL file, keyed by ``example_index``."""
    results: Dict[int, Dict[str, Any]] = {}
    if not os.path.exists(details_file):
        return results
    try:
        with open(details_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                results[entry["example_index"]] = entry
    except (json.JSONDecodeError, IOError, KeyError) as exc:
        warnings.warn(f"Could not fully load {details_file}: {exc}")
    return results


def save_faithfulness_details(details_file: str, results: Dict[int, Dict[str, Any]]) -> None:
    """Save per-example NLI results to a JSONL file (sorted by ``example_index``)."""
    os.makedirs(os.path.dirname(details_file), exist_ok=True)
    with open(details_file, "w", encoding="utf-8") as f:
        for idx in sorted(results.keys()):
            f.write(json.dumps(results[idx], ensure_ascii=False) + "\n")
