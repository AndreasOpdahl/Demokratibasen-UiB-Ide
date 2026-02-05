"""
Metrics for in-training validation and final evaluation of summarisation models for 
Norwegian public documents, assuming that LLM-generated reference summaries are available.

TODO: Rename to "summarisation_evaluation.py", which is more descriptive and
      contrasts the "capability_retention_evaluation.py" script.

TODO: Do not run NLI-based faithfulness metrics on the whole validation set for every checkpoint. 
      Either: 
      - Run only on a subset of the validation set for every checkpoint.
      - Run on the full validation set only for every N checkpoints.
      The full set can be used for final model evaluation, and a subset for validation to monitor training progress.

Types of metrics:
- Reference-based metrics (weak signals, used for monitoring)
    - ROUGE without lemmatisation (mean Lsum)
    - BERTScore with Norwegian encoder (f1_mean)
- Hygiene metrics (strong signals, used for alarm)
    - Mean repetition of n-grams (n=3)
    - Mean compression ratio
    - Ratio of endings with punctuation (to avoid truncation)
- NLI-based faithfulness metrics (strong signals, used for alarm)
    - Mean entailment score (NLI entailment aggregate mean)
    - Mean contradiction score (NLI contradiction aggregatemean)
    - Mean outlier rate (NLI outlier mean - either low entailment or high contradiction)
- TODO: QA-based faithfulness metrics (strong signals, used for alarm)
- TODO: Language-quality based metrics (weak signals)

Suggested “dashboard” fields:
- faithfulness_mean + % below threshold (every N*M steps, primary gate):
  check on a small fixed subset (e.g., 50–100 docs)
- rougeLsum, bertscore_f1_mean (every N steps, secondary monitors)
- rep_3gram, compression_ratio statistics (every N steps, tertiary alarms)
"""

import itertools
import json
import os
import re

from transformers import AutoTokenizer


# the checkpoint data to evaluate   
DATA_DIR = "small_eval_results/"  # test_eval_results/ contains a larger dataset
FILE_MASK = r"^checkpoint-(\d+)-inputs-refs-preds.jsonl$"

# the tokenizer to use
MODEL = "RuterNorway/Llama-2-13b-chat-norwegian"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL)


# Reference-based

import evaluate

rouge = evaluate.load("rouge")

# Load BERTScore lazily - it requires bert_score package which may not be installed
_bertscore = None
def _get_bertscore():
    """Lazy load BERTScore metric."""
    global _bertscore
    if _bertscore is None:
        try:
            _bertscore = evaluate.load("bertscore")
        except Exception as e:
            raise ImportError(
                f"BERTScore could not be loaded. Install with: pip install bert_score. "
                f"Original error: {e}"
            )
    return _bertscore

# Load BERT tokenizer for truncation (same model as used in BERTScore)
_bert_tokenizer = None

def _get_bert_tokenizer():
    """Lazy load BERT tokenizer for truncation."""
    global _bert_tokenizer
    if _bert_tokenizer is None:
        _bert_tokenizer = AutoTokenizer.from_pretrained("NbAiLab/nb-bert-large")
    return _bert_tokenizer

def _truncate_text_for_bert(text, max_tokens=510):
    """Truncate text to fit within BERT's max sequence length (512 tokens).
    
    Uses 510 to leave room for [CLS] and [SEP] tokens.
    """
    tokenizer = _get_bert_tokenizer()
    # Tokenize and truncate
    tokens = tokenizer.encode(text, add_special_tokens=False, max_length=max_tokens, truncation=True)
    # Decode back to text
    return tokenizer.decode(tokens, skip_special_tokens=True)

def eval_reference(pred_summaries, ref_summaries, include_bertscore=True):
    """Compute reference-based metrics (ROUGE + optionally BERTScore).
    
    Args:
        pred_summaries: List of predicted summaries
        ref_summaries: List of reference summaries
        include_bertscore: If True, compute BERTScore (slower, ~1.5-2 min for 500 examples)
    
    Returns:
        Dictionary with ROUGE metrics and optionally BERTScore
    """
    r = rouge.compute(
        predictions=pred_summaries,
        references=ref_summaries,
        use_stemmer=False,  # avoid for Norwegian
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
    )
    
    result = {**r}
    
    if include_bertscore:
        try:
            bertscore = _get_bertscore()
            # Truncate texts to avoid BERT max length errors (512 tokens)
            # BERTScore will handle this internally, but we need to ensure texts aren't too long
            pred_summaries_truncated = [_truncate_text_for_bert(text) for text in pred_summaries]
            ref_summaries_truncated = [_truncate_text_for_bert(text) for text in ref_summaries]
            
            b = bertscore.compute(
                predictions=pred_summaries_truncated,
                references=ref_summaries_truncated,
                model_type="NbAiLab/nb-bert-large",  # Norwegian encoder
                num_layers=24,  # BERT-large has 24 layers (must specify manually as model not in BERTScore registry)
                rescale_with_baseline=False  # Baseline not available for this model
            )
            result["bertscore_f1_mean"] = sum(b["f1"]) / len(b["f1"])
        except ImportError as e:
            # BERTScore not available - skip it but don't fail the whole evaluation
            print(f"Warning: BERTScore not available: {e}")
            print("Continuing without BERTScore...")
    
    return result
    
# TODO: also bleurt

    
# Hygiene metrics
    
from collections import Counter

def ngram_repetition(doc, n=3):
    tokens = re.findall(r"\w+|[^\w\s]", doc.lower())
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    c = Counter(ngrams)
    total = sum(c.values())
    repeated = sum(v for v in c.values() if v > 1)
    return repeated / total if total else 0.0

def hygiene(doc, pred_summary):
    doc_words = len(re.findall(r"\w+", doc))
    pred_sum_words = len(re.findall(r"\w+", pred_summary))
    return {
        "pred_summary_words": pred_sum_words,
        "doc_words": doc_words,
        "compression_ratio": (pred_sum_words / doc_words) if doc_words else None,
        "rep_3gram": ngram_repetition(pred_summary, n=3),
        "ends_with_punct": bool(re.search(r"[.!?]\s*$", pred_summary.strip())),
    }
    
def eval_hygiene(docs, pred_summaries):
    hygiene_out = []
    for doc, pred_summary in zip(docs, pred_summaries):
        hygiene_out.append(hygiene(doc, pred_summary))
    # return mean compression ratio and mean repetition of n-grams (n=3)
    # and ratio of endings with punctuation
    # Filter out None values for compression_ratio (in case doc_words is 0)
    compression_ratios = [h["compression_ratio"] for h in hygiene_out if h["compression_ratio"] is not None]
    mean_compression_ratio = sum(compression_ratios) / len(compression_ratios) if compression_ratios else None
    mean_rep_3gram = sum(h["rep_3gram"] for h in hygiene_out) / len(hygiene_out)
    ratio_ends_with_punct = sum(h["ends_with_punct"] for h in hygiene_out) / len(hygiene_out)
    return {
        "mean_compression_ratio": mean_compression_ratio,
        "mean_rep_3gram": mean_rep_3gram,
        "ratio_ends_with_punct": ratio_ends_with_punct,
    }

    
# NLI-based faithfulness
from typing import List, Dict, Any, Optional, Tuple
import time
import warnings

# Suppress PyTorch/CUDA pynvml deprecation FutureWarning (before torch is imported)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.utils import logging as hf_logging

# Suppress "overflowing tokens are not returned..." and similar tokenizer warnings
hf_logging.set_verbosity_error()


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


# Improved chunking constants from summarisation_evaluation.py
MAX_NLI_TOKENS = 512
PREMISE_CHUNK_OVERLAP = 96
PREMISE_SENT_MAX_TOKENS = 96
PREMISE_SENT_PART_TOKENS = 72
PREMISE_SENT_PART_OVERLAP = 24

def _build_overlapping_chunks(token_ids: List[int], max_tokens: int, overlap: int) -> List[List[int]]:
    """Build overlapping chunks in a single forward pass."""
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
    """Split over-long sentence token IDs into overlapping parts."""
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
    """Chunk by sentences while ensuring at least overlap tokens between chunks."""
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
            # Carry over last overlap tokens as a minimum requirement
            if overlap > 0:
                current = current[-overlap:] + ids
            else:
                current = ids[:]

        if len(current) > max_tokens:
            # If a single sentence is too long, split it with overlap
            overflow_chunks = _build_overlapping_chunks(current, max_tokens, overlap)
            chunks.extend(overflow_chunks[:-1])
            current = overflow_chunks[-1] if overflow_chunks else []

    if current:
        chunks.append(current)

    return chunks

def _split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs based on blank lines."""
    paragraphs = re.split(r"\n\s*\n+", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]

def _build_paragraph_aware_chunks(text: str, tokenizer, max_tokens: int, overlap: int) -> List[List[int]]:
    """Chunk by paragraphs, using sentence-aware chunking inside long paragraphs."""
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: List[List[int]] = []
    for paragraph in paragraphs:
        # Normalize whitespace inside paragraph for tokenization stability
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
    """Improved chunking function from summarisation_evaluation.py."""
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
        # if device is None:
        #     device = "cuda" if torch.cuda.is_available() else "cpu"
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
            truncation=False,  # Changed from True - we handle truncation in batch processing
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
        # --- Suggested starting thresholds for CI (tune with a small human-audited set) ---
        entailment_mean_min: float = 0.6,  # Lowered from 0.72 to match summarisation_evaluation.py
        entailment_sentence_min: float = 0.50,  # Lowered from 0.60 to match summarisation_evaluation.py
        max_low_entailment_sentences: int = 1,  # Changed from 0 to match summarisation_evaluation.py
        contradiction_sentence_max: float = 0.50,  # Raised from 0.35 to match summarisation_evaluation.py
        max_high_contradiction_sentences: int = 1,  # Changed from 0 to match summarisation_evaluation.py
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

        # Build all pair inputs once (tokenized), then batch
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

        # Sort by sequence length to reduce padding overhead
        pair_inputs.sort(key=lambda x: len(x[2]), reverse=True)
        num_premise_sentence_pairs = len(pair_inputs)

        start = 0
        while start < len(pair_inputs):
            cur_batch_size = batch_size
            while True:
                batch = pair_inputs[start:start + cur_batch_size]
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
                    "worst_contradiction": worst_contradictions[idx],
                    "worst_contradiction_chunk_idx": worst_chunk_idx,
                }
            )

        # Aggregates
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
        
        # Calculate outlier rate: proportion of sentences that fail at least one threshold
        # (low entailment OR high contradiction)
        if len(best_entailments) > 0:
            outlier_count = sum(
                1 for i in range(len(best_entailments))
                if (best_entailments[i] < entailment_sentence_min or 
                    worst_contradictions[i] > contradiction_sentence_max)
            )
            outlier_rate = outlier_count / len(best_entailments)
        else:
            outlier_rate = 0.0

        # Failing pairs: premise-hypothesis pairs that cause failure (low entailment or high contradiction)
        failing_pairs: List[Dict[str, Any]] = []
        for idx, sent in enumerate(sents):
            prem_low = doc_chunks[best_ent_chunk_idx[idx]] if 0 <= best_ent_chunk_idx[idx] < len(doc_chunks) else None
            prem_high = doc_chunks[worst_con_chunk_idx[idx]] if 0 <= worst_con_chunk_idx[idx] < len(doc_chunks) else None
            if best_entailments[idx] < entailment_sentence_min:
                failing_pairs.append({
                    "premise": prem_low or prem_high,
                    "hypothesis": sent,
                    "entailment": best_entailments[idx],
                    "contradiction": worst_contradictions[idx],
                    "reason": "low_entailment",
                })
            if worst_contradictions[idx] > contradiction_sentence_max:
                failing_pairs.append({
                    "premise": prem_high or prem_low,
                    "hypothesis": sent,
                    "entailment": best_entailments[idx],
                    "contradiction": worst_contradictions[idx],
                    "reason": "high_contradiction",
                })

        # Gate logic
        reasons = []
        if entail_mean < entailment_mean_min:
            reasons.append(
                f"low_mean_entailment {entail_mean:.3f} < {entailment_mean_min:.3f}"
            )
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

        passed = (len(reasons) == 0)

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
        
    def eval_faithfulness(self, docs, pred_summaries):
        faithfulness_out = []
        for doc, pred_summary in zip(docs, pred_summaries):
            faithfulness_out.append(self.score_and_gate(doc, pred_summary))
        # return ratio of that passed the gate and list of reasons for failure
        ratio_passed_documents = sum(1 for f in faithfulness_out if f["passed"]) / len(faithfulness_out)
        reasons_failed = [f["reasons"] for f in faithfulness_out if not f["passed"]]
        num_premise_sentence_pairs = sum(f["num_premise_sentence_pairs"] for f in faithfulness_out)
        # return mean entailment score, mean contradiction score, and mean outlier rate
        mean_entailment_score = sum(f["faithfulness"]["entailment_mean"] for f in faithfulness_out) / len(faithfulness_out)
        min_entailment_score = min(f["faithfulness"]["entailment_min"] for f in faithfulness_out)  # Don't divide by length - this is the minimum across all examples
        mean_ratio_low_entailment_sentences = sum(f["faithfulness"]["low_entailment_sentences"] for f in faithfulness_out) / len(faithfulness_out)
        max_contradiction_score = max(f["faithfulness"]["contradiction_max"] for f in faithfulness_out)  # Don't divide by length - this is the maximum across all examples
        mean_ratio_high_contradiction_sentences = sum(f["faithfulness"]["high_contradiction_sentences"] for f in faithfulness_out) / len(faithfulness_out)
        mean_ratio_outliers = sum(f["faithfulness"]["outlier_rate"] for f in faithfulness_out) / len(faithfulness_out)
        return {
            "mean_entailment_score": mean_entailment_score,
            "min_entailment_score": min_entailment_score,
            "mean_ratio_low_entailment_sentences": mean_ratio_low_entailment_sentences,
            "max_contradiction_score": max_contradiction_score,
            "mean_ratio_high_contradiction_sentences": mean_ratio_high_contradiction_sentences,
            "mean_ratio_outliers": mean_ratio_outliers,
            "ratio_passed_documents": ratio_passed_documents,
            "reasons_failed": reasons_failed,
            "num_premise_sentence_pairs": num_premise_sentence_pairs,
        }


def extended_evaluate(input_texts, prediction_texts, reference_texts, print_output=False, 
                      include_bertscore=True, include_faithfulness=False):
    """Compute evaluation metrics with selective inclusion of expensive metrics.
    
    Args:
        input_texts: List of input documents
        prediction_texts: List of predicted summaries
        reference_texts: List of reference summaries
        print_output: If True, print detailed results
        include_bertscore: If True, compute BERTScore (~1.5-2 min for 500 examples)
        include_faithfulness: If True, compute NLI faithfulness (~37 min for 500 examples)
    
    Returns:
        Dictionary with computed metrics
    """
    # Reference-based metrics (ROUGE always, BERTScore optional)
    reference_out = eval_reference(prediction_texts, reference_texts, include_bertscore=include_bertscore)
    if print_output:
        print("REFERENCE:")
        print(json.dumps(reference_out, indent=2, ensure_ascii=False, default=str))

    # Hygiene metrics (always fast)
    hygiene_out = eval_hygiene(input_texts, prediction_texts)
    if print_output:
        print("\nHYGIENE:")
        print(json.dumps(hygiene_out, indent=2, ensure_ascii=False, default=str))

    result = {
        "reference": reference_out,
        "hygiene": hygiene_out,
    }

    # NLI-based faithfulness metrics (very slow, optional)
    if include_faithfulness:
        gate = NLIFaithfulnessGate()
        faithfulness_out = gate.eval_faithfulness(input_texts, prediction_texts)
        if print_output:
            print("\nFAITHFULNESS:")
            print(json.dumps(faithfulness_out, indent=2, ensure_ascii=False, default=str))
        result["faithfulness"] = faithfulness_out
    else:
        result["faithfulness"] = None

    return result


def dummy_data_test():
    doc = (
        "Oslo kommune opplyser at budsjettet for 2026 øker med 3 prosent. "
        "Samtidig varsles det ingen økning i eiendomsskatten."
    )
    pred_summ = "Budsjettet i Oslo øker i 2026, mens eiendomsskatten forblir uendret."
    ref_summ = "Oslo kommune øker budsjettet med 3 prosent i 2026, uten å heve eiendomsskatten."

    extended_evaluate([doc], [pred_summ], [ref_summ], include_bertscore=True, include_faithfulness=False)


def detokenize_data(data):
    return TOKENIZER.batch_decode(data, skip_special_tokens=True)


def find_files(data_dir=DATA_DIR, file_mask=FILE_MASK):
    data_files = sorted(
        f for f in os.listdir(data_dir) 
        if re.match(file_mask, f)
    )
    return data_files


def load_texts(input_file):
    
    def collect(data, key):
        fields = []
        for d in data:
            fields.extend(d[key])
        return fields
    
    data = []
    with open(os.path.join(DATA_DIR, input_file), "r") as f:
        for line in f:
            data.append(json.loads(line))

    inputs = collect(data, "input")
    predictions = collect(data, "prediction")
    references = collect(data, "reference")
    
    assert len(inputs) == len(references) == len(predictions)
    print(f"Loaded {len(inputs)} input-reference-prediction examples from {input_file}")
    
    input_texts = detokenize_data(inputs)
    prediction_texts = detokenize_data(predictions)
    reference_texts = detokenize_data(references)
    
    # remove the prompt from the input text
    # TODO: the prompt should not have been saved in the first place
    input_texts = [text.replace('[INST] Oppsummer følgende tekst:\n\nDokument: ', '') for text in input_texts]

    return input_texts, prediction_texts, reference_texts


def save_results(eval_results, input_file, data_dir=DATA_DIR, file_mask=FILE_MASK):
    match = re.match(file_mask, input_file)
    checkpoint_id = match.group(1) if match and match.group(1) else 'UNKNOWN'
    output_file = os.path.join(data_dir, f"checkpoint-{checkpoint_id}-eval-results.json")
    with open(output_file, "w") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Saved evaluation results to {output_file}")


if __name__ == "__main__":

    data_files = find_files(DATA_DIR, FILE_MASK)    
    for input_file in data_files:

        input_texts, prediction_texts, reference_texts = load_texts(input_file)
        eval_results = extended_evaluate(input_texts, prediction_texts, reference_texts)
        save_results(eval_results, input_file)
        