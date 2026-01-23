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
bertscore = evaluate.load("bertscore")

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

def eval_reference(pred_summaries, ref_summaries):
    r = rouge.compute(
        predictions=pred_summaries,
        references=ref_summaries,
        use_stemmer=False,  # avoid for Norwegian
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
    )
    
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
    return {
        **r,
        "bertscore_f1_mean": sum(b["f1"]) / len(b["f1"]),
    }
    
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
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


def chunk_document(doc: str, tokenizer, max_tokens: int = 350) -> List[str]:
    doc = re.sub(r"\s+", " ", doc).strip()
    if not doc:
        return [""]

    ids = tokenizer.encode(doc, add_special_tokens=False)
    chunks = []
    for i in range(0, len(ids), max_tokens):
        chunk_ids = ids[i : i + max_tokens]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
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
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**inputs).logits[0]
        return torch.softmax(logits, dim=-1)

    def score_and_gate(
        self,
        document: str,
        summary: str,
        doc_chunk_tokens: int = 350,
        # --- Suggested starting thresholds for CI (tune with a small human-audited set) ---
        entailment_mean_min: float = 0.72,
        entailment_sentence_min: float = 0.60,
        max_low_entailment_sentences: int = 0,
        contradiction_sentence_max: float = 0.35,
        max_high_contradiction_sentences: int = 0,
    ) -> Dict[str, Any]:
        sents = split_sentences(summary)
        doc_chunks = chunk_document(document, self.tokenizer, max_tokens=doc_chunk_tokens)

        per_sentence: List[Dict[str, Any]] = []
        best_entailments: List[float] = []
        worst_contradictions: List[float] = []

        for sent in sents:
            best_ent = -1.0
            best_ent_chunk = -1
            worst_con = -1.0
            worst_con_chunk = -1

            for j, chunk in enumerate(doc_chunks):
                probs = self._probs(chunk, sent)
                p_ent = float(probs[self.entailment_idx].cpu().item())
                p_con = float(probs[self.contradiction_idx].cpu().item())

                if p_ent > best_ent:
                    best_ent = p_ent
                    best_ent_chunk = j
                if p_con > worst_con:
                    worst_con = p_con
                    worst_con_chunk = j

            best_entailments.append(best_ent)
            worst_contradictions.append(worst_con)

            per_sentence.append(
                {
                    "sentence": sent,
                    "best_entailment": best_ent,
                    "best_entailment_chunk_idx": best_ent_chunk,
                    "worst_contradiction": worst_con,
                    "worst_contradiction_chunk_idx": worst_con_chunk,
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

        # Gate logic
        reasons = []
        if entail_mean < entailment_mean_min:
            reasons.append(
                f"mean_entailment {entail_mean:.3f} < {entailment_mean_min:.3f}"
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
        }
        
    def eval_faithfulness(self, docs, pred_summaries):
        faithfulness_out = []
        for doc, pred_summary in zip(docs, pred_summaries):
            faithfulness_out.append(self.score_and_gate(doc, pred_summary))
        # return ratio of that passed the gate and list of reasons for failure
        ratio_passed = sum(1 for f in faithfulness_out if f["passed"]) / len(faithfulness_out)
        reasons_failed = [f["reasons"] for f in faithfulness_out if not f["passed"]]
        # return mean entailment score, mean contradiction score, and mean outlier rate
        mean_entailment_score = sum(f["faithfulness"]["entailment_mean"] for f in faithfulness_out) / len(faithfulness_out)
        min_entailment_score = min(f["faithfulness"]["entailment_min"] for f in faithfulness_out)  # Don't divide by length - this is the minimum across all examples
        mean_low_entailment_sentences = sum(f["faithfulness"]["low_entailment_sentences"] for f in faithfulness_out) / len(faithfulness_out)
        max_contradiction_score = max(f["faithfulness"]["contradiction_max"] for f in faithfulness_out)  # Don't divide by length - this is the maximum across all examples
        mean_high_contradiction_sentences = sum(f["faithfulness"]["high_contradiction_sentences"] for f in faithfulness_out) / len(faithfulness_out)
        mean_outlier_rate = sum(f["faithfulness"]["outlier_rate"] for f in faithfulness_out) / len(faithfulness_out)
        return {
            "mean_entailment_score": mean_entailment_score,
            "min_entailment_score": min_entailment_score,
            "mean_low_entailment_sentences": mean_low_entailment_sentences,
            "max_contradiction_score": max_contradiction_score,
            "mean_high_contradiction_sentences": mean_high_contradiction_sentences,
            "mean_outlier_rate": mean_outlier_rate,
            "ratio_passed": ratio_passed,
            "reasons_failed": reasons_failed,
        }


def evaluate(input_texts, prediction_texts, reference_texts, print_output=False):
    # Reference-based metrics
    reference_out = eval_reference(prediction_texts, reference_texts)
    if print_output:
        print("REFERENCE:")
        print(json.dumps(reference_out, indent=2, ensure_ascii=False, default=str))

    # Hygiene metrics
    hygiene_out = eval_hygiene(input_texts, prediction_texts)
    if print_output:
        print("\nHYGIENE:")
        print(json.dumps(hygiene_out, indent=2, ensure_ascii=False, default=str))

    # NLI-based faithfulness metrics
    gate = NLIFaithfulnessGate()
    faithfulness_out = gate.eval_faithfulness(input_texts, prediction_texts)
    if print_output:
        print("\nFAITHFULNESS:")
        print(json.dumps(faithfulness_out, indent=2, ensure_ascii=False, default=str))

    return {
        "reference": reference_out,
        "hygiene": hygiene_out,
        "faithfulness": faithfulness_out,
    }


def dummy_data_test():
    doc = (
        "Oslo kommune opplyser at budsjettet for 2026 øker med 3 prosent. "
        "Samtidig varsles det ingen økning i eiendomsskatten."
    )
    pred_summ = "Budsjettet i Oslo øker i 2026, mens eiendomsskatten forblir uendret."
    ref_summ = "Oslo kommune øker budsjettet med 3 prosent i 2026, uten å heve eiendomsskatten."

    evaluate([doc], [pred_summ], [ref_summ])


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
    checkpoint_id = re.match(file_mask, input_file).group(1)
    output_file = os.path.join(data_dir, f"checkpoint-{checkpoint_id}-eval-results.json")
    with open(output_file, "w") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Saved evaluation results to {output_file}")


if __name__ == "__main__":

    data_files = find_files(DATA_DIR, FILE_MASK)    
    for input_file in data_files:

        input_texts, prediction_texts, reference_texts = load_texts(input_file)
        eval_results = evaluate(input_texts, prediction_texts, reference_texts)
        save_results(eval_results, input_file)
        