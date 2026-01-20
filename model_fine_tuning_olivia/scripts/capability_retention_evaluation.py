"""
Metrics for general capability retention evaluation (both in-training validation and final evaluation) 
of summarisation models for Norwegian public documents, assuming that LLM-generated reference summaries are available.

TODO: Below follows the ambition - not yet implemented.

The evaluation regime measures:
* Regression vs the base model
* Distributional drift
* Trade-offs between specialization and generality

Types of metrics:
* Norwegian general-domain NLL / perplexity (cheap early signal)
  * text-prediction on fixed corpus
* Base-vs-tuned divergence (early warning): measures distributional drift via token-level KL divergence or log-prob deltas
  * Norwegian anchor prompts (general, not summarisation)
* Anchor-suite retention (primary signal): mean and worst-case retention of general capability after specialization
  * Norwegian reading comprehension / QA
  * General instruction following (Norwegian)
  * Simple reasoning / logic in Norwegian
  * Language modeling probes (cloze / continuation)
  * ...compare delta = tuned - base scores
  
* Evaluation data:
  * in-domain, out-of-domain
  * short (50-100 x 512B), middle (50-100 x 2kB) and long (50-100 x 4kB) bands
  
In-training validation (cheap panel):
* every N steps
* run on a small fixed subset (e.g., 50–100 docs)

In-training validation (major panel):
* every N*M steps
* run on a subset of the validation set (e.g., 500–1000 docs)

Final evaluation (full panel):

"""

import itertools
import json
import os

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


POLITICAL_RETENTION_DATA_FOLDER = "../../datasets_from_demokratibasen/cleaned_datasets/text_summary_dataset_202601/capability_retention_data"

STAGES = ["regular", "major", "final"]
BANDS = ["ultra_narow", "narrow", "medium", "broad"]
TASKS = ["3060_c2_text_prediction", "9650_c3_prompt_continuation"]

C3_PROMPT_QUESTION_SIZES = {
    "ultra_narow": (96, 96),
    "narrow": (128, 128),
    "medium": (512, 1536),
    "broad": (1024, 7168),
}

# --- C2: Norwegian general-domain NLL (cheap early signal) ---
"""
Text-prediction on fixed corpus.
* “After fine-tuning, did my model get worse at predicting general Norwegian text?”
* absolute language modeling competence:
  * measures whether the tuned model is still a good Norwegian language model, independent of the base model — Δ vs base just makes that judgment fair and comparable.

Approach:

Report:
* mean per-window ΔNLL (tuned − base) for each band and corpus
* OPTIONAL: p90 ΔNLL across windows (very useful)
* ΔPPL is less linear and harder to interpret

Evaluation data:
* in-domain Norwegian (political)
* out-of-domain Norwegian (general)
  * Wikipedia
  * general news
  * administrative / informational text

In-training validation:
* narrow band: 1000 x 512B
* medium band: 200 x 2kB
* wide band: 50-100 x 8kB (or 4kB)

Regular: 50 narrow, 10 medium
Major: 250 narrow, 50 medium, 10 broad

Final evaluation:

Final: 1000 narrow, 200 medium, 50 broad
"""

def compute_sequence_nll(model, tokenizer, text):
    """
    Compute negative log-likelihood for a single text sequence.
    
    Args:
        model: The language model (AutoModelForCausalLM)
        tokenizer: The tokenizer (AutoTokenizer)
        text: Input text string
        
    Returns:
        avg_nll: Average negative log-likelihood per token
        total_nll: Total negative log-likelihood
        num_tokens: Number of tokens in the sequence (excluding first token)
    """
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"][0]  # Remove batch dimension
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]  # Remove batch dimension: [seq_len, vocab_size]
        
        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # For each position i, get log prob of token at position i+1
        # (logits[i] predicts token[i+1])
        total_nll = 0.0
        num_tokens = 0
        
        for i in range(len(input_ids) - 1):
            next_token_id = input_ids[i + 1].item()
            token_log_prob = log_probs[i, next_token_id].item()
            total_nll -= token_log_prob  # NLL is negative log prob
            num_tokens += 1
        
        avg_nll = total_nll / num_tokens if num_tokens > 0 else 0.0
        
    return avg_nll, total_nll, num_tokens


def compute_average_nll(model, text_sequences):
    """
    Compute average negative log-likelihood across multiple text sequences.
        
    Args:
        model: DummyModel instance (or any object with .model and .tokenizer attributes)
        text_sequences: List of text strings (or token ID lists that will be decoded)
        
    Returns:
        Average NLL per token across all sequences
    """
    total_nll = 0.0
    total_tokens = 0
    
    for sequence in text_sequences:
        # Convert token IDs to text if needed
        if isinstance(sequence, list) and len(sequence) > 0 and isinstance(sequence[0], int):
            # It's a list of token IDs, decode to text
            text = model.tokenizer.decode(sequence, skip_special_tokens=True)
        else:
            # Assume it's already text
            text = sequence
            
        avg_nll, _, num_tokens = compute_sequence_nll(model.model, model.tokenizer, text)
        total_nll += avg_nll * num_tokens
        total_tokens += num_tokens
    
    return total_nll / total_tokens if total_tokens > 0 else 0.0


# --- C3: Base-vs-tuned divergence (early warning) ---
"""
Measures distributional drift via token-level log-prob deltas
* “how much the (tuned) model has moved (away from the base model)"
* full KL divergence is too costly
* Norwegian anchor prompts (general texts, not summaries)
  
Approach: Continuation windows
For each prompt, compute per-token ΔNLL on:
* next 256 tokens (cheap and stable)
* optionally also next 512 tokens for final eval

Suggested scale: 
* 1,000 prompts per set
* If that’s too large, 300–500 is still useful.

Report for each set: mean per-token ΔNLL, p95 ΔNLL

Interpretation: Drift should be small and stable; spikes usually correlate with regressions.
If drift is large on general prompts but small on political prompts, you are over-specializing.

Evaluation data:
* political prompts (in-domain), first tokens of a corpus-like political document (exclude summaries)
* general prompts (out-of-domain), first tokens of general public text
* OPTIONAL: instruction prompts (general Norwegian)

Short instruction prompts to detect instruction-following drift (not summarization)

In-training validation:
* prompt: first 256 tokens of a political document
* continuation window: next 256 tokens (cheap and stable)
* also use middle (512+1536B) and wide (1024+7168B, or 1024+3072B) bands...

Regular: 300 narrow, 50 medium, 0 broad
Major: 1000 narrow, 200 medium, 50 broad

Final evaluation:
* prompt: first 128–256 tokens of a political document
* continuation window: next 512 tokens - use longer continuations only for major or final checks
* also use middle and wide bands...

Final: 2000 narrow, 500 medium, 100 broad
"""


def compute_sequence_delta_nll(base_model, base_tokenizer, tuned_model, tuned_tokenizer, text, prompt_size, question_size):
    """
    Compute per-token delta NLL (tuned - base) for continuation tokens given a prompt.
    
    For a text T, this function:
    1. Uses the first PROMPT tokens as context
    2. Evaluates both models on the next QUESTION tokens
    3. Returns per-token ΔNLL = tuned_NLL - base_NLL
    
    Args:
        base_model: Base language model (AutoModelForCausalLM)
        base_tokenizer: Base tokenizer (AutoTokenizer)
        tuned_model: Tuned language model (AutoModelForCausalLM)
        tuned_tokenizer: Tuned tokenizer (AutoTokenizer)
        text: Input text string
        prompt_size: Number of tokens to use as prompt (context)
        question_size: Number of tokens to evaluate (continuation)
        
    Returns:
        delta_nlls: List of per-token ΔNLL values (tuned_NLL - base_NLL) for each continuation token
        mean_delta_nll: Mean delta NLL across continuation tokens
    """
    # Tokenize text (we'll truncate to only what we need)
    # Note: base and tuned models use the same tokenizer, so we tokenize once
    inputs = base_tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"][0]  # Remove batch dimension
    
    # Need at least prompt_size + 1 tokens (prompt + at least one continuation token)
    if len(input_ids) < prompt_size + 1:
        # Not enough tokens even for minimal evaluation
        return [], 0.0
    
    # Use as many continuation tokens as available (up to question_size)
    # This allows us to use partial data rather than discarding short texts
    available_continuation_tokens = len(input_ids) - prompt_size
    actual_question_size = min(question_size, available_continuation_tokens)
    
    # Truncate to only what we need: prompt_size + actual_question_size tokens
    # This saves computation and memory for long texts
    max_tokens_needed = prompt_size + actual_question_size
    input_ids_truncated = input_ids[:max_tokens_needed]
    
    # Create new input tensors with truncated sequences
    inputs_truncated = {"input_ids": input_ids_truncated.unsqueeze(0)}
    
    with torch.no_grad():
        # Run base model on truncated sequence (only what we need)
        base_outputs = base_model(**inputs_truncated)
        base_logits = base_outputs.logits[0]  # [seq_len, vocab_size]
        base_log_probs = torch.nn.functional.log_softmax(base_logits, dim=-1)
        
        # Run tuned model on truncated sequence (only what we need)
        tuned_outputs = tuned_model(**inputs_truncated)
        tuned_logits = tuned_outputs.logits[0]  # [seq_len, vocab_size]
        tuned_log_probs = torch.nn.functional.log_softmax(tuned_logits, dim=-1)
    
    # Extract continuation tokens (tokens at positions [prompt_size, prompt_size+actual_question_size-1])
    # Logits at position i predict token at position i+1
    # So logits at position [prompt_size-1, prompt_size+actual_question_size-2] predict continuation tokens
    
    delta_nlls = []
    
    for i in range(actual_question_size):
        # Position in the continuation
        token_pos = prompt_size + i  # Actual token position in the sequence
        logit_pos = token_pos - 1    # Logit position that predicts this token
        
        if logit_pos < 0 or token_pos >= len(input_ids_truncated):
            continue
            
        # Get the actual token ID (same for both models since they use the same tokenizer)
        token_id = input_ids_truncated[token_pos].item()
        
        # Get log probabilities for the actual token
        base_log_prob = base_log_probs[logit_pos, token_id].item()
        tuned_log_prob = tuned_log_probs[logit_pos, token_id].item()
        
        # Convert to NLL (negative log probability)
        base_nll = -base_log_prob
        tuned_nll = -tuned_log_prob
        
        # Compute delta NLL (tuned - base)
        delta_nll = tuned_nll - base_nll
        delta_nlls.append(delta_nll)
    
    mean_delta_nll = sum(delta_nlls) / len(delta_nlls) if len(delta_nlls) > 0 else 0.0
    
    return delta_nlls, mean_delta_nll


def compute_average_delta_nll(base_model_wrapper, tuned_model_wrapper, text_sequences, prompt_size, question_size):
    """
    Compute average delta NLL (tuned - base) across multiple text sequences.
    
    For each text sequence, extracts a prompt of PROMPT tokens and evaluates both
    models on the next QUESTION tokens, computing per-token ΔNLL.
    
    Args:
        base_model_wrapper: TestModel instance (or any object with .model and .tokenizer attributes)
        tuned_model_wrapper: TestModel instance
        text_sequences: List of text strings
        prompt_size: Number of tokens to use as prompt (context)
        question_size: Number of tokens to evaluate (continuation)
        
    Returns:
        mean_delta_nll: Mean delta NLL per token across all sequences
        per_token_deltas: List of all per-token delta NLL values (for computing percentiles)
    """
    all_delta_nlls = []
    total_tokens = 0
    total_delta_nll = 0.0
    
    for text in text_sequences:
        delta_nlls, mean_delta = compute_sequence_delta_nll(
            base_model_wrapper.model,
            base_model_wrapper.tokenizer,
            tuned_model_wrapper.model,
            tuned_model_wrapper.tokenizer,
            text,
            prompt_size,
            question_size
        )
        
        if len(delta_nlls) > 0:
            all_delta_nlls.extend(delta_nlls)
            total_delta_nll += sum(delta_nlls)
            total_tokens += len(delta_nlls)
    
    overall_mean = total_delta_nll / total_tokens if total_tokens > 0 else 0.0
    
    return overall_mean, all_delta_nlls


# --- C1: Anchor-suite retention (primary signal) ---
"""
Mean and worst-case retention of general capability after specialization
  * Norwegian reading comprehension / QA
  * General instruction following (Norwegian)
  * Simple reasoning / logic in Norwegian
  * Language modeling probes (cloze / continuation)
  * ...compare delta = tuned - base scores
  
Approach:


Evaluation data:


In-training validation:


Final evaluation:

"""

# --- Data loading ---

def load_political_retention_data(folder, task, stage):
    stages = (["major", "regular"] if stage=="major" else [stage])
    data = {}
    for band in BANDS:
        data[band] = []
        fn = f"{task}_{band}_examples.jsonl"
        path = os.path.join(folder, fn)
        with open(path, "r") as f:
            for line in f:
                obj = json.loads(line)
                if obj["stage"] in stages:
                    data[band].append(obj["text"])
    return data


# --- Test model ---

class TestModel:
    """Minimal model+tokenizer wrapper"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


if __name__ == "__main__":

    BASE_MODEL = 'google/gemma-2b'
    TUNED_MODEL = '../../model_fine_tuning/models/gemma2b_finetuned_5200_steps_200_eval_stride/checkpoint-5200'
    
    base_model = TestModel(BASE_MODEL)
    tuned_model = TestModel(TUNED_MODEL)

    # Assert that tokenizers are the same (required for correct evaluation)
    test_text = "Dette er en test for å verifisere at tokenizere er identiske."
    base_tokens = base_model.tokenizer(test_text, return_tensors="pt")
    tuned_tokens = tuned_model.tokenizer(test_text, return_tensors="pt")
    
    assert torch.equal(base_tokens["input_ids"], tuned_tokens["input_ids"]), \
        f"Tokenizers produce different tokenizations!\n" \
        f"Base: {base_tokens['input_ids']}\n" \
        f"Tuned: {tuned_tokens['input_ids']}"
    
    assert base_model.tokenizer.vocab_size == tuned_model.tokenizer.vocab_size, \
        f"Tokenizers have different vocab sizes: base={base_model.tokenizer.vocab_size}, " \
        f"tuned={tuned_model.tokenizer.vocab_size}"
    
    assert type(base_model.tokenizer) == type(tuned_model.tokenizer), \
        f"Tokenizers are different types: base={type(base_model.tokenizer)}, " \
        f"tuned={type(tuned_model.tokenizer)}"
    
    print("✓ Tokenizers verified to be identical")

    political_retention_data = load_political_retention_data(POLITICAL_RETENTION_DATA_FOLDER, "9650_c3_prompt_continuation", "major")
    len_ = len(list(itertools.chain.from_iterable(political_retention_data.values())))
    print(f'Loaded {len_} political texts')

    # Test with narrow band
    band = "narrow"

    # --- C2 negative log-likelihood test ---
    print("\n--- C2: Negative log-likelihood test ---")

    windows = political_retention_data[band][:5]
    print(f'Testing on first {len(windows)} {band}-band texts')

    nll_base  = compute_average_nll(base_model,  windows)
    nll_tuned = compute_average_nll(tuned_model, windows)
    
    delta_nll = nll_tuned - nll_base
    print(f"Base NLL: {nll_base:.4f}")
    print(f"Tuned NLL: {nll_tuned:.4f}")
    print(f"Delta NLL: {delta_nll:.4f}")
    
    # --- C3 text prediction divergence test ---
    print("\n--- C3: Base-vs-tuned divergence (early warning) ---")
    
    prompt_size, question_size = C3_PROMPT_QUESTION_SIZES[band]
    print(f"\nTesting {band} band: prompt={prompt_size} tokens, question={question_size} tokens")
    
    # Use a subset of texts for testing
    test_texts = political_retention_data[band][:5]  # Use first 5 texts for faster testing
    print(f"Testing on first {len(test_texts)} {band}-band texts")
    
    mean_delta_nll, all_delta_nlls = compute_average_delta_nll(
        base_model, tuned_model, test_texts, prompt_size, question_size
    )
    
    if len(all_delta_nlls) > 0:
        p95_delta_nll = np.percentile(all_delta_nlls, 95)
        
        print(f"Mean per-token ΔNLL (tuned - base): {mean_delta_nll:.4f}")
        print(f"P95 per-token ΔNLL: {p95_delta_nll:.4f}")
        print(f"Total tokens evaluated: {len(all_delta_nlls)}")
        
        # Interpretation hint
        if mean_delta_nll > 0:
            print(f"→ Tuned model has higher NLL (worse) on continuation by {mean_delta_nll:.4f} on average")
        else:
            print(f"→ Tuned model has lower NLL (better) on continuation by {abs(mean_delta_nll):.4f} on average")
    else:
        print("No tokens evaluated (texts may be too short)")
