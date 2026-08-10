# Utils

Shared utilities for fine-tuning and evaluation. Import via `from utils import ...` when running from the `scripts/` directory.

---

## File Summary

| File | Description |
|------|-------------|
| **checkpoint_utils.py** | Checkpoint path parsing and resolution. Extracts step numbers, normalizes checkpoint names, detects major checkpoints (every 500 steps), and derives model directory from checkpoint path (including backup dirs). |
| **data_collators.py** | Evaluation data collator. Left-pads `input_ids` and right-pads `labels` for decoder-only models; produces batched tensors for the evaluation loop. |
| **dataset_loading.py** | JSONL dataset loader. Loads training/validation JSONL with Git LFS pointer detection, file size checks, and structured error handling. |
| **eval_results.py** | Evaluation results I/O. Manages paths for centralized `all_eval_results/` layout, loads/saves eval JSON and summary files, tracks which checkpoints have been evaluated. |
| **formatting.py** | Example formatting for training and evaluation. Applies model-specific prompt templates (chat vs plain) using `model_configs`; supports batched formatting for fast `dataset.map()`. |
| **metrics.py** | ROUGE metrics and text cleaning. Computes ROUGE from predictions/labels, cleans decoded text (removes special tokens), optionally logs to WandB. |
| **nli_subset.py** | Fixed NLI evaluation subset. Default size 100; set size ≥ eval length (usually = `val_data_size`) for full-val NLI. Seed 42 for random subsets. Indices saved for comparability across checkpoints. |
| **tokenization.py** | Tokenization for training and evaluation. Tokenizes examples, tracks prompt length for loss masking, handles model context limits and truncation. |

---

## Modules

### checkpoint_utils.py

**Description:** Parses and resolves checkpoint paths. Extracts step numbers from directory names (`checkpoint-123`, `regular-checkpoint-123`, `major-checkpoint-123`), normalizes to canonical form, identifies major checkpoints (every 500 steps for BERTScore), and derives the parent model directory from a checkpoint path—including when checkpoints live in `regular_checkpoints/` or `major_checkpoints/` subdirs.

**Important functions:**
- `extract_checkpoint_step(checkpoint_path)` — Returns step number from path, or -1 if parsing fails. Handles `checkpoint-123`, `regular-checkpoint-123`, `major-checkpoint-123`.
- `get_checkpoint_name_and_step(checkpoint_path)` — Returns `(normalized_name, step)` e.g. `("checkpoint-100", 100)`.
- `is_major_checkpoint(checkpoint_step, major_checkpoint_interval=500)` — True if step is a multiple of interval (used for BERTScore on major checkpoints only).
- `get_model_dir_from_checkpoint(checkpoint_dir)` — Returns parent model dir; handles both main checkpoints and backup subdirs.

**Local dependencies:** None

---

### data_collators.py

**Description:** Custom data collator for evaluation batches. Left-pads `input_ids` (decoder-only convention) and right-pads `labels` with -100 (ignored in loss). Supports optional `pad_to_multiple_of` for efficiency. Used by the evaluation script when batching tokenized prompts and labels.

**Important classes:**
- `EvalDataCollator` — `__call__(features)` returns batched `input_ids`, `attention_mask`, `labels` tensors.

**Local dependencies:** `torch`, `transformers`

---

### dataset_loading.py

**Description:** Loads JSONL datasets (training, validation, etc.) from disk. Detects Git LFS pointers (small files with `version https://git-lfs.github.com/...`), validates file size, parses JSON lines with error handling, and returns a list of dicts. Used by training and evaluation scripts for dataset loading.

**Important functions:**
- `load_jsonl_dataset(file_path, dataset_type="dataset", raise_on_error=False)` — Loads JSONL; returns list of dicts or None on error. Detects LFS pointers and reports helpful message.

**Local dependencies:** None

---

### eval_results.py

**Description:** Handles evaluation results file I/O. Manages paths for the centralized `all_eval_results/` layout (e.g. `model_dir/all_eval_results/checkpoint-500-gen0-eval-results-1000-examples.json`) and legacy per-checkpoint locations. Loads and saves eval JSON, predictions JSONL, and the `gen0_evaluation_summary.json`/`genN_evaluation_summary.json` files that aggregate metrics across checkpoints. Used by the evaluation and monitor scripts.

**Important functions:**
- `get_eval_results_path(checkpoint_dir, model_dir, examples_suffix)` — Path to `checkpoint-N-genG-eval-results-X-examples.json`.
- `get_predictions_file_path(checkpoint_dir, model_dir, examples_suffix)` — Path to `checkpoint-N-genG-inputs-refs-preds-X-examples.jsonl`.
- `get_old_eval_results_path(checkpoint_dir)` — Legacy path: `checkpoint_dir/eval_results/eval_results.json`.
- `load_eval_results(checkpoint_dir, model_dir)` — Loads eval JSON; falls back to old location if needed.
- `save_eval_results(checkpoint_dir, results, model_dir, examples_suffix)` — Saves to centralized location.
- `get_evaluated_checkpoint_steps(model_dir)` — Returns set of step numbers that have eval results.
- `update_evaluation_summary(model_dir, checkpoint_name, results, examples_suffix)` — Updates `genN_evaluation_summary.json` with latest checkpoint metrics.

**Local dependencies:** `checkpoint_utils` (for `get_model_dir_from_checkpoint`)

---

### formatting.py

**Description:** Formats training and evaluation examples using model-specific prompt configs from `model_configs`. Supports chat templates (Llama-2/3, Mistral, ChatML) and plain text. Uses a cached model config to avoid repeated lookups during `dataset.map()`. The batched formatter can skip `apply_chat_template` for ~100x speedup when using manual templates. Used by training and evaluation for prompt construction.

**Important functions:**
- `format_train_example(example, model_name, tokenizer)` — Formats single example for training. Returns `{"text": formatted_string}`. Uses `model_configs.get_model_config_by_hf_name` and `PromptConfig.format_train`.
- `format_train_examples_batch(examples, model_name, tokenizer, model_config, use_fast_format=True)` — Batched formatting for `dataset.map()`. When `use_fast_format=True`, skips `apply_chat_template` for ~100x speedup; uses manual template instead.
- `format_eval_example(example, model_name, tokenizer)` — Formats for evaluation. Returns `{"prompt": formatted_prompt, "target_summary": output_text}`.
- `_get_model_config_cached(model_name)` — Internal cache to avoid repeated config lookups.

**Local dependencies:** `model_configs` (sibling in `scripts/`)

---

### metrics.py

**Description:** Computes ROUGE metrics and cleans decoded text for evaluation. Decodes token IDs to strings, removes special tokens (e.g. `[/INST]`, `<s>`), normalizes whitespace, and computes ROUGE-1/2/L/Lsum. Handles quantization edge cases (clips token IDs). Optionally logs metrics to WandB. Used by the evaluation script and in-training validation.

**Important functions:**
- `compute_rouge_metrics(eval_pred, tokenizer, log_to_wandb=False, step=None, is_main_process=True, verbose=True)` — Decodes predictions/labels, cleans text, computes ROUGE (rouge1, rouge2, rougeL, rougeLsum), optionally logs to WandB. Clips token IDs for quantization compatibility.
- `clean_decoded_text(text)` — Removes `[/INST]`, `[INST]`, `</s>`, `<s>`, backslashes; normalizes whitespace.

**Local dependencies:** `evaluate`, `transformers`, `wandb`

---

### nli_subset.py

**Description:** Manages a fixed subset of examples for NLI faithfulness evaluation. Default `subset_size` is **100** (`NLI_DEFAULT_SUBSET_SIZE`). Use `subset_size >= total_examples` (typically equal to `val_data_size`) for full-val NLI. Smaller sizes use a sorted **seed-42** random sample or first N when extending eval size. Same indices are reused across checkpoints via `nli_fixed_subset_indices.json`.

**Important functions:**
- `get_nli_subset_file_path(model_dir)` — Path to `all_eval_results/nli_fixed_subset_indices.json`.
- `create_fixed_nli_subset(total_examples, subset_size, seed, model_dir)` — `subset_size` None → default 100; `subset_size >= total` → full set; else seeded random subset.
- `load_fixed_nli_subset(model_dir)` — Loads indices from file; returns None if missing.
- `get_or_create_fixed_nli_subset(total_examples, model_dir, subset_size, seed, use_first_n_for_extended)` — Loads existing or creates new. When `use_first_n_for_extended=True` and total > subset_size, uses first N indices (for backward comparability when extending eval set).
- `apply_fixed_subset(input_texts, prediction_texts, reference_texts, indices)` — Filters lists by indices; returns `(filtered_inputs, filtered_preds, filtered_refs)`.

**Constants:** `NLI_DEFAULT_SUBSET_SIZE = 100`, `NLI_FIXED_SUBSET_SEED = 42`; `NLI_FIXED_SUBSET_SIZE = 500` retained for legacy imports only.

**Local dependencies:** None

---

### tokenization.py

**Description:** Tokenizes training and evaluation examples. For training: tokenizes full text and computes `prompt_length` for loss masking (only assistant response is trained). For evaluation: tokenizes prompt and target separately. Infers model context limits from tokenizer name when not set (e.g. Llama-3.1→128K, Mistral→32K). Pre-truncates at character level for speed. Used by training and evaluation pipelines.

**Important functions:**
- `tokenize_train_examples(examples, tokenizer, max_input_text_tokens, max_extra_prompt_tokens, max_output_summary_tokens)` — Tokenizes full training text. Returns dict with `input_ids`, `attention_mask`, `prompt_length` (for masking in data collator). Infers `model_max_length` from tokenizer name (Llama-3.1→128K, Mistral→32K, etc.) if not set.
- `tokenize_eval_examples(examples, tokenizer, ...)` — Tokenizes prompt and target separately. Returns `input_ids`, `attention_mask`, `labels` (target token IDs).
- `_compute_prompt_length_chat(input_ids, tokenizer)` — Finds prompt end via `[/INST]`, `<|end_header_id|>`, `<|im_start|>` markers.
- `_compute_prompt_length_plain(input_ids, tokenizer)` — Finds "Oppsummering:\n\n###\n\n" marker for plain format.

**Local dependencies:** `transformers`

---

## Dependency Graph

```
formatting.py   → model_configs (scripts/)
eval_results.py → checkpoint_utils
```

All other modules have no internal utils dependencies.

---

## Package Exports (`utils/__init__.py`)

| Category | Exports |
|----------|---------|
| Data collators | `EvalDataCollator` |
| Metrics | `compute_rouge_metrics`, `clean_decoded_text` |
| Checkpoint | `extract_checkpoint_step`, `get_checkpoint_name_and_step`, `is_major_checkpoint`, `get_model_dir_from_checkpoint` |
| Eval results | `get_eval_results_path`, `get_predictions_file_path`, `get_old_eval_results_path`, `load_eval_results`, `save_eval_results`, `get_evaluated_checkpoint_steps`, `update_evaluation_summary` |
| Dataset | `load_jsonl_dataset` |
| Tokenization | `tokenize_train_examples`, `tokenize_eval_examples` |
| Formatting | `format_train_example`, `format_train_examples_batch`, `format_eval_example` |
| NLI subset | `get_or_create_fixed_nli_subset`, `apply_fixed_subset`, `NLI_DEFAULT_SUBSET_SIZE`, `NLI_FIXED_SUBSET_SEED` |

---

## Usage

Ensure `scripts/` is on `PYTHONPATH`:

```python
import sys
import os
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from utils import load_jsonl_dataset, format_eval_example, compute_rouge_metrics
```
