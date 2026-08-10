# Scripts

Python scripts for fine-tuning, evaluation, and analysis of Norwegian summarisation models.

---

## Core Training & Evaluation

### wandb_finetune.py

Main fine-tuning script. Supports single-GPU, multi-GPU DDP/FSDP, and optional 4-bit/8-bit quantization. Uses LoRA for parameter-efficient training and Weights & Biases for logging. Handles checkpoint backups (regular + major), early stopping signals, and prompt masking for causal LM loss.

**Important functions/classes:**
- `fine_tune_model()` — Main entry point; orchestrates dataset loading, model loading, LoRA setup, and training
- `load_model_with_optional_quantization()` — Loads base model with optional BitsAndBytes 4-bit/8-bit quantization
- `prepare_model_for_lora()` — Prepares model for LoRA (gradient checkpointing, target modules)
- `CausalLMTrainer` — Custom Trainer with prompt-length masking (only compute loss on assistant response)
- `CheckpointBackupCallback` — Backs up checkpoints to `regular_checkpoints/` and `major_checkpoints/`
- `EarlyStoppingMonitorCallback` — Watches for early-stopping signal file written by monitor script
- `GPUMemoryCallback` — Logs GPU memory usage during training

**Local dependencies:** `model_configs`, `utils`

**Example:**
```bash
python wandb_finetune.py --model gemma-2b --quantization none --train_dataset data/train.jsonl --val_dataset data/val.jsonl --hf_token YOUR_TOKEN
```

---

### distributed_finetune.py

Alternative training script with ROUGE callback during validation. Uses DDP/FSDP for multi-GPU training. Computes ROUGE scores on a sampled validation subset at evaluation steps.

**Important functions/classes:**
- `fine_tune_model()` — Main entry point
- `ROUGECallback` — Computes ROUGE during `on_evaluate` for monitoring

**Local dependencies:** `transformers`, `datasets`, `peft`, `evaluate`

**Example:**
```bash
torchrun --nproc_per_node=4 distributed_finetune.py --model gemma-7b --fsdp
```

---

### model_configs.py

Model and prompt configuration definitions. Centralizes LoRA parameters, learning rates, model name mappings (short → HuggingFace), and prompt templates for each model family.

**Important functions/classes:**
- `get_model_config(short_name)` — Get config by short name (e.g. `gemma-2b`)
- `get_model_config_by_hf_name(hf_name)` — Get config by HuggingFace name (e.g. `google/gemma-2b`)
- `get_model_name_mapping()` — Dict mapping short names to HuggingFace IDs
- `get_doc_type_norwegian(doc_type)` — Map English doc types (e.g. `case_minutes`) to Norwegian (e.g. `vedtak`)
- `ModelConfig` — Dataclass with LoRA config, prompt config, batch size hints
- `PromptConfig` — Dataclass with `format_train()` and `format_eval()`; supports plain, Llama-2/3, Mistral, ChatML templates

**Local dependencies:** `peft`

**Usage:** Imported by other scripts; not run directly.

---

### evaluate_distributed_checkpoints_multigpu.py

Multi-GPU evaluation of PEFT checkpoints using model parallelism (`device_map="auto"`). Splits large models across GPUs to avoid FSDP/DDP issues with `model.generate()`. Computes ROUGE, BERTScore (on major checkpoints), NLI faithfulness, and hygiene metrics. Writes results to `all_eval_results/` and optional WandB.

**Important functions:**
- `evaluate_checkpoint()` — Main entry point; loads model+adapter, generates predictions, runs extended metrics, saves results
- `load_model_and_peft_checkpoint()` — Loads base model and PEFT adapter with optional multi-GPU
- `sample_validation_data_reproducibly()` — Samples validation subset with fixed seed
- `get_model_batch_size()` — Returns model-specific eval batch size
- `check_gpu_memory_utilization()` — Reports GPU memory before/after load
- `AlreadyEvaluatedError` — Raised when checkpoint already has eval results (skip re-eval)

**Local dependencies:** `model_configs`, `utils`

**Example:**
```bash
python evaluate_distributed_checkpoints_multigpu.py --model gemma-7b --checkpoint_dir models/gemma-7b/checkpoint-500 --val_dataset data/val.jsonl --use_multi_gpu
```

---

### monitor_and_evaluate_checkpoints.py

Runs in parallel with FSDP training. Polls the output directory for new checkpoints, evaluates each as it appears, logs to WandB, and implements early stopping by writing a signal file when validation metrics plateau.

**Important functions:**
- `monitor_and_evaluate()` — Main loop; discovers checkpoints, calls `evaluate_checkpoint()`, checks early stopping
- `find_checkpoints()` — Lists checkpoint dirs, optionally filtered by max step
- `get_current_training_step()` — Reads `trainer_state.json` for latest global step
- `check_early_stopping_signal()` / `write_early_stopping_signal()` — Signal file for training script
- `get_best_checkpoint_metric()` — Reads best metric from eval summary for early-stopping decision
- `check_training_complete()` — Detects if training has finished

**Local dependencies:** `model_configs`, `evaluate_distributed_checkpoints_multigpu`

**Example:**
```bash
python monitor_and_evaluate_checkpoints.py --output_dir models/gemma-7b-apptainer-fsdp --model gemma-7b --val_dataset data/val.jsonl
```

---

## Evaluation & Analysis

### run_nli_faithfulness_subset.py

Run NLI faithfulness evaluation on a subset of predictions from a JSONL file. Useful when you want to re-run faithfulness on a smaller sample without full checkpoint re-evaluation.

**Important functions:**
- `load_predictions_from_jsonl()` — Loads input/prediction pairs, optionally samples subset
- `main()` — CLI entry; loads predictions, runs `NLIFaithfulnessGate.eval_faithfulness()`, saves JSON

**Local dependencies:** `utils.faithfulness`

**Example:**
```bash
python run_nli_faithfulness_subset.py --predictions_file models/gemma-7b/all_eval_results/checkpoint-100-gen0-inputs-refs-preds-1000-examples.jsonl --subset_size 100
```

---

### analyze_predictions.py

Analyzes prediction files for repetition issues, empty outputs, and repetitive sequences. Produces per-document stats (3-gram repetition, doc length) and optional matplotlib visualizations (repetition distribution, worst examples).

**Important functions:**
- `analyze_predictions()` — Main analysis; computes repetition, finds worst docs
- `ngram_repetition()` — 3-gram repetition rate
- `find_repetitive_sequences()` — Finds n-grams that repeat ≥ N times
- `remove_markup()` — Strips ###, **, etc. before metrics
- `create_visualizations()` — Plots repetition distribution (if matplotlib available)

**Local dependencies:** `matplotlib`, `numpy` (optional)

**Example:**
```bash
python analyze_predictions.py --predictions_file models/gemma-2-9b/all_eval_results/checkpoint-4100-gen0-inputs-refs-preds-1000-examples.jsonl --output_dir analysis_results
```

---

### analyze_checkpoint_5000_all_models.py

Cross-model analysis of checkpoint-N `inputs-refs-preds` files. Checks repetition, empty outputs, special token artifacts (e.g. `</s>`, `[INST]`), prompt format consistency, and off-topic/hallucination indicators.

**Important functions:**
- `analyze_file()` — Per-file analysis; returns dict with repetition, special tokens, prompt format, off-topic flags
- `ngram_repetition()` — 3-gram repetition rate
- `detect_prompt_format()` — Infers format from prompt string (chatml, mistral, alpaca, plain)
- `has_special_tokens_in_output()` — Detects leaked special tokens
- `off_topic_indicators()` — Heuristics for hallucination/off-topic

**Local dependencies:** None

**Example:**
```bash
python analyze_checkpoint_5000_all_models.py
python analyze_checkpoint_5000_all_models.py --checkpoint 6000
```

---

### generate_checkpoint_6000_analyses.py

Generates heuristic summary quality analyses for all models with checkpoint-6000 data. Detects failure modes: alpaca instruction leakage, "Response:" labels, hashtag spam, XX placeholders, Swedish/Danish mixing, truncated outputs.

**Important functions:**
- `load_samples()` — Loads first/middle/longest samples from checkpoint-6000 JSONL
- `analyze_predictions()` — Counts issues per category
- `generate_report()` — Produces markdown report per model
- `load_eval_results()` — Loads eval-results JSON for metrics

**Local dependencies:** None

**Example:**
```bash
python generate_checkpoint_6000_analyses.py
```

---

### visualize_checkpoint_results.py

Loads evaluation results from `all_eval_results/` and creates plots (matplotlib PNG, plotly HTML). Can compare multiple models. Optionally logs to WandB.

**Important functions:**
- `load_checkpoint_results()` — Loads all `checkpoint-N-genG-eval-results-X-examples.json` from model dir
- `extract_metrics()` — Extracts metric series (rougeLsum, faithfulness, etc.) for plotting
- `create_matplotlib_plots()` — PNG plots
- `create_plotly_html()` — Interactive HTML
- `log_to_wandb()` — Logs metrics to WandB run
- `visualize_checkpoints()` — Main entry; orchestrates load, plot, save

**Local dependencies:** `matplotlib`, `seaborn`, `plotly`, `wandb` (optional)

**Example:**
```bash
python visualize_checkpoint_results.py --model_dir models/gemma-2-9b-apptainer-fsdp --output_dir visualizations
```

---

### learning_curve.py

Plots training loss and rougeLsum from `trainer_state.json`. Uses Savitzky-Golay filter for loss smoothing and cubic spline interpolation for ROUGE.

**Important functions:**
- `load_trainer_state()` — Loads `trainer_state.json`
- `extract_metrics()` — Extracts loss and rougeLsum from log_history
- `compute_smooth_trend()` — Savitzky-Golay smoothing
- `interpolate_metrics()` — Cubic spline for ROUGE
- `plot_learning_curves()` — Creates and saves plot

**Local dependencies:** `matplotlib`, `numpy`, `scipy`

**Example:**
```bash
python learning_curve.py models/gemma-7b/checkpoint-500
```

---

## Testing & Utilities

### test_e2e.py

End-to-end tests for refactored code. Uses minimal config (10 examples, 5 steps, WandB disabled). Covers training, evaluation, utilities, file I/O, error handling, monitor integration, edge cases.

**Important functions:**
- `test_training()` — Runs `fine_tune_model` with minimal data
- `test_evaluation()` — Runs checkpoint evaluation
- `test_utilities()` — Tests utils (formatting, checkpoint paths, etc.)
- `test_extended_evaluation_metrics()` — BERTScore, hygiene, NLI
- `test_multigpu_evaluation()` — Model parallelism
- `test_error_handling()` — Invalid inputs, missing files
- `test_file_io_persistence()` — Results persistence, JSONL
- `test_monitor_script_integration()` — Early stopping signals
- `test_edge_cases()` — Empty datasets, long sequences
- `create_minimal_test_dataset()` — Creates temp JSONL for tests

**Local dependencies:** `wandb_finetune`, `model_configs`, `utils`

**Example:**
```bash
python test_e2e.py --model gemma-2b
```

---

### test_base_models.py

Queries base models (before fine-tuning) and saves predictions. Verifies chat templates and prompt formats work correctly.

**Important functions:**
- `load_base_model()` — Loads base model (no adapter), optional multi-GPU
- `test_model_generation()` — Generates predictions for test dataset
- `save_predictions()` — Saves to JSONL with timestamp

**Local dependencies:** `model_configs`, `utils.formatting`, `utils.dataset_loading`

**Example:**
```bash
python test_base_models.py --models normistral-7b-instruct,llama-3.1-8b-instruct --test_dataset data/val.jsonl --num_examples 10
```

---

### test_normistral_prompt.py

Tests prompt format for normistral-7b-instruct. Compares `apply_chat_template` vs manual formats.

**Important functions:**
- `test_normistral_prompt_format()` — Runs format tests, prints results

**Local dependencies:** None

**Example:**
```bash
python test_normistral_prompt.py
```

---

### check_checkpoint_batch_size.py

Inspects training arguments from a checkpoint. Reads `training_args.bin` and `trainer_state.json`.

**Important functions:**
- `check_checkpoint_batch_size()` — Prints per_device batch size, gradient accumulation, effective batch, learning rate, etc.

**Local dependencies:** `transformers`

**Example:**
```bash
python check_checkpoint_batch_size.py models/gemma-2-9b-apptainer-fsdp/checkpoint-5000
```

---

## Experimental / TODO

### capability_retention_evaluation.py

Placeholder for capability retention metrics: general-domain NLL, base-vs-tuned divergence, anchor-suite retention. Partially implemented (NLL, delta-NLL, data loading).

**Important functions:**
- `compute_sequence_nll()` — NLL for a single sequence
- `compute_average_delta_nll()` — Delta NLL (tuned − base)
- `load_political_retention_data()` — Loads capability retention datasets

**Local dependencies:** `transformers`, `torch`

---

## Archive

The `archive/` folder contains legacy scripts: `finetune.py`, `fsdp_finetune.py`, `evaluate_distributed_checkpoints.py`, etc. Kept for reference; not actively used.

---

## Running Scripts

Run from the project root or `model_fine_tuning_olivia/`:

```bash
cd model_fine_tuning_olivia
python scripts/wandb_finetune.py ...
```

Or from the scripts directory:

```bash
cd model_fine_tuning_olivia/scripts
python wandb_finetune.py ...
```
