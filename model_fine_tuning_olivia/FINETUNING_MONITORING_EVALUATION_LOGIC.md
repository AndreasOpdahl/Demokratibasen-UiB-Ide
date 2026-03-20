# Fine-Tuning, Monitoring, and Evaluation: Logic and Design

This document describes the logic of the fine-tuning, monitoring, and evaluation pipeline in plain language. It is intended for readers who want to understand what the system does, how the pieces fit together, and what options and edge cases exist—without needing to read code.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Fine-Tuning Logic](#2-fine-tuning-logic)
3. [Monitoring Logic](#3-monitoring-logic)
4. [Evaluation Logic](#4-evaluation-logic)
5. [How the Three Components Interact](#5-how-the-three-components-interact)
6. [Options and Alternatives](#6-options-and-alternatives)
7. [Edge Cases and Considerations](#7-edge-cases-and-considerations)
8. [File Reference](#8-file-reference)

---

## 1. High-Level Overview

The project fine-tunes Norwegian language models for **summarization of public documents** (e.g., meeting minutes, case presentations). The pipeline has three main components:

| Component | Purpose |
|-----------|---------|
| **Fine-tuning** | Trains the model on document–summary pairs. Saves checkpoints (snapshots of the model) at regular intervals. |
| **Monitoring** | Runs in parallel with training. Watches for new checkpoints, evaluates each one, and can signal the trainer to stop early if quality stops improving. |
| **Evaluation** | Measures how well a checkpoint performs: ROUGE (overlap with reference summaries), BERTScore (semantic similarity), hygiene (repetition, length), and optionally NLI faithfulness (does the summary follow the source?). |

**Typical workflow:**
- Start training (e.g., via `run_finetune_multinode.sbatch`).
- Optionally start the monitor (e.g., via `run_monitor_evaluation.sbatch`) to evaluate checkpoints as they appear and enable early stopping.
- For models trained without monitoring, run evaluation afterward (e.g., via `run_evaluate_multiple.sh`).

---

## 2. Fine-Tuning Logic

### What Happens

1. **Load the base model** (e.g., Gemma, Normistral) from Hugging Face.
2. **Apply LoRA** (Low-Rank Adaptation): only a small set of parameters is trained, reducing memory and time.
3. **Load training and validation data** from JSONL files (one document–summary pair per line).
4. **Format examples** using model-specific prompts (e.g., chat format for Mistral, plain text for Gemma).
5. **Train** for a fixed number of steps or epochs. At intervals (e.g., every 100 steps), save a checkpoint.
6. **Back up checkpoints** to `regular_checkpoints/` and `major_checkpoints/` so they are not lost when the main directory is pruned.
7. **Check for early stopping** periodically: if the monitor has written a signal file, training stops.

### Important Concepts

- **Checkpoint:** A saved snapshot of the model at a given training step (e.g., step 500, 1000).
- **Regular vs. major checkpoints:** Every Nth step (default 500) is a “major” checkpoint. Major checkpoints get BERTScore evaluation; regular ones get ROUGE and hygiene only (BERTScore is slower).
- **Prompt masking:** During training, the loss is computed only on the summary tokens, not on the prompt. This focuses learning on the summarization task.
- **FSDP vs. DDP:** Two ways to use multiple GPUs. FSDP shards the model across GPUs (saves memory); DDP replicates it. FSDP is preferred for large models.

### Options (Command-Line / Configuration)

| Option | Effect |
|--------|--------|
| `--model` | Which base model to fine-tune (e.g., gemma-2-9b, normistral-7b). |
| `--quantization` | `none` (full precision), `4bit`, or `8bit`. Quantization reduces memory but is not recommended with multi-GPU. |
| `--train_dataset` / `--val_dataset` | Paths to training and validation JSONL files. |
| `--max_steps` | Stop after N training steps (overrides epochs if set). |
| `--num_train_epochs` | Stop after N passes over the training data. |
| `--resume_checkpoint` | Resume from a saved checkpoint (path or `"latest"`). |
| `--force_restart` | Ignore existing checkpoints and start from scratch. |
| `--fsdp` / `--ddp` | Use FSDP or DDP for multi-GPU training. |
| `--train_batch_size` / `--val_batch_size` | Batch sizes per GPU (defaults come from model config). |
| `--val_steps` | Save a checkpoint and run validation every N steps. |

### Edge Cases

- **Quantization + multi-GPU:** Not well supported; use single GPU or no quantization.
- **Resuming with FSDP/DDP + LoRA:** PEFT limitations mean resuming from checkpoints in distributed mode is unreliable; the system may ignore resume and start fresh.
- **Git LFS pointers:** If the dataset file is a Git LFS pointer (tiny file) instead of real data, loading fails with a clear error. Run `git lfs pull` to fetch the data.
- **Missing or invalid data:** Examples with missing `input` or `output` are filtered out before training.

---

## 3. Monitoring Logic

### What Happens

1. **Wait for training to start:** The monitor looks for a `training_started.txt` file in the output directory. It waits up to 1 hour by default.
2. **Poll for checkpoints:** Every N seconds (default 60), it lists checkpoint directories (main + backups).
3. **Pick the next unevaluated checkpoint:** It evaluates checkpoints in order (oldest first). Skips checkpoints that already have evaluation results.
4. **Ensure checkpoint is stable:** Waits until the checkpoint has not been modified for a set time (default 120 seconds) to avoid evaluating incomplete checkpoints.
5. **Run evaluation:** Calls the evaluation script on the checkpoint.
6. **Update best metrics:** Tracks the best ROUGE-Lsum and BERTScore.
7. **Decide on early stopping:** If there is no improvement for several evaluations, or if ROUGE is zero for several checkpoints, or if BERTScore is very low, it writes a `.early_stop` file.
8. **Stop when training completes:** Stops when it sees a `.training_complete` file.

### Important Concepts

- **Checkpoint stability:** The trainer may still be writing files when a checkpoint first appears. The monitor waits until the checkpoint is “stable” (unchanged for a configurable period) before evaluating.
- **Backup preference:** If a checkpoint exists in both the main directory and a backup (`regular_checkpoints/` or `major_checkpoints/`), the monitor prefers the backup for evaluation (it is guaranteed stable).
- **Continue runs:** When resuming training (e.g., from step 1000 to 10000), the monitor skips re-evaluating old checkpoints (e.g., 100–900) that were already evaluated in earlier runs.

### Options

| Option | Effect |
|--------|--------|
| `--check_interval` | How often to check for new checkpoints (seconds). Default: 60. |
| `--early_stopping_patience` | Stop if no improvement for N evaluations. Default: 3. |
| `--timeout_minutes` | Stop if no new checkpoints appear for N minutes. Default: 30. |
| `--include_nli_faithfulness` | Run NLI faithfulness evaluation (slow, ~37 min for 500 examples). Default: off. |
| `--val_data_size` | Number of validation examples (500 or 1000). 1000 uses more data and writes to separate files (e.g. `-examples_1000.json`). |
| `--checkpoint_stability_seconds` | Wait until checkpoint has not been modified for this many seconds. Default: 120. |

### Early Stopping Triggers

The monitor can signal early stopping in these cases:

| Trigger | Condition | Action |
|---------|-----------|--------|
| No improvement | ROUGE-Lsum has not improved for `early_stopping_patience` evaluations | Write `.early_stop` |
| Zero ROUGE | All ROUGE scores are 0.00 for 5 consecutive checkpoints (model collapse) | Write `.early_stop` |
| Low BERTScore | BERTScore F1 &lt; 0.25 for 2 consecutive major checkpoints | Write `.early_stop` |
| Timeout | No new checkpoints for `timeout_minutes` minutes | Stop monitor |
| Training complete | `.training_complete` file exists | Stop monitor |

### Edge Cases

- **Monitor starts before training:** Waits up to 1 hour for `training_started.txt`. If training never starts, the monitor exits with an error.
- **Stale evaluation results:** If evaluation results are older than the checkpoint (e.g., from a previous run), the monitor re-evaluates.
- **Checkpoint deleted before evaluation:** If the main checkpoint was pruned and the backup is missing, the monitor skips and marks it as evaluated to avoid infinite retries.
- **Adapter files missing:** A checkpoint must have `adapter_model.safetensors`. If it is incomplete or cleaned up, the monitor skips it.

---

## 4. Evaluation Logic

### What Happens

1. **Load the base model** and the PEFT adapter from the checkpoint directory.
2. **Load validation data** and sample a fixed number of examples (default 500) with a fixed seed for reproducibility.
3. **Format prompts** using the same model-specific format as training.
4. **Generate summaries** for each validation example (beam search or greedy decoding).
5. **Compute metrics:**
   - **ROUGE:** Overlap with reference summaries (fast).
   - **Hygiene:** Repetition, compression ratio, punctuation (fast).
   - **BERTScore:** Semantic similarity (only on major checkpoints; ~1.5–2 min for 500 examples).
   - **NLI faithfulness:** Optional; checks if summary sentences are entailed by the source (slow; ~37 min for 500 examples on a fixed subset).
6. **Save results** to `all_eval_results/` (per-checkpoint JSON and summary file).

### Important Concepts

- **Model parallelism:** For large models, evaluation can split the model across multiple GPUs (`device_map="auto"`). This avoids FSDP/DDP issues with generation.
- **Fixed NLI subset:** NLI faithfulness runs on a fixed subset (default 500 examples) with a fixed seed. The same subset is used for all checkpoints so results are comparable.
- **Major vs. normal checkpoints:** BERTScore is computed only for major checkpoints (every 500 steps), to save time and cost.

### Options

| Option | Effect |
|--------|--------|
| `--checkpoint_dir` | Path to the checkpoint to evaluate. |
| `--val_dataset` | Path to validation JSONL. |
| `--val_data_size` | Number of validation examples (500 or 1000). |
| `--use_multi_gpu` | Use model parallelism across GPUs. |
| `--use_greedy` | Use greedy decoding instead of beam search (faster, slightly lower quality). |
| `--keep_existing` | Skip checkpoints that already have evaluation results. Default: on when used from `run_evaluate_multiple.sh`. |
| `--force_recompute` | Re-evaluate even when results exist (overwrites). |
| `--include_nli_faithfulness` | Run NLI faithfulness (slow). |
| `--major_checkpoint_interval` | Every Nth step is major (default: 500). |
| `--wandb_project` / `--wandb_disabled` | Log to WandB or disable logging. |

### Edge Cases

- **Checkpoint already evaluated:** With `--keep_existing`, the script skips and exits successfully (so batch scripts can continue).
- **Missing adapter files:** If `adapter_model.safetensors` is missing, evaluation fails (checkpoint may be incomplete or cleaned up).
- **Extended evaluation failed:** If BERTScore or NLI fails, the script continues with ROUGE and hygiene only and logs the error.
- **Different evaluation sizes:** 500 vs 1000 examples are written to different files (e.g. `checkpoint-500-eval-results.json` vs `checkpoint-500-eval-results-examples_1000.json`).

---

## 5. How the Three Components Interact

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING (wandb_finetune.py)                     │
│  • Creates training_started.txt at start                                │
│  • Saves checkpoints every val_steps (e.g. 100)                         │
│  • Backs up to regular_checkpoints/ and major_checkpoints/              │
│  • Checks for .early_stop every 100 steps → stops if present            │
│  • Creates .training_complete at end                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │  (checkpoints)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    MONITORING (monitor_and_evaluate_checkpoints.py)     │
│  • Waits for training_started.txt                                        │
│  • Polls for new checkpoints                                             │
│  • Evaluates each checkpoint (calls evaluate_checkpoint)                 │
│  • Writes .early_stop if no improvement / zero ROUGE / low BERTScore     │
│  • Stops when .training_complete or timeout                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │  (evaluate_checkpoint)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              EVALUATION (evaluate_distributed_checkpoints_multigpu.py)  │
│  • Loads model + adapter, generates predictions                         │
│  • Computes ROUGE, hygiene, BERTScore (major), NLI (optional)            │
│  • Saves to all_eval_results/                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

**Signal files:**

| File | Written by | Read by |
|------|------------|---------|
| `training_started.txt` | Training | Monitor (waits for it) |
| `.early_stop` | Monitor | Training (stops when seen) |
| `.training_complete` | Training | Monitor (stops when seen) |

---

## 6. Options and Alternatives

### Training Modes

| Mode | Use case |
|------|----------|
| Single GPU | Small models, quick experiments, or when using quantization. |
| Multi-GPU DDP | Full model replication; faster for smaller models. |
| Multi-GPU FSDP | Large models; shards model across GPUs to fit memory. |

### Evaluation Modes

| Mode | Use case |
|------|----------|
| **In-training (monitor)** | Evaluate checkpoints as they appear; enable early stopping. |
| **Post-training (run_evaluate_multiple.sh)** | Evaluate all checkpoints for models that were not monitored. |
| **Single checkpoint** | Evaluate one checkpoint manually (e.g., for debugging). |

### Task Limits

| Option | Effect |
|--------|--------|
| `--max_steps N` | Stop after N training steps (common for reproducibility). |
| `--num_train_epochs N` | Stop after N passes over the training data. |

### Validation Data Size

| Size | Effect |
|------|--------|
| 500 | Default; faster; results in `checkpoint-N-eval-results.json`. |
| 1000 | More examples; results in `checkpoint-N-eval-results-examples_1000.json`. The first 500 are the same subset for NLI comparability. |

---

## 7. Edge Cases and Considerations

### Checkpoint Management

- **Backup vs. main:** When the main directory is pruned (e.g., only 10 checkpoints kept), older checkpoints remain in `regular_checkpoints/` and `major_checkpoints/`. The monitor and evaluation scripts look in both.
- **Naming:** `checkpoint-500` and `regular-checkpoint-500` / `major-checkpoint-500` refer to the same step; the system normalizes names internally.
- **Resume from backup:** If the main checkpoint was deleted, evaluation can still use the backup if it exists.

### Stale or Corrupted Results

- **Re-evaluation:** Use `--force_recompute` when running the evaluation script (or `run_evaluate_multiple.sh --force_recompute`) to overwrite existing results (e.g., after fixing prompts or config). The monitor automatically re-evaluates when it detects stale results (eval older than checkpoint or than `training_started.txt`).
- **Stale from previous run:** The monitor compares eval file timestamps with `training_started.txt`. If eval is older, it re-evaluates.

### Time and Cost

| Metric | Approximate time (500 examples) |
|--------|--------------------------------|
| ROUGE + hygiene | &lt; 1 second |
| BERTScore | ~1.5–2 minutes |
| NLI faithfulness | ~37 minutes |

Recommendation: Use NLI only when needed (e.g., final evaluation or major checkpoints); keep it off for routine monitoring.

### Resource Requirements

- **Training:** Typically 4 GPUs, 512 GB RAM, 25+ hours for 10k steps.
- **Monitoring:** 2 GPUs, 192 GB RAM; runs in parallel with training.
- **Post-training evaluation:** One job per model; can run many checkpoints sequentially.

### Job Dependencies

- **Recommended:** Submit the monitor with `--dependency=afterok:TRAINING_JOB_ID` so it starts only after training starts.
- **Alternative:** Submit both; the monitor waits up to 1 hour for `training_started.txt`.

---

## 8. File Reference

| File | Role |
|------|------|
| `scripts/wandb_finetune.py` | Main training script; LoRA, checkpoint backup, early stopping check. |
| `scripts/monitor_and_evaluate_checkpoints.py` | Polls for checkpoints, evaluates them, writes early stopping signal. |
| `scripts/evaluate_distributed_checkpoints_multigpu.py` | Evaluates a single checkpoint; ROUGE, BERTScore, hygiene, NLI. |
| `run_finetune_multinode.sbatch` | SLURM job script for training. |
| `run_monitor_evaluation.sbatch` | SLURM job script for the monitor. |
| `run_evaluate_multiple.sh` | Batch evaluation of multiple models; submits one job per model. |
| `run_evaluate_distributed_checkpoints_multigpu.sbatch` | SLURM job script for evaluation. |
| `scripts/model_configs.py` | Model-specific settings (LoRA, prompts, batch sizes). |
| `scripts/summarisation_evaluation.py` | Metric implementations (ROUGE, BERTScore, hygiene, NLI). |
| `scripts/utils/` | Shared utilities (formatting, tokenization, checkpoint paths, etc.). |

---

## 9. Summary

- **Fine-tuning** trains the model, saves checkpoints, and backs them up. It can stop early if the monitor writes `.early_stop`.
- **Monitoring** evaluates checkpoints as they appear, tracks best metrics, and triggers early stopping when quality stops improving or when the model collapses.
- **Evaluation** measures checkpoint quality with ROUGE, BERTScore, hygiene, and optionally NLI. It can run in training (via monitor) or afterward (via `run_evaluate_multiple.sh`).

The pipeline is designed for the Olivia HPC cluster (SLURM) but can be run manually. All paths are configurable; dataset paths are not hardcoded. For timing details, see `EVALUATION_TIME_ESTIMATES.md`.
