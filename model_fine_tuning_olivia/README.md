# Olivia Model Fine-Tuning

Fine-tuning Norwegian language models for **summarization of public documents** (Demokratibasen / UiB-Ide). This project trains and evaluates summarization models on Norwegian administrative texts such as meeting minutes, case presentations, and agendas.

## Overview

- **Task:** Abstractive summarization of Norwegian public documents
- **Training:** FSDP (Fully Sharded Data Parallel) fine-tuning with PEFT/LoRA
- **Evaluation:** ROUGE, BERTScore, hygiene metrics, and optional NLI-based faithfulness
- **Platform:** Designed for the Olivia HPC cluster (SLURM)

## Main Components

| Component | Purpose |
|-----------|---------|
| **Training** | `run_finetune_multinode.sbatch`, `scripts/distributed_finetune.py` |
| **In-training evaluation** | `run_monitor_evaluation.sbatch`, `scripts/monitor_and_evaluate_checkpoints.py` |
| **Post-training evaluation** | `run_evaluate_multiple.sh`, `run_evaluate_distributed_checkpoints_multigpu.sbatch` |
| **Metrics** | `scripts/summarisation_evaluation.py`, `scripts/capability_retention_evaluation.py` |
| **Model configs** | `scripts/model_configs.py` |

## Supported Models

Gemma (2B–27B), Viking (7B–33B), Normistral (7B, 11B), Llama (2/3), NorwAI Mistral, Norskgpt, NB-GPT-J, EuroLLM, and others. See `run_evaluate_multiple.sh --help` for the full list.

## Quick Start

**Training:**
```bash
sbatch --account=YOUR_PROJECT run_finetune_multinode.sbatch --model=gemma-2-9b
```

**Evaluation (models not monitored during training):**
```bash
./run_evaluate_multiple.sh --models="gemma-2-9b,normistral-7b" --account=YOUR_PROJECT
```

**Requirements:** `.env` with `WANDB_API_KEY`, `HUGGINGFACE_TOKEN`; `SLURM_ACCOUNT` for sbatch jobs.

## Dataset Format

Training and validation data are **JSONL** (one JSON object per line). Each line must have:

| Field | Required | Description |
|-------|----------|-------------|
| `input` | Yes | Document text to summarize |
| `output` | Yes | Reference summary (target) |
| `metadata` | No | Optional; may include `doc_type` (e.g. `vedtak`, `møtereferat`, `saksforelegg`) for prompt tuning |

Example:
```json
{"input": "Dette er dokumentteksten...", "output": "Dette er oppsummeringen...", "metadata": {"doc_type": "vedtak"}}
```

Dataset paths are passed via `--train_dataset` and `--val_dataset` (or `TRAIN_DATASET` / `VAL_DATASET` in sbatch). Current defaults are `data/dataset_149978_examples/149978_text_summary_examples_train.jsonl` and `data/dataset_149978_examples/149978_text_summary_examples_val.jsonl`, but any path can be used.

## Further Documentation

- `EVALUATION_TIME_ESTIMATES.md` — Metric timing and costs
- `scripts/README_TESTING.md` — Testing and e2e scripts
- `checkpoint_analyses/` — Checkpoint summaries and analysis reports
