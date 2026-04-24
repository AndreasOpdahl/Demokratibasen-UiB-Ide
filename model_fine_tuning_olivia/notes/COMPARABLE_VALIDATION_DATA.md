# Validation Data Consistency Report

**Generated**: 2026-03-24  
**Data location**: `~/OneDrive/Shared/Demokratibasen-UiB-Ide/TrainingRuns/olivia/ajay_finetunes/<model>/all_eval_results`

## Executive Summary

The 500-example evaluations used **different random validation subsets** for nearly every checkpoint due to a bug (unseeded `random.sample`) that was fixed on 2026-03-17 (commit `6565b45d`). The 1000-example evaluations, run later with the fixed code, are **fully consistent** across all 14 models that have them, sharing the same fingerprint (`e4d0785f315c`).

**Recommendation**: Re-run all missing 1000-example evaluations using `--val_data_size=1000 --keep_existing` so every checkpoint is evaluated on the same canonical dataset. The 500-example results are not directly comparable across checkpoints.

## Root Cause

Before commit `6565b45d` (2026-03-17), the evaluation script sampled validation data with:

```python
val_data = random.sample(val_data, min(val_data_size, len(val_data)))
```

Without calling `random.seed()` first. Since the sbatch loop spawns a **fresh Python process** (via `apptainer exec`) for each checkpoint, every process got a different OS-entropy seed and therefore a different 500-example subset.

The fix introduced `sample_validation_data_reproducibly()` with a fixed seed (42).

## 1000-Example Evaluations (Consistent)

All 14 models with 1000-example results share fingerprint `e4d0785f315c`. These results are **directly comparable** across checkpoints and across models.

| Model | 1000-eval checkpoints | Coverage |
|-------|----------------------|----------|
| nb-gpt-j-6b | 67 | **Complete** |
| normistral-7b | 50 | **Complete** |
| normistral-7b-instruct | 50 | **Complete** |
| eurollm-9b-instruct | 10 / 50 | Partial |
| gemma-2-9b | 36 / 94 | Partial |
| gemma-2b | 19 / 46 | Partial |
| gemma-7b | 18 / 62 | Partial |
| llama-2-13b-chat-norwegian | 15 / 100 | Partial |
| llama-3.1-8b-instruct | 18 / 100 | Partial |
| normistral-11b | 22 / 100 | Partial |
| norskgpt-llama3-8b | 13 / 50 | Partial |
| norwai-mistral-7b-instruct | 32 / 100 | Partial |
| viking-13b | 15 / 50 | Partial |
| viking-7b | 30 / 50 | Partial |
| gemma-2-27b | 0 / 27 | **None** |
| viking-33b | 0 / 7 | **None** |

## 500-Example Evaluations (Inconsistent)

The 500-example results split into two main fingerprint groups plus 3 unique outliers, none of which match across all models:

- **Fingerprint `f06171fdd0e9`** (7 models): gemma-2-9b, gemma-2b, llama-2-13b-chat-norwegian, llama-3.1-8b-instruct, nb-gpt-j-6b, normistral-11b, norwai-mistral-7b-instruct
- **Fingerprint `b8f228a85fde`** (6 models): eurollm-9b-instruct, gemma-2-27b, gemma-7b, normistral-7b-instruct, viking-13b, viking-33b
- **Unique fingerprints**: normistral-7b, norskgpt-llama3-8b, viking-7b

Within each model, most 500-example checkpoints have unique (non-comparable) subsets. Only a minority belong to the largest consistent group.

## Checkpoints Needing 1000-Example Re-Evaluation

Total: **628 checkpoint evaluations** across 13 models. Three models are already complete.

### Already complete (0 re-runs needed)

- **nb-gpt-j-6b** (67/67)
- **normistral-7b** (50/50)
- **normistral-7b-instruct** (50/50)

### Partially complete

| Model | Missing | Steps to evaluate |
|-------|---------|-------------------|
| eurollm-9b-instruct | 40 | 100 200 300 400 600 700 800 900 1100 1200 1300 1400 1600 1700 1800 1900 2100 2200 2300 2400 2600 2700 2800 2900 3100 3200 3300 3400 3600 3700 3800 3900 4100 4200 4300 4400 4600 4700 4800 4900 |
| gemma-2-9b | 58 | 400 600 700 800 900 3600 3700 3800 3900 4100 4200 4300 4400 4600 4700 4800 4900 5100 5200 5300 5400 5600 5700 5800 5900 6100 6200 6300 6400 6600 6700 6800 6900 7000 7100 7200 7300 7400 7500 7600 7700 7800 7900 8000 8100 8200 8300 8400 8500 8600 8700 8800 8900 9000 9100 9200 9300 9400 |
| gemma-2b | 27 | 200 300 400 5100 5200 5300 5400 5500 5600 5700 5800 5900 6000 6100 6200 6300 6400 6500 6600 6700 6800 6900 7000 7100 7200 7300 7400 |
| gemma-7b | 44 | 200 300 400 600 700 800 900 1700 1800 1900 2100 2200 2300 2400 2600 2700 2800 2900 3100 3200 3300 3400 3600 3700 3800 3900 4100 4200 4300 4400 4600 4700 4800 4900 5100 5200 5300 5400 5600 5700 5800 5900 6100 6200 |
| llama-2-13b-chat-norwegian | 85 | 200 300 400 600 700 800 900 1600 1700 1800 1900 2100 2200 2300 2400 2600 2700 2800 2900 3100 3200 3300 3400 3600 3700 3800 3900 4100 4200 4300 4400 4600 4700 4800 4900 5100 5200 5300 5400 5500 5600 5700 5800 5900 6000 6100 6200 6300 6400 6500 6600 6700 6800 6900 7000 7100 7200 7300 7400 7500 7600 7700 7800 7900 8000 8100 8200 8300 8400 8500 8600 8700 8800 8900 9000 9100 9200 9300 9400 9500 9600 9700 9800 9900 10000 |
| llama-3.1-8b-instruct | 82 | 200 300 400 600 700 800 900 1900 2100 2200 2300 2400 2600 2700 2800 2900 3100 3200 3300 3400 3600 3700 3800 3900 4100 4200 4300 4400 4600 4700 4800 4900 5100 5200 5300 5400 5500 5600 5700 5800 5900 6000 6100 6200 6300 6400 6500 6600 6700 6800 6900 7000 7100 7200 7300 7400 7500 7600 7700 7800 7900 8000 8100 8200 8300 8400 8500 8600 8700 8800 8900 9000 9100 9200 9300 9400 9500 9600 9700 9800 9900 10000 |
| normistral-11b | 78 | 300 400 600 700 800 900 2300 2400 2600 2700 2800 2900 3100 3200 3300 3400 3600 3700 3800 3900 4100 4200 4300 4400 4600 4700 4800 4900 5100 5200 5300 5400 5500 5600 5700 5800 5900 6000 6100 6200 6300 6400 6500 6600 6700 6800 6900 7000 7100 7200 7300 7400 7500 7600 7700 7800 7900 8000 8100 8200 8300 8400 8500 8600 8700 8800 8900 9000 9100 9200 9300 9400 9500 9600 9700 9800 9900 10000 |
| norskgpt-llama3-8b | 37 | 200 300 400 600 700 800 900 1300 1400 1600 1700 1800 1900 2100 2200 2300 2400 2600 2700 2800 2900 3100 3200 3300 3400 3600 3700 3800 3900 4100 4200 4300 4400 4600 4700 4800 4900 |
| norwai-mistral-7b-instruct | 68 | 400 600 700 800 900 3400 3600 3700 3800 3900 4100 4200 4300 4400 4600 4700 4800 4900 5100 5200 5300 5400 5500 5600 5700 5800 5900 6000 6100 6200 6300 6400 6500 6600 6700 6800 6900 7000 7100 7200 7300 7400 7500 7600 7700 7800 7900 8000 8100 8200 8300 8400 8500 8600 8700 8800 8900 9000 9100 9200 9300 9400 9500 9600 9700 9800 9900 10000 |
| viking-13b | 35 | 200 300 400 600 700 800 900 1600 1700 1800 1900 2100 2200 2300 2400 2600 2700 2800 2900 3100 3200 3300 3400 3600 3700 3800 3900 4100 4200 4300 4400 4600 4700 4800 4900 |
| viking-7b | 20 | 400 600 700 800 900 3200 3300 3400 3600 3700 3800 3900 4100 4200 4300 4400 4600 4700 4800 4900 |

### No 1000-example evaluations at all

| Model | Total checkpoints | Steps to evaluate |
|-------|-------------------|-------------------|
| gemma-2-27b | 27 | 1100 1200 1300 1400 2000 2100 2200 2300 2400 2500 2600 2700 2800 2900 3000 3100 3200 4100 4200 4300 4400 4500 4600 4700 4800 4900 5000 |
| viking-33b | 7 | 1100 2100 2200 2300 2400 2500 2600 |

## Commands to Complete All 1000-Example Evaluations

All commands use `--val_data_size=1000 --keep_existing` to avoid re-running checkpoints that already have 1000-example results. Run from the working directory on Olivia.

```bash
# eurollm-9b-instruct (40 missing)
sbatch --account=YOUR_ACCOUNT run_evaluate_distributed_checkpoints_multigpu.sbatch \
  --model=eurollm-9b-instruct \
  --val_data_size=1000 --keep_existing \
  --specific_checkpoints="100 200 300 400 600 700 800 900 1100 1200 1300 1400 1600 1700 1800 1900 2100 2200 2300 2400 2600 2700 2800 2900 3100 3200 3300 3400 3600 3700 3800 3900 4100 4200 4300 4400 4600 4700 4800 4900"

# gemma-2-9b (58 missing)
sbatch --account=YOUR_ACCOUNT run_evaluate_distributed_checkpoints_multigpu.sbatch \
  --model=gemma-2-9b \
  --val_data_size=1000 --keep_existing \
  --specific_checkpoints="400 600 700 800 900 3600 3700 3800 3900 4100 4200 4300 4400 4600 4700 4800 4900 5100 5200 5300 5400 5600 5700 5800 5900 6100 6200 6300 6400 6600 6700 6800 6900 7000 7100 7200 7300 7400 7500 7600 7700 7800 7900 8000 8100 8200 8300 8400 8500 8600 8700 8800 8900 9000 9100 9200 9300 9400"

# gemma-2-27b (27 missing -- no 1000-example results at all)
sbatch --account=YOUR_ACCOUNT run_evaluate_distributed_checkpoints_multigpu.sbatch \
  --model=gemma-2-27b \
  --val_data_size=1000 --keep_existing \
  --specific_checkpoints="1100 1200 1300 1400 2000 2100 2200 2300 2400 2500 2600 2700 2800 2900 3000 3100 3200 4100 4200 4300 4400 4500 4600 4700 4800 4900 5000"

# gemma-2b (27 missing)
sbatch --account=YOUR_ACCOUNT run_evaluate_distributed_checkpoints_multigpu.sbatch \
  --model=gemma-2b \
  --val_data_size=1000 --keep_existing \
  --specific_checkpoints="200 300 400 5100 5200 5300 5400 5500 5600 5700 5800 5900 6000 6100 6200 6300 6400 6500 6600 6700 6800 6900 7000 7100 7200 7300 7400"

# gemma-7b (44 missing)
sbatch --account=YOUR_ACCOUNT run_evaluate_distributed_checkpoints_multigpu.sbatch \
  --model=gemma-7b \
  --val_data_size=1000 --keep_existing \
  --specific_checkpoints="200 300 400 600 700 800 900 1700 1800 1900 2100 2200 2300 2400 2600 2700 2800 2900 3100 3200 3300 3400 3600 3700 3800 3900 4100 4200 4300 4400 4600 4700 4800 4900 5100 5200 5300 5400 5600 5700 5800 5900 6100 6200"

# llama-2-13b-chat-norwegian (85 missing)
sbatch --account=YOUR_ACCOUNT run_evaluate_distributed_checkpoints_multigpu.sbatch \
  --model=llama-2-13b-chat-norwegian \
  --val_data_size=1000 --keep_existing \
  --specific_checkpoints="200 300 400 600 700 800 900 1600 1700 1800 1900 2100 2200 2300 2400 2600 2700 2800 2900 3100 3200 3300 3400 3600 3700 3800 3900 4100 4200 4300 4400 4600 4700 4800 4900 5100 5200 5300 5400 5500 5600 5700 5800 5900 6000 6100 6200 6300 6400 6500 6600 6700 6800 6900 7000 7100 7200 7300 7400 7500 7600 7700 7800 7900 8000 8100 8200 8300 8400 8500 8600 8700 8800 8900 9000 9100 9200 9300 9400 9500 9600 9700 9800 9900 10000"

# llama-3.1-8b-instruct (82 missing)
sbatch --account=YOUR_ACCOUNT run_evaluate_distributed_checkpoints_multigpu.sbatch \
  --model=llama-3.1-8b-instruct \
  --val_data_size=1000 --keep_existing \
  --specific_checkpoints="200 300 400 600 700 800 900 1900 2100 2200 2300 2400 2600 2700 2800 2900 3100 3200 3300 3400 3600 3700 3800 3900 4100 4200 4300 4400 4600 4700 4800 4900 5100 5200 5300 5400 5500 5600 5700 5800 5900 6000 6100 6200 6300 6400 6500 6600 6700 6800 6900 7000 7100 7200 7300 7400 7500 7600 7700 7800 7900 8000 8100 8200 8300 8400 8500 8600 8700 8800 8900 9000 9100 9200 9300 9400 9500 9600 9700 9800 9900 10000"

# normistral-11b (78 missing)
sbatch --account=YOUR_ACCOUNT run_evaluate_distributed_checkpoints_multigpu.sbatch \
  --model=normistral-11b \
  --val_data_size=1000 --keep_existing \
  --specific_checkpoints="300 400 600 700 800 900 2300 2400 2600 2700 2800 2900 3100 3200 3300 3400 3600 3700 3800 3900 4100 4200 4300 4400 4600 4700 4800 4900 5100 5200 5300 5400 5500 5600 5700 5800 5900 6000 6100 6200 6300 6400 6500 6600 6700 6800 6900 7000 7100 7200 7300 7400 7500 7600 7700 7800 7900 8000 8100 8200 8300 8400 8500 8600 8700 8800 8900 9000 9100 9200 9300 9400 9500 9600 9700 9800 9900 10000"

# norskgpt-llama3-8b (37 missing)
sbatch --account=YOUR_ACCOUNT run_evaluate_distributed_checkpoints_multigpu.sbatch \
  --model=norskgpt-llama3-8b \
  --val_data_size=1000 --keep_existing \
  --specific_checkpoints="200 300 400 600 700 800 900 1300 1400 1600 1700 1800 1900 2100 2200 2300 2400 2600 2700 2800 2900 3100 3200 3300 3400 3600 3700 3800 3900 4100 4200 4300 4400 4600 4700 4800 4900"

# norwai-mistral-7b-instruct (68 missing)
sbatch --account=YOUR_ACCOUNT run_evaluate_distributed_checkpoints_multigpu.sbatch \
  --model=norwai-mistral-7b-instruct \
  --val_data_size=1000 --keep_existing \
  --specific_checkpoints="400 600 700 800 900 3400 3600 3700 3800 3900 4100 4200 4300 4400 4600 4700 4800 4900 5100 5200 5300 5400 5500 5600 5700 5800 5900 6000 6100 6200 6300 6400 6500 6600 6700 6800 6900 7000 7100 7200 7300 7400 7500 7600 7700 7800 7900 8000 8100 8200 8300 8400 8500 8600 8700 8800 8900 9000 9100 9200 9300 9400 9500 9600 9700 9800 9900 10000"

# viking-13b (35 missing)
sbatch --account=YOUR_ACCOUNT run_evaluate_distributed_checkpoints_multigpu.sbatch \
  --model=viking-13b \
  --val_data_size=1000 --keep_existing \
  --specific_checkpoints="200 300 400 600 700 800 900 1600 1700 1800 1900 2100 2200 2300 2400 2600 2700 2800 2900 3100 3200 3300 3400 3600 3700 3800 3900 4100 4200 4300 4400 4600 4700 4800 4900"

# viking-33b (7 missing -- no 1000-example results at all)
sbatch --account=YOUR_ACCOUNT run_evaluate_distributed_checkpoints_multigpu.sbatch \
  --model=viking-33b \
  --val_data_size=1000 --keep_existing \
  --specific_checkpoints="1100 2100 2200 2300 2400 2500 2600"

# viking-7b (20 missing)
sbatch --account=YOUR_ACCOUNT run_evaluate_distributed_checkpoints_multigpu.sbatch \
  --model=viking-7b \
  --val_data_size=1000 --keep_existing \
  --specific_checkpoints="400 600 700 800 900 3200 3300 3400 3600 3700 3800 3900 4100 4200 4300 4400 4600 4700 4800 4900"
```

### Models already complete (no action needed)

```bash
# nb-gpt-j-6b:          67/67 checkpoints have 1000-example eval
# normistral-7b:        50/50 checkpoints have 1000-example eval
# normistral-7b-instruct: 50/50 checkpoints have 1000-example eval
```

## Verification

After re-running, verify consistency with:

```bash
python model_fine_tuning_olivia/scripts/test_inputs_refs_consistency.py \
  --results_dir <model_all_eval_results_dir> \
  --checkpoints <step1> <step2> <step3> ... \
  --examples_suffix examples_1000
```
