#!/usr/bin/env bash
#SBATCH -J gpu-monitor-eval
#SBATCH -p accel
#SBATCH --gres=gpu:2  # Use fewer GPUs for evaluation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=48:00:00  # Long time to match training duration
#SBATCH -A nn12075k
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# ===== CONFIGURATION =====
# These should match your training script
MODEL=gemma-2-9b
OUTPUT_DIR="models/${MODEL}-apptainer-fsdp"
VAL_DATASET="data/output/new_processed_data_val.jsonl"
CHECK_INTERVAL=60  # Check every 60 seconds
EARLY_STOPPING_PATIENCE=3  # Stop if no improvement for 3 evaluations

# Load environment
source .env

# Apptainer image
export IMAGE=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif

# Run monitoring script
EVAL_CMD="export HF_HOME=\$HF_HOME && export HF_DATASETS_CACHE=\$HF_DATASETS_CACHE && export HF_MODULES_CACHE=\$HF_MODULES_CACHE && export HF_METRICS_CACHE=\$HF_METRICS_CACHE && export TOKENIZERS_PARALLELISM=false && python -m wandb login \$WANDB_API_KEY && python scripts/monitor_and_evaluate_checkpoints.py --output_dir $OUTPUT_DIR --model $MODEL --val_dataset $VAL_DATASET --hf_token \$HUGGINGFACE_TOKEN --check_interval $CHECK_INTERVAL --early_stopping_patience $EARLY_STOPPING_PATIENCE --wandb_project lm-finetuning --use_multi_gpu"

COMMAND="pip install --upgrade pip setuptools wheel > /dev/null 2>&1 && pip install --break-system-packages --only-binary=:all: transformers==4.45.2 tokenizers==0.20.3 datasets==3.0.0 sentencepiece==0.2.1 > /dev/null 2>&1 && pip install --break-system-packages --no-binary=rouge-score rouge-score==0.1.2 > /dev/null 2>&1 && pip install --break-system-packages --no-deps accelerate==1.0.1 peft==0.13.2 > /dev/null 2>&1 && pip install --break-system-packages --upgrade pyyaml psutil typing-extensions packaging==24.2 safetensors > /dev/null 2>&1 && pip install --break-system-packages --upgrade wandb evaluate > /dev/null 2>&1 && $EVAL_CMD"

apptainer exec --nv $IMAGE bash -c "$COMMAND"
