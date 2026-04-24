#!/usr/bin/env bash
#SBATCH -J regen-bad-hygiene
#SBATCH -p accel
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
# Account: REQUIRED - Pass via: sbatch --account=YOUR_ACCOUNT run_regenerate_bad_hygiene_predictions.sbatch ...
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# Slurm stages scripts under /var/spool; use submit dir when available.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SBATCH_DIR="$SLURM_SUBMIT_DIR"
else
    SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SBATCH_DIR" || { echo "ERROR: Cannot cd to $SBATCH_DIR"; exit 1; }
echo "Working directory for this job: $SBATCH_DIR"

if [ -f .env ]; then
    source .env
else
    echo "WARNING: .env file not found. Using environment variables/defaults."
fi

# ===== CONFIGURATION =====
MODEL=${MODEL:-llama-2-13b-chat-norwegian}
NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}
DISTRIBUTION_FLAG=${DISTRIBUTION_FLAG:-"--fsdp"}   # kept for interface compatibility
PRED_BATCH_SIZE=${PRED_BATCH_SIZE:-8}
EXAMPLES_SUFFIX=${EXAMPLES_SUFFIX:-1000-examples}
MAJOR_INTERVAL=${MAJOR_INTERVAL:-500}
HYGIENE_FOLDER=${HYGIENE_FOLDER:-hygiene_filtering/${MODEL}-apptainer-fsdp}
MODEL_FOLDER=${MODEL_FOLDER:-models/${MODEL}-apptainer-fsdp}
MIN_NEW_TOKENS=${MIN_NEW_TOKENS:-40}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-180}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-2048}
NUM_BEAMS=${NUM_BEAMS:-1}
NO_LORA=${NO_LORA:-false}
OVERWRITE=${OVERWRITE:-false}
DRY_RUN=${DRY_RUN:-false}

declare -a PRED_DATASETS=()

for arg in "$@"; do
    case $arg in
        --model=*)
            MODEL="${arg#*=}"
            ;;
        --num_gpus=*)
            NUM_GPUS="${arg#*=}"
            ;;
        --distribution_flag=*)
            DISTRIBUTION_FLAG="${arg#*=}"
            ;;
        --pred_dataset=*)
            PRED_DATASETS+=("${arg#*=}")
            ;;
        --pred_batch_size=*)
            PRED_BATCH_SIZE="${arg#*=}"
            ;;
        --examples_suffix=*)
            EXAMPLES_SUFFIX="${arg#*=}"
            ;;
        --major_interval=*)
            MAJOR_INTERVAL="${arg#*=}"
            ;;
        --hygiene_folder=*)
            HYGIENE_FOLDER="${arg#*=}"
            ;;
        --model_folder=*)
            MODEL_FOLDER="${arg#*=}"
            ;;
        --min_new_tokens=*)
            MIN_NEW_TOKENS="${arg#*=}"
            ;;
        --max_new_tokens=*)
            MAX_NEW_TOKENS="${arg#*=}"
            ;;
        --max_input_tokens=*)
            MAX_INPUT_TOKENS="${arg#*=}"
            ;;
        --num_beams=*)
            NUM_BEAMS="${arg#*=}"
            ;;
        --no_lora)
            NO_LORA=true
            ;;
        --overwrite)
            OVERWRITE=true
            ;;
        --dry-run|--dry_run|--debug)
            DRY_RUN=true
            ;;
        *)
            echo "ERROR: Unknown argument: $arg"
            echo ""
            echo "Usage: sbatch --account=YOUR_PROJECT_ID run_regenerate_bad_hygiene_predictions.sbatch [OPTIONS]"
            echo ""
            echo "Input selection (choose one mode):"
            echo "  Mode A (scan folder):"
            echo "    --hygiene_folder=DIR           Specific model hygiene folder (must contain all_eval_results)"
            echo "  Mode B (explicit files):"
            echo "    --pred_dataset=PATH            Path to checkpoint-<N>-bad-hygiene-<suffix>.jsonl (repeatable)"
            echo ""
            echo "Optional options:"
            echo "  --model=MODEL                    Model short name filter (default: llama-2-13b-chat-norwegian)"
            echo "  --num_gpus=N                     Number of GPUs (default: 1)"
            echo "  --distribution_flag=FLAG         Kept for interface compatibility (--fsdp or --ddp)"
            echo "  --pred_batch_size=N              Regeneration batch size (default: 8)"
            echo "  --examples_suffix=S              Hygiene suffix filter (default: 1000-examples)"
            echo "  --major_interval=N               Process only N divisible by interval (default: 500)"
            echo "  --hygiene_folder=DIR             Hygiene folder (default: hygiene_filtering/\${MODEL}-apptainer-fsdp)"
            echo "  --model_folder=DIR               Model folder (default: models/\${MODEL}-apptainer-fsdp)"
            echo "  --min_new_tokens=N               Min generated tokens (default: 40)"
            echo "  --max_new_tokens=N               Max generated tokens (default: 180)"
            echo "  --max_input_tokens=N             Prompt truncation length (default: 2048)"
            echo "  --num_beams=N                    Beam size, 1=greedy (default: 1)"
            echo "  --overwrite                      Overwrite existing gen1 output files"
            echo "  --no_lora                        Kept for interface compatibility (ignored by Python script)"
            echo "  --dry-run                        Validate and print command, do not execute"
            exit 1
            ;;
    esac
done

validate_model() {
    local model=$1
    case "$model" in
        gemma-2b|gemma-7b-it|gemma-2-9b|gemma-2-27b|gemma-3-12b|gemma-3-27b| \
        viking-7b|viking-13b|viking-33b| \
        normistral-7b|normistral-11b|normistral-11b-long|normistral-7b-instruct| \
        norskgpt-llama3-8b|llama-3.1-8b-instruct|llama-2-13b-chat-norwegian| \
        eurollm-9b-instruct|norwai-mistral-7b-instruct|nb-gpt-j-6b|mt5)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

validate_distribution_flag() {
    local flag=$1
    case "$flag" in
        --fsdp|--ddp|"")
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

ERRORS=0
if ! validate_model "$MODEL"; then
    echo "ERROR: Invalid model name: $MODEL"
    ERRORS=$((ERRORS + 1))
fi
if ! validate_distribution_flag "$DISTRIBUTION_FLAG"; then
    echo "ERROR: Invalid distribution_flag: $DISTRIBUTION_FLAG (allowed: --fsdp, --ddp, empty)"
    ERRORS=$((ERRORS + 1))
fi
if [ ! -d "$HYGIENE_FOLDER" ]; then
    echo "ERROR: hygiene_folder does not exist: $HYGIENE_FOLDER"
    ERRORS=$((ERRORS + 1))
fi
if [ ! -d "$HYGIENE_FOLDER/all_eval_results" ]; then
    echo "ERROR: hygiene_folder missing all_eval_results: $HYGIENE_FOLDER/all_eval_results"
    ERRORS=$((ERRORS + 1))
fi
if [ ! -d "$MODEL_FOLDER" ]; then
    echo "ERROR: model_folder does not exist: $MODEL_FOLDER"
    ERRORS=$((ERRORS + 1))
fi

if [ "${#PRED_DATASETS[@]}" -gt 0 ]; then
    for ds in "${PRED_DATASETS[@]}"; do
        if [ ! -f "$ds" ]; then
            echo "ERROR: pred_dataset does not exist: $ds"
            ERRORS=$((ERRORS + 1))
            continue
        fi
        base=$(basename "$ds")
        if [[ "$base" != checkpoint-*-bad-hygiene-*.jsonl ]]; then
            echo "ERROR: pred_dataset must match checkpoint-<N>-bad-hygiene-<suffix>.jsonl: $base"
            ERRORS=$((ERRORS + 1))
        fi
    done
fi

if [ $ERRORS -gt 0 ]; then
    echo "Validation failed with $ERRORS error(s)."
    exit 1
fi

echo "Configuration:"
echo "  MODEL: $MODEL"
echo "  NUM_GPUS: $NUM_GPUS"
echo "  DISTRIBUTION_FLAG: $DISTRIBUTION_FLAG (compatibility only)"
echo "  PRED_BATCH_SIZE: $PRED_BATCH_SIZE"
echo "  EXAMPLES_SUFFIX: $EXAMPLES_SUFFIX"
echo "  MAJOR_INTERVAL: $MAJOR_INTERVAL"
echo "  HYGIENE_FOLDER: $HYGIENE_FOLDER"
echo "  MODEL_FOLDER: $MODEL_FOLDER"
echo "  MIN_NEW_TOKENS: $MIN_NEW_TOKENS"
echo "  MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
echo "  MAX_INPUT_TOKENS: $MAX_INPUT_TOKENS"
echo "  NUM_BEAMS: $NUM_BEAMS"
echo "  NO_LORA: $NO_LORA"
echo "  OVERWRITE: $OVERWRITE"
echo "  DRY_RUN: $DRY_RUN"
if [ "${#PRED_DATASETS[@]}" -gt 0 ]; then
    echo "  INPUT_MODE: explicit --pred_dataset list"
    echo "  PRED_DATASETS:"
    for ds in "${PRED_DATASETS[@]}"; do echo "    - $ds"; done
else
    echo "  INPUT_MODE: scan --hygiene_folder/all_eval_results"
fi

if [ "$DRY_RUN" = "true" ]; then
    echo ""
    echo "Dry-run complete. No execution performed."
    exit 0
fi

if [ -z "${HUGGINGFACE_TOKEN:-}" ]; then
    echo "ERROR: HUGGINGFACE_TOKEN is required (set in .env or environment)."
    exit 1
fi

# ********** JOB-LOCAL CACHE **********
export SLURM_TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID:-$$}}
mkdir -p "$SLURM_TMPDIR"

export HF_HOME="$SLURM_TMPDIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_METRICS_CACHE="$HF_HOME/metrics"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$HF_METRICS_CACHE"

PROJECT_HF_DIR=/cluster/projects/YOUR_PROJECT_ID/shared/.cache/huggingface
if [ -d "$PROJECT_HF_DIR/hub" ]; then
    mkdir -p "$HF_HOME/hub"
    # Non-fatal cache warmup
    rsync -aq "$PROJECT_HF_DIR/hub/" "$HF_HOME/hub/" || true
fi
if [ -d "$PROJECT_HF_DIR/metrics" ]; then
    rsync -aq "$PROJECT_HF_DIR/metrics/" "$HF_METRICS_CACHE/" || true
fi
if [ -d "$PROJECT_HF_DIR/modules" ]; then
    rsync -aq "$PROJECT_HF_DIR/modules/" "$HF_MODULES_CACHE/" || true
fi
# ********** END JOB-LOCAL CACHE **********

module --force purge
module load BuildEnv/NeoverseV2
module load NRIS/GPU
module load CUDA/12.6.0
module load PyTorch/2.8.0

export IMAGE=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif

PRED_FLAGS=""
for ds in "${PRED_DATASETS[@]}"; do
    PRED_FLAGS="$PRED_FLAGS --pred_dataset \"$ds\""
done
OVERWRITE_FLAG=""
if [ "$OVERWRITE" = "true" ]; then
    OVERWRITE_FLAG="--overwrite"
fi
NO_LORA_FLAG=""
if [ "$NO_LORA" = "true" ]; then
    NO_LORA_FLAG="--no_lora"
fi

PY_CMD="export HF_HOME=$HF_HOME && export HF_DATASETS_CACHE=$HF_DATASETS_CACHE && export HF_MODULES_CACHE=$HF_MODULES_CACHE && export HF_METRICS_CACHE=$HF_METRICS_CACHE && export TOKENIZERS_PARALLELISM=false && python scripts/regenerate_bad_hygiene_predictions.py --model \"$MODEL\" --hygiene_folder \"$HYGIENE_FOLDER\" --model_folder \"$MODEL_FOLDER\" --examples_suffix \"$EXAMPLES_SUFFIX\" --major_interval \"$MAJOR_INTERVAL\" --pred_batch_size \"$PRED_BATCH_SIZE\" --min_new_tokens \"$MIN_NEW_TOKENS\" --max_new_tokens \"$MAX_NEW_TOKENS\" --max_input_tokens \"$MAX_INPUT_TOKENS\" --num_beams \"$NUM_BEAMS\" --hf_token \"$HUGGINGFACE_TOKEN\" $OVERWRITE_FLAG $NO_LORA_FLAG $PRED_FLAGS"

# Keep install stack similar to evaluation script for predictable runtime env.
COMMAND="pip install --upgrade pip setuptools wheel > /dev/null 2>&1 && pip install --break-system-packages --only-binary=:all: transformers==4.45.2 tokenizers==0.20.3 datasets==3.0.0 sentencepiece==0.2.1 > /dev/null 2>&1 && pip install --break-system-packages --no-binary=rouge-score rouge-score==0.1.2 > /dev/null 2>&1 && pip install --break-system-packages --no-deps accelerate==1.0.1 peft==0.13.2 > /dev/null 2>&1 && pip install --break-system-packages --upgrade pyyaml psutil typing-extensions packaging==24.2 safetensors > /dev/null 2>&1 && $PY_CMD"

echo "Starting regeneration job..."
apptainer exec --nv "$IMAGE" bash -lc "$COMMAND"
echo "Regeneration completed."
