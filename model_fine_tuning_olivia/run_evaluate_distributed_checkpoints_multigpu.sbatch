#!/usr/bin/env bash
#SBATCH -J gpu-eval-checkpoints-multigpu
#SBATCH -p accel
#SBATCH --gres=gpu:4  # Request 2 GPUs (adjust if needed)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16  # Increase CPUs for multi-GPU
#SBATCH --mem=384G  # Increase memory for multi-GPU
#SBATCH --time=72:00:00  # 3 days - many checkpoints × ~1hr each
# Account: REQUIRED - Pass via: sbatch --account=YOUR_ACCOUNT run_evaluate_distributed_checkpoints_multigpu.sbatch ...
# Example: sbatch --account=YOUR_PROJECT_ID run_evaluate_distributed_checkpoints_multigpu.sbatch --model=gemma-2-9b
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# Slurm stages the batch script under /var/spool/slurmd/job*/ on compute nodes, so
# ${BASH_SOURCE[0]} points there—not your project. Use SLURM_SUBMIT_DIR (cwd when you ran sbatch).
# Without Slurm, fall back to the directory containing this file.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SBATCH_DIR="$SLURM_SUBMIT_DIR"
else
    SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SBATCH_DIR" || { echo "ERROR: Cannot cd to $SBATCH_DIR"; exit 1; }
echo "Working directory for this job: $SBATCH_DIR"

# Load environment variables from .env file
# .env should contain: SLURM_ACCOUNT, WANDB_API_KEY, HUGGINGFACE_TOKEN
if [ -f .env ]; then
    source .env
else
    echo "WARNING: .env file not found. Using defaults or environment variables."
fi

# Note: Account is set in #SBATCH -A line above (or via --account flag)
# Check if account was overridden via --account flag (SLURM sets this)
# If not, the #SBATCH -A line value will be used

# ===== CONFIGURATION =====
# Parse command-line arguments or use defaults
# Usage: sbatch --export=SLURM_ACCOUNT=YOUR_PROJECT_ID run_evaluate_distributed_checkpoints_multigpu.sbatch [--model=MODEL] [--num_gpus=N] [--checkpoint_base_dir=DIR] [--val_dataset=PATH] [--specific_checkpoints="100 200 300"]
MODEL=${MODEL:-llama-2-13b-chat-norwegian}
NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-"models/${MODEL}-apptainer-fsdp"}
VAL_DATASET=${VAL_DATASET:-"data/dataset_149978_examples/149978_text_summary_examples_val.jsonl"}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-32}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-1000}
SPECIFIC_CHECKPOINTS=${SPECIFIC_CHECKPOINTS:-""}
DRY_RUN=${DRY_RUN:-false}
# NLI Faithfulness evaluation flags: "auto" (default) = on for major checkpoints only,
# "true" = on for all checkpoints, "false" = off for all checkpoints
INCLUDE_NLI_FAITHFULNESS=${INCLUDE_NLI_FAITHFULNESS:-auto}
# Empty = Python default (100 NLI examples). Set equal to VAL_DATA_SIZE for full-val NLI.
NLI_SUBSET_SIZE=${NLI_SUBSET_SIZE:-}
KEEP_EXISTING=${KEEP_EXISTING:-false}
FORCE_RECOMPUTE=${FORCE_RECOMPUTE:-false}
# Checkpoint type filtering
CHECKPOINT_FILTER=${CHECKPOINT_FILTER:-"all"}   # "all", "major", "minor"
MAJOR_CHECKPOINT_INTERVAL=${MAJOR_CHECKPOINT_INTERVAL:-500}
# Python evaluation mode flags
EVAL_MODE=${EVAL_MODE:-""}        # "", "predict_only", "metrics_only"
UPDATE_METRICS=${UPDATE_METRICS:-""}  # comma-separated: "rouge", "rouge,hygiene", etc.
# Length-control options (disabled by default to preserve current behavior)
LENGTH_CONTROLLED=${LENGTH_CONTROLLED:-false}
MIN_NEW_TOKENS=${MIN_NEW_TOKENS:-40}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-180}
USE_BEAM_LENGTH_CONTROL=${USE_BEAM_LENGTH_CONTROL:-false}
VAL_BEAM_SIZE=${VAL_BEAM_SIZE:-4}
LENGTH_PENALTY=${LENGTH_PENALTY:-1.05}

# Parse command-line arguments if provided
for arg in "$@"; do
    case $arg in
        --model=*)
            MODEL="${arg#*=}"
            # Update CHECKPOINT_BASE_DIR if it wasn't explicitly set
            if [ -z "${CHECKPOINT_BASE_DIR_SET:-}" ]; then
                CHECKPOINT_BASE_DIR="models/${MODEL}-apptainer-fsdp"
            fi
            ;;
        --num_gpus=*)
            NUM_GPUS="${arg#*=}"
            ;;
        --checkpoint_base_dir=*)
            CHECKPOINT_BASE_DIR="${arg#*=}"
            CHECKPOINT_BASE_DIR_SET=1
            ;;
        --val_dataset=*)
            VAL_DATASET="${arg#*=}"
            ;;
        --val_batch_size=*)
            VAL_BATCH_SIZE="${arg#*=}"
            ;;
        --val_data_size=*)
            VAL_DATA_SIZE="${arg#*=}"
            ;;
        --specific_checkpoints=*)
            SPECIFIC_CHECKPOINTS="${arg#*=}"
            ;;
        --include_nli_faithfulness|--include-nli-faithfulness)
            INCLUDE_NLI_FAITHFULNESS=true
            ;;
        --no_nli_faithfulness|--no-nli-faithfulness)
            INCLUDE_NLI_FAITHFULNESS=false
            ;;
        --keep_existing)
            KEEP_EXISTING=true
            ;;
        --no-keep_existing)
            KEEP_EXISTING=false
            ;;
        --force_recompute)
            FORCE_RECOMPUTE=true
            ;;
        --nli_subset_size=*|--nli-subset-size=*)
            NLI_SUBSET_SIZE="${arg#*=}"
            ;;
        --major-only|--major_only)
            CHECKPOINT_FILTER="major"
            ;;
        --minor-only|--minor_only)
            CHECKPOINT_FILTER="minor"
            ;;
        --major-interval=*|--major_checkpoint_interval=*)
            MAJOR_CHECKPOINT_INTERVAL="${arg#*=}"
            ;;
        --predict-only|--predict_only)
            EVAL_MODE="predict_only"
            ;;
        --metrics-only|--metrics_only)
            EVAL_MODE="metrics_only"
            ;;
        --update-rouge|--update_rouge)
            UPDATE_METRICS="${UPDATE_METRICS:+$UPDATE_METRICS,}rouge"
            ;;
        --update-hygiene|--update_hygiene)
            UPDATE_METRICS="${UPDATE_METRICS:+$UPDATE_METRICS,}hygiene"
            ;;
        --update-bertscore|--update_bertscore)
            UPDATE_METRICS="${UPDATE_METRICS:+$UPDATE_METRICS,}bertscore"
            ;;
        --update-faithfulness|--update_faithfulness)
            UPDATE_METRICS="${UPDATE_METRICS:+$UPDATE_METRICS,}faithfulness"
            ;;
        --dry-run|--debug)
            DRY_RUN=true
            ;;
        --length_controlled|--length-controlled)
            LENGTH_CONTROLLED=true
            ;;
        --no_length_controlled|--no-length-controlled)
            LENGTH_CONTROLLED=false
            ;;
        --min_new_tokens=*)
            MIN_NEW_TOKENS="${arg#*=}"
            ;;
        --max_new_tokens=*)
            MAX_NEW_TOKENS="${arg#*=}"
            ;;
        --use_beam_length_control|--use-beam-length-control)
            USE_BEAM_LENGTH_CONTROL=true
            ;;
        --val_beam_size=*)
            VAL_BEAM_SIZE="${arg#*=}"
            ;;
        --length_penalty=*)
            LENGTH_PENALTY="${arg#*=}"
            ;;
        *)
            echo "ERROR: Unknown argument: $arg"
            echo ""
            echo "Usage: sbatch --export=SLURM_ACCOUNT=YOUR_PROJECT_ID run_evaluate_distributed_checkpoints_multigpu.sbatch [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model=MODEL                    Model short name (default: llama-2-13b-chat-norwegian)"
            echo "  --num_gpus=N                     Number of GPUs (default: 4)"
            echo "  --checkpoint_base_dir=DIR         Base directory containing checkpoints (default: models/\${MODEL}-apptainer-fsdp)"
            echo "  --val_dataset=PATH               Path to validation dataset JSONL file (default: data/dataset_149978_examples/149978_text_summary_examples_val.jsonl)"
            echo "  --val_batch_size=N                Validation batch size per device (default: 32; auto-adjusted by Python script based on model size)"
            echo "  --val_data_size=N                 Number of validation examples (default: 1000; output files end with -N-examples)"
            echo "  --specific_checkpoints=\"N1 N2\"   Space-separated checkpoint numbers to evaluate (default: all)"
            echo "  --force_recompute                  Re-evaluate even when results exist (e.g. backfill missed checkpoints)"
            echo "  --include_nli_faithfulness        Enable NLI-based faithfulness for ALL checkpoints (default: auto = major only)"
            echo "  --no_nli_faithfulness             Disable NLI-based faithfulness for all checkpoints (override auto)"
            echo "  --nli_subset_size=N               NLI examples (default 100 in Python if omitted); set N = val_data_size for full val NLI"
            echo "  --major-only                      Only evaluate major checkpoints (every Nth step)"
            echo "  --minor-only                      Only evaluate minor checkpoints (not every Nth step)"
            echo "  --major-interval=N                Major checkpoint interval (default: 500)"
            echo "  --predict-only                    Generate predictions (JSONL) only; skip metrics"
            echo "  --metrics-only                    Compute metrics from existing JSONL; no model loading"
            echo "  --update-rouge                    Recompute ROUGE from existing JSONL (combinable)"
            echo "  --update-hygiene                  Recompute hygiene from existing JSONL (combinable)"
            echo "  --update-bertscore                Recompute BERTScore from existing JSONL (combinable)"
            echo "  --update-faithfulness             Recompute NLI faithfulness from existing JSONL (combinable)"
            echo "  --length_controlled               Write results to length_controlled_eval_results and apply length controls"
            echo "  --min_new_tokens=N                Minimum generated tokens for length-controlled runs (default: 40)"
            echo "  --max_new_tokens=N                Maximum generated tokens for length-controlled runs (default: 180)"
            echo "  --use_beam_length_control         Use beam decoding with length_penalty for length-controlled runs"
            echo "  --val_beam_size=N                 Beam size when --use_beam_length_control is enabled (default: 4)"
            echo "  --length_penalty=F                Length penalty when --use_beam_length_control is enabled (default: 1.05)"
            echo "  --dry-run|--debug                Debug mode: validate inputs and show what would run (no job submission)"
            echo ""
            echo "Valid models: gemma-2b, gemma-7b-it, gemma-2-9b, gemma-2-27b, gemma-3-12b, gemma-3-27b,"
            echo "              viking-7b, viking-13b, viking-33b, normistral-7b, normistral-11b, normistral-11b-long, normistral-7b-instruct,"
            echo "              norskgpt-llama3-8b, llama-3.1-8b-instruct, llama-2-13b-chat-norwegian,"
            echo "              eurollm-9b-instruct, norwai-mistral-7b-instruct, nb-gpt-j-6b, mt5"
            exit 1
            ;;
    esac
done

# Validation functions
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

validate_num_gpus() {
    local num=$1
    if ! [[ "$num" =~ ^[0-9]+$ ]] || [ "$num" -lt 1 ] || [ "$num" -gt 8 ]; then
        return 1
    fi
    return 0
}

validate_checkpoint_numbers() {
    local checkpoints=$1
    if [ -z "$checkpoints" ]; then
        return 0  # Empty is valid (evaluate all)
    fi
    for ckpt in $checkpoints; do
        if ! [[ "$ckpt" =~ ^[0-9]+$ ]]; then
            return 1
        fi
    done
    return 0
}

# Validate inputs
ERRORS=0

if ! validate_model "$MODEL"; then
    echo "ERROR: Invalid model name: $MODEL"
    echo "Valid models: gemma-2b, gemma-7b-it, gemma-2-9b, gemma-2-27b, gemma-3-12b, gemma-3-27b,"
    echo "              viking-7b, viking-13b, viking-33b, normistral-7b, normistral-11b, normistral-11b-long, normistral-7b-instruct,"
    echo "              norskgpt-llama3-8b, llama-3.1-8b-instruct, llama-2-13b-chat-norwegian,"
    echo "              eurollm-9b-instruct, norwai-mistral-7b-instruct, nb-gpt-j-6b, mt5"
    ERRORS=$((ERRORS + 1))
fi

if ! validate_num_gpus "$NUM_GPUS"; then
    echo "ERROR: Invalid num_gpus: $NUM_GPUS"
    echo "num_gpus must be a positive integer between 1 and 8"
    ERRORS=$((ERRORS + 1))
fi

if [ ! -d "$CHECKPOINT_BASE_DIR" ]; then
    echo "ERROR: Checkpoint base directory does not exist: $CHECKPOINT_BASE_DIR"
    ERRORS=$((ERRORS + 1))
fi

if [ ! -f "$VAL_DATASET" ]; then
    echo "ERROR: Validation dataset file does not exist: $VAL_DATASET"
    ERRORS=$((ERRORS + 1))
fi

if ! validate_checkpoint_numbers "$SPECIFIC_CHECKPOINTS"; then
    echo "ERROR: Invalid specific_checkpoints format: $SPECIFIC_CHECKPOINTS"
    echo "specific_checkpoints must be space-separated numbers, e.g., \"100 200 300\""
    ERRORS=$((ERRORS + 1))
fi

if [ -n "${NLI_SUBSET_SIZE:-}" ]; then
    if ! [[ "$NLI_SUBSET_SIZE" =~ ^[0-9]+$ ]] || [ "$NLI_SUBSET_SIZE" -lt 1 ]; then
        echo "ERROR: nli_subset_size must be a positive integer (got: $NLI_SUBSET_SIZE)"
        ERRORS=$((ERRORS + 1))
    fi
fi

# Check if NUM_GPUS matches #SBATCH --gres (warning only, skip in dry-run)
if [ "$DRY_RUN" != "true" ]; then
    SBATCH_GPUS=$(grep "^#SBATCH --gres=gpu:" "$0" | sed 's/.*gpu:\([0-9]*\).*/\1/' | head -1)
    if [ -n "$SBATCH_GPUS" ] && [ "$NUM_GPUS" != "$SBATCH_GPUS" ]; then
        echo "WARNING: NUM_GPUS ($NUM_GPUS) does not match #SBATCH --gres=gpu:$SBATCH_GPUS"
        echo "         Consider updating the #SBATCH --gres line or using --num_gpus=$SBATCH_GPUS"
    fi
fi

if [ $ERRORS -gt 0 ]; then
    echo ""
    echo "Validation failed with $ERRORS error(s). Please fix the errors above and try again."
    exit 1
fi

echo "Configuration:"
echo "  MODEL: $MODEL"
echo "  NUM_GPUS: $NUM_GPUS"
echo "  CHECKPOINT_BASE_DIR: $CHECKPOINT_BASE_DIR"
echo "  VAL_DATASET: $VAL_DATASET"
echo "  VAL_BATCH_SIZE: $VAL_BATCH_SIZE"
echo "  VAL_DATA_SIZE: $VAL_DATA_SIZE (output files end with -${VAL_DATA_SIZE}-examples.json/.jsonl)"
echo "  SPECIFIC_CHECKPOINTS: ${SPECIFIC_CHECKPOINTS:-all}"
echo "  INCLUDE_NLI_FAITHFULNESS: $INCLUDE_NLI_FAITHFULNESS (auto = major checkpoints only)"
echo "  LENGTH_CONTROLLED: $LENGTH_CONTROLLED"
if [ "$LENGTH_CONTROLLED" = "true" ]; then
    echo "  MIN_NEW_TOKENS: $MIN_NEW_TOKENS"
    echo "  MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
    echo "  USE_BEAM_LENGTH_CONTROL: $USE_BEAM_LENGTH_CONTROL"
    if [ "$USE_BEAM_LENGTH_CONTROL" = "true" ]; then
        echo "  VAL_BEAM_SIZE: $VAL_BEAM_SIZE"
        echo "  LENGTH_PENALTY: $LENGTH_PENALTY"
    fi
fi
if [ "$INCLUDE_NLI_FAITHFULNESS" != "false" ]; then
    if [ -n "${NLI_SUBSET_SIZE:-}" ]; then
        echo "  NLI_SUBSET_SIZE: $NLI_SUBSET_SIZE"
    else
        echo "  NLI_SUBSET_SIZE: (unset → Python default 100; use same value as VAL_DATA_SIZE for full val NLI)"
    fi
fi
echo "  DRY_RUN: $DRY_RUN"

# Verify required environment variables are set
if [ -z "${WANDB_API_KEY:-}" ] || [ -z "${HUGGINGFACE_TOKEN:-}" ]; then
    echo "ERROR: Required environment variables not set. Please check your .env file contains:"
    echo "  WANDB_API_KEY"
    echo "  HUGGINGFACE_TOKEN"
    exit 1
fi

# ===== DRY-RUN MODE =====
if [ "$DRY_RUN" = "true" ]; then
    echo ""
    echo "==================================================================="
    echo "DRY-RUN MODE: Validation and Preview"
    echo "==================================================================="
    echo ""
    
    # Show what would be executed
    echo "✓ All validations passed!"
    echo ""
    echo "Would execute with the following configuration:"
    echo "  Checkpoint base directory: $CHECKPOINT_BASE_DIR"
    echo "  Validation dataset: $VAL_DATASET"
    echo "  Number of GPUs: $NUM_GPUS"
    
    # Find checkpoints that would be evaluated
    if [ -n "$SPECIFIC_CHECKPOINTS" ]; then
        echo "  Specific checkpoints to evaluate: $SPECIFIC_CHECKPOINTS"
        CHECKPOINTS=""
        for ckpt_num in $SPECIFIC_CHECKPOINTS; do
            ckpt_dir="$CHECKPOINT_BASE_DIR/checkpoint-$ckpt_num"
            regular_ckpt_legacy="$CHECKPOINT_BASE_DIR/regular_checkpoints/regular-checkpoint-$ckpt_num"
            regular_ckpt_new="$CHECKPOINT_BASE_DIR/regular_checkpoints/checkpoint-$ckpt_num"
            major_ckpt_legacy="$CHECKPOINT_BASE_DIR/major_checkpoints/major-checkpoint-$ckpt_num"
            major_ckpt_new="$CHECKPOINT_BASE_DIR/major_checkpoints/checkpoint-$ckpt_num"
            
            if [ -d "$ckpt_dir" ]; then
                CHECKPOINTS="$CHECKPOINTS $ckpt_dir"
                echo "    ✓ Found: checkpoint-$ckpt_num (main)"
            elif [ -d "$major_ckpt_new" ] || [ -d "$major_ckpt_legacy" ]; then
                if [ -d "$major_ckpt_new" ]; then CHECKPOINTS="$CHECKPOINTS $major_ckpt_new"; else CHECKPOINTS="$CHECKPOINTS $major_ckpt_legacy"; fi
                echo "    ✓ Found: checkpoint-$ckpt_num (major backup)"
            elif [ -d "$regular_ckpt_new" ] || [ -d "$regular_ckpt_legacy" ]; then
                if [ -d "$regular_ckpt_new" ]; then CHECKPOINTS="$CHECKPOINTS $regular_ckpt_new"; else CHECKPOINTS="$CHECKPOINTS $regular_ckpt_legacy"; fi
                echo "    ✓ Found: checkpoint-$ckpt_num (regular backup)"
            else
                echo "    ⚠ Not found: checkpoint-$ckpt_num (checked main and backups)"
            fi
        done
    else
        echo "  Would evaluate ALL checkpoints in: $CHECKPOINT_BASE_DIR (including backups)"
        if [ -d "$CHECKPOINT_BASE_DIR" ]; then
            MAIN_COUNT=$(find "$CHECKPOINT_BASE_DIR" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | wc -l)
            REGULAR_COUNT=$(find "$CHECKPOINT_BASE_DIR/regular_checkpoints" -mindepth 1 -maxdepth 1 -type d \( -name "checkpoint-*" -o -name "regular-checkpoint-*" \) 2>/dev/null | wc -l)
            MAJOR_COUNT=$(find "$CHECKPOINT_BASE_DIR/major_checkpoints" -mindepth 1 -maxdepth 1 -type d \( -name "checkpoint-*" -o -name "major-checkpoint-*" \) 2>/dev/null | wc -l)
            echo "    Main: $MAIN_COUNT, Regular backups: $REGULAR_COUNT, Major backups: $MAJOR_COUNT"
        fi
    fi
    
    echo ""
    echo "File/Directory checks:"
    if [ -d "$CHECKPOINT_BASE_DIR" ]; then
        echo "  ✓ Checkpoint base directory exists"
    else
        echo "  ⚠ Checkpoint base directory not found: $CHECKPOINT_BASE_DIR"
    fi
    
    if [ -f "$VAL_DATASET" ]; then
        echo "  ✓ Validation dataset exists"
    else
        echo "  ⚠ Validation dataset not found: $VAL_DATASET"
    fi
    
    echo ""
    echo "Environment variables:"
    echo "  ✓ SLURM_ACCOUNT: ${SLURM_ACCOUNT:-NOT SET}"
    echo "  ✓ WANDB_API_KEY: ${WANDB_API_KEY:+SET (hidden)}"
    echo "  ✓ HUGGINGFACE_TOKEN: ${HUGGINGFACE_TOKEN:+SET (hidden)}"
    echo ""
    echo "Evaluation command would be:"
    echo "  python scripts/evaluate_distributed_checkpoints_multigpu.py"
    echo "    --model $MODEL"
    echo "    --checkpoint_dir [checkpoint-dir]"
    echo "    --val_dataset $VAL_DATASET"
    echo "    --hf_token [REDACTED]"
    echo "    --output_dir [optional, defaults to model_dir/all_eval_results]"
    echo "    --wandb_project lm-evaluation"
    echo "    --use_greedy --use_multi_gpu"
    if [ "$INCLUDE_NLI_FAITHFULNESS" = "true" ]; then
        echo "    --include_nli_faithfulness  (all checkpoints)"
    elif [ "$INCLUDE_NLI_FAITHFULNESS" = "auto" ]; then
        echo "    --include_nli_faithfulness  (auto: major checkpoints only)"
    fi
    if [ "$INCLUDE_NLI_FAITHFULNESS" != "false" ]; then
        if [ -n "${NLI_SUBSET_SIZE:-}" ]; then
            echo "    --nli_subset_size $NLI_SUBSET_SIZE"
        else
            echo "    --nli_subset_size 100   # Python default; set = VAL_DATA_SIZE for full val NLI"
        fi
    fi
    echo ""
    echo "==================================================================="
    echo "Dry-run complete. No job was submitted."
    echo "To actually run, remove --dry-run flag."
    echo "==================================================================="
    exit 0
fi

echo "==================================================================="
echo "Multi-GPU Evaluation: $NUM_GPUS GPUs"
echo "Evaluating checkpoints from: $CHECKPOINT_BASE_DIR"
echo "Model: $MODEL"
echo "==================================================================="

# ********** JOB-LOCAL CACHE **********

# define job-local cache for training
export SLURM_TMPDIR=/tmp/${USER}_$SLURM_JOB_ID
mkdir -p "$SLURM_TMPDIR"

# create a job-local HF cache, overrides the global HF_home cache
export HF_HOME="$SLURM_TMPDIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_METRICS_CACHE="$HF_HOME/metrics"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$HF_METRICS_CACHE"

# stage model/dataset into job-local cache (fast local read, writable)
PROJECT_HF_DIR=/cluster/projects/YOUR_PROJECT_ID/shared/.cache/huggingface
mkdir -p "$HF_HOME/hub"

# Convert model name to HuggingFace cache directory format
convert_model_to_cache_dir() {
    local model=$1
    echo "models--${model//\//--}"
}

# Determine the full model name from the short name
get_full_model_name() {
    local short_name=$1
    case "$short_name" in
        gemma-2b) echo "google/gemma-2b" ;;
        gemma-7b-it) echo "google/gemma-7b-it" ;;
        gemma-2-9b) echo "google/gemma-2-9b" ;;
        gemma-2-27b) echo "google/gemma-2-27b" ;;
        gemma-3-12b) echo "google/gemma-3-12b-pt" ;;
        gemma-3-27b) echo "google/gemma-3-27b-pt" ;;
        viking-7b) echo "LumiOpen/Viking-7B" ;;
        viking-13b) echo "LumiOpen/Viking-13B" ;;
        viking-33b) echo "LumiOpen/Viking-33B" ;;
        normistral-7b) echo "norallm/normistral-7b-warm" ;;
        normistral-11b) echo "norallm/normistral-11b-warm" ;;
        normistral-11b-long) echo "norallm/normistral-11b-long" ;;
        normistral-7b-instruct) echo "norallm/normistral-7b-warm-instruct" ;;
        norskgpt-llama3-8b) echo "bineric/norskgpt-llama3-8b" ;;
        llama-3.1-8b-instruct) echo "meta-llama/Llama-3.1-8B-Instruct" ;;
        llama-2-13b-chat-norwegian) echo "ruternorway/llama-2-13b-chat-norwegian" ;;
        eurollm-9b-instruct) echo "utter-project/EuroLLM-9B-Instruct-2512" ;;
        norwai-mistral-7b-instruct) echo "NorwAI/NorwAI-Mistral-7B-instruct" ;;
        nb-gpt-j-6b) echo "NbAiLab/nb-gpt-j-6B-torgersen-alpaca" ;;
        *) echo "$short_name" ;;
    esac
}

FULL_MODEL_NAME=$(get_full_model_name "$MODEL")
MODEL_CACHE_DIR=$(convert_model_to_cache_dir "$FULL_MODEL_NAME")
SOURCE_MODEL_DIR="$PROJECT_HF_DIR/hub/$MODEL_CACHE_DIR"

# Copy model from read-only project storage to writable job-local cache
echo "Copying model $FULL_MODEL_NAME to job-local cache..."
if [ -d "$SOURCE_MODEL_DIR" ]; then
    rsync -aq "$SOURCE_MODEL_DIR/" "$HF_HOME/hub/$MODEL_CACHE_DIR/"
    echo "Model cache copy complete"
else
    echo "WARNING: Model $FULL_MODEL_NAME not found in shared cache at $SOURCE_MODEL_DIR"
    echo "Model will be downloaded during evaluation (requires internet access)"
fi

[ -d "$PROJECT_HF_DIR/metrics" ] && rsync -aq "$PROJECT_HF_DIR/metrics/" "$HF_METRICS_CACHE/"
[ -d "$PROJECT_HF_DIR/modules" ] && rsync -aq "$PROJECT_HF_DIR/modules/" "$HF_MODULES_CACHE/"

echo "Cache staging complete"

# ********** END JOB-LOCAL CACHE **********

# env setup
echo "DEBUG: About to load modules..."
module --force purge || { echo "ERROR: Failed to purge modules"; exit 1; }
echo "DEBUG: Modules purged, loading new ones..."
module load BuildEnv/NeoverseV2 || { echo "ERROR: Failed to load BuildEnv/NeoverseV2"; exit 1; }
module load NRIS/GPU || { echo "ERROR: Failed to load NRIS/GPU"; exit 1; }
module load CUDA/12.6.0 || { echo "ERROR: Failed to load CUDA/12.6.0"; exit 1; }
module load PyTorch/2.8.0 || { echo "ERROR: Failed to load PyTorch/2.8.0"; exit 1; }
echo "DEBUG: All modules loaded successfully"

# .env already sourced above, variables should be available

# Apptainer image
export IMAGE=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif
echo "DEBUG: Apptainer image set to: $IMAGE"

# Find all checkpoint directories (main + backups)
# Priority: main checkpoints first, then backups if main doesn't exist
echo "DEBUG: Starting checkpoint discovery..."
echo "DEBUG: CHECKPOINT_BASE_DIR=$CHECKPOINT_BASE_DIR"
echo "DEBUG: SPECIFIC_CHECKPOINTS=$SPECIFIC_CHECKPOINTS"

if [ -n "$SPECIFIC_CHECKPOINTS" ]; then
    echo "DEBUG: Using specific checkpoints mode"
    CHECKPOINTS=""
    for ckpt_num in $SPECIFIC_CHECKPOINTS; do
        # Try main checkpoint directory first
        ckpt_dir="$CHECKPOINT_BASE_DIR/checkpoint-$ckpt_num"
        if [ -d "$ckpt_dir" ]; then
            CHECKPOINTS="$CHECKPOINTS $ckpt_dir"
        else
            # Try backup directories. Prefer major over regular for major steps (500, 1000, ...).
            # Support both new naming (checkpoint-X) and legacy (regular-checkpoint-X / major-checkpoint-X).
            major_new="$CHECKPOINT_BASE_DIR/major_checkpoints/checkpoint-$ckpt_num"
            major_legacy="$CHECKPOINT_BASE_DIR/major_checkpoints/major-checkpoint-$ckpt_num"
            regular_new="$CHECKPOINT_BASE_DIR/regular_checkpoints/checkpoint-$ckpt_num"
            regular_legacy="$CHECKPOINT_BASE_DIR/regular_checkpoints/regular-checkpoint-$ckpt_num"
            major_ckpt=""
            regular_ckpt=""
            [ -d "$major_new" ] && major_ckpt="$major_new" || [ -d "$major_legacy" ] && major_ckpt="$major_legacy"
            [ -d "$regular_new" ] && regular_ckpt="$regular_new" || [ -d "$regular_legacy" ] && regular_ckpt="$regular_legacy"
            
            if [ -n "$major_ckpt" ] && [ -f "$major_ckpt/adapter_config.json" ]; then
                CHECKPOINTS="$CHECKPOINTS $major_ckpt"
                echo "Using backup checkpoint: $major_ckpt"
            elif [ -n "$regular_ckpt" ] && [ -f "$regular_ckpt/adapter_config.json" ]; then
                CHECKPOINTS="$CHECKPOINTS $regular_ckpt"
                echo "Using backup checkpoint: $regular_ckpt"
            elif [ -n "$major_ckpt" ]; then
                CHECKPOINTS="$CHECKPOINTS $major_ckpt"
                echo "Using backup checkpoint: $major_ckpt"
            elif [ -n "$regular_ckpt" ]; then
                CHECKPOINTS="$CHECKPOINTS $regular_ckpt"
                echo "Using backup checkpoint: $regular_ckpt"
            else
                echo "WARNING: Checkpoint not found in main or backup directories: checkpoint-$ckpt_num"
            fi
        fi
    done
else
    # Find all checkpoints: main first, then backups
    # Add error handling to prevent script exit on find errors
    MAIN_CHECKPOINTS=$(find "$CHECKPOINT_BASE_DIR" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | sort -V || true)
    REGULAR_CHECKPOINTS=$(find "$CHECKPOINT_BASE_DIR/regular_checkpoints" -mindepth 1 -maxdepth 1 -type d \( -name "checkpoint-*" -o -name "regular-checkpoint-*" \) 2>/dev/null | sort -V || true)
    MAJOR_CHECKPOINTS=$(find "$CHECKPOINT_BASE_DIR/major_checkpoints" -mindepth 1 -maxdepth 1 -type d \( -name "checkpoint-*" -o -name "major-checkpoint-*" \) 2>/dev/null | sort -V || true)
    
    # Create a map to avoid duplicates (prioritize main checkpoints)
    # Check bash version for associative array support
    if [ "${BASH_VERSION%%.*}" -lt 4 ]; then
        echo "ERROR: Bash version 4+ required for associative arrays. Current version: $BASH_VERSION"
        exit 1
    fi
    declare -A CHECKPOINT_MAP

    # Bash < 4.4 treats empty associative arrays as unbound under set -u;
    # disable -u for the map-building section.
    set +u

    # Add main checkpoints first
    echo "DEBUG: Processing main checkpoints..."
    for ckpt in $MAIN_CHECKPOINTS; do
        if [ -n "$ckpt" ]; then
            step=$(basename "$ckpt" | sed 's/checkpoint-//')
            CHECKPOINT_MAP[$step]="$ckpt"
        fi
    done
    echo "DEBUG: Found ${#CHECKPOINT_MAP[@]} main checkpoints"

    # Add backup checkpoints only if main doesn't exist.
    # Process MAJOR before REGULAR so step 500 uses major-checkpoint-500 (major steps are
    # stored only in major_checkpoints/, not regular_checkpoints/ - regular-checkpoint-500 may be empty).
    echo "DEBUG: Processing backup checkpoints (major first, then regular)..."
    for ckpt in $MAJOR_CHECKPOINTS $REGULAR_CHECKPOINTS; do
        if [ -n "$ckpt" ]; then
            step=$(basename "$ckpt" | sed -E 's/^(regular-checkpoint-|major-checkpoint-|checkpoint-)//')
            if [ -z "${CHECKPOINT_MAP[$step]:-}" ]; then
                CHECKPOINT_MAP[$step]="$ckpt"
            fi
        fi
    done
    echo "DEBUG: Total checkpoints after adding backups: ${#CHECKPOINT_MAP[@]}"

    # Sort by step number and build CHECKPOINTS string
    CHECKPOINTS=""
    for step in $(printf '%s\n' "${!CHECKPOINT_MAP[@]}" | sort -n); do
        CHECKPOINTS="$CHECKPOINTS ${CHECKPOINT_MAP[$step]}"
    done
    CHECKPOINTS=$(echo $CHECKPOINTS | xargs)  # Trim leading/trailing spaces

    set -u
    echo "DEBUG: Checkpoint discovery complete (all checkpoints mode)"
fi

# Filter checkpoints by type (major/minor) if requested
if [ "$CHECKPOINT_FILTER" != "all" ]; then
    FILTERED=""
    for CKPT_PATH in $CHECKPOINTS; do
        step=$(basename "$CKPT_PATH" | sed -E 's/^(regular-checkpoint-|major-checkpoint-|checkpoint-)//')
        remainder=$((step % MAJOR_CHECKPOINT_INTERVAL))
        if [ "$CHECKPOINT_FILTER" = "major" ] && [ "$remainder" -eq 0 ]; then
            FILTERED="$FILTERED $CKPT_PATH"
        elif [ "$CHECKPOINT_FILTER" = "minor" ] && [ "$remainder" -ne 0 ]; then
            FILTERED="$FILTERED $CKPT_PATH"
        fi
    done
    CHECKPOINTS=$(echo $FILTERED | xargs)
    echo "Filtered to $CHECKPOINT_FILTER checkpoints (interval=$MAJOR_CHECKPOINT_INTERVAL): $(echo $CHECKPOINTS | wc -w) remaining"
fi

# Count checkpoints
NUM_CHECKPOINTS=$(echo "$CHECKPOINTS" | wc -w)
echo "DEBUG: Checkpoint count calculation complete"
echo "Found $NUM_CHECKPOINTS checkpoints to evaluate"

if [ $NUM_CHECKPOINTS -eq 0 ]; then
    echo "ERROR: No checkpoints found in $CHECKPOINT_BASE_DIR"
    exit 1
fi

# Create results directory
RESULTS_DIR="$CHECKPOINT_BASE_DIR/all_eval_results"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo "Evaluation summary JSON will be automatically generated by the Python script"
echo ""

# Evaluate each checkpoint
CHECKPOINT_NUM=0
for CHECKPOINT_DIR in $CHECKPOINTS; do
    CHECKPOINT_NUM=$((CHECKPOINT_NUM + 1))
    CHECKPOINT_NAME=$(basename "$CHECKPOINT_DIR")
    
    STARTED_AT=$(date '+%Y-%m-%d %H:%M:%S')
    echo "==================================================================="
    echo "[$CHECKPOINT_NUM/$NUM_CHECKPOINTS] Evaluating: $CHECKPOINT_NAME | model=$MODEL | gpus=$NUM_GPUS | val_size=$VAL_DATA_SIZE | batch=$VAL_BATCH_SIZE | started=$STARTED_AT"
    echo "  dir: $CHECKPOINT_DIR"
    echo "==================================================================="
    
    # Command to run inside apptainer - use model parallelism, not torchrun
    # Model parallelism (device_map="auto") doesn't need torchrun
    # Build NLI flags (per-checkpoint: "auto" enables NLI only for major checkpoints)
    NLI_FLAGS=""
    NLI_SUBSET_FLAG=""
    CKPT_STEP=$(basename "$CHECKPOINT_DIR" | sed -E 's/^(regular-checkpoint-|major-checkpoint-|checkpoint-)//')
    if [ "$INCLUDE_NLI_FAITHFULNESS" = "true" ]; then
        NLI_FLAGS="--include_nli_faithfulness"
    elif [ "$INCLUDE_NLI_FAITHFULNESS" = "auto" ]; then
        CKPT_REMAINDER=$((CKPT_STEP % MAJOR_CHECKPOINT_INTERVAL))
        if [ "$CKPT_REMAINDER" -eq 0 ]; then
            NLI_FLAGS="--include_nli_faithfulness"
        fi
    fi
    if [ -n "$NLI_FLAGS" ] && [ -n "${NLI_SUBSET_SIZE:-}" ]; then
        NLI_SUBSET_FLAG=" --nli_subset_size $NLI_SUBSET_SIZE"
    fi
    KEEP_FLAG=""
    if [ "$KEEP_EXISTING" = "true" ]; then
        KEEP_FLAG="--keep_existing"
    fi
    FORCE_FLAG=""
    if [ "$FORCE_RECOMPUTE" = "true" ]; then
        FORCE_FLAG="--force_recompute"
    fi

    # Python evaluation mode flags
    MODE_FLAGS=""
    if [ "$EVAL_MODE" = "predict_only" ]; then
        MODE_FLAGS="--predict_only"
    elif [ "$EVAL_MODE" = "metrics_only" ]; then
        MODE_FLAGS="--metrics_only"
    fi
    # --update-* flags (can be combined)
    if [ -n "$UPDATE_METRICS" ]; then
        IFS=',' read -ra METRICS_ARR <<< "$UPDATE_METRICS"
        for m in "${METRICS_ARR[@]}"; do
            MODE_FLAGS="$MODE_FLAGS --update_$m"
        done
    fi
    # Decode strategy + optional length-control flags
    DECODE_FLAGS="--use_greedy"
    LENGTH_FLAGS=""
    if [ "$LENGTH_CONTROLLED" = "true" ]; then
        LENGTH_FLAGS="--length_controlled --min_new_tokens $MIN_NEW_TOKENS --max_output_summary_tokens $MAX_NEW_TOKENS"
        if [ "$USE_BEAM_LENGTH_CONTROL" = "true" ]; then
            DECODE_FLAGS="--val_beam_size $VAL_BEAM_SIZE --length_penalty $LENGTH_PENALTY"
        fi
    fi
    
    EVAL_CMD="export HF_HOME=$HF_HOME && export HF_DATASETS_CACHE=$HF_DATASETS_CACHE && export HF_MODULES_CACHE=$HF_MODULES_CACHE && export HF_METRICS_CACHE=$HF_METRICS_CACHE && export TOKENIZERS_PARALLELISM=false && python -m wandb login $WANDB_API_KEY && python scripts/evaluate_distributed_checkpoints_multigpu.py --model $MODEL --checkpoint_dir $CHECKPOINT_DIR --val_dataset $VAL_DATASET --val_batch_size $VAL_BATCH_SIZE --val_data_size $VAL_DATA_SIZE --hf_token $HUGGINGFACE_TOKEN --wandb_project lm-evaluation $DECODE_FLAGS --use_multi_gpu $LENGTH_FLAGS --major_checkpoint_interval $MAJOR_CHECKPOINT_INTERVAL $NLI_FLAGS $KEEP_FLAG $FORCE_FLAG$NLI_SUBSET_FLAG $MODE_FLAGS"

    # Apptainer commands - all on one line
    # Install bert_score for BERTScore evaluation (required by evaluate library)
    COMMAND="pip install --upgrade pip setuptools wheel > /dev/null 2>&1 && pip install --break-system-packages --only-binary=:all: transformers==4.45.2 tokenizers==0.20.3 datasets==3.0.0 sentencepiece==0.2.1 > /dev/null 2>&1 && pip install --break-system-packages --no-binary=rouge-score rouge-score==0.1.2 > /dev/null 2>&1 && pip install --break-system-packages --no-deps accelerate==1.0.1 peft==0.13.2 > /dev/null 2>&1 && pip install --break-system-packages --upgrade pyyaml psutil typing-extensions packaging==24.2 safetensors > /dev/null 2>&1 && pip install --break-system-packages --upgrade wandb evaluate sentence-transformers bert_score simplemma > /dev/null 2>&1 && $EVAL_CMD"
    
    # Run evaluation directly (already in SLURM job context from sbatch)
    apptainer exec --nv $IMAGE bash -c "$COMMAND"
    
    EVAL_STATUS=$?
    
    # Check evaluation status
    if [ $EVAL_STATUS -eq 0 ]; then
        echo "✓ Evaluation completed successfully"
    else
        echo "✗ Evaluation failed with exit code: $EVAL_STATUS"
    fi
    
    echo ""
done

# Print summary from JSON file (generated by Python script)
SUMMARY_FILE="$RESULTS_DIR/evaluation_summary.json"
if [ -f "$SUMMARY_FILE" ]; then
    echo "==================================================================="
    echo "EVALUATION SUMMARY"
    echo "==================================================================="
    python3 -c "
import json
import sys

summary_file = sys.argv[1]

try:
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    print(f\"Model: {summary.get('model', 'unknown')}\")
    stats = summary.get('statistics', {})
    print(f\"Total checkpoints: {stats.get('total_checkpoints', 0)}\")
    print(f\"Successful: {stats.get('successful', 0)}\")
    print(f\"Failed: {stats.get('failed', 0)}\")
    print()
    
    if summary.get('best_checkpoint'):
        print(f\"Best checkpoint: {summary['best_checkpoint']}\")
        print(f\"Best ROUGE-Lsum: {summary.get('best_rouge_lsum', 0):.2f}\")
        print()
    
    print('Checkpoint results:')
    for ckpt in summary.get('checkpoints', []):
        status_icon = '✓' if ckpt.get('status') == 'success' else '✗'
        print(f\"  {status_icon} {ckpt.get('checkpoint', 'unknown')}: \", end='')
        if ckpt.get('status') == 'success':
            rouge1 = ckpt.get('rouge1', 0)
            rouge2 = ckpt.get('rouge2', 0)
            rougeL = ckpt.get('rougel', 0)
            rougeLsum = ckpt.get('rougelsum', 0)
            print(f\"R1={rouge1:.2f}, R2={rouge2:.2f}, RL={rougeL:.2f}, RLsum={rougeLsum:.2f}\")
        else:
            print(f\"Status: {ckpt.get('status', 'unknown')}\")
except FileNotFoundError:
    print('Summary file not found yet. It will be created after first evaluation.')
except Exception as e:
    print(f'Error reading summary: {e}')
" "$SUMMARY_FILE"
    echo ""
    echo "==================================================================="
    echo "Detailed results saved to: $RESULTS_DIR"
    echo "JSON summary saved to: $SUMMARY_FILE"
    echo "==================================================================="
else
    echo "==================================================================="
    echo "Summary file will be created after evaluations complete"
    echo "==================================================================="
fi
