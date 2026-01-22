#!/usr/bin/env bash
#SBATCH -J gpu-eval-checkpoints-multigpu
#SBATCH -p accel
#SBATCH --gres=gpu:4  # Request 4 GPUs (adjust if needed)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16  # Increase CPUs for multi-GPU
#SBATCH --mem=384G  # Increase memory for multi-GPU
#SBATCH --time=12:00:00
# Account: REQUIRED - Pass via: sbatch --account=YOUR_ACCOUNT run_evaluate_distributed_checkpoints_multigpu.sbatch ...
# Example: sbatch --account=nn12075k run_evaluate_distributed_checkpoints_multigpu.sbatch --model=gemma-2-9b
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

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
# Usage: sbatch --export=SLURM_ACCOUNT=nn12075k run_evaluate_distributed_checkpoints_multigpu.sbatch [--model=MODEL] [--num_gpus=N] [--checkpoint_base_dir=DIR] [--val_dataset=PATH] [--specific_checkpoints="100 200 300"]
MODEL=${MODEL:-llama-2-13b-chat-norwegian}
NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-"models/${MODEL}-apptainer-fsdp"}
VAL_DATASET=${VAL_DATASET:-"data/output/new_processed_data_val.jsonl"}
SPECIFIC_CHECKPOINTS=${SPECIFIC_CHECKPOINTS:-""}
DRY_RUN=${DRY_RUN:-false}

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
        --specific_checkpoints=*)
            SPECIFIC_CHECKPOINTS="${arg#*=}"
            ;;
        --dry-run|--debug)
            DRY_RUN=true
            ;;
        *)
            echo "ERROR: Unknown argument: $arg"
            echo ""
            echo "Usage: sbatch --export=SLURM_ACCOUNT=nn12075k run_evaluate_distributed_checkpoints_multigpu.sbatch [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model=MODEL                    Model short name (default: llama-2-13b-chat-norwegian)"
            echo "  --num_gpus=N                     Number of GPUs (default: 4)"
            echo "  --checkpoint_base_dir=DIR         Base directory containing checkpoints (default: models/\${MODEL}-apptainer-fsdp)"
            echo "  --val_dataset=PATH               Path to validation dataset JSONL file (default: data/output/new_processed_data_val.jsonl)"
            echo "  --specific_checkpoints=\"N1 N2\"   Space-separated checkpoint numbers to evaluate (default: all)"
            echo "  --dry-run|--debug                Debug mode: validate inputs and show what would run (no job submission)"
            echo ""
            echo "Valid models: gemma-2b, gemma-7b, gemma-2-9b, gemma-2-27b, gemma-3-12b, gemma-3-27b,"
            echo "              viking-7b, viking-13b, viking-33b, normistral-7b, normistral-11b,"
            echo "              norskgpt-llama3-8b, llama-2-13b-chat-norwegian, mt5"
            exit 1
            ;;
    esac
done

# Validation functions
validate_model() {
    local model=$1
    case "$model" in
        gemma-2b|gemma-7b|gemma-2-9b|gemma-2-27b|gemma-3-12b|gemma-3-27b| \
        viking-7b|viking-13b|viking-33b| \
        normistral-7b|normistral-11b| \
        norskgpt-llama3-8b|llama-2-13b-chat-norwegian|mt5)
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
    echo "Valid models: gemma-2b, gemma-7b, gemma-2-9b, gemma-2-27b, gemma-3-12b, gemma-3-27b,"
    echo "              viking-7b, viking-13b, viking-33b, normistral-7b, normistral-11b,"
    echo "              norskgpt-llama3-8b, llama-2-13b-chat-norwegian, mt5"
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
echo "  SPECIFIC_CHECKPOINTS: ${SPECIFIC_CHECKPOINTS:-all}"
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
            if [ -d "$ckpt_dir" ]; then
                CHECKPOINTS="$CHECKPOINTS $ckpt_dir"
                echo "    ✓ Found: checkpoint-$ckpt_num"
            else
                echo "    ⚠ Not found: checkpoint-$ckpt_num"
            fi
        done
    else
        echo "  Would evaluate ALL checkpoints in: $CHECKPOINT_BASE_DIR"
        if [ -d "$CHECKPOINT_BASE_DIR" ]; then
            FOUND_CHECKPOINTS=$(find "$CHECKPOINT_BASE_DIR" -maxdepth 1 -type d -name "checkpoint-*" | wc -l)
            echo "    Found $FOUND_CHECKPOINTS checkpoint directory(ies)"
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
    echo "    --output_dir [checkpoint-dir]/eval_results"
    echo "    --wandb_project lm-evaluation"
    echo "    --use_greedy --use_multi_gpu"
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
PROJECT_HF_DIR=/cluster/projects/nn12075k/shared/.cache/huggingface
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
        gemma-7b) echo "google/gemma-7b" ;;
        gemma-2-9b) echo "google/gemma-2-9b" ;;
        gemma-2-27b) echo "google/gemma-2-27b" ;;
        gemma-3-12b) echo "google/gemma-3-12b-pt" ;;
        gemma-3-27b) echo "google/gemma-3-27b-pt" ;;
        viking-7b) echo "LumiOpen/Viking-7B" ;;
        viking-13b) echo "LumiOpen/Viking-13B" ;;
        viking-33b) echo "LumiOpen/Viking-33B" ;;
        normistral-7b) echo "norallm/normistral-7b-warm" ;;
        normistral-11b) echo "norallm/normistral-11b-warm" ;;
        norskgpt-llama3-8b) echo "bineric/norskgpt-llama3-8b" ;;
        llama-2-13b-chat-norwegian) echo "ruternorway/llama-2-13b-chat-norwegian" ;;
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
module --force purge
module load BuildEnv/NeoverseV2
module load NRIS/GPU
module load CUDA/12.6.0
module load PyTorch/2.8.0

# .env already sourced above, variables should be available

# Apptainer image
export IMAGE=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif

# Find all checkpoint directories
if [ -n "$SPECIFIC_CHECKPOINTS" ]; then
    CHECKPOINTS=""
    for ckpt_num in $SPECIFIC_CHECKPOINTS; do
        ckpt_dir="$CHECKPOINT_BASE_DIR/checkpoint-$ckpt_num"
        if [ -d "$ckpt_dir" ]; then
            CHECKPOINTS="$CHECKPOINTS $ckpt_dir"
        else
            echo "WARNING: Checkpoint not found: $ckpt_dir"
        fi
    done
else
    CHECKPOINTS=$(find "$CHECKPOINT_BASE_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V)
fi

# Count checkpoints
NUM_CHECKPOINTS=$(echo "$CHECKPOINTS" | wc -w)
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
    
    echo "==================================================================="
    echo "[$CHECKPOINT_NUM/$NUM_CHECKPOINTS] Evaluating: $CHECKPOINT_NAME"
    echo "Using $NUM_GPUS GPUs for parallel evaluation"
    echo "==================================================================="
    
    # Command to run inside apptainer - use model parallelism, not torchrun
    # Model parallelism (device_map="auto") doesn't need torchrun
    EVAL_CMD="export HF_HOME=$HF_HOME && export HF_DATASETS_CACHE=$HF_DATASETS_CACHE && export HF_MODULES_CACHE=$HF_MODULES_CACHE && export HF_METRICS_CACHE=$HF_METRICS_CACHE && export TOKENIZERS_PARALLELISM=false && python -m wandb login $WANDB_API_KEY && python scripts/evaluate_distributed_checkpoints_multigpu.py --model $MODEL --checkpoint_dir $CHECKPOINT_DIR --val_dataset $VAL_DATASET --hf_token $HUGGINGFACE_TOKEN --output_dir $CHECKPOINT_DIR/eval_results --wandb_project lm-evaluation --use_greedy --use_multi_gpu"

    # Apptainer commands - all on one line
    COMMAND="pip install --upgrade pip setuptools wheel > /dev/null 2>&1 && pip install --break-system-packages --only-binary=:all: transformers==4.45.2 tokenizers==0.20.3 datasets==3.0.0 sentencepiece==0.2.1 > /dev/null 2>&1 && pip install --break-system-packages --no-binary=rouge-score rouge-score==0.1.2 > /dev/null 2>&1 && pip install --break-system-packages --no-deps accelerate==1.0.1 peft==0.13.2 > /dev/null 2>&1 && pip install --break-system-packages --upgrade pyyaml psutil typing-extensions packaging==24.2 safetensors > /dev/null 2>&1 && pip install --break-system-packages --upgrade wandb evaluate > /dev/null 2>&1 && $EVAL_CMD"
    
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
