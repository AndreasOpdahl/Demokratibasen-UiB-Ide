#!/usr/bin/env bash
#SBATCH -J gpu-finetune-apptainer
#SBATCH -p accel
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=25:00:00
# Account: REQUIRED - Pass via: sbatch --account=YOUR_ACCOUNT run_finetune_multinode.sbatch ...
# Example: sbatch --account=nnXDXFWF run_finetune_multinode.sbatch --model=gemma-2-9b
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

# Note: SLURM_ACCOUNT is hardcoded in #SBATCH -A line above (nnXDXFWF)
# If you need a different account, update the #SBATCH -A line directly
# SLURM_ACCOUNT from .env is still used for informational purposes

# Parse command-line arguments or use defaults
# Usage: sbatch --export=SLURM_ACCOUNT=nnXDXFWF run_finetune_multinode.sbatch [--model=MODEL] [--num_gpus=N] [--train_dataset=PATH] [--val_dataset=PATH] [--train_batch_size=N] [--val_batch_size=N] [--resume_checkpoint=CHECKPOINT] [--task_limit="--max_steps N"] [--distribution_flag="--fsdp"] [--dry-run]
MODEL=${MODEL:-gemma-2-9b}
NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}
TRAIN_DATASET=${TRAIN_DATASET:-"data/output/new_processed_data_train.jsonl"}
VAL_DATASET=${VAL_DATASET:-"data/output/new_processed_data_val.jsonl"}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-""}  # Empty = use model config default
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-""}      # Empty = use model config default
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-""}
TASK_LIMIT=${TASK_LIMIT:-"--max_steps 10000"}
DISTRIBUTION_FLAG=${DISTRIBUTION_FLAG:-"--fsdp"}
DRY_RUN=${DRY_RUN:-false}

# Parse command-line arguments if provided
for arg in "$@"; do
    case $arg in
        --model=*)
            MODEL="${arg#*=}"
            ;;
        --num_gpus=*)
            NUM_GPUS="${arg#*=}"
            ;;
        --resume_checkpoint=*)
            RESUME_CHECKPOINT="${arg#*=}"
            ;;
        --task_limit=*)
            TASK_LIMIT="${arg#*=}"
            ;;
        --distribution_flag=*)
            DISTRIBUTION_FLAG="${arg#*=}"
            ;;
        --train_dataset=*)
            TRAIN_DATASET="${arg#*=}"
            ;;
        --val_dataset=*)
            VAL_DATASET="${arg#*=}"
            ;;
        --train_batch_size=*)
            TRAIN_BATCH_SIZE="${arg#*=}"
            ;;
        --val_batch_size=*)
            VAL_BATCH_SIZE="${arg#*=}"
            ;;
        --dry-run|--debug)
            DRY_RUN=true
            ;;
        *)
            echo "ERROR: Unknown argument: $arg"
            echo ""
            echo "Usage: sbatch --export=SLURM_ACCOUNT=YOUR_PROJECT_ID run_finetune_multinode.sbatch [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model=MODEL              Model short name (default: gemma-2-9b)"
            echo "  --num_gpus=N               Number of GPUs (default: 4)"
            echo "  --train_dataset=PATH       Path to training dataset JSONL file (default: data/output/new_processed_data_train.jsonl)"
            echo "  --val_dataset=PATH         Path to validation dataset JSONL file (default: data/output/new_processed_data_val.jsonl)"
            echo "  --train_batch_size=N       Training batch size per device (default: use model config default)"
            echo "  --val_batch_size=N         Validation batch size per device (default: use model config default)"
            echo "  --resume_checkpoint=DIR    Checkpoint directory to resume from (default: none)"
            echo "  --task_limit=\"OPTIONS\"     Task limit, e.g. \"--max_steps 10000\" (default: --max_steps 10000)"
            echo "  --distribution_flag=FLAG   Distribution flag: --fsdp or --ddp (default: --fsdp)"
            echo "  --dry-run|--debug          Debug mode: validate inputs and show what would run (no job submission)"
            echo ""
            echo "Valid models: gemma-2b, gemma-7b, gemma-2-9b, gemma-2-27b, gemma-3-12b, gemma-3-27b,"
            echo "              viking-7b, viking-13b, viking-33b, normistral-7b, normistral-11b, normistral-7b-instruct,"
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
        gemma-2b|gemma-7b|gemma-2-9b|gemma-2-27b|gemma-3-12b|gemma-3-27b| \
        viking-7b|viking-13b|viking-33b| \
        normistral-7b|normistral-11b|normistral-7b-instruct| \
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

validate_task_limit() {
    local limit=$1
    if [[ "$limit" =~ ^--max_steps\ [0-9]+$ ]] || [[ "$limit" =~ ^--num_train_epochs\ [0-9]+(\.[0-9]+)?$ ]] || [ -z "$limit" ]; then
        return 0
    fi
    return 1
}

# Validate inputs
ERRORS=0

if ! validate_model "$MODEL"; then
    echo "ERROR: Invalid model name: $MODEL"
    echo "Valid models: gemma-2b, gemma-7b, gemma-2-9b, gemma-2-27b, gemma-3-12b, gemma-3-27b,"
    echo "              viking-7b, viking-13b, viking-33b, normistral-7b, normistral-11b, normistral-7b-instruct,"
    echo "              norskgpt-llama3-8b, llama-3.1-8b-instruct, llama-2-13b-chat-norwegian,"
    echo "              eurollm-9b-instruct, norwai-mistral-7b-instruct, nb-gpt-j-6b, mt5"
    ERRORS=$((ERRORS + 1))
fi

if ! validate_num_gpus "$NUM_GPUS"; then
    echo "ERROR: Invalid num_gpus: $NUM_GPUS"
    echo "num_gpus must be a positive integer between 1 and 8"
    ERRORS=$((ERRORS + 1))
fi

if ! validate_distribution_flag "$DISTRIBUTION_FLAG"; then
    echo "ERROR: Invalid distribution_flag: $DISTRIBUTION_FLAG"
    echo "distribution_flag must be: --fsdp, --ddp, or empty"
    ERRORS=$((ERRORS + 1))
fi

if ! validate_task_limit "$TASK_LIMIT"; then
    echo "ERROR: Invalid task_limit format: $TASK_LIMIT"
    echo "task_limit must be in format: \"--max_steps N\" or \"--num_train_epochs N\" or empty"
    ERRORS=$((ERRORS + 1))
fi

if [ -n "$RESUME_CHECKPOINT" ]; then
    # Determine output_dir for validation (same logic as later in script)
    case "$MODEL" in
        gemma-2b) FULL_MODEL_NAME="google/gemma-2b" ;;
        gemma-7b) FULL_MODEL_NAME="google/gemma-7b" ;;
        gemma-2-9b) FULL_MODEL_NAME="google/gemma-2-9b" ;;
        gemma-2-27b) FULL_MODEL_NAME="google/gemma-2-27b" ;;
        gemma-3-12b) FULL_MODEL_NAME="google/gemma-3-12b-pt" ;;
        gemma-3-27b) FULL_MODEL_NAME="google/gemma-3-27b-pt" ;;
        viking-7b) FULL_MODEL_NAME="LumiOpen/Viking-7B" ;;
        viking-13b) FULL_MODEL_NAME="LumiOpen/Viking-13B" ;;
        viking-33b) FULL_MODEL_NAME="LumiOpen/Viking-33B" ;;
        normistral-7b) FULL_MODEL_NAME="norallm/normistral-7b-warm" ;;
        normistral-11b) FULL_MODEL_NAME="norallm/normistral-11b-warm" ;;
        normistral-7b-instruct) FULL_MODEL_NAME="norallm/normistral-7b-warm-instruct" ;;
        norskgpt-llama3-8b) FULL_MODEL_NAME="bineric/norskgpt-llama3-8b" ;;
        llama-3.1-8b-instruct) FULL_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" ;;
        llama-2-13b-chat-norwegian) FULL_MODEL_NAME="ruternorway/llama-2-13b-chat-norwegian" ;;
        eurollm-9b-instruct) FULL_MODEL_NAME="utter-project/EuroLLM-9B-2512-instruct" ;;
        norwai-mistral-7b-instruct) FULL_MODEL_NAME="NorwAI/NorwAI-Mistral-7B-instruct" ;;
        nb-gpt-j-6b) FULL_MODEL_NAME="NbAiLab/nb-gpt-j-6B-torgersen-alpaca" ;;
        *) FULL_MODEL_NAME="$MODEL" ;;
    esac
    OUTPUT_DIR_VALIDATION="models/${MODEL}-apptainer-fsdp"
    
    # Try multiple resolution strategies (same as Python script)
    CHECKPOINT_FOUND=false
    
    # Strategy 1: Absolute path
    if [[ "$RESUME_CHECKPOINT" == /* ]]; then
        if [ -d "$RESUME_CHECKPOINT" ]; then
            CHECKPOINT_FOUND=true
        fi
    fi
    
    # Strategy 2: Simple name (e.g., "checkpoint-5000") - check relative to output_dir
    if [ "$CHECKPOINT_FOUND" = false ]; then
        # Check if it's a simple name (no path separators)
        if [[ "$RESUME_CHECKPOINT" != */* ]] && [[ "$RESUME_CHECKPOINT" != *\\* ]]; then
            CHECKPOINT_PATH="${OUTPUT_DIR_VALIDATION}/${RESUME_CHECKPOINT}"
            if [ -d "$CHECKPOINT_PATH" ]; then
                CHECKPOINT_FOUND=true
            fi
        fi
    fi
    
    # Strategy 3: Relative to current directory
    if [ "$CHECKPOINT_FOUND" = false ]; then
        if [ -d "$RESUME_CHECKPOINT" ]; then
            CHECKPOINT_FOUND=true
        fi
    fi
    
    # Strategy 4: Relative to output_dir (even if it has path separators)
    if [ "$CHECKPOINT_FOUND" = false ]; then
        CHECKPOINT_PATH="${OUTPUT_DIR_VALIDATION}/${RESUME_CHECKPOINT}"
        if [ -d "$CHECKPOINT_PATH" ]; then
            CHECKPOINT_FOUND=true
        fi
    fi
    
    if [ "$CHECKPOINT_FOUND" = false ]; then
        echo "ERROR: Resume checkpoint directory not found: $RESUME_CHECKPOINT"
        echo "       Tried:"
        if [[ "$RESUME_CHECKPOINT" == /* ]]; then
            echo "         - $RESUME_CHECKPOINT (absolute path)"
        fi
        if [[ "$RESUME_CHECKPOINT" != */* ]] && [[ "$RESUME_CHECKPOINT" != *\\* ]]; then
            echo "         - ${OUTPUT_DIR_VALIDATION}/${RESUME_CHECKPOINT} (relative to output_dir)"
        fi
        echo "         - $RESUME_CHECKPOINT (relative to current directory)"
        echo "         - ${OUTPUT_DIR_VALIDATION}/${RESUME_CHECKPOINT} (relative to output_dir)"
        echo ""
        echo "       Please provide:"
        echo "         - Simple name: checkpoint-5000 (recommended, resolved relative to ${OUTPUT_DIR_VALIDATION})"
        echo "         - Absolute path: /full/path/to/checkpoint-5000"
        echo "         - Relative path: models/gemma-2-9b-apptainer-fsdp/checkpoint-5000"
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
echo "  TRAIN_DATASET: $TRAIN_DATASET"
echo "  VAL_DATASET: $VAL_DATASET"
echo "  TRAIN_BATCH_SIZE: ${TRAIN_BATCH_SIZE:-use model config default}"
echo "  VAL_BATCH_SIZE: ${VAL_BATCH_SIZE:-use model config default}"
echo "  RESUME_CHECKPOINT: ${RESUME_CHECKPOINT:-none}"
echo "  TASK_LIMIT: $TASK_LIMIT"
echo "  DISTRIBUTION_FLAG: $DISTRIBUTION_FLAG"
echo "  DRY_RUN: $DRY_RUN"

# Verify required environment variables are set
if [ -z "${WANDB_API_KEY:-}" ] || [ -z "${HUGGINGFACE_TOKEN:-}" ]; then
    echo "ERROR: Required environment variables not set. Please check your .env file contains:"
    echo "  WANDB_API_KEY"
    echo "  HUGGINGFACE_TOKEN"
    exit 1
fi

# ===== DRY-RUN MODE =====
# Note: Functions are defined later, but we'll compute values here for dry-run
if [ "$DRY_RUN" = "true" ]; then
    echo ""
    echo "==================================================================="
    echo "DRY-RUN MODE: Validation and Preview"
    echo "==================================================================="
    echo ""
    
    # Compute model info (functions defined later, but we can compute manually for preview)
    case "$MODEL" in
        gemma-2b) FULL_MODEL_NAME="google/gemma-2b" ;;
        gemma-7b) FULL_MODEL_NAME="google/gemma-7b" ;;
        gemma-2-9b) FULL_MODEL_NAME="google/gemma-2-9b" ;;
        gemma-2-27b) FULL_MODEL_NAME="google/gemma-2-27b" ;;
        gemma-3-12b) FULL_MODEL_NAME="google/gemma-3-12b-pt" ;;
        gemma-3-27b) FULL_MODEL_NAME="google/gemma-3-27b-pt" ;;
        viking-7b) FULL_MODEL_NAME="LumiOpen/Viking-7B" ;;
        viking-13b) FULL_MODEL_NAME="LumiOpen/Viking-13B" ;;
        viking-33b) FULL_MODEL_NAME="LumiOpen/Viking-33B" ;;
        normistral-7b) FULL_MODEL_NAME="norallm/normistral-7b-warm" ;;
        normistral-11b) FULL_MODEL_NAME="norallm/normistral-11b-warm" ;;
        normistral-7b-instruct) FULL_MODEL_NAME="norallm/normistral-7b-warm-instruct" ;;
        norskgpt-llama3-8b) FULL_MODEL_NAME="bineric/norskgpt-llama3-8b" ;;
        llama-3.1-8b-instruct) FULL_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" ;;
        llama-2-13b-chat-norwegian) FULL_MODEL_NAME="ruternorway/llama-2-13b-chat-norwegian" ;;
        eurollm-9b-instruct) FULL_MODEL_NAME="utter-project/EuroLLM-9B-2512-instruct" ;;
        norwai-mistral-7b-instruct) FULL_MODEL_NAME="NorwAI/NorwAI-Mistral-7B-instruct" ;;
        nb-gpt-j-6b) FULL_MODEL_NAME="NbAiLab/nb-gpt-j-6B-torgersen-alpaca" ;;
        *) FULL_MODEL_NAME="$MODEL" ;;
    esac
    MODEL_CACHE_DIR="models--${FULL_MODEL_NAME//\//--}"
    OUTPUT_DIR="models/${MODEL}-apptainer-fsdp"
    
    echo "✓ All validations passed!"
    echo ""
    echo "Would execute with the following configuration:"
    echo "  Full model name: $FULL_MODEL_NAME"
    echo "  Model cache dir: $MODEL_CACHE_DIR"
    echo "  Output directory: $OUTPUT_DIR"
    echo "  Training command would be:"
    echo "    torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS ..."
    echo "    scripts/wandb_finetune.py --model $MODEL --quantization none $DISTRIBUTION_FLAG"
    echo "    --hf_token [REDACTED]"
    echo "    --train_dataset $TRAIN_DATASET"
    echo "    --val_dataset $VAL_DATASET"
    echo "    $TASK_LIMIT"
    echo "    --output_dir $OUTPUT_DIR ${RESUME_CHECKPOINT:+--resume_checkpoint $RESUME_CHECKPOINT}"
    echo ""
    
    # Check if files/directories exist (non-blocking in dry-run)
    echo "File/Directory checks:"
    if [ -f "$TRAIN_DATASET" ]; then
        echo "  ✓ Training dataset exists: $TRAIN_DATASET"
    else
        echo "  ⚠ Training dataset not found: $TRAIN_DATASET"
    fi
    
    if [ -f "$VAL_DATASET" ]; then
        echo "  ✓ Validation dataset exists: $VAL_DATASET"
    else
        echo "  ⚠ Validation dataset not found: $VAL_DATASET"
    fi
    
    if [ -n "$RESUME_CHECKPOINT" ]; then
        if [ -d "$RESUME_CHECKPOINT" ]; then
            echo "  ✓ Resume checkpoint directory exists"
        else
            echo "  ⚠ Resume checkpoint directory not found: $RESUME_CHECKPOINT"
        fi
    fi
    
    echo ""
    echo "Environment variables:"
    echo "  ✓ SLURM_ACCOUNT: ${SLURM_ACCOUNT:-NOT SET}"
    echo "  ✓ WANDB_API_KEY: ${WANDB_API_KEY:+SET (hidden)}"
    echo "  ✓ HUGGINGFACE_TOKEN: ${HUGGINGFACE_TOKEN:+SET (hidden)}"
    echo ""
    echo "==================================================================="
    echo "Dry-run complete. No job was submitted."
    echo "To actually run, remove --dry-run flag."
    echo "==================================================================="
    exit 0
fi

export SLURM_TMPDIR=/tmp/${USER}_$SLURM_JOB_ID
mkdir -p "$SLURM_TMPDIR"

export HF_HOME="$SLURM_TMPDIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_METRICS_CACHE="$HF_HOME/metrics"
# Set PYTHONUSERBASE to writable location to avoid .local read-only filesystem errors
export PYTHONUSERBASE="$SLURM_TMPDIR/.local"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$HF_METRICS_CACHE" "$PYTHONUSERBASE"

PROJECT_HF_DIR=/cluster/projects/YOUR_PROJECT_ID/shared/.cache/huggingface
mkdir -p "$HF_HOME/hub"

# Convert model name to HuggingFace cache directory format
# e.g., "google/gemma-7b" -> "models--google--gemma-7b"
# e.g., "NorwAI/NorMistral-7B-warm-IT" -> "models--NorwAI--NorMistral-7B-warm-IT"
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
        viking-7b) echo "LumiOpen/Viking-7B" ;;
        viking-13b) echo "LumiOpen/Viking-13B" ;;
        normistral-7b) echo "norallm/normistral-7b-warm" ;;
        normistral-11b) echo "norallm/normistral-11b-warm" ;;
        normistral-7b-instruct) echo "norallm/normistral-7b-warm-instruct" ;;
        norskgpt-llama3-8b) echo "bineric/norskgpt-llama3-8b" ;;
        llama-3.1-8b-instruct) echo "meta-llama/Llama-3.1-8B-Instruct" ;;
        llama-2-13b-chat-norwegian) echo "ruternorway/llama-2-13b-chat-norwegian" ;;
        eurollm-9b-instruct) echo "utter-project/EuroLLM-9B-2512-instruct" ;;
        norwai-mistral-7b-instruct) echo "NorwAI/NorwAI-Mistral-7B-instruct" ;;
        nb-gpt-j-6b) echo "NbAiLab/nb-gpt-j-6B-torgersen-alpaca" ;;
        gemma-2-9b) echo "google/gemma-2-9b" ;;
        gemma-2-27b) echo "google/gemma-2-27b" ;;
        gemma-3-12b) echo "google/gemma-3-12b-pt" ;;
        gemma-3-27b) echo "google/gemma-3-27b-pt" ;;
        viking-33b) echo "LumiOpen/Viking-33B" ;;
        *) echo "$short_name" ;;  # Assume it's already a full name
    esac
}

FULL_MODEL_NAME=$(get_full_model_name "$MODEL")
MODEL_CACHE_DIR=$(convert_model_to_cache_dir "$FULL_MODEL_NAME")
SOURCE_MODEL_DIR="$PROJECT_HF_DIR/hub/$MODEL_CACHE_DIR"

# Copy model from read-only project storage to writable job-local cache
if [ -d "$SOURCE_MODEL_DIR" ]; then
    echo "Copying model $FULL_MODEL_NAME to job-local cache..."
    rsync -aq "$SOURCE_MODEL_DIR/" "$HF_HOME/hub/$MODEL_CACHE_DIR/"
    echo "Model cache copy complete"
else
    echo "WARNING: Model $FULL_MODEL_NAME not found in shared cache at $SOURCE_MODEL_DIR"
    echo "Model will be downloaded during training (requires internet access)"
fi

[ -d "$PROJECT_HF_DIR/metrics" ] && rsync -aq "$PROJECT_HF_DIR/metrics/" "$HF_METRICS_CACHE/"
[ -d "$PROJECT_HF_DIR/modules" ] && rsync -aq "$PROJECT_HF_DIR/modules/" "$HF_MODULES_CACHE/"

# Check for container command BEFORE loading modules (like the working monitor script)
# The monitor script works without modules, so apptainer should be available in base environment
CONTAINER_CMD=""
if command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
    echo "Found apptainer command (before module load)"
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
    echo "Found singularity command (before module load)"
else
    echo "WARNING: Container command not found before module load. Will check again after."
fi

module --force purge
module load BuildEnv/NeoverseV2
module load NRIS/GPU
module load CUDA/12.6.0
module load PyTorch/2.8.0

source .env

# Re-check container command after loading modules (in case it's provided by a module)
if [ -z "$CONTAINER_CMD" ]; then
    if command -v apptainer &> /dev/null; then
        CONTAINER_CMD="apptainer"
        echo "Found apptainer command (after module load)"
    elif command -v singularity &> /dev/null; then
        CONTAINER_CMD="singularity"
        echo "Found singularity command (after module load)"
    else
        echo "ERROR: Neither 'apptainer' nor 'singularity' command found."
        echo "Please check if Apptainer/Singularity is available in your environment."
        echo "Current PATH: $PATH"
        exit 1
    fi
fi

export IMAGE=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif
export CONTAINER_CMD  # Export so it's available in the script
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)

# Calculate MASTER_PORT after modules are loaded (modules may provide python)
# Try python3 first, then python, then use a default
MASTER_PORT=""
if command -v python3 &> /dev/null; then
    MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()' 2>/dev/null)
elif command -v python &> /dev/null; then
    MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()' 2>/dev/null)
fi

# Fallback: use a random port in the ephemeral range if python is not available
if [ -z "$MASTER_PORT" ]; then
    # Generate a random port between 29500-30000 (common range for distributed training)
    MASTER_PORT=$((29500 + RANDOM % 500))
    echo "WARNING: Could not use python to get available port, using random port: $MASTER_PORT"
fi

export MASTER_PORT

[[ -n "$RESUME_CHECKPOINT" ]] && RESUME_ARG="--resume_checkpoint $RESUME_CHECKPOINT" || RESUME_ARG=""

# Write commands to a script file on SHARED filesystem (PWD is shared)
# Use absolute path to ensure it's accessible from within apptainer container
SCRIPT_DIR="${PWD}/logs"
mkdir -p "$SCRIPT_DIR"
SCRIPT_FILE="${SCRIPT_DIR}/run_training_${SLURM_JOB_ID}.sh"
cat > "$SCRIPT_FILE" << 'SCRIPT_EOF'
#!/bin/bash
set -e

echo "=== Step 1: Environment setup ==="
export HF_HOME=HF_HOME_PLACEHOLDER
export HF_MODULES_CACHE=HF_MODULES_CACHE_PLACEHOLDER
export HF_METRICS_CACHE=HF_METRICS_CACHE_PLACEHOLDER
export PYTHONUSERBASE=PYTHONUSERBASE_PLACEHOLDER
mkdir -p "$PYTHONUSERBASE"
mkdir -p "$PYTHONUSERBASE/bin"
# Add PYTHONUSERBASE/bin to PATH so pip-installed scripts are available
export PATH="$PYTHONUSERBASE/bin:$PATH"
export NCCL_DEBUG=NCCL_DEBUG_PLACEHOLDER
export NCCL_IB_DISABLE=NCCL_IB_DISABLE_PLACEHOLDER
export NCCL_P2P_DISABLE=NCCL_P2P_DISABLE_PLACEHOLDER
export NCCL_SOCKET_IFNAME=
export TORCH_NCCL_ASYNC_ERROR_HANDLING=TORCH_NCCL_ASYNC_ERROR_HANDLING_PLACEHOLDER
export MASTER_ADDR=MASTER_ADDR_PLACEHOLDER
export MASTER_PORT=MASTER_PORT_PLACEHOLDER

echo "=== Step 2: Install packages ==="
INSTALL_DONE_FLAG="INSTALL_FLAG_PLACEHOLDER"
# PYTHONUSERBASE is already set in Step 1 to redirect --user installs to writable location

if [ "$SLURM_NODEID" = "0" ]; then
    echo "Node 0: Installing packages..."
    pip install --break-system-packages --upgrade pip setuptools wheel
    pip install --break-system-packages --only-binary=:all: transformers==4.45.2 tokenizers==0.20.3 datasets==3.0.0 sentencepiece==0.2.1
    pip install --break-system-packages --no-binary=rouge-score rouge-score==0.1.2
    pip install --break-system-packages --no-deps accelerate==1.0.1 peft==0.13.2
    pip install --break-system-packages --upgrade pyyaml psutil typing-extensions packaging==24.2 safetensors
    pip install --break-system-packages --upgrade wandb evaluate
    echo ">>> All packages installed"
    touch "$INSTALL_DONE_FLAG"
else
    echo "Node $SLURM_NODEID: Waiting for node 0 to finish installing..."
    while [ ! -f "$INSTALL_DONE_FLAG" ]; do sleep 2; done
    echo "Node $SLURM_NODEID: Continuing..."
fi

echo "=== Step 3: Verify imports ==="
# Use python3 if available, otherwise python (after pip install, python3 should be available)
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi
$PYTHON_CMD -c "import transformers; print('transformers:', transformers.__version__)"
$PYTHON_CMD -c "import datasets; print('datasets:', datasets.__version__)"
$PYTHON_CMD -c "import accelerate; print('accelerate:', accelerate.__version__)"
$PYTHON_CMD -c "import peft; print('peft:', peft.__version__)"
$PYTHON_CMD -c "import torch; print('torch:', torch.__version__)"
$PYTHON_CMD -c "import numpy; print('numpy:', numpy.__version__)"

echo "=== Step 4: Login to wandb ==="
$PYTHON_CMD -m wandb login WANDB_API_KEY_PLACEHOLDER

echo "=== Network interfaces ==="
ip link show | grep -E "^[0-9]+" | cut -d: -f2

echo "=== Step 5: Launch training ==="
torchrun --nnodes=SLURM_NNODES_PLACEHOLDER --nproc_per_node=SLURM_GPUS_ON_NODE_PLACEHOLDER --node_rank=$SLURM_NODEID --rdzv_backend=c10d --rdzv_endpoint=MASTER_ADDR_PLACEHOLDER:MASTER_PORT_PLACEHOLDER scripts/wandb_finetune.py --model MODEL_PLACEHOLDER --quantization none DISTRIBUTION_FLAG_PLACEHOLDER --hf_token HUGGINGFACE_TOKEN_PLACEHOLDER --train_dataset TRAIN_DATASET_PLACEHOLDER --val_dataset VAL_DATASET_PLACEHOLDER TASK_LIMIT_PLACEHOLDER --output_dir models/MODEL_PLACEHOLDER-apptainer-fsdp RESUME_ARG_PLACEHOLDER
SCRIPT_EOF

# Replace placeholders with actual values
sed -i "s|HF_HOME_PLACEHOLDER|$HF_HOME|g" "$SCRIPT_FILE"
sed -i "s|HF_DATASETS_CACHE_PLACEHOLDER|$HF_DATASETS_CACHE|g" "$SCRIPT_FILE"
sed -i "s|HF_MODULES_CACHE_PLACEHOLDER|$HF_MODULES_CACHE|g" "$SCRIPT_FILE"
sed -i "s|HF_METRICS_CACHE_PLACEHOLDER|$HF_METRICS_CACHE|g" "$SCRIPT_FILE"
sed -i "s|PYTHONUSERBASE_PLACEHOLDER|$PYTHONUSERBASE|g" "$SCRIPT_FILE"
sed -i "s|NCCL_DEBUG_PLACEHOLDER|$NCCL_DEBUG|g" "$SCRIPT_FILE"
sed -i "s|NCCL_IB_DISABLE_PLACEHOLDER|$NCCL_IB_DISABLE|g" "$SCRIPT_FILE"
sed -i "s|NCCL_P2P_DISABLE_PLACEHOLDER|$NCCL_P2P_DISABLE|g" "$SCRIPT_FILE"
sed -i "s|NCCL_SOCKET_IFNAME_PLACEHOLDER|$NCCL_SOCKET_IFNAME|g" "$SCRIPT_FILE"
sed -i "s|TORCH_NCCL_ASYNC_ERROR_HANDLING_PLACEHOLDER|$TORCH_NCCL_ASYNC_ERROR_HANDLING|g" "$SCRIPT_FILE"
sed -i "s|MASTER_ADDR_PLACEHOLDER|$MASTER_ADDR|g" "$SCRIPT_FILE"
sed -i "s|MASTER_PORT_PLACEHOLDER|$MASTER_PORT|g" "$SCRIPT_FILE"
sed -i "s|WANDB_API_KEY_PLACEHOLDER|$WANDB_API_KEY|g" "$SCRIPT_FILE"
sed -i "s|SLURM_NNODES_PLACEHOLDER|$SLURM_NNODES|g" "$SCRIPT_FILE"
sed -i "s|SLURM_GPUS_ON_NODE_PLACEHOLDER|$SLURM_GPUS_ON_NODE|g" "$SCRIPT_FILE"
sed -i "s|MODEL_PLACEHOLDER|$MODEL|g" "$SCRIPT_FILE"
sed -i "s|DISTRIBUTION_FLAG_PLACEHOLDER|$DISTRIBUTION_FLAG|g" "$SCRIPT_FILE"
sed -i "s|HUGGINGFACE_TOKEN_PLACEHOLDER|$HUGGINGFACE_TOKEN|g" "$SCRIPT_FILE"
sed -i "s|TASK_LIMIT_PLACEHOLDER|$TASK_LIMIT|g" "$SCRIPT_FILE"
sed -i "s|TRAIN_DATASET_PLACEHOLDER|$TRAIN_DATASET|g" "$SCRIPT_FILE"
sed -i "s|VAL_DATASET_PLACEHOLDER|$VAL_DATASET|g" "$SCRIPT_FILE"
# Add batch size arguments if provided (empty string if not provided)
TRAIN_BATCH_SIZE_ARG=""
VAL_BATCH_SIZE_ARG=""
[ -n "$TRAIN_BATCH_SIZE" ] && TRAIN_BATCH_SIZE_ARG="--train_batch_size $TRAIN_BATCH_SIZE"
[ -n "$VAL_BATCH_SIZE" ] && VAL_BATCH_SIZE_ARG="--val_batch_size $VAL_BATCH_SIZE"
sed -i "s|TRAIN_BATCH_SIZE_PLACEHOLDER|$TRAIN_BATCH_SIZE_ARG|g" "$SCRIPT_FILE"
sed -i "s|VAL_BATCH_SIZE_PLACEHOLDER|$VAL_BATCH_SIZE_ARG|g" "$SCRIPT_FILE"
sed -i "s|RESUME_ARG_PLACEHOLDER|$RESUME_ARG|g" "$SCRIPT_FILE"
sed -i "s|INSTALL_FLAG_PLACEHOLDER|${PWD}/logs/install_done_${SLURM_JOB_ID}|g" "$SCRIPT_FILE"

chmod +x "$SCRIPT_FILE"

echo "=== Script to execute ==="
cat "$SCRIPT_FILE"
echo "========================="

# Use absolute path for script file to ensure it's accessible from within apptainer
SCRIPT_FILE_ABS=$(cd "$(dirname "$SCRIPT_FILE")" && pwd)/$(basename "$SCRIPT_FILE")

# Verify script file exists and is executable
if [ ! -f "$SCRIPT_FILE_ABS" ]; then
    echo "ERROR: Script file not found: $SCRIPT_FILE_ABS"
    exit 1
fi

# Verify container command is available (should be set earlier)
if [ -z "$CONTAINER_CMD" ]; then
    echo "ERROR: CONTAINER_CMD not set. This should not happen."
    exit 1
fi

# Verify IMAGE exists
if [ ! -f "$IMAGE" ]; then
    echo "ERROR: Container image not found: $IMAGE"
    exit 1
fi

echo "Using script file: $SCRIPT_FILE_ABS"
echo "Using container command: $CONTAINER_CMD"
echo "Using container image: $IMAGE"

# Verify script file is readable
if [ ! -r "$SCRIPT_FILE_ABS" ]; then
    echo "ERROR: Script file is not readable: $SCRIPT_FILE_ABS"
    exit 1
fi

# Test if we can at least see the container command before srun
echo "Testing container command: $CONTAINER_CMD"
if ! $CONTAINER_CMD --version &> /dev/null; then
    echo "WARNING: Container command test failed, but continuing..."
fi

# Use /bin/bash explicitly to avoid PATH issues inside container
# Also ensure the script file path is absolute and accessible
echo "Launching training with srun..."
echo "Command: srun --ntasks=$SLURM_NNODES --cpus-per-task=${SLURM_CPUS_PER_TASK} --gres=gpu:${NUM_GPUS} $CONTAINER_CMD exec --nv --bind \"${PWD}:${PWD}\" $IMAGE /bin/bash \"$SCRIPT_FILE_ABS\""

# Ensure CONTAINER_CMD and PATH are exported for srun
# srun needs access to the container command and the environment
export CONTAINER_CMD
export PATH

# Launch with srun - export all environment variables to ensure apptainer is available
# Use --export=ALL to pass all environment variables (including PATH with apptainer) to srun tasks
srun --ntasks=$SLURM_NNODES \
     --cpus-per-task=${SLURM_CPUS_PER_TASK} \
     --gres=gpu:${NUM_GPUS} \
     --export=ALL \
    $CONTAINER_CMD exec \
        --nv \
        --bind "${PWD}:${PWD}" \
        $IMAGE \
        /bin/bash "$SCRIPT_FILE_ABS"
