#!/usr/bin/env bash
#SBATCH -J gpu-monitor-eval
#SBATCH -p accel
#SBATCH --gres=gpu:2  # Use fewer GPUs for evaluation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=48:00:00  # Long time to match training duration
# Account: REQUIRED - Pass via: sbatch --account=YOUR_ACCOUNT run_monitor_evaluation.sbatch ...
# Example: sbatch --account=YOUR_PROJECT_ID run_monitor_evaluation.sbatch --model=gemma-2-9b
#
# OPTIONAL: Use SLURM job dependency to ensure training starts first:
#   1. Submit training job first: TRAINING_JOB_ID=$(sbatch --account=YOUR_PROJECT_ID run_finetune_multinode.sbatch --model=gemma-2-9b | grep -oP '\d+')
#   2. Submit monitor job with dependency: sbatch --account=YOUR_PROJECT_ID --dependency=afterok:$TRAINING_JOB_ID run_monitor_evaluation.sbatch --model=gemma-2-9b
#
# If not using job dependency, the monitor script will wait up to 1 hour for training to start
# by checking for the training_started.txt signal file in the output directory.
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# Load environment variables from .env file
# .env should contain: WANDB_API_KEY, HUGGINGFACE_TOKEN
if [ -f .env ]; then
    source .env
else
    echo "WARNING: .env file not found. Using defaults or environment variables."
fi

# Verify required environment variables are set
if [ -z "${WANDB_API_KEY:-}" ] || [ -z "${HUGGINGFACE_TOKEN:-}" ]; then
    echo "ERROR: Required environment variables not set. Please check your .env file contains:"
    echo "  WANDB_API_KEY"
    echo "  HUGGINGFACE_TOKEN"
    exit 1
fi

# ===== CONFIGURATION =====
# Parse command-line arguments or use defaults
# Usage: sbatch run_monitor_evaluation.sbatch [--model=MODEL] [--output_dir=DIR] [--val_dataset=PATH] [--check_interval=N] [--early_stopping_patience=N] [--include_nli_faithfulness] [--dry-run]
MODEL=${MODEL:-gemma-2-9b}
OUTPUT_DIR=${OUTPUT_DIR:-"models/${MODEL}-apptainer-fsdp"}
VAL_DATASET=${VAL_DATASET:-"data/output/new_processed_data_val.jsonl"}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-500}  # 500 (default) or 1000; 1000 → separate -examples_1000.json/.jsonl files
CHECK_INTERVAL=${CHECK_INTERVAL:-60}  # Check every 60 seconds
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-3}  # Stop if no improvement for 3 evaluations
INCLUDE_NLI_FAITHFULNESS=${INCLUDE_NLI_FAITHFULNESS:-false}
DRY_RUN=${DRY_RUN:-false}

# Parse command-line arguments if provided
for arg in "$@"; do
    case $arg in
        --model=*)
            MODEL="${arg#*=}"
            # Update OUTPUT_DIR if it wasn't explicitly set
            if [ -z "${OUTPUT_DIR_SET:-}" ]; then
                OUTPUT_DIR="models/${MODEL}-apptainer-fsdp"
            fi
            ;;
        --output_dir=*)
            OUTPUT_DIR="${arg#*=}"
            OUTPUT_DIR_SET=1
            ;;
        --val_dataset=*)
            VAL_DATASET="${arg#*=}"
            ;;
        --val_data_size=*)
            VAL_DATA_SIZE="${arg#*=}"
            ;;
        --check_interval=*)
            CHECK_INTERVAL="${arg#*=}"
            ;;
        --early_stopping_patience=*)
            EARLY_STOPPING_PATIENCE="${arg#*=}"
            ;;
        --include_nli_faithfulness)
            INCLUDE_NLI_FAITHFULNESS=true
            ;;
        --dry-run|--debug)
            DRY_RUN=true
            ;;
        *)
            echo "ERROR: Unknown argument: $arg"
            echo ""
            echo "Usage: sbatch run_monitor_evaluation.sbatch [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model=MODEL                    Model short name (default: gemma-2-9b)"
            echo "  --output_dir=DIR                 Output directory containing checkpoints (default: models/\${MODEL}-apptainer-fsdp)"
            echo "  --val_dataset=PATH               Path to validation dataset JSONL file (default: data/output/new_processed_data_val.jsonl)"
            echo "  --val_data_size=N                 Validation examples: 500 (default) or 1000 (writes to -examples_1000.json/.jsonl)"
            echo "  --check_interval=N               Check for new checkpoints every N seconds (default: 60)"
            echo "  --early_stopping_patience=N      Stop if no improvement for N evaluations (default: 3)"
            echo "  --include_nli_faithfulness        Enable NLI-based faithfulness evaluation (slow: ~37 min for 500 examples)"
            echo "  --dry-run|--debug                Debug mode: validate inputs and show what would run (no job submission)"
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

validate_positive_integer() {
    local num=$1
    local name=$2
    if ! [[ "$num" =~ ^[0-9]+$ ]] || [ "$num" -lt 1 ]; then
        echo "ERROR: Invalid $name: $num"
        echo "$name must be a positive integer"
        return 1
    fi
    return 0
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

if ! validate_positive_integer "$CHECK_INTERVAL" "check_interval"; then
    ERRORS=$((ERRORS + 1))
fi

if ! validate_positive_integer "$EARLY_STOPPING_PATIENCE" "early_stopping_patience"; then
    ERRORS=$((ERRORS + 1))
fi

if [ ! -f "$VAL_DATASET" ]; then
    echo "ERROR: Validation dataset file does not exist: $VAL_DATASET"
    ERRORS=$((ERRORS + 1))
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "WARNING: Output directory does not exist: $OUTPUT_DIR"
    echo "         It will be created when training starts"
fi

if [ $ERRORS -gt 0 ]; then
    echo ""
    echo "Validation failed with $ERRORS error(s). Please fix the errors above and try again."
    exit 1
fi

echo "Configuration:"
echo "  MODEL: $MODEL"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  VAL_DATASET: $VAL_DATASET"
echo "  VAL_DATA_SIZE: $VAL_DATA_SIZE (1000 → separate -examples_1000.json/.jsonl files)"
echo "  CHECK_INTERVAL: $CHECK_INTERVAL seconds"
echo "  EARLY_STOPPING_PATIENCE: $EARLY_STOPPING_PATIENCE evaluations"
echo "  INCLUDE_NLI_FAITHFULNESS: $INCLUDE_NLI_FAITHFULNESS"
echo "  DRY_RUN: $DRY_RUN"

# ===== DRY-RUN MODE =====
if [ "$DRY_RUN" = "true" ]; then
    echo ""
    echo "==================================================================="
    echo "DRY-RUN MODE: Validation and Preview"
    echo "==================================================================="
    echo ""
    
    echo "✓ All validations passed!"
    echo ""
    echo "Would execute with the following configuration:"
    echo "  Monitoring checkpoints in: $OUTPUT_DIR"
    echo "  Validation dataset: $VAL_DATASET"
    echo "  Validation examples: $VAL_DATA_SIZE"
    echo "  Check interval: $CHECK_INTERVAL seconds"
    echo "  Early stopping patience: $EARLY_STOPPING_PATIENCE evaluations"
    echo "  Include NLI faithfulness: $INCLUDE_NLI_FAITHFULNESS"
    echo ""
    echo "Monitoring command would be:"
    echo "  python scripts/monitor_and_evaluate_checkpoints.py"
    echo "    --output_dir $OUTPUT_DIR"
    echo "    --model $MODEL"
    echo "    --val_dataset $VAL_DATASET"
    echo "    --val_data_size $VAL_DATA_SIZE"
    echo "    --hf_token [REDACTED]"
    echo "    --check_interval $CHECK_INTERVAL"
    echo "    --early_stopping_patience $EARLY_STOPPING_PATIENCE"
    if [ "$INCLUDE_NLI_FAITHFULNESS" = "true" ]; then
        echo "    --include_nli_faithfulness"
    fi
    echo "    --wandb_project lm-finetuning"
    echo "    --use_multi_gpu"
    echo ""
    echo "File/Directory checks:"
    if [ -d "$OUTPUT_DIR" ]; then
        echo "  ✓ Output directory exists"
    else
        echo "  ⚠ Output directory not found: $OUTPUT_DIR (will be created during training)"
    fi
    
    if [ -f "$VAL_DATASET" ]; then
        echo "  ✓ Validation dataset exists"
    else
        echo "  ⚠ Validation dataset not found: $VAL_DATASET"
    fi
    
    echo ""
    echo "Environment variables:"
    echo "  ✓ WANDB_API_KEY: ${WANDB_API_KEY:+SET (hidden)}"
    echo "  ✓ HUGGINGFACE_TOKEN: ${HUGGINGFACE_TOKEN:+SET (hidden)}"
    echo ""
    echo "==================================================================="
    echo "Dry-run complete. No job was submitted."
    echo "To actually run, remove --dry-run flag."
    echo "==================================================================="
    exit 0
fi

# Apptainer image
export IMAGE=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif

# Run monitoring script
# Build command with optional NLI faithfulness flag
NLI_FLAG=""
if [ "$INCLUDE_NLI_FAITHFULNESS" = "true" ]; then
    NLI_FLAG="--include_nli_faithfulness"
fi

EVAL_CMD="export HF_HOME=\$HF_HOME && export HF_DATASETS_CACHE=\$HF_DATASETS_CACHE && export HF_MODULES_CACHE=\$HF_MODULES_CACHE && export HF_METRICS_CACHE=\$HF_METRICS_CACHE && export TOKENIZERS_PARALLELISM=false && python -m wandb login \$WANDB_API_KEY && python scripts/monitor_and_evaluate_checkpoints.py --output_dir $OUTPUT_DIR --model $MODEL --val_dataset $VAL_DATASET --val_data_size $VAL_DATA_SIZE --hf_token \$HUGGINGFACE_TOKEN --check_interval $CHECK_INTERVAL --early_stopping_patience $EARLY_STOPPING_PATIENCE $NLI_FLAG --wandb_project lm-finetuning --use_multi_gpu"

# Run pip installs; hide stdout only so pip errors (stderr) appear in logs/gpu-monitor-eval-*.err
COMMAND="pip install --upgrade pip setuptools wheel >/dev/null && pip install --break-system-packages --only-binary=:all: transformers==4.45.2 tokenizers==0.20.3 datasets==3.0.0 sentencepiece==0.2.1 >/dev/null && pip install --break-system-packages --no-binary=rouge-score rouge-score==0.1.2 >/dev/null && pip install --break-system-packages --no-deps accelerate==1.0.1 peft==0.13.2 >/dev/null && pip install --break-system-packages --upgrade pyyaml psutil typing-extensions packaging==24.2 safetensors >/dev/null && pip install --break-system-packages --upgrade wandb evaluate >/dev/null && $EVAL_CMD"

apptainer exec --nv $IMAGE bash -c "$COMMAND"
