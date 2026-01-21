#!/usr/bin/env bash
#SBATCH -J gpu-eval-checkpoints-multigpu
#SBATCH -p accel
#SBATCH --gres=gpu:4  # Request 4 GPUs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16  # Increase CPUs for multi-GPU
#SBATCH --mem=384G  # Increase memory for multi-GPU
#SBATCH --time=12:00:00
#SBATCH -A nn12075k
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# ===== CONFIGURATION =====
# Set the model and checkpoint directory to evaluate
# Supported models: gemma-2b, gemma-7b, viking-7b, viking-13b, normistral-7b, normistral-11b,
#                   norskgpt-llama3-8b, llama-2-13b-chat-norwegian
MODEL=llama-2-13b-chat-norwegian
# Default checkpoint directory pattern (adjust if your naming convention differs)
CHECKPOINT_BASE_DIR="models/${MODEL}-apptainer-fsdp"  # Directory containing checkpoint-* subdirectories
VAL_DATASET="data/output/new_processed_data_val.jsonl"  # Match training script

# Optional: Evaluate specific checkpoints (space-separated numbers)
# Leave empty to evaluate ALL checkpoints
SPECIFIC_CHECKPOINTS=""  # e.g., "100 200 300" or "" for all

# Number of GPUs to use
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}

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

# Load environment variables
source .env

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
SUMMARY_FILE="$RESULTS_DIR/evaluation_summary.txt"

# Clear previous summary
> "$SUMMARY_FILE"

echo "Results will be saved to: $RESULTS_DIR"
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
    
    # Extract results if evaluation succeeded
    if [ $EVAL_STATUS -eq 0 ]; then
        RESULT_FILE="$CHECKPOINT_DIR/eval_results/eval_results.json"
        if [ -f "$RESULT_FILE" ]; then
            echo "✓ Evaluation completed successfully"
            echo "$CHECKPOINT_NAME:" >> "$SUMMARY_FILE"
            python3 -c "
import json
with open('$RESULT_FILE', 'r') as f:
    results = json.load(f)
    for key, value in results.items():
        if 'rouge' in key.lower():
            print(f'  {key}: {value:.2f}')
            with open('$SUMMARY_FILE', 'a') as summary:
                summary.write(f'  {key}: {value:.2f}\n')
" || echo "  Could not parse results"
            echo "" >> "$SUMMARY_FILE"
        else
            echo "⚠ WARNING: Result file not found: $RESULT_FILE"
        fi
    else
        echo "✗ Evaluation failed with exit code: $EVAL_STATUS"
        echo "$CHECKPOINT_NAME: FAILED (exit code $EVAL_STATUS)" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
    fi
    
    echo ""
done

# Print summary
echo "==================================================================="
echo "EVALUATION SUMMARY"
echo "==================================================================="
cat "$SUMMARY_FILE"
echo "==================================================================="
echo "Detailed results saved to: $RESULTS_DIR"
echo "Summary saved to: $SUMMARY_FILE"
echo "==================================================================="
