#!/usr/bin/env bash
#SBATCH -J gpu-eval-checkpoints
#SBATCH -p accel
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH -A nn12075k
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# ===== CONFIGURATION =====
# Set the model and checkpoint directory to evaluate
MODEL=gemma-7b
CHECKPOINT_BASE_DIR="models/gemma-7b-apptainer-fsdp"  # Directory containing checkpoint-* subdirectories
VAL_DATASET="data/output/processed_data_val.jsonl"

# Optional: Evaluate specific checkpoints (space-separated numbers)
# Leave empty to evaluate ALL checkpoints
SPECIFIC_CHECKPOINTS=""  # e.g., "100 200 300" or "" for all

echo "==================================================================="
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
echo "Copying model to job-local cache..."
PROJECT_HF_DIR=/cluster/projects/nn12075k/shared/.cache/huggingface
mkdir -p "$HF_HOME/hub"
rsync -av "$PROJECT_HF_DIR/hub/models--google--gemma-7b/" \
          "$HF_HOME/hub/models--google--gemma-7b/"

# Also copy metrics cache if it exists (needed for ROUGE)
if [ -d "$PROJECT_HF_DIR/metrics" ]; then
    echo "Copying metrics cache..."
    rsync -av "$PROJECT_HF_DIR/metrics/" "$HF_METRICS_CACHE/"
else
    echo "WARNING: No metrics cache found at $PROJECT_HF_DIR/metrics"
    echo "ROUGE metric may need to be downloaded during job (requires internet)"
fi

# Copy modules cache if it exists (needed for evaluate library)
if [ -d "$PROJECT_HF_DIR/modules" ]; then
    echo "Copying modules cache..."
    rsync -av "$PROJECT_HF_DIR/modules/" "$HF_MODULES_CACHE/"
fi

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
    # Evaluate specific checkpoints
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
    # Find all checkpoints, sorted by checkpoint number
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
    echo "==================================================================="
    
    # Command to run inside apptainer - all on one line to avoid escaping issues
    EVAL_CMD="export HF_HOME=$HF_HOME && export HF_DATASETS_CACHE=$HF_DATASETS_CACHE && export HF_MODULES_CACHE=$HF_MODULES_CACHE && export HF_METRICS_CACHE=$HF_METRICS_CACHE && python scripts/evaluate_distributed_checkpoints.py --model $MODEL --checkpoint_dir $CHECKPOINT_DIR --val_dataset $VAL_DATASET --hf_token $HUGGINGFACE_TOKEN --output_dir $CHECKPOINT_DIR/eval_results"

    # Apptainer commands - all on one line
    COMMAND="pip install --upgrade pip setuptools wheel > /dev/null 2>&1 && pip install --only-binary=:all: transformers==4.45.2 tokenizers==0.20.3 datasets==4.0.0 sentencepiece==0.2.1 > /dev/null 2>&1 && pip install --no-binary=rouge-score rouge-score==0.1.2 > /dev/null 2>&1 && pip install --no-deps accelerate==1.10.1 peft==0.17.1 > /dev/null 2>&1 && pip install --upgrade pyyaml psutil typing-extensions packaging==24.2 > /dev/null 2>&1 && $EVAL_CMD"
    
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
