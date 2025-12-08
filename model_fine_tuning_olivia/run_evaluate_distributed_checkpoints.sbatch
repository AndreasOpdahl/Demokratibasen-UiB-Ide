#!/usr/bin/env bash
#SBATCH -J eval-gemma-3-12b-pt-checkpoints
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
MODEL=${MODEL:="gemma-3-12b-pt"}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:="training_runs/${MODEL}-apptainer-fsdp-with-collator"}
SPECIFIC_CHECKPOINTS=${SPECIFIC_CHECKPOINTS:=""}

# Shared project files location
PROJECT_DIR="/cluster/projects/nn12075k/shared"
DATASET_DIR="$PROJECT_DIR/datasets/dataset_43221_examples"
VAL_DATASET=${VAL_DATASET:="${DATASET_DIR}/processed_data_val.jsonl"}

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
PROJECT_HF_DIR="$PROJECT_DIR/.cache/huggingface"
mkdir -p "$HF_HOME/hub"

# Symlink read-only model from read-only project storage (instant, no copy time!)
printf "Linking model to job-local cache (symlink for speed)...\n"
ln -s "$PROJECT_HF_DIR/hub/models--google--$MODEL" \
      "$HF_HOME/hub/models--google--$MODEL"

# Copy metrics cache if it exists (needed for ROUGE - needs write access)
if [ -d "$PROJECT_HF_DIR/metrics" ]; then
    echo "Copying metrics cache..."
    rsync -a "$PROJECT_HF_DIR/metrics/" "$HF_METRICS_CACHE/"
else
    echo "WARNING: No metrics cache found at $PROJECT_HF_DIR/metrics"
    echo "ROUGE metric may need to be downloaded during job (requires internet)"
fi

# Copy modules cache if it exists (needed for evaluate library - needs write access)
if [ -d "$PROJECT_HF_DIR/modules" ]; then
    echo "Copying modules cache..."
    rsync -a "$PROJECT_HF_DIR/modules/" "$HF_MODULES_CACHE/"
fi

echo "Cache staging complete"

# ********** END JOB-LOCAL CACHE **********


# env setup
module --force purge
module load BuildEnv/NeoverseV2
module load NRIS/GPU
module load CUDA/12.6.0
module load PyTorch/2.8.0

# Apptainer image
export IMAGE=${HOME}/apptainer_images/pytorch_25.11-py3.sif
export BINDINGS="--bind ${DATASET_DIR} --bind ${PROJECT_HF_DIR}"

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
    
    # Install commands (matching training script setup)
    INSTALL_CMD="
    set -e
    pip install setuptools wheel
    pip install transformers>=4.45.0 tokenizers datasets sentencepiece evaluate
    pip install 'accelerate>=0.30.0' 'peft>=0.11.0,<0.18.0' wandb
    pip install --no-binary=rouge-score rouge-score==0.1.2
    pip install --upgrade packaging==24.2 psutil nvidia-ml-py pyyaml typing-extensions
    echo 'Pip installations complete'
    pip list | grep -E 'transformers|accelerate|peft|evaluate|wandb'
    "
    
    # Command to run inside apptainer
    EVAL_CMD="
    export HF_HOME=$HF_HOME
    export HF_DATASETS_CACHE=$HF_DATASETS_CACHE
    export HF_MODULES_CACHE=$HF_MODULES_CACHE
    export HF_METRICS_CACHE=$HF_METRICS_CACHE
    
    source ./.env
    
    python scripts/evaluate_distributed_checkpoints.py \
      --model $MODEL \
      --checkpoint_dir $CHECKPOINT_DIR \
      --val_dataset $VAL_DATASET \
      --output_dir $CHECKPOINT_DIR/eval_results
    "
    
    # Run evaluation directly (already in SLURM job context from sbatch)
    apptainer exec --nv ${BINDINGS} $IMAGE bash -c "
        $INSTALL_CMD 
        $EVAL_CMD
    "
    
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
