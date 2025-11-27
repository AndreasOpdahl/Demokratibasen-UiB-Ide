#!/usr/bin/env bash
#SBATCH -J gpu-finetune-apptainer
#SBATCH -p accel
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:10:00
#SBATCH -A nn12075k
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# define run control variables
FINETUNE_SCRIPT="scripts/wandb_finetune.py"  # was "scripts/finetune.py"
MODEL=gemma-7b  # this is not properly used
DISTRIBUTION_FLAG="--fsdp"  # this should be renamed
RESUME_CHECKPOINT="checkpoint-60"  # or "" or a specific checkpoint path
TASK_LIMIT="--max_steps 120"

NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
printf "Allocated $NUM_GPUS GPUs...\n"

# where to find shared project files, including training and evaluation datasets
PROJECT_DIR="/cluster/projects/nn12075k/shared"
DATASET_DIR="$PROJECT_DIR/datasets/dataset_43221_examples"
OUTPUT_DIR="training_runs"

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

# Copy model from read-only project storage to writable job-local cache
printf "Copying model to job-local cache...\n"
rsync -a "$PROJECT_HF_DIR/hub/models--google--gemma-7b/" \
         "$HF_HOME/hub/models--google--gemma-7b/"

# Also copy metrics cache if it exists (needed for ROUGE)
if [ -d "$PROJECT_HF_DIR/metrics" ]; then
    echo "Copying metrics cache..."
    rsync -a "$PROJECT_HF_DIR/metrics/" "$HF_METRICS_CACHE/"
else
    echo "WARNING: No metrics cache found at $PROJECT_HF_DIR/metrics"
    echo "ROUGE metric may need to be downloaded during job (requires internet)"
fi

# Copy modules cache if it exists (needed for evaluate library)
if [ -d "$PROJECT_HF_DIR/modules" ]; then
    echo "Copying modules cache..."
    rsync -a "$PROJECT_HF_DIR/modules/" "$HF_MODULES_CACHE/"
fi

echo "Cache staging complete..."

# export HF_HUB_OFFLINE=1 

# env setup
module --force purge
module load BuildEnv/NeoverseV2
module load NRIS/GPU
module load CUDA/12.6.0
module load PyTorch/2.8.0

# Load environment variables (currently only HUGGINGFACE_TOKEN is used)
source .env

# Apptainer version and image
apptainer --version
export IMAGE=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif
export BINDINGS="--bind $DATASET_DIR"

# Command to run inside apptainer
export FINETUNE="
export HF_HOME=$HF_HOME
export HF_DATASETS_CACHE=$HF_DATASETS_CACHE
export HF_MODULES_CACHE=$HF_MODULES_CACHE
export HF_METRICS_CACHE=$HF_METRICS_CACHE
# export HF_HUB_OFFLINE=1
TORCHRUN_ARGS=(
    torchrun
        --standalone
        --nproc_per_node="${NUM_GPUS}"
        $FINETUNE_SCRIPT
            --model $MODEL
            --quantization none
            $DISTRIBUTION_FLAG
            --hf_token "$HUGGINGFACE_TOKEN"
            --train_dataset "$DATASET_DIR/processed_data_train.jsonl"
            --val_dataset "$DATASET_DIR/processed_data_val.jsonl"
            $TASK_LIMIT
            --output_dir $OUTPUT_DIR/${MODEL}-apptainer-fsdp-with-collator
)
if [[ -n \"$RESUME_CHECKPOINT\" ]]; then
    TORCHRUN_ARGS+=(--resume_checkpoint "$RESUME_CHECKPOINT")
fi
\${TORCHRUN_ARGS[@]}
"

# Apptainer commands
export COMMAND="
    pip install --upgrade pip setuptools wheel
    pip install --only-binary=:all: \
        transformers==4.45.2 \
        tokenizers==0.20.3 \
        datasets==4.0.0 \
        sentencepiece==0.2.1
    pip install --no-binary=rouge-score rouge-score==0.1.2
    pip install --no-deps accelerate==1.10.1
    pip install --no-deps peft==0.17.1
    pip install --upgrade packaging==24.2 psutil pynvml==12.0.0 pyyaml typing-extensions wandb
    $FINETUNE
"

# Run finetune
echo "Starting apptainer..."
echo "Command to execute:"
echo "$COMMAND"
echo ""

# Launch ONE (SLURM_NTASKS=1) container that will spawn multiple GPU processes internally
srun --ntasks=${SLURM_NTASKS} \
     --cpus-per-task=${SLURM_CPUS_PER_TASK} \
     --gres=gpu:${NUM_GPUS} \
    apptainer exec --nv $BINDINGS $IMAGE bash -c "$COMMAND"
