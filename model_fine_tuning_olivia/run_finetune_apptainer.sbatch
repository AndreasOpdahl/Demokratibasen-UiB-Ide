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
MODEL=gemma-7b  # this is not properly used
DDP_FLAG="--fsdp"  # this should be renamed
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "Detected $NUM_GPUS GPUs"

# define job-local cache
export SLURM_TMPDIR=/tmp/${USER}_$SLURM_JOB_ID
mkdir -p "$SLURM_TMPDIR"

# create a job-local HF cache
export HF_HOME="$SLURM_TMPDIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_METRICS_CACHE="$HF_HOME/metrics"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$HF_METRICS_CACHE"

# stage model/dataset into job-local cache (fast local read, writable)
PROJECT_HF_DIR=/cluster/projects/nn12075k/shared/.cache/huggingface
mkdir -p "$HF_HOME/hub"

# Copy model from read-only project storage to writable job-local cache
echo "Copying model to job-local cache..."
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

# export HF_HUB_OFFLINE=1 

# env setup
module --force purge
module load BuildEnv/NeoverseV2
module load NRIS/GPU
module load CUDA/12.6.0
module load PyTorch/2.8.0

# Load environment variables
source .env

# Apptainer version and image
apptainer --version
export IMAGE=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif

# Command to run inside apptainer
export FINETUNE="
export HF_HOME=$HF_HOME
export HF_DATASETS_CACHE=$HF_DATASETS_CACHE
export HF_MODULES_CACHE=$HF_MODULES_CACHE
export HF_METRICS_CACHE=$HF_METRICS_CACHE
# export HF_HUB_OFFLINE=1
torchrun --standalone --nproc_per_node="${NUM_GPUS}" \\
    scripts/finetune.py \\
        --model $MODEL \\
        --quantization none \\
        $DDP_FLAG \\
        --hf_token "$HUGGINGFACE_TOKEN" \\
        --train_dataset data/output/processed_data_train.jsonl \\
        --val_dataset data/output/processed_data_val.jsonl \\
        --max_steps 60 \\
        --output_dir models/${MODEL}-apptainer-fsdp
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
    pip install --upgrade pyyaml psutil typing-extensions packaging==24.2
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
    apptainer exec \
        --nv \
        $IMAGE \
        bash -c "$COMMAND"
        # --bind /usr/lib64:/usr/lib64 \
