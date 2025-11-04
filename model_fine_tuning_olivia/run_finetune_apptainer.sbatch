#!/usr/bin/env bash
#SBATCH -J gpu-finetune-apptainer
#SBATCH -p accel
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH -A nn12075k
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# Define variables
MODEL=gemma-2b
DISTRIBUTED="--distributed"
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "Detected $NUM_GPUS GPUs"

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
torchrun --standalone --nproc_per_node="${NUM_GPUS}" \\
    scripts/finetune.py \\
        --model $MODEL \\
        --quantization none \\
        $DISTRIBUTED \\
        --hf_token "$HUGGINGFACE_TOKEN" \\
        --train_dataset data/output/processed_data_train.jsonl \\
        --val_dataset data/output/processed_data_val.jsonl \\
        --output_dir models/${MODEL}_apptainer
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
