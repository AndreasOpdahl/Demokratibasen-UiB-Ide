#!/usr/bin/env bash
#SBATCH -J gpu-finetune-apptainer
#SBATCH -p accel
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=25:00:00
#SBATCH -A nn12075k
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# Model selection: Use short name (gemma-7b, viking-7b, normistral-7b, etc.)
# Supported models: gemma-2b, gemma-7b, viking-7b, viking-13b, no
MODEL=gemma-2-9b
DISTRIBUTION_FLAG="--fsdp"
RESUME_CHECKPOINT="checkpoint-5000"
TASK_LIMIT="--max_steps 10000"
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}

export SLURM_TMPDIR=/tmp/${USER}_$SLURM_JOB_ID
mkdir -p "$SLURM_TMPDIR"

export HF_HOME="$SLURM_TMPDIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_METRICS_CACHE="$HF_HOME/metrics"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$HF_METRICS_CACHE"

PROJECT_HF_DIR=/cluster/projects/nn12075k/shared/.cache/huggingface
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
        norskgpt-llama3-8b) echo "bineric/norskgpt-llama3-8b" ;;
        llama-2-13b-chat-norwegian) echo "ruternorway/llama-2-13b-chat-norwegian" ;;
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

module --force purge
module load BuildEnv/NeoverseV2
module load NRIS/GPU
module load CUDA/12.6.0
module load PyTorch/2.8.0

source .env

export IMAGE=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')

[[ -n "$RESUME_CHECKPOINT" ]] && RESUME_ARG="--resume_checkpoint $RESUME_CHECKPOINT" || RESUME_ARG=""

# Write commands to a script file on SHARED filesystem (PWD is shared)
SCRIPT_FILE="${PWD}/logs/run_training_${SLURM_JOB_ID}.sh"
mkdir -p "${PWD}/logs"
cat > "$SCRIPT_FILE" << 'SCRIPT_EOF'
#!/bin/bash
set -e

echo "=== Step 1: Environment setup ==="
export HF_HOME=HF_HOME_PLACEHOLDER
export HF_MODULES_CACHE=HF_MODULES_CACHE_PLACEHOLDER
export HF_METRICS_CACHE=HF_METRICS_CACHE_PLACEHOLDER
export NCCL_DEBUG=NCCL_DEBUG_PLACEHOLDER
export NCCL_IB_DISABLE=NCCL_IB_DISABLE_PLACEHOLDER
export NCCL_P2P_DISABLE=NCCL_P2P_DISABLE_PLACEHOLDER
export NCCL_SOCKET_IFNAME=
export TORCH_NCCL_ASYNC_ERROR_HANDLING=TORCH_NCCL_ASYNC_ERROR_HANDLING_PLACEHOLDER
export MASTER_ADDR=MASTER_ADDR_PLACEHOLDER
export MASTER_PORT=MASTER_PORT_PLACEHOLDER

echo "=== Step 2: Install packages ==="
INSTALL_DONE_FLAG="INSTALL_FLAG_PLACEHOLDER"
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
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import datasets; print('datasets:', datasets.__version__)"
python -c "import accelerate; print('accelerate:', accelerate.__version__)"
python -c "import peft; print('peft:', peft.__version__)"
python -c "import torch; print('torch:', torch.__version__)"
python -c "import numpy; print('numpy:', numpy.__version__)"

echo "=== Step 4: Login to wandb ==="
python -m wandb login WANDB_API_KEY_PLACEHOLDER

echo "=== Network interfaces ==="
ip link show | grep -E "^[0-9]+" | cut -d: -f2

echo "=== Step 5: Launch training ==="
torchrun --nnodes=SLURM_NNODES_PLACEHOLDER --nproc_per_node=SLURM_GPUS_ON_NODE_PLACEHOLDER --node_rank=$SLURM_NODEID --rdzv_backend=c10d --rdzv_endpoint=MASTER_ADDR_PLACEHOLDER:MASTER_PORT_PLACEHOLDER scripts/wandb_finetune.py --model MODEL_PLACEHOLDER --quantization none DISTRIBUTION_FLAG_PLACEHOLDER --hf_token HUGGINGFACE_TOKEN_PLACEHOLDER --train_dataset data/output/new_processed_data_train.jsonl --val_dataset data/output/new_processed_data_val.jsonl TASK_LIMIT_PLACEHOLDER --output_dir models/MODEL_PLACEHOLDER-apptainer-fsdp RESUME_ARG_PLACEHOLDER
SCRIPT_EOF

# Replace placeholders with actual values
sed -i "s|HF_HOME_PLACEHOLDER|$HF_HOME|g" "$SCRIPT_FILE"
sed -i "s|HF_DATASETS_CACHE_PLACEHOLDER|$HF_DATASETS_CACHE|g" "$SCRIPT_FILE"
sed -i "s|HF_MODULES_CACHE_PLACEHOLDER|$HF_MODULES_CACHE|g" "$SCRIPT_FILE"
sed -i "s|HF_METRICS_CACHE_PLACEHOLDER|$HF_METRICS_CACHE|g" "$SCRIPT_FILE"
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
sed -i "s|RESUME_ARG_PLACEHOLDER|$RESUME_ARG|g" "$SCRIPT_FILE"
sed -i "s|INSTALL_FLAG_PLACEHOLDER|${PWD}/logs/install_done_${SLURM_JOB_ID}|g" "$SCRIPT_FILE"

chmod +x "$SCRIPT_FILE"

echo "=== Script to execute ==="
cat "$SCRIPT_FILE"
echo "========================="

srun --ntasks=$SLURM_NNODES \
     --cpus-per-task=${SLURM_CPUS_PER_TASK} \
     --gres=gpu:${NUM_GPUS} \
    apptainer exec \
        --nv \
        $IMAGE \
        bash "$SCRIPT_FILE"
