#!/usr/bin/env bash
#SBATCH -J gpu-finetune-apptainer
#SBATCH -p accel
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:10:00
#SBATCH -A YOUR_PROJECT_ID
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
echo "Starting job..."
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo

# define run control variables if not set from command line
FINETUNE_SCRIPT=${FINETUNE_SCRIPT:="scripts/wandb_finetune.py"}  # was "scripts/finetune.py"
MODEL=${MODEL:="gemma-3-12b-pt"}
DISTRIBUTION_FLAG=${DISTRIBUTION_FLAG:="--fsdp"}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:="latest"}  # this is safest; alternatively "" or a specific checkpoint path="checkpoint-60"
TASK_LIMIT=${TASK_LIMIT:=""}  # or "" or a specific task limit="--max_steps 120" or "--num_train_epochs 5"

# convenience
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
printf "Allocated $NUM_GPUS GPUs...\n"

# where to find shared project files, including training and evaluation datasets
PROJECT_DIR="/cluster/projects/YOUR_PROJECT_ID/shared"
DATASET_DIR="$PROJECT_DIR/datasets/dataset_43221_examples"
OUTPUT_DIR="training_runs"

# define job-local cache for training
export SLURM_TMPDIR=/tmp/${USER}_$SLURM_JOB_ID
mkdir -p "$SLURM_TMPDIR"

# create a job-local temporary HF_HOME cache, overrides the global HF_HOME cache
export HF_HOME="$SLURM_TMPDIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_METRICS_CACHE="$HF_HOME/metrics"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$HF_METRICS_CACHE"

# stage model/dataset into job-local cache (fast local read, writable)
PROJECT_HF_DIR="$PROJECT_DIR/.cache/huggingface"
mkdir -p "$HF_HOME/hub"

# symlink read-only model from read-only project storage
printf "Linking model to job-local cache (symlink for speed)...\n"
ln -s "$PROJECT_HF_DIR/hub/models--google--$MODEL" \
      "$HF_HOME/hub/models--google--$MODEL"
# OLD SOLUTION: copy model from read-only project storage to writable job-local cache
# printf "Copying model to job-local cache...\n"
# rsync -a "$PROJECT_HF_DIR/hub/models--google--$MODEL/" \
#          "$HF_HOME/hub/models--google--$MODEL/"

# also copy metrics cache if it exists (needed for ROUGE)
if [ -d "$PROJECT_HF_DIR/metrics" ]; then
    echo "Copying metrics cache..."
    rsync -a "$PROJECT_HF_DIR/metrics/" "$HF_METRICS_CACHE/"
else
    echo "WARNING: No metrics cache found at $PROJECT_HF_DIR/metrics"
    echo "ROUGE metric may need to be downloaded during job (requires internet)"
fi

# copy modules cache if it exists (needed for evaluate library)
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

# apptainer install commands
INSTALL_CMD="
set -e
pip install --upgrade pip setuptools wheel
# utilities
pip install packaging psutil nvidia-ml-py pyyaml typing-extensions wandb
# core HuggingFace dependencies
pip install transformers>=4.45.0 tokenizers datasets sentencepiece evaluate
# FSDP tools
pip install accelerate>=0.30.0 peft>=0.11.0
# evaluation metrics
pip install --no-binary=rouge-score rouge-score==0.1.2
echo 'Pip installations complete'
pip list | grep -E 'wandb|transformers|accelerate|peft|evaluate'
"
echo "INSTALL_CMD: ${INSTALL_CMD}"

FINETUNE_SCRIPT_ARGS="\
    --model ${MODEL} \
    --quantization none \
    ${DISTRIBUTION_FLAG} \
    --train_dataset ${DATASET_DIR}/processed_data_train.jsonl \
    --val_dataset ${DATASET_DIR}/processed_data_val.jsonl \
    --output_dir ${OUTPUT_DIR}/${MODEL}-apptainer-fsdp-with-collator"
if [[ -n "${TASK_LIMIT}" ]]; then
    FINETUNE_SCRIPT_ARGS+=" ${TASK_LIMIT}"
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    FINETUNE_SCRIPT_ARGS+=" --resume_checkpoint \"${RESUME_CHECKPOINT}\""
fi
echo "FINETUNE_SCRIPT_ARGS: ${FINETUNE_SCRIPT_ARGS}"

# torchrun command
TORCHRUN_CMD="
export HF_HOME=${HF_HOME}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE}
export HF_MODULES_CACHE=${HF_MODULES_CACHE}
export HF_METRICS_CACHE=${HF_METRICS_CACHE}

source ./.env


torchrun \
    --standalone \
    --nproc_per_node=${NUM_GPUS} \
    $FINETUNE_SCRIPT $FINETUNE_SCRIPT_ARGS
"
echo "TORCHRUN_CMD: ${TORCHRUN_CMD}"

# apptainer version and image
export APPTAINER=$(which apptainer)
# export IMAGE=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif
export IMAGE=${HOME}/apptainer_images/pytorch_25.11-py3.sif
export BINDINGS="--bind ${DATASET_DIR} --bind ${PROJECT_HF_DIR}"

# run finetune
echo "Starting apptainer:"
$APPTAINER --version
echo $IMAGE
echo ""
echo "Dependencies to install:"
echo "$INSTALL_CMD"
echo ""
echo "Commands to execute:"
echo "$TORCHRUN_CMD"
echo ""

# launch ONE (SLURM_NTASKS=1) container that will spawn multiple GPU processes internally
srun --ntasks=${SLURM_NTASKS} \
     --cpus-per-task=${SLURM_CPUS_PER_TASK} \
     --gres=gpu:${NUM_GPUS} \
    ${APPTAINER} exec --nv ${BINDINGS} ${IMAGE} \
        bash -c "
            ${INSTALL_CMD}
            ${TORCHRUN_CMD}
        "
