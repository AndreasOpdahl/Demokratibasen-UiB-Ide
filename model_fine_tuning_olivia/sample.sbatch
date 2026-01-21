#!/bin/bash
#SBATCH --job-name=simple_nn_training
#SBATCH --account=<project_number>
#SBATCH --output=singlenode_with_container_%j.out
#SBATCH --error=singlenode_with_container_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=accel
#SBATCH --nodes=1     # Use one compute node
##SBATCH --nodelist=x1000c2s4b1n0
#SBATCH --ntasks-per-node=1 #  Single task per node
#SBATCH --cpus-per-gpu=72  # Reserve enough CPU cores for full workload with each GPU
#SBATCH --mem-per-gpu=110G   # Request 110 GB of CPU RAM per GPU
#SBATCH --gpus-per-node=4   # Reserve 4 GPUs on node

# Path to the container
CONTAINER_PATH="/cluster/work/support/container/pytorch_nvidia_25.05_arm64.sif"

# Path to the training script
TRAINING_SCRIPT="/cluster/work/projects/<project_number>/<user_name>/PyTorch/private/simple_nn_project/singlenode.py --batch-size 1024 --epochs 100 --base-lr 0.04 --target-accuracy 0.95 --patience 2"

# Bind directories
BIND_DIR="/cluster/work/projects/<project_number>/<user_name>/PyTorch/private"

# Set PYTHONPATH to include the shared directory
export PYTHONPATH=/cluster/work/projects/<project_number>/<user_name>/PyTorch/private/shared:$PYTHONPATH

# Explicitly specify the full path to torchrun
TORCHRUN_PATH="/usr/local/bin/torchrun"

apptainer exec --nv $CONTAINER_PATH which torchrun
# Start GPU utilization monitoring in the background
GPU_LOG_FILE="gpu_utilization_multinode_container.log"
echo "Starting GPU utilization monitoring..."
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > $GPU_LOG_FILE &

# Run the training script with torchrun inside the container
apptainer exec --nv --bind $BIND_DIR $CONTAINER_PATH $TORCHRUN_PATH --standalone --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_GPUS_ON_NODE $TRAINING_SCRIPT

# Stop GPU utilization monitoring
echo "Stopping GPU utilization monitoring..."
pkill -f "nvidia-smi --query-gpu"