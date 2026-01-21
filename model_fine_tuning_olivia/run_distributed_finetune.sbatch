#!/usr/bin/env bash
#SBATCH -J gpu-smoke
#SBATCH --partition=accel
#SBATCH --nodes=1     # Use one compute node
##SBATCH --nodelist=x1000c2s4b1n0
#SBATCH --ntasks-per-node=1 #  Single task per node
#SBATCH --cpus-per-gpu=72  # Reserve enough CPU cores for full workload with each GPU
#SBATCH --mem-per-gpu=110G   # Request 110 GB of CPU RAM per GPU
#SBATCH --gpus-per-node=4   # Reserve 4 GPUs on node
#SBATCH --time=05:00:00
#SBATCH -A nn12075k
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Path to the container

CONTAINER_PATH="/cluster/work/support/container/pytorch_nvidia_25.05_arm64.sif"

source .env
CONTAINER_PATH="/cluster/work/support/container/pytorch_nvidia_25.05_arm64.sif"
TRAINING_SCRIPT="/cluster/work/projects/nn12075k/ajayv/PyTorch/private/model_finetuning_distributed/scripts/run_training.py"

TRAINING_ARGS="--model gemma-7b \
    --hf_token $HUGGINGFACE_TOKEN \
    --train_dataset data/output/processed_data_train.jsonl \
    --val_dataset data/output/processed_data_val.jsonl \
    --test_dataset data/output/processed_data_test.jsonl \
    --output_dir model/gemma-7b-finetuned \
    --max_input_text_tokens 2048 \
    --max_extra_prompt_tokens 40 \
    --max_output_summary_tokens 512 \
    --max_epochs 1 \
    --train_batch_size 4 \
    --val_batch_size 5 \
    --val_data_size 20 \
    --val_beam_size 4 \
    --val_steps 100"

BIND_DIR="/cluster/work/projects/nn12075k/ajayv/PyTorch/private/"
export PYTHONPATH=/cluster/work/projects/nn12075k/ajayv/PyTorch/private/shared:$PYTHONPATH

# Start GPU monitoring
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > gpu_utilization.log &

echo "=== Creating virtual environment in container ==="

apptainer exec --nv --bind $BIND_DIR $CONTAINER_PATH /bin/bash -c "
    set -e
    
    # Create virtual environment
    VENV_PATH=\"/tmp/training_venv_$SLURM_JOB_ID\"
    python -m venv \$VENV_PATH
    source \$VENV_PATH/bin/activate
    
    echo '=== Installing CUDA-enabled PyTorch FIRST ==='
    
    # Install CUDA-enabled PyTorch FIRST with explicit index
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    echo '=== Verifying CUDA installation ==='
    python -c \"import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Number of GPUs: {torch.cuda.device_count()}')\"
    
    # Now install other packages
    echo '=== Installing other packages ==='
    pip install transformers tokenizers datasets \
        pyarrow multiprocess xxhash sentencepiece \
        absl-py six sympy bitsandbytes wandb weave \
        evaluate jinja2 rouge-score accelerate peft
    
    # Final verification
    echo '=== Final CUDA check ==='
    python -c \"import torch; print(f'Final - CUDA available: {torch.cuda.is_available()}')\"
    
    # Run training
    export PYTHONPATH=\$PYTHONPATH:$(dirname $TRAINING_SCRIPT)
    torchrun --standalone --nnodes=\$SLURM_JOB_NUM_NODES --nproc_per_node=\$SLURM_GPUS_ON_NODE $TRAINING_SCRIPT $TRAINING_ARGS
"

# Stop GPU utilization monitoring
echo "Stopping GPU utilization monitoring..."
pkill -f "nvidia-smi --query-gpu"