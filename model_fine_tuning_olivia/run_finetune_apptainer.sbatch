#!/usr/bin/env bash
#SBATCH -J gpu-finetune-apptainer
#SBATCH -p accel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:03:00
#SBATCH -A nn12075k
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# env setup
# module purge
# module load CrayEnv
# module load cuda/12.6
# module load BuildEnv/NeoverseV2
# module load NRIS/GPU
# module load PyTorch/2.8.0

# Check version
apptainer --version

# Choose a cache dir in your $HOME or project (avoids /tmp quotas)
export APPTAINER_CACHEDIR=$HOME/.cache/apptainer
mkdir -p "$APPTAINER_CACHEDIR"


IMAGE=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif

# run finetune
echo "Starting apptainer..."
source .env
srun --ntasks=1 \
     --cpus-per-task=${SLURM_CPUS_PER_TASK:-2} \
    apptainer exec \
        --nv \
        --bind /usr/lib64:/usr/lib64 \
        $IMAGE \
        python3 -c "import os,ctypes; p='/usr/lib64/libnvidia-ml.so.1'; print(f'Checking {p}...'); print('✅ Found' if os.path.exists(p) else '❌ Missing'); \
       (lambda: print('✅ Loaded OK') if ctypes.CDLL(p) else None)() if os.path.exists(p) else None"
#
#        bash -c "
#            echo 'Inside container...';
#            nvidia-smi;
#            ls -l /usr/lib64/libnvidia-ml.so.1 || echo 'lib not found';
#            file /usr/lib64/libnvidia-ml.so.1 || echo 'file not found';
#        "
