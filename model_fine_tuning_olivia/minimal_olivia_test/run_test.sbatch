#!/usr/bin/env bash
#SBATCH -J gpu-smoke
#SBATCH -p accel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH -A YOUR_PROJECT_ID
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# Slurm stages the batch script under /var/spool/slurmd/job*/; use SLURM_SUBMIT_DIR when set.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SBATCH_DIR="$SLURM_SUBMIT_DIR"
else
    SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SBATCH_DIR" || { echo "ERROR: Cannot cd to $SBATCH_DIR"; exit 1; }
echo "Working directory for this job: $SBATCH_DIR"

# env setup
module load CrayEnv
module load cuda/12.6
module load BuildEnv/NeoverseV2
module load NRIS/GPU
module load PyTorch/2.8.0

python -V
nvidia-smi
python test_gpu.py
