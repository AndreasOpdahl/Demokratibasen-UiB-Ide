#!/usr/bin/env bash
#SBATCH -J gpu-smoke
#SBATCH -p accel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH -A nn12075k
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# env setup
module purge
module load CrayEnv
module load cuda/12.6
module load BuildEnv/NeoverseV2
module load NRIS/GPU
module load PyTorch/2.8.0

# debug
python -V

# confirm the module exports PYTHONPATH so torch is visible inside the venv:
TORCH_VER="$(python - <<'PY'
try:
    import torch; print(torch.__version__)
except Exception as e:
    print("NO TORCH:", e)
PY
)"
echo "TORCH_VER: $TORCH_VER"

# virtual environment
VENV_DIR=venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR"
    python -m venv "$VENV_DIR"  # no --system-site-packages so we do not load cudf
    source "$VENV_DIR/bin/activate"

    # need to add symlink to torch inside the venv
    PY_VER=$(python -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    VENV_SP="$VENV_DIR/lib/python$PY_VER/site-packages"

    # this must point to the module-loaded system torch path
    SYS_SP="/usr/local/lib/python3.12/dist-packages"

    # create a symlink module-loaded torch inside the venv
    ln -snf "$SYS_SP/torch" "$VENV_SP/"
    ln -snf "$SYS_SP/torchgen" "$VENV_SP/"

    TORCH_DIST_DIR="$VENV_SP/torch-$TORCH_VER.dist-info"
    mkdir -p "$TORCH_DIST_DIR"
    cat > "$TORCH_DIST_DIR/METADATA" <<EOF
Metadata-Version: 2.1
Name: torch
Version: $TORCH_VER
Summary: Local metadata for system-provided PyTorch.
EOF
: > "$TORCH_DIST_DIR/RECORD"

    echo "VENV_SP: $VENV_SP"
    echo "SYS_SP: $SYS_SP"
    echo "PY_VER: $PY_VER"
    echo "TORCH_DIST_DIR: $TORCH_DIST_DIR"

    # install regular dependencies
    pip install --upgrade pip setuptools wheel
    pip install --upgrade 'numpy<2'  # downgrade numpy to avoid conflict with torch
    pip install --only-binary=:all: \
        transformers==4.45.2 \
        tokenizers==0.20.3 \
        datasets==4.1.1 \
        pyarrow==21.0.0 \
        multiprocess==0.70.16 \
        xxhash==3.6.0 \
        sentencepiece==0.2.1 \
        absl-py==2.1.0 \
        six==1.16.0 \
        sympy==1.12
    pip install evaluate==0.4.6  # had to install this one manually on the GPU node
<<<<<<< HEAD
<<<<<<< HEAD
    pip install jinja2==3.1.6
=======
>>>>>>> a1f1896dc62263972a9586eb54f57fa3cbe1a47e
=======
>>>>>>> a1f1896dc62263972a9586eb54f57fa3cbe1a47e
    pip install --no-binary=rouge-score rouge-score==0.1.2
    pip install --no-deps accelerate==1.10.1
    pip install --no-deps peft==0.17.1  # Install peft *without* pulling torch
    pip install --upgrade pyyaml psutil typing-extensions packaging  # possible accelerate/peft deps
else
    echo "Using existing virtual environment at $VENV_DIR"
    source "$VENV_DIR/bin/activate"
fi

# test that torch is visible inside the venv
echo "Checking torch and friends are visible inside the venv..."
python - <<'PY'
import torch, importlib.metadata as md, numpy, transformers, accelerate, peft
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "avail:", torch.cuda.is_available())
print("torch (metadata):", md.version("torch"))
print("numpy:", numpy.__version__)
print("transformers:", transformers.__version__, "accelerate:", accelerate.__version__, "peft:", peft.__version__)
PY

# run finetune
echo "Running finetune..."
source .env
srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-2} \
    python scripts/finetune_olivia.py \
        --model gemma-2b \
        --hf_token "$HUGGINGFACE_TOKEN" \
        --train_dataset data/output/processed_data_train.jsonl \
        --val_dataset data/output/processed_data_val.jsonl \
        --output_dir data/output
