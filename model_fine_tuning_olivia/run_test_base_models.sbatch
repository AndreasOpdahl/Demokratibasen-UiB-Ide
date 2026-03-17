#!/bin/bash
#SBATCH --job-name=gpu-test-base-models
#SBATCH -p accel
#SBATCH --output=logs/gpu-test-base-models-%j.out
#SBATCH --error=logs/gpu-test-base-models-%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
# Account: REQUIRED - Pass via: sbatch --account=YOUR_ACCOUNT run_test_base_models.sbatch ...
# Example: sbatch --account=YOUR_PROJECT_ID run_test_base_models.sbatch --model=gemma-7b

# Test base models (before fine-tuning) to verify prompt formatting and generation

set -euo pipefail

# ===== VALIDATE MODEL =====
validate_model() {
    case "$MODEL" in
        gemma-2b|gemma-7b|gemma-2-9b|gemma-2-27b|gemma-3-12b|gemma-3-27b| \
        viking-7b|viking-13b|viking-33b| \
        normistral-7b|normistral-11b|normistral-7b-instruct| \
        norskgpt-llama3-8b|llama-3.1-8b-instruct|llama-2-13b-chat-norwegian| \
        eurollm-9b-instruct|norwai-mistral-7b-instruct|nb-gpt-j-6b|mt5)
            return 0
            ;;
        *)
            echo "ERROR: Unknown model: $MODEL"
            echo "Valid models: gemma-2b, gemma-7b, gemma-2-9b, gemma-2-27b, gemma-3-12b, gemma-3-27b,"
            echo "              viking-7b, viking-13b, viking-33b, normistral-7b, normistral-11b,"
            echo "              normistral-7b-instruct, norskgpt-llama3-8b, llama-3.1-8b-instruct,"
            echo "              llama-2-13b-chat-norwegian, eurollm-9b-instruct, norwai-mistral-7b-instruct,"
            echo "              nb-gpt-j-6b, mt5"
            exit 1
            ;;
    esac
}

# ===== PARSE ARGUMENTS =====
MODEL=""
TEST_DATASET="${TEST_DATASET:-data/output/new_processed_data_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-base_model_predictions}"
NUM_EXAMPLES="${NUM_EXAMPLES:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
USE_MULTI_GPU="${USE_MULTI_GPU:-false}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model=*) MODEL="${1#*=}" ;;
        --test_dataset=*) TEST_DATASET="${1#*=}" ;;
        --output_dir=*) OUTPUT_DIR="${1#*=}" ;;
        --num_examples=*) NUM_EXAMPLES="${1#*=}" ;;
        --max_new_tokens=*) MAX_NEW_TOKENS="${1#*=}" ;;
        --use_multi_gpu) USE_MULTI_GPU=true ;;
        *)
            echo "ERROR: Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

# ===== VALIDATE =====
if [ -z "$MODEL" ]; then
    echo "ERROR: --model is required"
    exit 1
fi

validate_model

if [ ! -f "$TEST_DATASET" ]; then
    echo "ERROR: Test dataset not found: $TEST_DATASET"
    exit 1
fi

# ===== SETUP =====
echo "================================================================================"
echo "Base Model Testing"
echo "================================================================================"
echo "Model: $MODEL"
echo "Test dataset: $TEST_DATASET"
echo "Output directory: $OUTPUT_DIR"
echo "Number of examples: $NUM_EXAMPLES"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Use multi-GPU: $USE_MULTI_GPU"
echo "================================================================================"
echo ""

# ********** JOB-LOCAL CACHE **********
# define job-local cache
export SLURM_TMPDIR=/tmp/${USER}_$SLURM_JOB_ID
mkdir -p "$SLURM_TMPDIR"

# create a job-local HF cache, overrides the global HF_home cache
export HF_HOME="$SLURM_TMPDIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_METRICS_CACHE="$HF_HOME/metrics"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$HF_METRICS_CACHE"

# Set PYTHONUSERBASE to writable location to avoid .local read-only filesystem errors
export PYTHONUSERBASE="$SLURM_TMPDIR/.local"
mkdir -p "$PYTHONUSERBASE"

# Check for container command BEFORE loading modules (like the working monitor script)
CONTAINER_CMD=""
if command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
    echo "Found apptainer command (before module load)"
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
    echo "Found singularity command (before module load)"
else
    echo "WARNING: Container command not found before module load. Will check again after."
fi

# Load modules (same pattern as run_test_e2e.sbatch and run_finetune_multinode.sbatch)
module --force purge
module load BuildEnv/NeoverseV2
module load NRIS/GPU
module load CUDA/12.6.0
module load PyTorch/2.8.0

# Source .env file for HF_TOKEN and other environment variables
if [ -f .env ]; then
    source .env
    echo "✓ Loaded environment variables from .env"
else
    echo "WARNING: .env file not found. Using environment variables if set."
fi

# Re-check container command after loading modules (in case it's provided by a module)
if [ -z "$CONTAINER_CMD" ]; then
    if command -v apptainer &> /dev/null; then
        CONTAINER_CMD="apptainer"
        echo "Found apptainer command (after module load)"
    elif command -v singularity &> /dev/null; then
        CONTAINER_CMD="singularity"
        echo "Found singularity command (after module load)"
    else
        echo "ERROR: Neither 'apptainer' nor 'singularity' command found."
        echo "Please ensure one is available in your environment."
        exit 1
    fi
fi

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Check HF token (support both HF_TOKEN and HUGGINGFACE_TOKEN)
HF_TOKEN_VALUE="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
if [ -z "$HF_TOKEN_VALUE" ]; then
    echo "WARNING: Neither HF_TOKEN nor HUGGINGFACE_TOKEN is set. Some models may require authentication."
else
    echo "✓ HF token is set (from HF_TOKEN or HUGGINGFACE_TOKEN)"
fi

# ===== RUN TEST =====
# Get absolute path to current directory (where the script is run from)
WORK_DIR=$(pwd)

# Apptainer image
export IMAGE=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif

# Build test command arguments
TEST_ARGS="--models $MODEL --test_dataset $TEST_DATASET --output_dir $OUTPUT_DIR --num_examples $NUM_EXAMPLES --max_new_tokens $MAX_NEW_TOKENS"

if [ "$USE_MULTI_GPU" = true ]; then
    TEST_ARGS="$TEST_ARGS --use_multi_gpu"
fi

# Build command to run inside Apptainer
# Export HF token as environment variable (huggingface_hub will use it automatically)
# Also pass it as argument to the script
TEST_CMD="export HF_HOME=\$HF_HOME && export HF_DATASETS_CACHE=\$HF_DATASETS_CACHE && export HF_MODULES_CACHE=\$HF_MODULES_CACHE && export HF_METRICS_CACHE=\$HF_METRICS_CACHE && export PYTHONUSERBASE=\$PYTHONUSERBASE && export TOKENIZERS_PARALLELISM=false"

# Export HF token environment variables inside container (huggingface_hub uses these)
if [ -n "$HF_TOKEN_VALUE" ]; then
    TEST_CMD="$TEST_CMD && export HF_TOKEN=\"$HF_TOKEN_VALUE\" && export HUGGINGFACE_HUB_TOKEN=\"$HF_TOKEN_VALUE\" && export HUGGINGFACE_TOKEN=\"$HF_TOKEN_VALUE\""
    TEST_ARGS="$TEST_ARGS --hf_token $HF_TOKEN_VALUE"
fi

TEST_CMD="$TEST_CMD && cd $WORK_DIR && python scripts/test_base_models.py $TEST_ARGS"

# Apptainer command - install packages and run test
COMMAND="pip install --upgrade pip setuptools wheel > /dev/null 2>&1 && pip install --break-system-packages --only-binary=:all: transformers==4.45.2 tokenizers==0.20.3 datasets==3.0.0 sentencepiece==0.2.1 > /dev/null 2>&1 && pip install --break-system-packages --no-binary=rouge-score rouge-score==0.1.2 > /dev/null 2>&1 && pip install --break-system-packages --no-deps accelerate==1.0.1 peft==0.13.2 > /dev/null 2>&1 && pip install --break-system-packages --upgrade pyyaml psutil typing-extensions packaging==24.2 safetensors > /dev/null 2>&1 && pip install --break-system-packages --upgrade wandb evaluate sentence-transformers bert_score > /dev/null 2>&1 && $TEST_CMD"

# Run test inside apptainer
echo "Running test inside apptainer..."
$CONTAINER_CMD exec --nv $IMAGE bash -c "$COMMAND"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✓ Base model test completed successfully"
    echo "================================================================================"
    echo "Predictions saved to: $OUTPUT_DIR/"
else
    echo ""
    echo "================================================================================"
    echo "✗ Base model test failed with exit code: $EXIT_CODE"
    echo "================================================================================"
fi

exit $EXIT_CODE
