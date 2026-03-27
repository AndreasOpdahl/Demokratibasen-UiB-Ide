#!/usr/bin/env bash
#SBATCH -J test-e2e
#SBATCH -p accel
#SBATCH --gres=gpu:1  # Single GPU for testing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00  # 1 hour should be enough for minimal testing
# Account: REQUIRED - Pass via: sbatch --account=YOUR_ACCOUNT run_test_e2e.sbatch ...
# Example: sbatch --account=YOUR_PROJECT_ID run_test_e2e.sbatch --model=gemma-2b
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# Slurm stages the batch script under /var/spool/slurmd/job*/ on compute nodes, so
# ${BASH_SOURCE[0]} points there—not your project. Use SLURM_SUBMIT_DIR (cwd when you ran sbatch).
# Without Slurm, fall back to the directory containing this file.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SBATCH_DIR="$SLURM_SUBMIT_DIR"
else
    SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SBATCH_DIR" || { echo "ERROR: Cannot cd to $SBATCH_DIR"; exit 1; }
echo "Working directory for this job: $SBATCH_DIR"

# Load environment variables from .env file
# .env should contain: SLURM_ACCOUNT, WANDB_API_KEY, HUGGINGFACE_TOKEN
if [ -f .env ]; then
    source .env
else
    echo "WARNING: .env file not found. Using defaults or environment variables."
fi

# ===== CONFIGURATION =====
# Parse command-line arguments or use defaults
# Usage: sbatch --export=SLURM_ACCOUNT=YOUR_PROJECT_ID run_test_e2e.sbatch [--model=MODEL] [--test_dir=DIR] [--skip_*] [--keep_test_data]
MODEL=${MODEL:-gemma-2b}
TEST_DIR=${TEST_DIR:-""}  # Empty = use temp directory
SKIP_TRAINING=${SKIP_TRAINING:-false}
SKIP_EVALUATION=${SKIP_EVALUATION:-false}
SKIP_EXTENDED_EVAL=${SKIP_EXTENDED_EVAL:-false}
SKIP_MULTIGPU=${SKIP_MULTIGPU:-false}
SKIP_ERROR_HANDLING=${SKIP_ERROR_HANDLING:-false}
SKIP_FILE_IO=${SKIP_FILE_IO:-false}
SKIP_MONITOR=${SKIP_MONITOR:-false}
SKIP_EDGE_CASES=${SKIP_EDGE_CASES:-false}
INCLUDE_NLI_FAITHFULNESS=${INCLUDE_NLI_FAITHFULNESS:-false}
KEEP_TEST_DATA=${KEEP_TEST_DATA:-false}
DRY_RUN=${DRY_RUN:-false}

# Parse command-line arguments if provided
for arg in "$@"; do
    case $arg in
        --model=*)
            MODEL="${arg#*=}"
            ;;
        --test_dir=*)
            TEST_DIR="${arg#*=}"
            ;;
        --skip_training)
            SKIP_TRAINING=true
            ;;
        --skip_evaluation)
            SKIP_EVALUATION=true
            ;;
        --skip_extended_eval)
            SKIP_EXTENDED_EVAL=true
            ;;
        --skip_multigpu)
            SKIP_MULTIGPU=true
            ;;
        --skip_error_handling)
            SKIP_ERROR_HANDLING=true
            ;;
        --skip_file_io)
            SKIP_FILE_IO=true
            ;;
        --skip_monitor)
            SKIP_MONITOR=true
            ;;
        --skip_edge_cases)
            SKIP_EDGE_CASES=true
            ;;
        --include_nli_faithfulness)
            INCLUDE_NLI_FAITHFULNESS=true
            ;;
        --keep_test_data)
            KEEP_TEST_DATA=true
            ;;
        --dry-run|--debug)
            DRY_RUN=true
            ;;
        *)
            echo "ERROR: Unknown argument: $arg"
            echo ""
            echo "Usage: sbatch --export=SLURM_ACCOUNT=YOUR_PROJECT_ID run_test_e2e.sbatch [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model=MODEL              Model short name (default: gemma-2b)"
            echo "  --test_dir=DIR             Directory for test outputs (default: temp directory)"
            echo "  --skip_training            Skip training test"
            echo "  --skip_evaluation          Skip evaluation test"
            echo "  --skip_extended_eval       Skip extended evaluation metrics test (BERTScore, NLI)"
            echo "  --skip_multigpu            Skip multi-GPU evaluation test"
            echo "  --skip_error_handling      Skip error handling tests"
            echo "  --skip_file_io             Skip file I/O and persistence tests"
            echo "  --skip_monitor             Skip monitor script integration tests"
            echo "  --skip_edge_cases          Skip edge case tests"
            echo "  --include_nli_faithfulness Include NLI faithfulness in extended metrics test"
            echo "  --keep_test_data           Keep test datasets and outputs after testing"
            echo "  --dry-run|--debug          Debug mode: validate inputs and show what would run"
            echo ""
            echo "Valid models: gemma-2b, gemma-7b, gemma-2-9b, gemma-2-27b, gemma-3-12b, gemma-3-27b,"
            echo "              viking-7b, viking-13b, viking-33b, normistral-7b, normistral-11b, normistral-11b-long, normistral-7b-instruct,"
            echo "              norskgpt-llama3-8b, llama-3.1-8b-instruct, llama-2-13b-chat-norwegian,"
            echo "              eurollm-9b-instruct, norwai-mistral-7b-instruct, nb-gpt-j-6b, mt5"
            exit 1
            ;;
    esac
done

# Validation functions
validate_model() {
    local model=$1
    case "$model" in
        gemma-2b|gemma-7b|gemma-2-9b|gemma-2-27b|gemma-3-12b|gemma-3-27b| \
        viking-7b|viking-13b|viking-33b| \
        normistral-7b|normistral-11b|normistral-11b-long|normistral-7b-instruct| \
        norskgpt-llama3-8b|llama-3.1-8b-instruct|llama-2-13b-chat-norwegian| \
        eurollm-9b-instruct|norwai-mistral-7b-instruct|nb-gpt-j-6b|mt5)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Validate model
if ! validate_model "$MODEL"; then
    echo "ERROR: Invalid model: $MODEL"
    echo "Valid models: gemma-2b, gemma-7b, gemma-2-9b, gemma-2-27b, gemma-3-12b, gemma-3-27b,"
    echo "              viking-7b, viking-13b, viking-33b, normistral-7b, normistral-11b, normistral-11b-long, normistral-7b-instruct,"
    echo "              norskgpt-llama3-8b, llama-3.1-8b-instruct, llama-2-13b-chat-norwegian,"
    echo "              eurollm-9b-instruct, norwai-mistral-7b-instruct, nb-gpt-j-6b, mt5"
    exit 1
fi

# Print configuration
echo "==================================================================="
echo "END-TO-END TEST CONFIGURATION"
echo "==================================================================="
echo "  Model: $MODEL"
echo "  Test directory: ${TEST_DIR:-<temporary>}"
echo "  Skip training: $SKIP_TRAINING"
echo "  Skip evaluation: $SKIP_EVALUATION"
echo "  Skip extended eval: $SKIP_EXTENDED_EVAL"
echo "  Skip multi-GPU: $SKIP_MULTIGPU"
echo "  Skip error handling: $SKIP_ERROR_HANDLING"
echo "  Skip file I/O: $SKIP_FILE_IO"
echo "  Skip monitor: $SKIP_MONITOR"
echo "  Skip edge cases: $SKIP_EDGE_CASES"
echo "  Include NLI faithfulness: $INCLUDE_NLI_FAITHFULNESS"
echo "  Keep test data: $KEEP_TEST_DATA"
echo "  WandB: DISABLED (for testing)"
echo "==================================================================="

# ===== DRY-RUN MODE =====
if [ "$DRY_RUN" = "true" ]; then
    echo ""
    echo "==================================================================="
    echo "DRY-RUN MODE: Validation and Preview"
    echo "==================================================================="
    echo ""
    echo "✓ All validations passed!"
    echo ""
    echo "Would execute with the following configuration:"
    echo "  Model: $MODEL"
    echo "  Test directory: ${TEST_DIR:-<temporary>}"
    echo ""
    echo "Test command would be:"
    echo "  python scripts/test_e2e.py"
    echo "    --model $MODEL"
    [ -n "$TEST_DIR" ] && echo "    --test_dir $TEST_DIR"
    [ "$SKIP_TRAINING" = "true" ] && echo "    --skip_training"
    [ "$SKIP_EVALUATION" = "true" ] && echo "    --skip_evaluation"
    [ "$SKIP_EXTENDED_EVAL" = "true" ] && echo "    --skip_extended_eval"
    [ "$SKIP_MULTIGPU" = "true" ] && echo "    --skip_multigpu"
    [ "$SKIP_ERROR_HANDLING" = "true" ] && echo "    --skip_error_handling"
    [ "$SKIP_FILE_IO" = "true" ] && echo "    --skip_file_io"
    [ "$SKIP_MONITOR" = "true" ] && echo "    --skip_monitor"
    [ "$SKIP_EDGE_CASES" = "true" ] && echo "    --skip_edge_cases"
    [ "$INCLUDE_NLI_FAITHFULNESS" = "true" ] && echo "    --include_nli_faithfulness"
    [ "$KEEP_TEST_DATA" = "true" ] && echo "    --keep_test_data"
    echo ""
    echo "Environment variables:"
    echo "  ✓ SLURM_ACCOUNT: ${SLURM_ACCOUNT:-NOT SET}"
    echo "  ✓ WANDB_API_KEY: ${WANDB_API_KEY:+SET (hidden)}"
    echo "  ✓ HUGGINGFACE_TOKEN: ${HUGGINGFACE_TOKEN:+SET (hidden)}"
    echo ""
    echo "==================================================================="
    echo "Dry-run complete. No job was submitted."
    echo "To actually run, remove --dry-run flag."
    echo "==================================================================="
    exit 0
fi

echo "==================================================================="
echo "Running End-to-End Test: $MODEL"
echo "Single GPU, minimal resources (5 steps, 10 examples)"
echo "==================================================================="

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

# stage model/dataset into job-local cache (fast local read, writable)
PROJECT_HF_DIR=/cluster/projects/YOUR_PROJECT_ID/shared/.cache/huggingface
mkdir -p "$HF_HOME/hub"

# Convert model name to HuggingFace cache directory format
convert_model_to_cache_dir() {
    local model=$1
    case "$model" in
        gemma-2b) echo "models--google--gemma-2b" ;;
        gemma-7b) echo "models--google--gemma-7b" ;;
        gemma-2-9b) echo "models--google--gemma-2-9b" ;;
        gemma-2-27b) echo "models--google--gemma-2-27b" ;;
        gemma-3-12b) echo "models--google--gemma-3-12b-pt" ;;
        gemma-3-27b) echo "models--google--gemma-3-27b-pt" ;;
        viking-7b) echo "models--LumiOpen--Viking-7B" ;;
        viking-13b) echo "models--LumiOpen--Viking-13B" ;;
        viking-33b) echo "models--LumiOpen--Viking-33B" ;;
        normistral-7b) echo "models--norallm--normistral-7b-warm" ;;
        normistral-11b) echo "models--norallm--normistral-11b-warm" ;;
        normistral-11b-long) echo "models--norallm--normistral-11b-long" ;;
        normistral-7b-instruct) echo "models--norallm--normistral-7b-warm-instruct" ;;
        norskgpt-llama3-8b) echo "models--bineric--norskgpt-llama3-8b" ;;
        llama-3.1-8b-instruct) echo "models--meta-llama--Llama-3.1-8B-Instruct" ;;
        llama-2-13b-chat-norwegian) echo "models--ruternorway--llama-2-13b-chat-norwegian" ;;
        eurollm-9b-instruct) echo "models--utter-project--EuroLLM-9B-2512-instruct" ;;
        norwai-mistral-7b-instruct) echo "models--NorwAI--NorwAI-Mistral-7B-instruct" ;;
        nb-gpt-j-6b) echo "models--NbAiLab--nb-gpt-j-6B-torgersen-alpaca" ;;
        *) echo "models--${model//\//--}" ;;
    esac
}

MODEL_CACHE_DIR=$(convert_model_to_cache_dir "$MODEL")

# Stage model cache if it exists in shared location
if [ -d "$PROJECT_HF_DIR/hub/$MODEL_CACHE_DIR" ]; then
    echo "Staging model cache from shared location..."
    mkdir -p "$HF_HOME/hub"
    rsync -aq "$PROJECT_HF_DIR/hub/$MODEL_CACHE_DIR/" "$HF_HOME/hub/$MODEL_CACHE_DIR/" || true
fi

# Stage other caches
[ -d "$PROJECT_HF_DIR/metrics" ] && rsync -aq "$PROJECT_HF_DIR/metrics/" "$HF_METRICS_CACHE/"
[ -d "$PROJECT_HF_DIR/modules" ] && rsync -aq "$PROJECT_HF_DIR/modules/" "$HF_MODULES_CACHE/"

# Load modules
module --force purge
module load BuildEnv/NeoverseV2
module load NRIS/GPU
module load CUDA/12.6.0
module load PyTorch/2.8.0

source .env

export IMAGE=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif

# Build test command arguments
TEST_ARGS="--model $MODEL"
[ -n "$TEST_DIR" ] && TEST_ARGS="$TEST_ARGS --test_dir $TEST_DIR"
[ "$SKIP_TRAINING" = "true" ] && TEST_ARGS="$TEST_ARGS --skip_training"
[ "$SKIP_EVALUATION" = "true" ] && TEST_ARGS="$TEST_ARGS --skip_evaluation"
[ "$SKIP_EXTENDED_EVAL" = "true" ] && TEST_ARGS="$TEST_ARGS --skip_extended_eval"
[ "$SKIP_MULTIGPU" = "true" ] && TEST_ARGS="$TEST_ARGS --skip_multigpu"
[ "$SKIP_ERROR_HANDLING" = "true" ] && TEST_ARGS="$TEST_ARGS --skip_error_handling"
[ "$SKIP_FILE_IO" = "true" ] && TEST_ARGS="$TEST_ARGS --skip_file_io"
[ "$SKIP_MONITOR" = "true" ] && TEST_ARGS="$TEST_ARGS --skip_monitor"
[ "$SKIP_EDGE_CASES" = "true" ] && TEST_ARGS="$TEST_ARGS --skip_edge_cases"
[ "$INCLUDE_NLI_FAITHFULNESS" = "true" ] && TEST_ARGS="$TEST_ARGS --include_nli_faithfulness"
[ "$KEEP_TEST_DATA" = "true" ] && TEST_ARGS="$TEST_ARGS --keep_test_data"

# Command to run inside apptainer
# Disable WandB for testing
export WANDB_DISABLED=true
export WANDB_MODE=disabled

# Get absolute path to current directory (where the script is run from)
WORK_DIR=$(pwd)

# If TEST_DIR is not specified, use a directory in WORK_DIR to ensure it persists
# (Apptainer containers may have ephemeral /tmp directories)
if [ -z "$TEST_DIR" ]; then
    TEST_DIR="$WORK_DIR/test_outputs/test_${SLURM_JOB_ID:-$$}_$(date +%s)"
    mkdir -p "$TEST_DIR"
    TEST_ARGS="$TEST_ARGS --test_dir $TEST_DIR"
    echo "Using test directory in WORK_DIR (persists on host): $TEST_DIR"
fi

TEST_CMD="export HF_HOME=$HF_HOME && export HF_DATASETS_CACHE=$HF_DATASETS_CACHE && export HF_MODULES_CACHE=$HF_MODULES_CACHE && export HF_METRICS_CACHE=$HF_METRICS_CACHE && export PYTHONUSERBASE=$PYTHONUSERBASE && export WANDB_DISABLED=true && export WANDB_MODE=disabled && export TOKENIZERS_PARALLELISM=false && cd $WORK_DIR && python scripts/test_e2e.py $TEST_ARGS"

# Apptainer command - install packages and run test
COMMAND="pip install --upgrade pip setuptools wheel > /dev/null 2>&1 && pip install --break-system-packages --only-binary=:all: transformers==4.45.2 tokenizers==0.20.3 datasets==3.0.0 sentencepiece==0.2.1 > /dev/null 2>&1 && pip install --break-system-packages --no-binary=rouge-score rouge-score==0.1.2 > /dev/null 2>&1 && pip install --break-system-packages --no-deps accelerate==1.0.1 peft==0.13.2 > /dev/null 2>&1 && pip install --break-system-packages --upgrade pyyaml psutil typing-extensions packaging==24.2 safetensors > /dev/null 2>&1 && pip install --break-system-packages --upgrade wandb evaluate sentence-transformers bert_score > /dev/null 2>&1 && $TEST_CMD"

# Run test inside apptainer
echo "Running test inside apptainer..."
apptainer exec --nv $IMAGE bash -c "$COMMAND"

TEST_STATUS=$?

# Check test status
if [ $TEST_STATUS -eq 0 ]; then
    echo ""
    echo "==================================================================="
    echo "✓ All tests passed! Refactorings are working correctly."
    echo "==================================================================="
else
    echo ""
    echo "==================================================================="
    echo "✗ Some tests failed. Check the output above for details."
    echo "==================================================================="
fi

exit $TEST_STATUS
