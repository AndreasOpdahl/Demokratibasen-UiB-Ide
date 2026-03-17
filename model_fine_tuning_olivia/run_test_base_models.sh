#!/usr/bin/env bash
# Script to test multiple base models (before fine-tuning) to verify prompt formatting
#
# Usage:
#   ./run_test_base_models.sh --models="model1,model2,model3" --account=YOUR_ACCOUNT [OPTIONS]

set -euo pipefail

# ===== DEFAULTS =====
MODELS=""
SLURM_ACCOUNT=""
TEST_DATASET="${TEST_DATASET:-data/output/new_processed_data_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-base_model_predictions}"
NUM_EXAMPLES="${NUM_EXAMPLES:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
USE_MULTI_GPU="${USE_MULTI_GPU:-false}"
DRY_RUN="${DRY_RUN:-false}"

# ===== PARSE ARGUMENTS =====
while [[ $# -gt 0 ]]; do
    case $1 in
        --models=*) MODELS="${1#*=}" ;;
        --account=*) SLURM_ACCOUNT="${1#*=}" ;;
        --test_dataset=*) TEST_DATASET="${1#*=}" ;;
        --output_dir=*) OUTPUT_DIR="${1#*=}" ;;
        --num_examples=*) NUM_EXAMPLES="${1#*=}" ;;
        --max_new_tokens=*) MAX_NEW_TOKENS="${1#*=}" ;;
        --use_multi_gpu) USE_MULTI_GPU=true ;;
        --dry-run) DRY_RUN=true ;;
        -h|--help)
            cat << EOF
Usage: $0 --models="MODEL1,MODEL2,..." --account=SLURM_ACCOUNT [OPTIONS]

Required:
  --models=LIST              Comma-separated list of model names
  --account=ACCOUNT          SLURM account/project ID

Optional:
  --test_dataset=PATH        Test dataset (default: data/output/new_processed_data_val.jsonl)
  --output_dir=DIR           Output directory for predictions (default: base_model_predictions)
  --num_examples=N           Number of examples to test per model (default: 10)
  --max_new_tokens=N         Maximum tokens to generate (default: 200)
  --use_multi_gpu            Use model parallelism across multiple GPUs
  --dry-run                  Show what would be submitted

Valid models: gemma-2b, gemma-7b, gemma-2-9b, gemma-2-27b, gemma-3-12b, gemma-3-27b,
              viking-7b, viking-13b, viking-33b, normistral-7b, normistral-11b,
              normistral-7b-instruct, norskgpt-llama3-8b, llama-3.1-8b-instruct,
              llama-2-13b-chat-norwegian, eurollm-9b-instruct, norwai-mistral-7b-instruct,
              nb-gpt-j-6b, mt5
EOF
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

# ===== VALIDATE =====
if [ -z "$MODELS" ] || [ -z "$SLURM_ACCOUNT" ]; then
    echo "ERROR: --models and --account are required"
    exit 1
fi

if [ ! -f "$TEST_DATASET" ]; then
    echo "ERROR: Test dataset file not found: $TEST_DATASET"
    exit 1
fi

# ===== PARSE MODELS =====
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
VALID_MODELS=()
for model in "${MODEL_ARRAY[@]}"; do
    model=$(echo "$model" | xargs)  # Trim whitespace
    VALID_MODELS+=("$model")
done

# ===== PRINT CONFIG =====
echo "================================================================================"
echo "Base Model Testing"
echo "================================================================================"
echo "Models: ${VALID_MODELS[*]}"
echo "Account: $SLURM_ACCOUNT"
echo "Test dataset: $TEST_DATASET"
echo "Output directory: $OUTPUT_DIR"
echo "Examples per model: $NUM_EXAMPLES"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Use multi-GPU: $USE_MULTI_GPU"
echo "================================================================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - No jobs will be submitted"
    echo ""
fi

# ===== SUBMIT JOBS =====
echo "Submitting base model test jobs..."
TEST_JOBS=()
for model in "${VALID_MODELS[@]}"; do
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] Would submit: sbatch --account=$SLURM_ACCOUNT run_test_base_models.sbatch --model=$model"
    else
        # Build command
        TEST_CMD=(sbatch --account="$SLURM_ACCOUNT" \
            --export="MODEL=$model,SLURM_ACCOUNT=$SLURM_ACCOUNT,TEST_DATASET=$TEST_DATASET,OUTPUT_DIR=$OUTPUT_DIR,NUM_EXAMPLES=$NUM_EXAMPLES,MAX_NEW_TOKENS=$MAX_NEW_TOKENS,USE_MULTI_GPU=$USE_MULTI_GPU")
        
        TEST_CMD+=(run_test_base_models.sbatch \
            --model="$model" \
            --test_dataset="$TEST_DATASET" \
            --output_dir="$OUTPUT_DIR" \
            --num_examples="$NUM_EXAMPLES" \
            --max_new_tokens="$MAX_NEW_TOKENS")
        
        if [ "$USE_MULTI_GPU" = true ]; then
            TEST_CMD+=(--use_multi_gpu)
        fi
        
        # Submit job
        set +e
        OUTPUT=$("${TEST_CMD[@]}" 2>&1)
        EXIT_CODE=$?
        set -e
        
        if [ $EXIT_CODE -eq 0 ] && echo "$OUTPUT" | grep -q "Submitted batch job"; then
            JOB_ID=$(echo "$OUTPUT" | sed -n 's/.*Submitted batch job \([0-9]*\).*/\1/p')
            echo "  ✓ $model: Job ID $JOB_ID"
            TEST_JOBS+=("$model:$JOB_ID")
        else
            echo "  ✗ $model: Failed (exit code: $EXIT_CODE) - $OUTPUT"
            TEST_JOBS+=("$model:FAILED")
        fi
    fi
done

# ===== SUMMARY =====
echo ""
echo "================================================================================"
echo "Summary"
echo "================================================================================"
echo "Test jobs submitted: ${#TEST_JOBS[@]}"
echo ""
echo "Check status: squeue -u \$USER"
echo "View logs: logs/gpu-test-base-models-*.out"
echo "Predictions will be saved to: $OUTPUT_DIR/"
echo "================================================================================"
