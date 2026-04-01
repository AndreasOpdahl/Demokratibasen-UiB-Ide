#!/usr/bin/env bash
# Mass evaluation script for finetuned models that were not monitored.
#
# Submits one Slurm evaluation job per model. Each job evaluates ALL checkpoints
# in that model's directory (main + regular_checkpoints/ + major_checkpoints/).
#
# Usage:
#   ./run_evaluate_multiple.sh --models="model1,model2,model3" --account=YOUR_ACCOUNT [OPTIONS]

set -euo pipefail

# ===== DEFAULTS =====
MODELS=""
SLURM_ACCOUNT=""
VAL_DATASET="${VAL_DATASET:-data/dataset_149978_examples/149978_text_summary_examples_val.jsonl}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-500}"  # 500 (default) or 1000; 1000 → separate -examples_1000.json/.jsonl files
INCLUDE_NLI_FAITHFULNESS="${INCLUDE_NLI_FAITHFULNESS:-false}"
KEEP_EXISTING="${KEEP_EXISTING:-true}"  # Skip already-evaluated checkpoints
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-false}"  # Re-evaluate even when results exist (overwrites)
SKIP_NO_CHECKPOINTS="${SKIP_NO_CHECKPOINTS:-true}"  # Skip models with no checkpoints
SPECIFIC_CHECKPOINTS="${SPECIFIC_CHECKPOINTS:-}"  # e.g. "6000" or "5000 6000" to evaluate only those
DRY_RUN="${DRY_RUN:-false}"

# ===== PARSE ARGUMENTS =====
while [[ $# -gt 0 ]]; do
    case $1 in
        --models=*) MODELS="${1#*=}" ;;
        --account=*) SLURM_ACCOUNT="${1#*=}" ;;
        --val_dataset=*) VAL_DATASET="${1#*=}" ;;
        --val_data_size=*) VAL_DATA_SIZE="${1#*=}" ;;
        --specific_checkpoints=*) SPECIFIC_CHECKPOINTS="${1#*=}" ;;
        --include_nli_faithfulness) INCLUDE_NLI_FAITHFULNESS=true ;;
        --no-keep_existing) KEEP_EXISTING=false ;;
        --keep_existing) KEEP_EXISTING=true ;;
        --force_recompute) FORCE_RECOMPUTE=true ;;
        --no-force_recompute) FORCE_RECOMPUTE=false ;;
        --no-skip_empty) SKIP_NO_CHECKPOINTS=false ;;
        --dry-run) DRY_RUN=true ;;
        -h|--help)
            cat << EOF
Usage: $0 --models="MODEL1,MODEL2,..." --account=SLURM_ACCOUNT [OPTIONS]

Mass evaluate all checkpoints for finetuned models that were not monitored.
Submits one evaluation job per model using run_evaluate_distributed_checkpoints_multigpu.sbatch.

Required:
  --models=LIST              Comma-separated list of model names
  --account=ACCOUNT          SLURM account/project ID

Optional:
  --val_dataset=PATH         Validation dataset (default: data/dataset_149978_examples/149978_text_summary_examples_val.jsonl)
  --val_data_size=N          Validation examples: 500 (default) or 1000 (writes to -examples_1000.json/.jsonl)
  --specific_checkpoints=LIST  Space-separated checkpoint numbers (e.g. "100 200 300"); default: all
  --include_nli_faithfulness  Enable NLI faithfulness evaluation (slow)
  --no-keep_existing         Re-evaluate checkpoints that already have results (default: skip them)
  --force_recompute          Re-evaluate even when results exist; overwrites existing (e.g. backfill)
  --no-skip_empty            Submit jobs even for models with no checkpoints (will fail)
  --dry-run                  Show what would be submitted without submitting

Valid models: gemma-2b, gemma-7b-it, gemma-2-9b, gemma-2-27b, gemma-3-12b, gemma-3-27b,
              viking-7b, viking-13b, viking-33b, normistral-7b, normistral-11b, normistral-11b-long,
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

if [ ! -f "$VAL_DATASET" ]; then
    echo "ERROR: Validation dataset not found: $VAL_DATASET"
    exit 1
fi

# ===== PARSE MODELS =====
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
VALID_MODELS=()
for model in "${MODEL_ARRAY[@]}"; do
    model=$(echo "$model" | xargs)
    [ -n "$model" ] && VALID_MODELS+=("$model")
done

# ===== COUNT CHECKPOINTS PER MODEL =====
count_checkpoints() {
    local model=$1
    local dir="models/${model}-apptainer-fsdp"
    local main=0 regular=0 major=0
    [ -d "$dir" ] || return 1
    main=$(find "$dir" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | wc -l)
    [ -d "$dir/regular_checkpoints" ] && regular=$(find "$dir/regular_checkpoints" -mindepth 1 -maxdepth 1 -type d \( -name "checkpoint-*" -o -name "regular-checkpoint-*" \) 2>/dev/null | wc -l)
    [ -d "$dir/major_checkpoints" ] && major=$(find "$dir/major_checkpoints" -mindepth 1 -maxdepth 1 -type d \( -name "checkpoint-*" -o -name "major-checkpoint-*" \) 2>/dev/null | wc -l)
    # Deduplicate: regular and major are backups, main may be missing; total unique steps
    echo $((main + regular + major))
}

# ===== PRINT CONFIG =====
echo "================================================================================"
echo "Mass Evaluation (unmonitored models)"
echo "================================================================================"
echo "Models: ${VALID_MODELS[*]}"
echo "Account: $SLURM_ACCOUNT"
echo "Validation: $VAL_DATASET (size: $VAL_DATA_SIZE)"
echo "Keep existing: $KEEP_EXISTING"
echo "Force recompute: $FORCE_RECOMPUTE"
echo "Skip empty: $SKIP_NO_CHECKPOINTS"
echo "Specific checkpoints: ${SPECIFIC_CHECKPOINTS:-all}"
echo "NLI faithfulness: $INCLUDE_NLI_FAITHFULNESS"
echo "================================================================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - No jobs will be submitted"
    echo ""
fi

# ===== SUBMIT JOBS =====
echo "Submitting evaluation jobs..."
EVAL_JOBS=()
for model in "${VALID_MODELS[@]}"; do
    dir="models/${model}-apptainer-fsdp"
    n_ckpt=$(count_checkpoints "$model" 2>/dev/null || echo "0")

    if [ "$SKIP_NO_CHECKPOINTS" = true ] && [ "${n_ckpt:-0}" -eq 0 ]; then
        echo "  ⚠ Skipping $model (no checkpoints in $dir)"
        EVAL_JOBS+=("$model:SKIPPED")
        continue
    fi

    if [ "$DRY_RUN" = true ]; then
        ckpt_info="~$n_ckpt checkpoints"
        [ -n "$SPECIFIC_CHECKPOINTS" ] && ckpt_info="checkpoints: $SPECIFIC_CHECKPOINTS"
        echo "  [DRY RUN] Would submit: sbatch --account=$SLURM_ACCOUNT run_evaluate_distributed_checkpoints_multigpu.sbatch --model=$model ($ckpt_info)"
        EVAL_JOBS+=("$model:DRY")
        continue
    fi

    # Build sbatch command
    SBATCH_CMD=(sbatch --account="$SLURM_ACCOUNT" \
        --export="MODEL=$model,CHECKPOINT_BASE_DIR=$dir,VAL_DATASET=$VAL_DATASET,INCLUDE_NLI_FAITHFULNESS=$INCLUDE_NLI_FAITHFULNESS,KEEP_EXISTING=$KEEP_EXISTING,FORCE_RECOMPUTE=$FORCE_RECOMPUTE")

    SBATCH_CMD+=(run_evaluate_distributed_checkpoints_multigpu.sbatch \
        --model="$model" \
        --val_dataset="$VAL_DATASET" \
        --val_data_size="$VAL_DATA_SIZE")

    if [ "$INCLUDE_NLI_FAITHFULNESS" = true ]; then
        SBATCH_CMD+=(--include_nli_faithfulness)
    fi

    if [ -n "$SPECIFIC_CHECKPOINTS" ]; then
        SBATCH_CMD+=(--specific_checkpoints="$SPECIFIC_CHECKPOINTS")
    fi

    if [ "$FORCE_RECOMPUTE" = true ]; then
        SBATCH_CMD+=(--force_recompute)
    fi

    set +e
    OUTPUT=$("${SBATCH_CMD[@]}" 2>&1)
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -eq 0 ] && echo "$OUTPUT" | grep -q "Submitted batch job"; then
        JOB_ID=$(echo "$OUTPUT" | sed -n 's/.*Submitted batch job \([0-9]*\).*/\1/p')
        echo "  ✓ $model: Job ID $JOB_ID (~$n_ckpt checkpoints)"
        EVAL_JOBS+=("$model:$JOB_ID")
    else
        echo "  ✗ $model: Failed (exit $EXIT_CODE) - $OUTPUT"
        EVAL_JOBS+=("$model:FAILED")
    fi
done

# ===== SUMMARY =====
echo ""
echo "================================================================================"
echo "Summary"
echo "================================================================================"
echo "Evaluation jobs: ${#EVAL_JOBS[@]}"
echo ""
echo "Check status: squeue -u \$USER"
echo "View logs: logs/gpu-eval-checkpoints-multigpu-*.out"
echo "Results: models/<model>-apptainer-fsdp/all_eval_results/"
echo "================================================================================"
