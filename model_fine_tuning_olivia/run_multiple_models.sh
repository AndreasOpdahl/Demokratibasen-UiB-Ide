#!/usr/bin/env bash
# Script to run finetune and monitor scripts for multiple models simultaneously
#
# Usage:
#   ./run_multiple_models.sh --models="model1,model2,model3" --account=YOUR_ACCOUNT [OPTIONS]

set -euo pipefail

# ===== DEFAULTS =====
MODELS=""
SLURM_ACCOUNT=""
TRAIN_DATASET="${TRAIN_DATASET:-data/output/new_processed_data_train.jsonl}"
VAL_DATASET="${VAL_DATASET:-data/output/new_processed_data_val.jsonl}"
NUM_GPUS="${NUM_GPUS:-4}"
TASK_LIMIT="${TASK_LIMIT:---max_steps 10000}"
DISTRIBUTION_FLAG="${DISTRIBUTION_FLAG:---fsdp}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
INCLUDE_NLI_FAITHFULNESS="${INCLUDE_NLI_FAITHFULNESS:-false}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
# Job dependency disabled - monitor script has built-in waiting logic
USE_JOB_DEPENDENCY=false
DRY_RUN="${DRY_RUN:-false}"

# ===== PARSE ARGUMENTS =====
while [[ $# -gt 0 ]]; do
    case $1 in
        --models=*) MODELS="${1#*=}" ;;
        --account=*) SLURM_ACCOUNT="${1#*=}" ;;
        --train_dataset=*) TRAIN_DATASET="${1#*=}" ;;
        --val_dataset=*) VAL_DATASET="${1#*=}" ;;
        --num_gpus=*) NUM_GPUS="${1#*=}" ;;
        --task_limit=*) TASK_LIMIT="${1#*=}" ;;
        --distribution_flag=*) DISTRIBUTION_FLAG="${1#*=}" ;;
        --check_interval=*) CHECK_INTERVAL="${1#*=}" ;;
        --early_stopping_patience=*) EARLY_STOPPING_PATIENCE="${1#*=}" ;;
        --include_nli_faithfulness) INCLUDE_NLI_FAITHFULNESS=true ;;
        --resume_checkpoint=*) RESUME_CHECKPOINT="${1#*=}" ;;
        --no-job-dependency) USE_JOB_DEPENDENCY=false ;;
        --dry-run) DRY_RUN=true ;;
        -h|--help)
            cat << EOF
Usage: $0 --models="MODEL1,MODEL2,..." --account=SLURM_ACCOUNT [OPTIONS]

Required:
  --models=LIST              Comma-separated list of model names
  --account=ACCOUNT          SLURM account/project ID

Optional:
  --train_dataset=PATH       Training dataset (default: data/output/new_processed_data_train.jsonl)
  --val_dataset=PATH         Validation dataset (default: data/output/new_processed_data_val.jsonl)
  --num_gpus=N               Number of GPUs per model (default: 4)
  --task_limit="OPTIONS"     Task limit, e.g. "--max_steps 10000" (default: --max_steps 10000)
  --resume_checkpoint=NAME   Resume from checkpoint (e.g. checkpoint-5000, resolved per model)
  --distribution_flag=FLAG   --fsdp or --ddp (default: --fsdp)
  --check_interval=N         Monitor check interval in seconds (default: 60)
  --early_stopping_patience=N Early stopping patience (default: 10)
  --include_nli_faithfulness  Enable NLI faithfulness evaluation
  --no-job-dependency         (Deprecated - dependency is disabled by default)
  --dry-run                   Show what would be submitted

Valid models: gemma-2b, gemma-7b, gemma-2-9b, gemma-2-27b, gemma-3-12b, gemma-3-27b,
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

if [ ! -f "$TRAIN_DATASET" ] || [ ! -f "$VAL_DATASET" ]; then
    echo "ERROR: Dataset files not found"
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
echo "Multi-Model Training and Monitoring"
echo "================================================================================"
echo "Models: ${VALID_MODELS[*]}"
echo "Account: $SLURM_ACCOUNT"
echo "Training: $TRAIN_DATASET"
echo "Validation: $VAL_DATASET"
echo "GPUs per model: $NUM_GPUS"
echo "Task limit: $TASK_LIMIT"
echo "Resume checkpoint: ${RESUME_CHECKPOINT:-none}"
echo "Distribution: $DISTRIBUTION_FLAG"
echo "================================================================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - No jobs will be submitted"
    echo ""
fi

# ===== SUBMIT JOBS =====
echo "Submitting training jobs..."
TRAIN_JOBS=()
for model in "${VALID_MODELS[@]}"; do
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] Would submit: sbatch --account=$SLURM_ACCOUNT run_finetune_multinode.sbatch --model=$model"
    else
        # Submit training job and capture job ID
        EXPORT_VARS="MODEL=$model,SLURM_ACCOUNT=$SLURM_ACCOUNT,TRAIN_DATASET=$TRAIN_DATASET,VAL_DATASET=$VAL_DATASET,NUM_GPUS=$NUM_GPUS,TASK_LIMIT=$TASK_LIMIT,DISTRIBUTION_FLAG=$DISTRIBUTION_FLAG"
        [ -n "$RESUME_CHECKPOINT" ] && EXPORT_VARS="$EXPORT_VARS,RESUME_CHECKPOINT=$RESUME_CHECKPOINT"
        SBATCH_ARGS=(--model="$model" --num_gpus="$NUM_GPUS" --train_dataset="$TRAIN_DATASET" --val_dataset="$VAL_DATASET" --task_limit="$TASK_LIMIT" --distribution_flag="$DISTRIBUTION_FLAG")
        [ -n "$RESUME_CHECKPOINT" ] && SBATCH_ARGS+=(--resume_checkpoint="$RESUME_CHECKPOINT")
        OUTPUT=$(sbatch --account="$SLURM_ACCOUNT" --export="$EXPORT_VARS" run_finetune_multinode.sbatch "${SBATCH_ARGS[@]}" 2>&1)
        
        if echo "$OUTPUT" | grep -q "Submitted batch job"; then
            JOB_ID=$(echo "$OUTPUT" | sed -n 's/.*Submitted batch job \([0-9]*\).*/\1/p')
            TRAIN_JOBS+=("$model:$JOB_ID")
            echo "  ✓ $model: Job ID $JOB_ID"
        else
            echo "  ✗ $model: Failed - $OUTPUT"
            TRAIN_JOBS+=("$model:FAILED")
        fi
    fi
done

echo ""
echo "Submitting monitor jobs..."
MONITOR_JOBS=()
for job_info in "${TRAIN_JOBS[@]}"; do
    model="${job_info%%:*}"
    train_job_id="${job_info##*:}"
    
    if [ "$train_job_id" = "FAILED" ]; then
        echo "  ⚠ Skipping monitor for $model (training job failed)"
        MONITOR_JOBS+=("$model:SKIPPED")
        continue
    fi

    # Clear stale .early_stop from previous run so this monitor doesn't exit immediately
    early_stop_file="models/${model}-apptainer-fsdp/.early_stop"
    if [ -f "$early_stop_file" ]; then
        rm -f "$early_stop_file"
        echo "  Cleared stale .early_stop for $model"
    fi

    if [ "$DRY_RUN" = true ]; then
        DEP_MSG=""
        [ "$USE_JOB_DEPENDENCY" = true ] && DEP_MSG=" (depends on $train_job_id)"
        echo "  [DRY RUN] Would submit: sbatch --account=$SLURM_ACCOUNT$DEP_MSG run_monitor_evaluation.sbatch --model=$model"
    else
        # Build monitor job command
        MONITOR_CMD=(sbatch --account="$SLURM_ACCOUNT" \
            --export="MODEL=$model,OUTPUT_DIR=models/${model}-apptainer-fsdp,VAL_DATASET=$VAL_DATASET,CHECK_INTERVAL=$CHECK_INTERVAL,EARLY_STOPPING_PATIENCE=$EARLY_STOPPING_PATIENCE,INCLUDE_NLI_FAITHFULNESS=$INCLUDE_NLI_FAITHFULNESS")
        
        # Note: We don't use job dependency because:
        # 1. The monitor script already waits for training_started.txt (up to 1 hour)
        # 2. This is simpler and more reliable
        # 3. Monitor can start immediately and wait for training to begin
        
        MONITOR_CMD+=(run_monitor_evaluation.sbatch \
            --model="$model" \
            --val_dataset="$VAL_DATASET" \
            --check_interval="$CHECK_INTERVAL" \
            --early_stopping_patience="$EARLY_STOPPING_PATIENCE")
        
        if [ "$INCLUDE_NLI_FAITHFULNESS" = true ]; then
            MONITOR_CMD+=(--include_nli_faithfulness)
        fi
        
        # Submit monitor job (disable exit on error temporarily to capture output)
        set +e
        OUTPUT=$("${MONITOR_CMD[@]}" 2>&1)
        EXIT_CODE=$?
        set -e
        
        if [ $EXIT_CODE -eq 0 ] && echo "$OUTPUT" | grep -q "Submitted batch job"; then
            JOB_ID=$(echo "$OUTPUT" | sed -n 's/.*Submitted batch job \([0-9]*\).*/\1/p')
            echo "  ✓ $model: Job ID $JOB_ID (will wait for training to start)"
            MONITOR_JOBS+=("$model:$JOB_ID")
        else
            echo "  ✗ $model: Failed (exit code: $EXIT_CODE) - $OUTPUT"
            MONITOR_JOBS+=("$model:FAILED")
        fi
    fi
done

# ===== SUMMARY =====
echo ""
echo "================================================================================"
echo "Summary"
echo "================================================================================"
echo "Training jobs: ${#TRAIN_JOBS[@]}"
echo "Monitor jobs: ${#MONITOR_JOBS[@]}"
echo ""
echo "Check status: squeue -u \$USER"
echo "View logs: logs/gpu-finetune-apptainer-*.out and logs/gpu-monitor-eval-*.out"
echo "================================================================================"
