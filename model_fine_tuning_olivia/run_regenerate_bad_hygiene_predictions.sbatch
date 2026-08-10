#!/usr/bin/env bash
#SBATCH -J regen-bad-preds
#SBATCH -p accel
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=72:00:00
# Account: REQUIRED - Pass via: sbatch --account=YOUR_ACCOUNT run_regenerate_bad_hygiene_predictions.sbatch ...
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# Slurm stages scripts under /var/spool; use submit dir when available.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SBATCH_DIR="$SLURM_SUBMIT_DIR"
else
    SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SBATCH_DIR" || { echo "ERROR: Cannot cd to $SBATCH_DIR"; exit 1; }
echo "Working directory for this job: $SBATCH_DIR"

if [ -f .env ]; then
    source .env
else
    echo "WARNING: .env file not found. Using environment variables/defaults."
fi

# ===== CONFIGURATION =====
MODEL=${MODEL:-llama-2-13b-chat-norwegian}
NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}
DISTRIBUTION_FLAG=${DISTRIBUTION_FLAG:-"--fsdp"}   # kept for interface compatibility
PRED_BATCH_SIZE=${PRED_BATCH_SIZE:-8}
EXAMPLES_SUFFIX=${EXAMPLES_SUFFIX:-1000-examples}
GEN=${GEN:-0}
MAJOR_INTERVAL=${MAJOR_INTERVAL:-500}
MODEL_FOLDER=${MODEL_FOLDER:-models/${MODEL}-apptainer-fsdp}
RESULTS_FOLDER=${RESULTS_FOLDER:-${HYGIENE_FOLDER:-$MODEL_FOLDER}}
MIN_NEW_TOKENS=${MIN_NEW_TOKENS:-40}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-180}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-2048}
NUM_BEAMS=${NUM_BEAMS:-1}
NUM_CANDIDATES=${NUM_CANDIDATES:-8}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.9}
TOP_K=${TOP_K:-50}
NO_REPEAT_NGRAM_SIZE=${NO_REPEAT_NGRAM_SIZE:-3}
ROUGELSUM_TOLERANCE=${ROUGELSUM_TOLERANCE:-0.95}
BERTSCORE_TOLERANCE=${BERTSCORE_TOLERANCE:-0.95}
FAITHFULNESS_TOLERANCE=${FAITHFULNESS_TOLERANCE:-0.95}
SEED=${SEED:-}
LEXICON=${LEXICON:-}
SPACY_MODEL=${SPACY_MODEL:-nb_core_news_md}
NO_LORA=${NO_LORA:-false}
OVERWRITE=${OVERWRITE:-false}
SKIP_EXISTING=${SKIP_EXISTING:-false}
DRY_RUN=${DRY_RUN:-false}

declare -a PRED_DATASETS=()

for arg in "$@"; do
    case $arg in
        --model=*)
            MODEL="${arg#*=}"
            if [ -z "${MODEL_FOLDER_SET:-}" ]; then
                MODEL_FOLDER="models/${MODEL}-apptainer-fsdp"
            fi
            if [ -z "${RESULTS_FOLDER_SET:-}" ] && [ -z "${HYGIENE_FOLDER:-}" ]; then
                RESULTS_FOLDER="$MODEL_FOLDER"
            fi
            ;;
        --num_gpus=*)
            NUM_GPUS="${arg#*=}"
            ;;
        --distribution_flag=*)
            DISTRIBUTION_FLAG="${arg#*=}"
            ;;
        --pred_dataset=*)
            PRED_DATASETS+=("${arg#*=}")
            ;;
        --pred_batch_size=*)
            PRED_BATCH_SIZE="${arg#*=}"
            ;;
        --examples_suffix=*)
            EXAMPLES_SUFFIX="${arg#*=}"
            ;;
        --generation=*|--gen=*)
            GEN="${arg#*=}"
            ;;
        --major_interval=*)
            MAJOR_INTERVAL="${arg#*=}"
            ;;
        --hygiene_folder=*)
            RESULTS_FOLDER="${arg#*=}"
            RESULTS_FOLDER_SET=1
            ;;
        --results_folder=*)
            RESULTS_FOLDER="${arg#*=}"
            RESULTS_FOLDER_SET=1
            ;;
        --model_folder=*)
            MODEL_FOLDER="${arg#*=}"
            MODEL_FOLDER_SET=1
            if [ -z "${RESULTS_FOLDER_SET:-}" ] && [ -z "${HYGIENE_FOLDER:-}" ]; then
                RESULTS_FOLDER="$MODEL_FOLDER"
            fi
            ;;
        --min_new_tokens=*)
            MIN_NEW_TOKENS="${arg#*=}"
            ;;
        --max_new_tokens=*)
            MAX_NEW_TOKENS="${arg#*=}"
            ;;
        --max_input_tokens=*)
            MAX_INPUT_TOKENS="${arg#*=}"
            ;;
        --num_beams=*)
            NUM_BEAMS="${arg#*=}"
            ;;
        --num_candidates=*)
            NUM_CANDIDATES="${arg#*=}"
            ;;
        --temperature=*)
            TEMPERATURE="${arg#*=}"
            ;;
        --top_p=*)
            TOP_P="${arg#*=}"
            ;;
        --top_k=*)
            TOP_K="${arg#*=}"
            ;;
        --no_repeat_ngram_size=*)
            NO_REPEAT_NGRAM_SIZE="${arg#*=}"
            ;;
        --rougeLsum_tolerance=*|--rougelsum_tolerance=*|--rogueLsum_tolerance=*)
            ROUGELSUM_TOLERANCE="${arg#*=}"
            ;;
        --bertscore_tolerance=*|--bertscore-tolerance=*|--bertscore_toleance=*)
            BERTSCORE_TOLERANCE="${arg#*=}"
            ;;
        --faithfulness_tolerance=*|--faithfulness-tolerance=*)
            FAITHFULNESS_TOLERANCE="${arg#*=}"
            ;;
        --seed=*)
            SEED="${arg#*=}"
            ;;
        --lexicon=*)
            LEXICON="${arg#*=}"
            ;;
        --spacy_model=*|--spacy-model=*)
            SPACY_MODEL="${arg#*=}"
            ;;
        --no_lora)
            NO_LORA=true
            ;;
        --overwrite)
            OVERWRITE=true
            ;;
        --skip-existing|--skip_existing)
            SKIP_EXISTING=true
            ;;
        --dry-run|--dry_run|--debug)
            DRY_RUN=true
            ;;
        *)
            echo "ERROR: Unknown argument: $arg"
            echo ""
            echo "Usage: sbatch --account=YOUR_PROJECT_ID run_regenerate_bad_hygiene_predictions.sbatch [OPTIONS]"
            echo ""
            echo "Input selection (choose one mode):"
            echo "  Mode A (scan folder):"
            echo "    --results_folder=DIR           Specific model results folder (must contain all_eval_results)"
            echo "  Mode B (explicit files):"
            echo "    --pred_dataset=PATH            Path to checkpoint-<N>-gen<M>-hygiene-bad-<suffix>.jsonl (repeatable)"
            echo ""
            echo "Optional options:"
            echo "  --model=MODEL                    Model short name filter (default: llama-2-13b-chat-norwegian)"
            echo "  --num_gpus=N                     Number of GPUs (default: 1)"
            echo "  --distribution_flag=FLAG         Kept for interface compatibility (--fsdp or --ddp)"
            echo "  --pred_batch_size=N              Regeneration batch size (default: 8)"
            echo "  --examples_suffix=S              Hygiene suffix filter (default: 1000-examples)"
            echo "  --gen=N                          Input hygiene generation to regenerate (default: 0; writes gen<N+1>)"
            echo "  --major_interval=N               Process only N divisible by interval (default: 500)"
            echo "  --results_folder=DIR             Results/model folder (default: models/\${MODEL}-apptainer-fsdp)"
            echo "  --hygiene_folder=DIR             Deprecated alias for --results_folder"
            echo "  --model_folder=DIR               Model folder (default: models/\${MODEL}-apptainer-fsdp)"
            echo "  --min_new_tokens=N               Min generated tokens (default: 40)"
            echo "  --max_new_tokens=N               Max generated tokens (default: 180)"
            echo "  --max_input_tokens=N             Prompt truncation length (default: 2048)"
            echo "  --num_beams=N                    Beam size, 1=greedy (default: 1)"
            echo "  --num_candidates=N               Sample N candidates per bad input (regeneration budget; default: 8)"
            echo "  --temperature=F                  Sampling temperature (default: 0.7)"
            echo "  --top_p=F                        Nucleus sampling probability (default: 0.9)"
            echo "  --top_k=N                        Top-k sampling cutoff, 0 disables (default: 50)"
            echo "  --no_repeat_ngram_size=N         No-repeat n-gram size, 0 disables (default: 3)"
            echo "  --rougeLsum_tolerance=F          Accept candidate only if ROUGE-Lsum >= F*original; 0 disables (default: 0.95)"
            echo "  --bertscore_tolerance=F          Accept candidate only if BERTScore F1 >= F*original; 0 disables (default: 0.95)"
            echo "  --faithfulness_tolerance=F       Accept candidate only if NLI entailment >= F*original; 0 disables (default: 0.95)"
            echo "  --seed=N                         Optional random seed for reproducible retries"
            echo "  --lexicon=PATH                   Optional external Norwegian lexicon"
            echo "  --spacy_model=MODEL              spaCy model for candidate scoring (default: nb_core_news_md)"
            echo "  --overwrite                      Overwrite existing output files"
            echo "  --skip-existing                  Skip targets whose output already exists (resume mode)"
            echo "  --no_lora                        Kept for interface compatibility (ignored by Python script)"
            echo "  --dry-run                        Validate and print command, do not execute"
            exit 1
            ;;
    esac
done

validate_model() {
    local model=$1
    case "$model" in
        gemma-2b|gemma-7b-it|gemma-2-9b|gemma-2-27b|gemma-3-12b|gemma-3-27b| \
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

validate_distribution_flag() {
    local flag=$1
    case "$flag" in
        --fsdp|--ddp|"")
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

ERRORS=0
if [ "$OVERWRITE" = "true" ] && [ "$SKIP_EXISTING" = "true" ]; then
    echo "ERROR: --overwrite and --skip-existing are mutually exclusive."
    ERRORS=$((ERRORS + 1))
fi
if ! validate_model "$MODEL"; then
    echo "ERROR: Invalid model name: $MODEL"
    ERRORS=$((ERRORS + 1))
fi
if ! validate_distribution_flag "$DISTRIBUTION_FLAG"; then
    echo "ERROR: Invalid distribution_flag: $DISTRIBUTION_FLAG (allowed: --fsdp, --ddp, empty)"
    ERRORS=$((ERRORS + 1))
fi
if [ ! -d "$RESULTS_FOLDER" ]; then
    echo "ERROR: results_folder does not exist: $RESULTS_FOLDER"
    ERRORS=$((ERRORS + 1))
fi
if [ ! -d "$RESULTS_FOLDER/all_eval_results" ]; then
    echo "ERROR: results_folder missing all_eval_results: $RESULTS_FOLDER/all_eval_results"
    ERRORS=$((ERRORS + 1))
fi
if [ ! -d "$MODEL_FOLDER" ]; then
    echo "ERROR: model_folder does not exist: $MODEL_FOLDER"
    ERRORS=$((ERRORS + 1))
fi
if ! [[ "$GEN" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --gen must be a non-negative integer: $GEN"
    ERRORS=$((ERRORS + 1))
fi
if [ -n "$LEXICON" ] && [ ! -f "$LEXICON" ]; then
    echo "ERROR: lexicon file does not exist: $LEXICON"
    ERRORS=$((ERRORS + 1))
fi
validate_tolerance() {
    local name=$1 val=$2
    if ! awk "BEGIN{exit !($val >= 0 && $val <= 1)}" </dev/null 2>/dev/null; then
        echo "ERROR: $name must be a number in [0, 1]: $val"
        ERRORS=$((ERRORS + 1))
    fi
}
validate_tolerance "--rougeLsum_tolerance" "$ROUGELSUM_TOLERANCE"
validate_tolerance "--bertscore_tolerance" "$BERTSCORE_TOLERANCE"
validate_tolerance "--faithfulness_tolerance" "$FAITHFULNESS_TOLERANCE"

if [ "${#PRED_DATASETS[@]}" -gt 0 ]; then
    for ds in "${PRED_DATASETS[@]}"; do
        if [ ! -f "$ds" ]; then
            echo "ERROR: pred_dataset does not exist: $ds"
            ERRORS=$((ERRORS + 1))
            continue
        fi
        base=$(basename "$ds")
        if ! [[ "$base" =~ checkpoint-[0-9]+-gen[0-9]+-hygiene-bad-[0-9]+-examples\.jsonl$ ]]; then
            echo "ERROR: pred_dataset must match [<prefix>]checkpoint-<N>-gen<M>-hygiene-bad-<suffix>.jsonl: $base"
            ERRORS=$((ERRORS + 1))
        elif ! [[ "$base" =~ checkpoint-[0-9]+-gen${GEN}-hygiene-bad-[0-9]+-examples\.jsonl$ ]]; then
            echo "ERROR: pred_dataset generation does not match --gen=$GEN: $base"
            ERRORS=$((ERRORS + 1))
        fi
    done
fi

if [ $ERRORS -gt 0 ]; then
    echo "Validation failed with $ERRORS error(s)."
    exit 1
fi

echo "Configuration:"
echo "  MODEL: $MODEL"
echo "  NUM_GPUS: $NUM_GPUS"
echo "  DISTRIBUTION_FLAG: $DISTRIBUTION_FLAG (compatibility only)"
echo "  PRED_BATCH_SIZE: $PRED_BATCH_SIZE"
echo "  EXAMPLES_SUFFIX: $EXAMPLES_SUFFIX"
echo "  GEN: $GEN (input hygiene generation; output gen$((GEN + 1)))"
echo "  MAJOR_INTERVAL: $MAJOR_INTERVAL"
echo "  RESULTS_FOLDER: $RESULTS_FOLDER"
echo "  MODEL_FOLDER: $MODEL_FOLDER"
echo "  MIN_NEW_TOKENS: $MIN_NEW_TOKENS"
echo "  MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
echo "  MAX_INPUT_TOKENS: $MAX_INPUT_TOKENS"
echo "  NUM_BEAMS: $NUM_BEAMS"
echo "  NUM_CANDIDATES: $NUM_CANDIDATES"
echo "  TEMPERATURE: $TEMPERATURE"
echo "  TOP_P: $TOP_P"
echo "  TOP_K: $TOP_K"
echo "  NO_REPEAT_NGRAM_SIZE: $NO_REPEAT_NGRAM_SIZE"
echo "  ROUGELSUM_TOLERANCE: $ROUGELSUM_TOLERANCE (0 disables ROUGE gate)"
echo "  BERTSCORE_TOLERANCE: $BERTSCORE_TOLERANCE (0 disables BERTScore gate)"
echo "  FAITHFULNESS_TOLERANCE: $FAITHFULNESS_TOLERANCE (0 disables NLI faithfulness gate)"
echo "  SEED: ${SEED:-"(not set)"}"
echo "  SPACY_MODEL: $SPACY_MODEL"
echo "  LEXICON: ${LEXICON:-"(none)"}"
echo "  NO_LORA: $NO_LORA"
echo "  OVERWRITE: $OVERWRITE"
echo "  SKIP_EXISTING: $SKIP_EXISTING"
echo "  DRY_RUN: $DRY_RUN"
if [ "${#PRED_DATASETS[@]}" -gt 0 ]; then
    echo "  INPUT_MODE: explicit --pred_dataset list"
    echo "  PRED_DATASETS:"
    for ds in "${PRED_DATASETS[@]}"; do echo "    - $ds"; done
else
    echo "  INPUT_MODE: scan --results_folder/all_eval_results"
fi

if [ "$DRY_RUN" = "true" ]; then
    echo ""
    echo "Dry-run complete. No execution performed."
    exit 0
fi

if [ -z "${HUGGINGFACE_TOKEN:-}" ]; then
    echo "ERROR: HUGGINGFACE_TOKEN is required (set in .env or environment)."
    exit 1
fi

# ********** JOB-LOCAL CACHE **********
source scripts/olivia_job_local_cache.sh
# ********** END JOB-LOCAL CACHE **********

module --force purge
module load BuildEnv/NeoverseV2
module load NRIS/GPU
module load CUDA/12.6.0
module load PyTorch/2.8.0

export IMAGE=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif

PRED_FLAGS=""
for ds in "${PRED_DATASETS[@]}"; do
    PRED_FLAGS="$PRED_FLAGS --pred_dataset \"$ds\""
done
OVERWRITE_FLAG=""
if [ "$OVERWRITE" = "true" ]; then
    OVERWRITE_FLAG="--overwrite"
elif [ "$SKIP_EXISTING" = "true" ]; then
    OVERWRITE_FLAG="--skip-existing"
fi
NO_LORA_FLAG=""
if [ "$NO_LORA" = "true" ]; then
    NO_LORA_FLAG="--no_lora"
fi
SEED_FLAG=""
if [ -n "$SEED" ]; then
    SEED_FLAG="--seed \"$SEED\""
fi
LEXICON_FLAG=""
if [ -n "$LEXICON" ]; then
    LEXICON_FLAG="--lexicon \"$LEXICON\""
fi

PY_CMD="export HF_HOME=$HF_HOME && export HF_DATASETS_CACHE=$HF_DATASETS_CACHE && export HF_MODULES_CACHE=$HF_MODULES_CACHE && export HF_METRICS_CACHE=$HF_METRICS_CACHE && export TOKENIZERS_PARALLELISM=false && python scripts/regenerate_bad_hygiene_predictions.py --model \"$MODEL\" --results_folder \"$RESULTS_FOLDER\" --model_folder \"$MODEL_FOLDER\" --examples_suffix \"$EXAMPLES_SUFFIX\" --gen \"$GEN\" --major_interval \"$MAJOR_INTERVAL\" --pred_batch_size \"$PRED_BATCH_SIZE\" --min_new_tokens \"$MIN_NEW_TOKENS\" --max_new_tokens \"$MAX_NEW_TOKENS\" --max_input_tokens \"$MAX_INPUT_TOKENS\" --num_beams \"$NUM_BEAMS\" --num_candidates \"$NUM_CANDIDATES\" --temperature \"$TEMPERATURE\" --top_p \"$TOP_P\" --top_k \"$TOP_K\" --no_repeat_ngram_size \"$NO_REPEAT_NGRAM_SIZE\" --rougeLsum_tolerance \"$ROUGELSUM_TOLERANCE\" --bertscore_tolerance \"$BERTSCORE_TOLERANCE\" --faithfulness_tolerance \"$FAITHFULNESS_TOLERANCE\" --spacy_model \"$SPACY_MODEL\" --hf_token \"$HUGGINGFACE_TOKEN\" $SEED_FLAG $LEXICON_FLAG $OVERWRITE_FLAG $NO_LORA_FLAG $PRED_FLAGS"

# Keep install stack similar to evaluation script for predictable runtime env.
COMMAND="pip install --upgrade pip setuptools wheel > /dev/null 2>&1 && pip install --break-system-packages --only-binary=:all: transformers==4.45.2 tokenizers==0.20.3 datasets==3.0.0 sentencepiece==0.2.1 evaluate spacy > /dev/null 2>&1 && python -m spacy download \"$SPACY_MODEL\" > /dev/null 2>&1 && pip install --break-system-packages --no-binary=rouge-score rouge-score==0.1.2 > /dev/null 2>&1 && pip install --break-system-packages --no-deps accelerate==1.0.1 peft==0.13.2 > /dev/null 2>&1 && pip install --break-system-packages --upgrade pyyaml psutil typing-extensions packaging==24.2 safetensors > /dev/null 2>&1 && pip install --break-system-packages bert_score sentence-transformers > /dev/null 2>&1 && $PY_CMD"

echo "Starting regeneration job..."
apptainer exec --nv "$IMAGE" bash -lc "$COMMAND"
echo "Regeneration completed."
