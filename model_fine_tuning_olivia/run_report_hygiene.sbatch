#!/usr/bin/env bash
#SBATCH -J report-hygiene
#SBATCH -p accel
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
# Account: REQUIRED - Pass via: sbatch --account=YOUR_ACCOUNT run_report_hygiene.sbatch ...
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
FOLDER=""
EXAMPLES_SUFFIX=${EXAMPLES_SUFFIX:-1000-examples}
GEN=${GEN:-0}
OVERWRITE=${OVERWRITE:-false}
SKIP_EXISTING=${SKIP_EXISTING:-false}
DRY_RUN=${DRY_RUN:-false}
MAJOR_ONLY=${MAJOR_ONLY:-false}
LEXICON=${LEXICON:-""}
SPACY_MODEL=${SPACY_MODEL:-nb_core_news_md}

declare -a INPUT_FILES=()
declare -a THRESHOLD_ARGS=()

for arg in "$@"; do
    case $arg in
        --folder=*)
            FOLDER="${arg#*=}"
            ;;
        --input_file=*)
            INPUT_FILES+=("${arg#*=}")
            ;;
        --examples_suffix=*)
            EXAMPLES_SUFFIX="${arg#*=}"
            ;;
        --gen=*)
            GEN="${arg#*=}"
            ;;
        --lexicon=*)
            LEXICON="${arg#*=}"
            ;;
        --spacy_model=*|--spacy-model=*)
            SPACY_MODEL="${arg#*=}"
            ;;
        --overwrite)
            OVERWRITE=true
            ;;
        --skip-existing|--skip_existing)
            SKIP_EXISTING=true
            ;;
        --dry-run|--dry_run)
            DRY_RUN=true
            ;;
        --major-only|--major_only)
            MAJOR_ONLY=true
            ;;
        --max_rep_3gram=*|--min_compression_ratio=*|--min_pred_chars=*| \
        --max_pred_ref_char_ratio=*|--max_markup_ratio=*| \
        --min_punctuation_score=*|--min_complete_sentence_ratio=*| \
        --min_known_word_ratio=*)
            THRESHOLD_ARGS+=("$arg")
            ;;
        *)
            echo "ERROR: Unknown argument: $arg"
            echo ""
            echo "Usage: sbatch --account=YOUR_PROJECT_ID run_report_hygiene.sbatch [OPTIONS]"
            echo ""
            echo "Input selection (choose one or both):"
            echo "  --folder=DIR                     Folder to scan for *-gen<N>-inputs-refs-preds-<suffix>.jsonl"
            echo "  --input_file=PATH                Explicit input file (repeatable)"
            echo ""
            echo "Options:"
            echo "  --examples_suffix=S              Suffix filter for folder scan (default: 1000-examples)"
            echo "  --gen=N                          Generation number to process (default: 0)"
            echo "  --lexicon=PATH                   External Norwegian lexicon file (one word per line)"
            echo "  --spacy_model=MODEL              spaCy model name (default: nb_core_news_md)"
            echo "  --overwrite                      Allow overwriting existing hygiene output files"
            echo "  --skip-existing                  Skip files whose outputs already exist (resume mode)"
            echo "  --major-only                     Only process major checkpoints (step divisible by 500)"
            echo "  --dry-run                        Show matched files and exit without processing"
            echo ""
            echo "Hygiene thresholds (passed through to report_hygiene.py):"
            echo "  --max_rep_3gram=F                (default: 0.2)"
            echo "  --min_compression_ratio=F        (default: 1.0)"
            echo "  --min_pred_chars=N               (default: 50)"
            echo "  --max_pred_ref_char_ratio=F      (default: 1.5)"
            echo "  --max_markup_ratio=F             (default: 0.01)"
            echo "  --min_punctuation_score=F        (default: -0.1)"
            echo "  --min_complete_sentence_ratio=F  (default: 0.6)"
            echo "  --min_known_word_ratio=F         (default: 0.9)"
            exit 1
            ;;
    esac
done

# ===== VALIDATION =====
ERRORS=0

if [ "$OVERWRITE" = "true" ] && [ "$SKIP_EXISTING" = "true" ]; then
    echo "ERROR: --overwrite and --skip-existing are mutually exclusive."
    ERRORS=$((ERRORS + 1))
fi
if [ -z "$FOLDER" ] && [ "${#INPUT_FILES[@]}" -eq 0 ]; then
    echo "ERROR: At least one of --folder or --input_file is required."
    ERRORS=$((ERRORS + 1))
fi

if [ -n "$FOLDER" ] && [ ! -d "$FOLDER" ]; then
    echo "ERROR: folder does not exist: $FOLDER"
    ERRORS=$((ERRORS + 1))
fi

if ! [[ "$GEN" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --gen must be a non-negative integer (got: $GEN)"
    ERRORS=$((ERRORS + 1))
fi

for f in "${INPUT_FILES[@]+"${INPUT_FILES[@]}"}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: input_file does not exist: $f"
        ERRORS=$((ERRORS + 1))
    fi
done

if [ -n "$LEXICON" ] && [ ! -f "$LEXICON" ]; then
    echo "ERROR: lexicon file does not exist: $LEXICON"
    ERRORS=$((ERRORS + 1))
fi

if [ $ERRORS -gt 0 ]; then
    echo "Validation failed with $ERRORS error(s)."
    exit 1
fi

# ===== COLLECT INPUT FILES =====
declare -a FILES=()

if [ -n "$FOLDER" ]; then
    declare -a PATTERNS=()
    if [ "$GEN" -eq 0 ]; then
        # Checkpoint files must include an explicit generation marker.
        PATTERNS+=(
            "$FOLDER"/*-gen0-inputs-refs-preds-"${EXAMPLES_SUFFIX}".jsonl
            "$FOLDER"/*-gen0_inputs_refs_preds-"${EXAMPLES_SUFFIX}".jsonl
        )
    else
        PATTERNS+=(
            "$FOLDER"/*-gen"${GEN}"-inputs-refs-preds-"${EXAMPLES_SUFFIX}".jsonl
            "$FOLDER"/*-gen"${GEN}"_inputs_refs_preds-"${EXAMPLES_SUFFIX}".jsonl
        )
    fi
    for pattern in "${PATTERNS[@]}"; do
        for f in $pattern; do
            base="$(basename "$f")"
            if ! [[ "$base" =~ checkpoint-[0-9]+-gen${GEN}[-_]inputs[-_]refs[-_]preds[-_]${EXAMPLES_SUFFIX}\.jsonl$ ]]; then
                continue
            fi
            [ -f "$f" ] && FILES+=("$f")
        done
    done
fi

for f in "${INPUT_FILES[@]+"${INPUT_FILES[@]}"}"; do
    FILES+=("$f")
done

if [ "${#FILES[@]}" -gt 0 ]; then
    declare -A SEEN_FILES=()
    declare -a UNIQUE_FILES=()
    for f in "${FILES[@]}"; do
        if [ -z "${SEEN_FILES[$f]+x}" ]; then
            SEEN_FILES["$f"]=1
            UNIQUE_FILES+=("$f")
        fi
    done
    FILES=("${UNIQUE_FILES[@]}")
fi

if [ "$MAJOR_ONLY" = "true" ] && [ "${#FILES[@]}" -gt 0 ]; then
    declare -a MAJOR_FILES=()
    for f in "${FILES[@]}"; do
        basename_f="$(basename "$f")"
        step=$(echo "$basename_f" | sed -E 's/^.*checkpoint-([0-9]+).*/\1/')
        if [[ "$step" =~ ^[0-9]+$ ]] && [ $(( step % 500 )) -eq 0 ]; then
            MAJOR_FILES+=("$f")
        fi
    done
    FILES=("${MAJOR_FILES[@]+"${MAJOR_FILES[@]}"}")
fi

if [ "${#FILES[@]}" -gt 0 ]; then
    mapfile -t FILES < <(printf '%s\n' "${FILES[@]}" | sort -V)
fi

if [ "${#FILES[@]}" -eq 0 ]; then
    echo "No matching input files found."
    exit 0
fi

echo "Configuration:"
echo "  FOLDER: ${FOLDER:-"(not set)"}"
echo "  EXAMPLES_SUFFIX: $EXAMPLES_SUFFIX"
echo "  GEN: $GEN"
echo "  SPACY_MODEL: $SPACY_MODEL"
echo "  LEXICON: ${LEXICON:-"(none)"}"
echo "  OVERWRITE: $OVERWRITE"
echo "  SKIP_EXISTING: $SKIP_EXISTING"
echo "  MAJOR_ONLY: $MAJOR_ONLY"
echo "  DRY_RUN: $DRY_RUN"
if [ "${#THRESHOLD_ARGS[@]}" -gt 0 ]; then
    echo "  THRESHOLD_ARGS: ${THRESHOLD_ARGS[*]}"
fi
echo ""
echo "Input files (${#FILES[@]}):"
for f in "${FILES[@]}"; do
    echo "  - $f"
done

if [ "$DRY_RUN" = "true" ]; then
    echo ""
    echo "Dry-run complete. No execution performed."
    exit 0
fi

# ===== ENVIRONMENT SETUP =====
export SLURM_TMPDIR=/tmp/${USER}_${SLURM_JOB_ID:-$$}
mkdir -p "$SLURM_TMPDIR"
export PYTHONUSERBASE="$SLURM_TMPDIR/.local"
mkdir -p "$PYTHONUSERBASE"

module --force purge
module load BuildEnv/NeoverseV2
module load NRIS/GPU
module load CUDA/12.6.0
module load PyTorch/2.8.0

export IMAGE=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif

# ===== BUILD INNER SCRIPT =====
# A single apptainer invocation runs pip setup once, then loops over all files.
INNER_SCRIPT="$SLURM_TMPDIR/run_hygiene_inner.sh"
{
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    echo "export PYTHONUSERBASE=$PYTHONUSERBASE"
    echo 'pip install --break-system-packages evaluate transformers spacy > /dev/null 2>&1'
    echo "python -m spacy download $SPACY_MODEL > /dev/null 2>&1"
    echo ''
    echo 'PROCESSED=0'
    echo 'FAILED=0'

    # Common flags shared by every report_hygiene.py call.
    COMMON_FLAGS="--gen $GEN --spacy-model $SPACY_MODEL"
    if [ -n "$LEXICON" ]; then
        COMMON_FLAGS="$COMMON_FLAGS --lexicon $LEXICON"
    fi
    if [ "$OVERWRITE" = "true" ]; then
        COMMON_FLAGS="$COMMON_FLAGS --overwrite"
    elif [ "$SKIP_EXISTING" = "true" ]; then
        COMMON_FLAGS="$COMMON_FLAGS --skip-existing"
    fi
    for ta in "${THRESHOLD_ARGS[@]+"${THRESHOLD_ARGS[@]}"}"; do
        COMMON_FLAGS="$COMMON_FLAGS $ta"
    done

    for input_file in "${FILES[@]}"; do
        echo ''
        echo "echo 'Processing: $input_file'"
        echo 'tmp_log=$(mktemp)'
        echo "if python scripts/report_hygiene.py '$input_file' $COMMON_FLAGS >\"\$tmp_log\" 2>&1; then"
        echo '    cat "$tmp_log"'
        echo '    rm -f "$tmp_log"'
        echo '    PROCESSED=$((PROCESSED + 1))'
        echo 'else'
        echo '    status=$?'
        echo '    cat "$tmp_log" >&2'
        echo '    if grep -q "output files already exist" "$tmp_log"; then'
        echo "        echo 'FAILED because hygiene files exist already (use --overwrite to replace): $input_file'"
        echo '    else'
        echo "        echo 'FAILED (exit code '\$status'): $input_file; see this job'\''s .err log for the detailed Python error.'"
        echo '    fi'
        echo '    rm -f "$tmp_log"'
        echo '    FAILED=$((FAILED + 1))'
        echo 'fi'
    done

    echo ''
    echo 'echo ""'
    echo 'echo "Hygiene reporting complete. Processed: $PROCESSED, Failed: $FAILED"'
    echo 'if [ $FAILED -gt 0 ]; then exit 1; fi'
} > "$INNER_SCRIPT"
chmod +x "$INNER_SCRIPT"

echo ""
echo "Starting hygiene reporting (${#FILES[@]} files)..."
apptainer exec "$IMAGE" bash -l "$INNER_SCRIPT"
echo "All done."
