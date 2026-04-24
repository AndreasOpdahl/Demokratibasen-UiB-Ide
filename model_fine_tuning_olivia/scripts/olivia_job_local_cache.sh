#!/usr/bin/env bash
# ********** JOB-LOCAL CACHE **********
#
# Shared cache setup sourced by all SLURM sbatch scripts.
#
# Prerequisites (must be set by the caller before sourcing):
#   $MODEL            — short model name (e.g. "llama-3.1-8b-instruct")
#   $SLURM_JOB_ID     — standard SLURM variable
#   $SLURM_PROJECT_ID — from sbatch --account=
#   $USER             — standard env variable
#
# Exports (available to the caller after sourcing):
#   SLURM_TMPDIR  HF_HOME  HF_DATASETS_CACHE
#   HF_MODULES_CACHE  HF_METRICS_CACHE  PYTHONUSERBASE

export SLURM_TMPDIR=/tmp/${USER}_$SLURM_JOB_ID
mkdir -p "$SLURM_TMPDIR"

export HF_HOME="$SLURM_TMPDIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_METRICS_CACHE="$HF_HOME/metrics"
export PYTHONUSERBASE="$SLURM_TMPDIR/.local"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$HF_METRICS_CACHE" "$PYTHONUSERBASE"

_PROJECT_HF_DIR=/cluster/projects/${SLURM_PROJECT_ID}/shared/.cache/huggingface
mkdir -p "$HF_HOME/hub"

_convert_model_to_cache_dir() {
    local _model=$1
    case "$_model" in
        gemma-2b) echo "models--google--gemma-2b" ;;
        gemma-7b) echo "models--google--gemma-7b" ;;
        gemma-7b-it) echo "models--google--gemma-7b-it" ;;
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
        eurollm-9b-instruct) echo "models--utter-project--EuroLLM-9B-Instruct-2512" ;;
        norwai-mistral-7b-instruct) echo "models--NorwAI--NorwAI-Mistral-7B-instruct" ;;
        nb-gpt-j-6b) echo "models--NbAiLab--nb-gpt-j-6B-torgersen-alpaca" ;;
        *) echo "models--${_model//\//--}" ;;
    esac
}

_MODEL_CACHE_DIR=$(_convert_model_to_cache_dir "$MODEL")
_SOURCE_MODEL_DIR="$_PROJECT_HF_DIR/hub/$_MODEL_CACHE_DIR"

if [ -d "$_SOURCE_MODEL_DIR" ]; then
    echo "Copying model $MODEL to job-local cache..."
    rsync -aq "$_SOURCE_MODEL_DIR/" "$HF_HOME/hub/$_MODEL_CACHE_DIR/"
    echo "Model cache copy complete"
else
    echo "WARNING: Model $MODEL not found in shared cache at $_SOURCE_MODEL_DIR"
    echo "Model will be downloaded at runtime (requires internet access)"
fi

[ -d "$_PROJECT_HF_DIR/metrics" ] && rsync -aq "$_PROJECT_HF_DIR/metrics/" "$HF_METRICS_CACHE/"
[ -d "$_PROJECT_HF_DIR/modules" ] && rsync -aq "$_PROJECT_HF_DIR/modules/" "$HF_MODULES_CACHE/"

echo "Cache staging complete"

# ********** END JOB-LOCAL CACHE **********
