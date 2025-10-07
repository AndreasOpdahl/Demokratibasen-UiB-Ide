#!/usr/bin/bash

# Set your Hugging Face token as an environment variable
source .env

# Run the fine-tuning script
python scripts/finetune.py --model gemma-7b --hf_token "$HUGGINGFACE_TOKEN"
python scripts/finetune.py --model gemma-7b --hf_token "$HUGGINGFACE_TOKEN" --train_dataset data/output/processed_data_train.jsonl --val_dataset data/output/processed_data_val.jsonl

