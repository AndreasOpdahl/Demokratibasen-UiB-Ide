import json
import random
import pandas as pd
from datasets import Dataset
from pympler import asizeof
from transformers import AutoTokenizer


# --- CONFIG ---

TRAIN_FILE = "../../datasets_from_demokratibasen/datasets/dataset_43221_examples/processed_data_train.jsonl"  # replace with actual path
VAL_FILE = "../../datasets_from_demokratibasen/datasets/dataset_43221_examples/processed_data_val.jsonl"
MODEL_NAME = "google/gemma-2b"
VALIDATION_SIZE = 100
MAX_INPUT_TEXT_TOKENS = 2048
MAX_EXTRA_PROMPT_TOKENS = 256
MAX_OUTPUT_SUMMARY_TOKENS = 256


# --- LOAD TOKENIZER ---

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

max_input_prompt_tokens = MAX_INPUT_TEXT_TOKENS + MAX_EXTRA_PROMPT_TOKENS


# --- LOAD DATA ---

# Read training JSONL file
train_data = []
try:
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        for json_line in f:
            json_dict = json.loads(json_line)
            train_data.append(json_dict)
except Exception as e:
    print(f"Error reading train dataset: {e}")
    exit(1)

# Read validation JSONL file
val_data = []
try:
    with open(VAL_FILE, 'r', encoding='utf-8') as f:
        for json_line in f:
            json_dict = json.loads(json_line)
            val_data.append(json_dict)
except Exception as e:
    print(f"Error reading validation dataset: {e}")
    exit(1)


# --- TRAIN DATA ---

train_df = pd.DataFrame(train_data)
train_df = train_df[train_df['output'].notna()]

assert train_df['input'].apply(lambda x: x is not None and x != '').all()
assert train_df['input'].notna().all()
assert train_df['output'].apply(lambda x: x is not None and x != '').all()
assert train_df['output'].notna().all()

train_dataset = Dataset.from_pandas(train_df)
print(f"*** training dataset size: {len(train_dataset)} examples ***")


# --- SAMPLE VALIDATION DATA ---

val_df = pd.DataFrame(val_data)
val_df = val_df[val_df['output'].notna()]
val_df = val_df.sample(n=VALIDATION_SIZE)

assert val_df['input'].apply(lambda x: x is not None and x != '').all()
assert val_df['input'].notna().all()
assert val_df['output'].apply(lambda x: x is not None and x != '').all()
assert val_df['output'].notna().all()

val_dataset = Dataset.from_pandas(val_df)
print(f"*** validation dataset size: {len(val_dataset)} examples ***")


# --- FORMAT & TOKENIZE TRAINING DATA ---

def format_example_train(example):
    # Format the input-output pair for the model (TRAINING: full text for teacher forcing)
    text = f"Oppgave: Oppsummer følgende tekst:\n\n###\n\n{example['input']}\n\n###\n\nOppsummering:\n\n###\n\n{example['output']}\n\n###\n"
    return {"text": text}

def tokenize_function_train(examples):

    max_input_text_tokens = 2048  # max tokens for input to summarisation
    max_extra_prompt_tokens = 40  # max extra tokens for input prompt (the task description)
    max_input_prompt_tokens = max_input_text_tokens + max_extra_prompt_tokens
    max_output_summary_tokens = 512  # max tokens for output from summarisation

    # Tokenize the formatted text for training
    max_input_prompt_tokens = max_input_text_tokens + max_extra_prompt_tokens
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_input_prompt_tokens + max_output_summary_tokens,
        padding=True
    )

formatted_train_dataset = train_dataset.map(format_example_train)
tokenized_train_dataset = formatted_train_dataset.map(tokenize_function_train, batched=True)

size = asizeof.asizeof(tokenized_train_dataset)


# --- FORMAT & TOKENIZE VALIDATION DATA ---

def format_example_eval(example):
    prompt = f"Oppgave: Oppsummer følgende tekst:\n\n###\n\n{example['input']}\n\n###\n\nOppsummering:\n\n###\n\n"
    return {
        "prompt": prompt,
        "target_summary": example["output"]
    }

def tokenize_function_eval(examples):
    tokenized_prompts = tokenizer(
        examples["prompt"],
        truncation=True,
        max_length=max_input_prompt_tokens,
        padding=False,
    )
    tokenized_targets = tokenizer(
        examples["target_summary"],
        truncation=True,
        max_length=MAX_OUTPUT_SUMMARY_TOKENS,
        padding=False,
    )
    tokenized_prompts["labels"] = tokenized_targets["input_ids"]
    return tokenized_prompts

formatted_val_dataset = val_dataset.map(format_example_eval)
tokenized_val_dataset = formatted_val_dataset.map(tokenize_function_eval, batched=True)
