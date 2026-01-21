import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, MT5Tokenizer
from typing import Dict, List, Any

def load_jsonl_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSONL file"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading dataset {file_path}: {e}")
        raise
    return data

def load_and_preprocess_datasets(train_path: str, val_path: str, test_path: str = None):
    """Load and convert datasets to Hugging Face format"""
    # Load training data
    train_data = load_jsonl_dataset(train_path)
    train_df = pd.DataFrame(train_data)
    train_dataset = Dataset.from_pandas(train_df)
    
    # Load validation data
    val_data = load_jsonl_dataset(val_path)
    val_df = pd.DataFrame(val_data)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Load test data if provided
    test_dataset = None
    if test_path:
        test_data = load_jsonl_dataset(test_path)
        test_df = pd.DataFrame(test_data)
        test_dataset = Dataset.from_pandas(test_df)
    
    return train_dataset, val_dataset, test_dataset

def format_train_example(example: Dict[str, Any]) -> Dict[str, str]:
    """Format the input-output pair for training (with answer for teacher forcing)"""
    text = f"### Oppgave: Oppsummer følgende tekst\n{example['input']}\n\n### Svar: {example['output']}"
    return {"text": text}

def format_eval_example(example: Dict[str, Any]) -> Dict[str, str]:
    """Format for evaluation/generation (without answer but WITH prompt)"""
    input_text = f"### Oppgave: Oppsummer følgende tekst\n{example['input']}\n\n### Svar:"
    return {
        "input_text": input_text,  # This includes the prompt
        "input": example['input'],  # Keep original for reference
        "output": example['output']
    }

def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int, format_fn, batched: bool = True):
    """Tokenize dataset with the given formatting function"""
    formatted_dataset = dataset.map(format_fn)
    tokenized_dataset = formatted_dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False
        ),
        batched=batched
    )
    return formatted_dataset, tokenized_dataset

def get_tokenizer(model_name: str, hf_token: str = None):
    """Load appropriate tokenizer for the model"""
    try:
        if model_name == 'google/mt5-base':
            tokenizer = MT5Tokenizer.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token if hf_token else None
            )
        
        # Set padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise