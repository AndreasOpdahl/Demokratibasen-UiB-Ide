"""
Before runnning:
export PPYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from datasets import Dataset
import evaluate
from huggingface_hub import login
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction


VAL_DATA_SIZE=100  # number of examples to use for validation
MAX_TOKENS=2048  # maximum number of tokens to use for input and output


class CausalLMTrainer(Trainer):
    def __init__(self, *args, 
                 generation_max_length: Optional[int] = None,
                 generation_num_beams: Optional[int] = None,
                 **kwargs) -> None:
        # 1. Store generation parameters
        self.generation_max_length = generation_max_length
        self.generation_num_beams = generation_num_beams
        # 2. Call parent constructor
        super().__init__(*args, **kwargs)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        # 1. Compute Loss (Standard behavior)
        if prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        # If we are here, we are evaluating the model
        print('*** evaluation: prediction_step ***')
        torch.cuda.empty_cache()

        # 2. Get Input IDs and Labels
        # We need the input_ids to use as the prompt for generation
        if 'input_ids' in inputs:
            input_ids = inputs["input_ids"]
        else:
            # Handle cases where DataCollator might rename input_ids
            # This logic needs to be robust based on your data collator
            raise KeyError("input_ids not found in batch. Check your DataCollator setup.")

        # Get the labels for metric calculation (target summary)
        labels = inputs.get("labels")

        print('*** evaluation: input_ids ***', input_ids.shape)

        # 3. Autoregressive Generation (The Key Step)
        # Call generate() using the model and the input IDs
        # Pass any necessary generation arguments (e.g., num_beams, max_new_tokens, etc.)
        generated_ids = model.generate(
            input_ids=input_ids,
            use_cache=True,
            # Use generation settings defined in TrainingArguments or passed here
            max_new_tokens=self.generation_max_length or model.config.max_length, 
            num_beams=self.generation_num_beams,  # was: 1 or model.config.num_beams,
            do_sample=False,  # Set to True for sampling, False for beam search
            # Add any other required generation arguments
        )
        
        # CRITICAL FIX: Remove the input prompt from generated_ids
        # model.generate() returns [input_ids + new_tokens], we only want the new tokens for ROUGE
        input_length = input_ids.shape[1]
        generated_ids = generated_ids[:, input_length:]
        
        print('*** evaluation: generated_ids (without prompt) ***', generated_ids.shape)
        
        # 4. Handle Loss Calculation (Optional, but good practice)
        # You may want to calculate loss for logging alongside the ROUGE score
        # For evaluation, we typically don't need the logits, just the loss and generated IDs.
        
        torch.cuda.empty_cache()

        calc_loss_on_evaluation = False
        if calc_loss_on_evaluation:
            with torch.no_grad():
                # Process in chunks of 2 examples at a time
                losses = []
                chunk_size = 2
                for i in range(0, input_ids.shape[0], chunk_size):
                    chunk_inputs = {k: v[i:i+chunk_size] for k, v in inputs.items()}
                    chunk_loss = model(**chunk_inputs).loss
                    losses.append(chunk_loss.item())
                loss = sum(losses) / len(losses)
            torch.cuda.empty_cache()
        else:        
            # alternative if little memory is available
            loss = None
        
        # 5. Return Results
        # The Trainer expects: (loss, predictions, labels)
        # We return loss, the generated IDs (predictions), and the original labels.
        return (loss, generated_ids, labels)

# only do this once
rouge = evaluate.load("rouge")
          

def fine_tune_model(
    model_name: str,
    dataset_path: str,
    val_dataset_path: str,
    output_dir: str,
    hf_token: str = None
):
    
    def compute_metrics(eval_pred):
        print('*** evaluation: compute_metrics ***')
        
        preds, labels = eval_pred  # preds: generated ids; labels: target ids (with -100 masked)
        print('*** evaluation: preds ***', preds.shape)
        print('*** evaluation: labels ***', labels.shape)
        
        # Replace -100 so we can decode labels
        labels = np.where(labels != -100, labels, 0)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # (optional) strip/normalize for ROUGE
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        scores = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        print('*** evaluation: computed_metrics ***', scores)
        return {k: v * 100 for k, v in scores.items()}  # % values


    
    # Login to Hugging Face if token is provided
    if hf_token:
        print("Logging in to Hugging Face Hub...")
        login(token=hf_token)

    # Load tokenizer
    try:
        if model_name == 'google/mt5-base':
            tokenizer = MT5Tokenizer.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token if hf_token else None
            )
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Make sure you have the correct model name and authentication if needed.")
        return

    # Set padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with device_map if supported, otherwise fallback
    try:
        if model_name == 'google/mt5-base':
            model = MT5ForConditionalGeneration.from_pretrained(model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                # load_in_4bit=True,
                torch_dtype=torch.float16,
                device_map="auto",
                token=hf_token if hf_token else None
            )
    except Exception as e:
        print(f"Error loading model with device_map: {e}")
        try:
            # Fallback if device_map isn't supported
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                token=hf_token if hf_token else None
            ).cuda()
        except Exception as e2:
            print(f"Error loading model: {e2}")
            return

    # Load and preprocess dataset
    print(f"Loading dataset from: {dataset_path}")

    # Read JSONL file manually
    data = []
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return

    # Read JSONL file manually
    val_data = []
    try:
        with open(val_dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                val_data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return

    # Create a pandas DataFrame
    df = pd.DataFrame(data)

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    # randomly sample VAL_DATA_SIZE examples from val_data
    val_data = random.sample(val_data, VAL_DATA_SIZE)

    # Create a pandas DataFrame
    df = pd.DataFrame(val_data)

    # Convert to Hugging Face Dataset
    val_dataset = Dataset.from_pandas(df)
    print('*** evaluation: val_dataset ***', val_dataset.shape)

    def format_example(example):
        # Format the input-output pair for the model
        text = f"### Oppgave: Oppsummer følgende tekst\n{example['input']}\n\n### Svar: {example['output']}"
        return {"text": text}

    def tokenize_function(examples):
        # Tokenize the formatted text
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_TOKENS,
            padding=False
        )

    formatted_dataset = dataset.map(format_example)
    tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)

    # Format and tokenize the dataset
    formatted_val_dataset = val_dataset.map(format_example)
    tokenized_val_dataset = formatted_val_dataset.map(tokenize_function, batched=True)
    # Data collator for dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # cpu_eval_callback =  CPUEvaluationCallback(eval_dataset_path=val_dataset_path, tokenizer=tokenizer, num_samples=5, eval_steps=500, patience=5)
    
    # Manually prepare model for LoRA training (without quantization)
    # This replicates the key parts of prepare_model_for_kbit_training() without the quantization-specific code
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Enable gradients for input embeddings (critical for LoRA!)
    # Without this, the embeddings stay frozen and cause "does not require grad" errors
    for param in model.parameters():
        param.requires_grad = False  # Freeze all parameters first
    
    # Enable input embeddings to require gradients
    if hasattr(model, 'get_input_embeddings'):
        input_embeddings = model.get_input_embeddings()
        if input_embeddings is not None:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            input_embeddings.register_forward_hook(make_inputs_require_grad)
    
    # Disable cache for gradient checkpointing
    model.config.use_cache = False

    # Define LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # depends on model architecture
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA (this will unfreeze the LoRA parameters)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,  # was 2
        per_device_eval_batch_size=5,  # was 10 with beam 1  
        gradient_accumulation_steps=4,  # was 4
        learning_rate=2e-5,
        max_steps=1200,
        # num_train_epochs=None,
        fp16=False,
        logging_steps=10,
        
        # Validate + save on a schedule (needed for ES)
        eval_strategy="steps",
        eval_steps=50,  # was 500                    # align eval & save cadence
        save_strategy="steps",
        save_steps=50,  # was 500
        save_total_limit=10,         # keep disk usage sane

        # Pick the best checkpoint and restore it at the end
        load_best_model_at_end=True,
        metric_for_best_model="rougeLsum",   # was "eval_loss", or a custom metric key, e.g., "accuracy"
        greater_is_better=True,     # was False
                
        optim="adamw_torch",
        report_to="none",
        gradient_checkpointing=True,
        dataloader_pin_memory=False  # Can help with memory issues
    )
    
    # Optional: if you compute custom metrics, define compute_metrics=... in Trainer
    # Check if we're resuming from checkpoint
    import glob
    checkpoints_exist = len(glob.glob(os.path.join(output_dir, "checkpoint-*"))) > 0
    
    # Only add EarlyStoppingCallback when starting fresh (not resuming)
    # This avoids KeyError when resuming from checkpoint
    callbacks = []
    if not checkpoints_exist:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=10,           # stop if no improvement for n evals
            early_stopping_threshold=0.0         # require strictly better than best
        )
        callbacks.append(early_stopping)
        print("Adding EarlyStoppingCallback (fresh training)")
    else:
        print("Resuming from checkpoint - skipping EarlyStoppingCallback to avoid state errors")

    # Initialize Trainer
    trainer = CausalLMTrainer(
        # Generation settings (important so ROUGE is computed on model outputs)
        generation_max_length=MAX_TOKENS,  # Ample length for summaries (was MAX_TOKENS=2048)
        generation_num_beams=4,  # Greedy decoding for speed during training (was 4)
        # General Trainer settings
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_val_dataset, # <-- ES needs this
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=callbacks
    )
    

    # Start training
    if checkpoints_exist:
        print("Resuming training from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("Starting training from scratch...")
        trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"Training completed. Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune a language model')
    parser.add_argument('--model', type=str, required=True,
                       choices=['viking-7b', 'gemma-2b', 'mt5', 'gemma-7b'],
                       help='Model to fine-tune')
    parser.add_argument('--train_dataset', type=str, default='/app/data/output/processed_data_train.jsonl',
                       help='Path to processed dataset')
    parser.add_argument('--val_dataset', type=str, default='/app/data/output/processed_data_val.jsonl',
                       help='Path to processed val dataset')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for the fine-tuned model')
    parser.add_argument('--hf_token', type=str,
                       help='Hugging Face authentication token for private models')

    args = parser.parse_args()

    # Model mapping
    model_mapping = {
        'viking-7b': 'LumiOpen/Viking-7B',
        'gemma-2b': 'google/gemma-2b',
        'mt5': 'google/mt5-base',
        'gemma-7b': 'google/gemma-7b'
    }

    model_name = model_mapping[args.model]

    if args.model == 'viking-7b':
        output_dir = args.output_dir or "/app/models/viking_finetuned"
    else:
        output_dir = args.output_dir or "/app/models/gemma_finetuned"

    fine_tune_model(
        model_name=model_name,
        dataset_path=args.train_dataset,
        val_dataset_path=args.val_dataset,
        output_dir=output_dir,
        hf_token=args.hf_token
    )
