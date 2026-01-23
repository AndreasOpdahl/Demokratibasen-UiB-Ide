"""
Norwegian Text Summarization Fine-Tuning Script

This script fine-tunes language models for Norwegian text summarization using LoRA (Low-Rank Adaptation)
and evaluates performance using ROUGE metrics. Supports multiple models including Gemma, Viking, and mT5.

KEY FEATURES:
- LoRA fine-tuning for parameter-efficient training
- Automatic checkpoint resumption from best model
- ROUGE evaluation during training and testing
- Weights & Biases integration for experiment tracking
- Comprehensive testing with detailed metrics
- Support for both causal LM and sequence-to-sequence models

MODELS SUPPORTED:
- gemma-2b, gemma-7b (Google)
- viking-7b (LumiOpen)
- mt5 (Google)

USAGE:
1. Basic training from scratch:
   python script.py --model gemma-2b --train_dataset data/train.jsonl --val_dataset data/val.jsonl

2. Training with specific hyperparameters:
   python script.py --model gemma-2b --max_epochs 10 --train_batch_size 4 --val_steps 200

3. Resume training from best checkpoint:
   python script.py --model gemma-2b --output_dir models/gemma-2b-finetuned

4. Test a trained model:
   python script.py --model gemma-2b --test_dataset data/test.jsonl --output_dir models/gemma-2b-finetuned

REQUIREMENTS:
- transformers, datasets, peft, evaluate, rouge-score, wandb
- GPU with sufficient memory (>=16GB recommended for 7B models)
- Hugging Face token for private models (if needed)

ENVIRONMENT SETUP:
1. Install dependencies:
   pip install transformers datasets peft evaluate rouge-score wandb

2. Set up WandB (optional but recommended):
   wandb login

3. Set Hugging Face token (if using private models):
   export HUGGINGFACE_TOKEN=your_token_here

CHECKPOINT MANAGEMENT:
- Automatically finds and resumes from best checkpoint
- Saves checkpoints every --val_steps steps
- Keeps top 10 checkpoints (configurable)
- Supports both full fine-tuning and PEFT checkpoints

OUTPUT:
- Fine-tuned model in --output_dir
- Training logs and metrics in WandB
- Test results: test_results_<wandb_run_name>.json
- All predictions in JSONL format for analysis

TROUBLESHOOTING:
- If resuming fails, check checkpoint files exist and are valid
- For memory issues, reduce batch sizes or use gradient checkpointing
- For WandB issues, run without: --report_to none

EXAMPLE SLURM SCRIPT:
See accompanying .sbatch file for cluster execution
"""

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from huggingface_hub import login
import pandas as pd
import json
import glob
import subprocess
import tempfile
import torch
import argparse
import os
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import evaluate
import numpy as np
import datetime
import re
torch.cuda.empty_cache()

# Add safe globals before loading - prevents serialization warnings
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

import multiprocessing as mp
import random
import time
from tqdm import tqdm

# Import wandb for experiment tracking (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Weights & Biases not available. Install with: pip install wandb")


def is_valid_checkpoint(checkpoint_path):
    """
    Check if a checkpoint directory contains the necessary files to resume training
    Handles both full fine-tuning and PEFT (LoRA) checkpoints
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        bool: True if checkpoint is valid and can be used for resumption
    """
    # For PEFT models, we need adapter files
    peft_files = [
        'adapter_model.safetensors',  # PEFT adapter weights (safetensors format)
        'adapter_model.bin',          # PEFT adapter weights (bin format)
        'adapter_config.json',        # PEFT adapter configuration
    ]

    # For full fine-tuning models
    full_ft_files = [
        'pytorch_model.bin',
        'model.safetensors',
    ]

    checkpoint_files = os.listdir(checkpoint_path)

    # Check if it's a PEFT checkpoint
    has_peft_weights = any(f in checkpoint_files for f in ['adapter_model.safetensors', 'adapter_model.bin'])
    has_adapter_config = 'adapter_config.json' in checkpoint_files

    # Check if it's a full fine-tuning checkpoint
    has_full_ft_weights = any(f in checkpoint_files for f in full_ft_files)

    if has_peft_weights and has_adapter_config:
        print(f"✓ Valid PEFT checkpoint: {checkpoint_path}")
        return True
    elif has_full_ft_weights:
        print(f"✓ Valid full fine-tuning checkpoint: {checkpoint_path}")
        return True
    else:
        print(f"✗ Invalid checkpoint {checkpoint_path}: missing model weights")
        return False


def find_best_checkpoint(checkpoint_dir):
    """
    Find the best checkpoint by reading trainer_state.json and finding the one with lowest eval_loss
    Automatically handles both root-level and checkpoint-specific trainer_state files
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        str: Path to best checkpoint directory, or None if no valid checkpoints found
    """
    # First, check if there's a trainer_state.json in the root directory
    root_trainer_state = os.path.join(checkpoint_dir, "trainer_state.json")

    if os.path.exists(root_trainer_state):
        print(f"Found trainer_state.json in root directory")
        with open(root_trainer_state, 'r') as f:
            trainer_state = json.load(f)

        # Find the best model checkpoint (lowest eval_loss)
        if "best_model_checkpoint" in trainer_state and trainer_state["best_model_checkpoint"]:
            best_checkpoint = trainer_state["best_model_checkpoint"]
            if os.path.exists(best_checkpoint) and is_valid_checkpoint(best_checkpoint):
                return best_checkpoint
            else:
                print(f"Best checkpoint from trainer_state not found or invalid: {best_checkpoint}")

    # Fallback: check individual checkpoint directories
    print(f"No valid trainer_state.json in root. Checking individual checkpoints...")
    checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))

    if not checkpoint_dirs:
        print("No checkpoint directories found")
        return None

    # Sort by step number
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))

    # Try to find trainer_state.json in each checkpoint to determine the best one
    best_checkpoint = None
    best_eval_loss = float('inf')

    for checkpoint in checkpoint_dirs:
        checkpoint_state_path = os.path.join(checkpoint, "trainer_state.json")
        if os.path.exists(checkpoint_state_path):
            try:
                with open(checkpoint_state_path, 'r') as f:
                    checkpoint_state = json.load(f)

                # Look for the eval loss in log history
                if "log_history" in checkpoint_state:
                    eval_logs = [log for log in checkpoint_state["log_history"] if "eval_loss" in log]
                    if eval_logs:
                        current_eval_loss = min(log["eval_loss"] for log in eval_logs)
                        if current_eval_loss < best_eval_loss and is_valid_checkpoint(checkpoint):
                            best_eval_loss = current_eval_loss
                            best_checkpoint = checkpoint
            except Exception as e:
                print(f"Error reading trainer_state.json in {checkpoint}: {e}")

    if best_checkpoint:
        print(f"Found best checkpoint by eval_loss: {best_checkpoint} (loss: {best_eval_loss:.4f})")
        return best_checkpoint

    # Final fallback: use the latest valid checkpoint
    print("Using latest checkpoint as fallback")
    for checkpoint in reversed(checkpoint_dirs):
        if is_valid_checkpoint(checkpoint):
            return checkpoint

    return None


class ROUGECallback(TrainerCallback):
    """
    Custom callback to compute ROUGE scores during validation
    Evaluates summarization quality using ROUGE-1, ROUGE-2, ROUGE-L metrics
    """

    def __init__(self, tokenizer, val_dataset, compute_rouge_every_n_steps=100):
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.compute_rouge_every_n_steps = compute_rouge_every_n_steps
        self.rouge = evaluate.load('rouge')

    def on_evaluate(self, args, state, control, **kwargs):
        # Compute ROUGE scores during evaluation (not every eval to save time)
        if state.global_step % self.compute_rouge_every_n_steps == 0:
            self._compute_rouge(**kwargs)

    def _compute_rouge(self, model, **kwargs):
        """
        Compute ROUGE scores on a subset of validation data
        Uses beam search for generation and compares with reference summaries
        """
        model.eval()
        predictions = []
        references = []

        # Sample a subset for ROUGE computation (for efficiency)
        subset_size = min(50, len(self.val_dataset))
        indices = random.sample(range(len(self.val_dataset)), subset_size)

        with torch.no_grad():
            for idx in indices:
                example = self.val_dataset[idx]

                # Use the pre-formatted input text (already includes the prompt)
                input_text = example['input_text']
                reference = example['output']

                # Tokenize input
                inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Generate prediction with beam search
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )

                # Decode prediction
                prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extract just the generated answer (remove the input part)
                if "### Svar:" in prediction:
                    prediction = prediction.split("### Svar:")[-1].strip()

                predictions.append(prediction)
                references.append(reference)

        # Compute ROUGE scores
        if predictions and references:
            rouge_results = self.rouge.compute(
                predictions=predictions,
                references=references,
                use_stemmer=True
            )

            # Log to wandb if available
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    "rouge1": rouge_results["rouge1"],
                    "rouge2": rouge_results["rouge2"],
                    "rougeL": rouge_results["rougeL"],
                    "rougeLsum": rouge_results["rougeLsum"]
                })

            print(f"ROUGE Scores - 1: {rouge_results['rouge1']:.4f}, 2: {rouge_results['rouge2']:.4f}, L: {rouge_results['rougeL']:.4f}")

        model.train()


def fine_tune_model(
    model_name: str,
    dataset_path: str,
    val_dataset_path: str,
    output_dir: str,
    max_input_text_tokens: int,
    max_extra_prompt_tokens: int,
    max_input_prompt_tokens: int,
    max_output_summary_tokens: int,
    max_epochs: int,
    train_batch_size: int,
    val_batch_size: int,
    val_data_size: int,
    val_beam_size: int,
    val_steps: int,
    resume_from_checkpoint: str = None,
    test_dataset_path: str = None,
    hf_token: str = None,
):
    """
    Main function to fine-tune a language model for Norwegian text summarization
    
    Args:
        model_name: Hugging Face model identifier
        dataset_path: Path to training data (JSONL format)
        val_dataset_path: Path to validation data (JSONL format)
        output_dir: Directory to save model and results
        max_input_text_tokens: Maximum tokens for input text
        max_extra_prompt_tokens: Additional tokens for prompt template
        max_input_prompt_tokens: Total input tokens (text + prompt)
        max_output_summary_tokens: Maximum tokens for generated summaries
        max_epochs: Maximum training epochs
        train_batch_size: Batch size for training
        val_batch_size: Batch size for validation
        val_data_size: Number of examples for validation
        val_beam_size: Beam size for validation generation
        val_steps: Run validation every N steps
        resume_from_checkpoint: Specific checkpoint to resume from (None for auto-detection)
        test_dataset_path: Path to test data for final evaluation
        hf_token: Hugging Face authentication token
    """
    # Ensure we're using the environment-set cache directory
    cache_dir = os.environ.get('HF_HOME', os.environ.get('TRANSFORMERS_CACHE', None))
    print(f"Using cache directory: {cache_dir}")

    # Initialize wandb if available
    if WANDB_AVAILABLE:
        wandb.init(
            project="text-summarization-finetuning",
            config={
                "model_name": model_name,
                "dataset": dataset_path,
                "output_dir": output_dir
            }
        )
    else:
        print("Weights & Biases not available. Training will proceed without logging.")

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
                load_in_4bit=True,
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

    # Read validation JSONL file manually
    val_data = []
    try:
        with open(val_dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                val_data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading validation dataset: {e}")
        return

    # Create a pandas DataFrame
    df = pd.DataFrame(data)
    val_df = pd.DataFrame(val_data)

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    val_dataset = Dataset.from_pandas(val_df)

    def format_train_example(example):
        # Format the input-output pair for the model (with answer for teacher forcing)
        text = f"### Oppgave: Oppsummer følgende tekst\n{example['input']}\n\n### Svar: {example['output']}"
        return {"text": text}

    def format_eval_example(example):
        # Format for evaluation/generation (without answer but WITH prompt)
        input_text = f"### Oppgave: Oppsummer følgende tekst\n{example['input']}\n\n### Svar:"
        return {
            "input_text": input_text,  # This includes the prompt
            "input": example['input'],  # Keep original for reference
            "output": example['output']
        }

    def tokenize_function(examples):
        # Tokenize the formatted text
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_input_prompt_tokens,
            padding=False
        )

    # Format and tokenize training dataset (with answers)
    formatted_train_dataset = dataset.map(format_train_example)
    tokenized_train_dataset = formatted_train_dataset.map(tokenize_function, batched=True)

    # Format validation dataset (with prompt but without answer)
    formatted_val_dataset = val_dataset.map(format_eval_example)

    # Also create tokenized version for loss computation (with answers)
    formatted_val_dataset_for_loss = val_dataset.map(format_train_example)
    tokenized_val_dataset = formatted_val_dataset_for_loss.map(tokenize_function, batched=True)

    # Data collator for dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    model = prepare_model_for_kbit_training(model)

    # Define LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # depends on model architecture
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    print(model.print_trainable_parameters())

    # Create callbacks
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]

    # Add ROUGE callback
    rouge_callback = ROUGECallback(
        tokenizer=tokenizer,
        val_dataset=formatted_val_dataset,  # Use eval format (with prompt, without answers)
        compute_rouge_every_n_steps=500  # Compute ROUGE every 500 steps
    )
    callbacks.append(rouge_callback)
    
    # Determine which checkpoint to resume from
    if resume_from_checkpoint is None and os.path.exists(output_dir):
        # Auto-detect best checkpoint in output directory
        resume_from_checkpoint = find_best_checkpoint(output_dir)

    if resume_from_checkpoint and is_valid_checkpoint(resume_from_checkpoint):
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")

        # For PEFT models, we need to load the adapter weights
        if any(f in os.listdir(resume_from_checkpoint) for f in ['adapter_model.safetensors', 'adapter_model.bin']):
            print("Loading PEFT adapter weights...")
            # The Trainer will automatically handle PEFT checkpoint loading
    else:
        print("Starting training from scratch (no valid checkpoint found)")
        resume_from_checkpoint = None


    # Training arguments with wandb logging
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=val_batch_size,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        num_train_epochs=max_epochs,
        fp16=False,
        bf16=False,
        logging_steps=10,
        eval_steps=val_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=val_steps,
        optim="adamw_torch",
        report_to="wandb" if WANDB_AVAILABLE else "none",  # Enable wandb logging
        gradient_checkpointing=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_accumulation_steps=2,
        warmup_steps=100,
        weight_decay=0.01,
        save_total_limit=10,  # Keep 10 best checkpoints
        dataloader_pin_memory=False,
        resume_from_checkpoint=resume_from_checkpoint
    )

    # Initialize Trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,  # Use tokenized version for loss computation
        data_collator=data_collator,
        callbacks=callbacks
    )

    if not resume_from_checkpoint:
        # Validate once before training (both loss and ROUGE)
        print("Running validation before training...")

        # Compute initial loss
        initial_eval = trainer.evaluate()
        print(f"Initial validation loss: {initial_eval['eval_loss']:.4f}")

        # Compute initial ROUGE
        print("Computing initial ROUGE scores...")
        rouge_callback._compute_rouge(model=model)

        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({"eval_loss": initial_eval['eval_loss'], "step": 0})

    else:
        print("Skipping initial validation (resuming from checkpoint)")

    # Start training
    print("Starting training...")
    trainer.train()

    # Save the best model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"Training completed. Best model saved to {output_dir}")

    # Run testing if test dataset is provided
    if test_dataset_path and os.path.exists(test_dataset_path):
        print(f"Running testing on test dataset: {test_dataset_path}")
        test_model_on_gpu(model, tokenizer, test_dataset_path, output_dir, 
                         max_new_tokens=max_output_summary_tokens, num_beams=val_beam_size)

    # Finish wandb run
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


def test_model_on_gpu(model, tokenizer, test_dataset_path, output_dir, max_new_tokens=256, num_beams=4):
    """Run testing on the trained model using GPU"""
    print("Loading test dataset...")

    # Read test JSONL file
    test_data = []
    try:
        with open(test_dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading test dataset: {e}")
        return

    # Create test dataset
    test_df = pd.DataFrame(test_data)
    test_dataset = Dataset.from_pandas(test_df)

    def format_test_example(example):
        # Format for generation (with prompt but without answer)
        input_text = f"### Oppgave: Oppsummer følgende tekst\n{example['input']}\n\n### Svar:"
        return {
            "input_text": input_text,
            "input": example['input'],
            "reference": example['output']
        }

    formatted_test_dataset = test_dataset.map(format_test_example)

    # Initialize ROUGE
    rouge = evaluate.load('rouge')

    model.eval()
    predictions = []
    references = []
    all_inputs = []

    print("Running inference on test set...")
    with torch.no_grad():
        for example in tqdm(formatted_test_dataset, desc="Testing"):
            # Use the pre-formatted input text (already includes the prompt)
            input_text = example['input_text']

            # Tokenize input
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate prediction
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                do_sample=False
            )

            # Decode prediction
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the generated answer (remove the input part)
            if "### Svar:" in prediction:
                prediction = prediction.split("### Svar:")[-1].strip()

            predictions.append(prediction)
            references.append(example['reference'])
            all_inputs.append(example['input_text'])

    # Compute ROUGE scores
    rouge_results = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )

    # Print and save results
    print("\n" + "="*50)
    print("TEST RESULTS:")
    print("="*50)
    for key, value in rouge_results.items():
        print(f"{key}: {value:.4f}")

    # Save test results to file with wandb name or timestamp
    if WANDB_AVAILABLE and wandb.run is not None:
        # Get wandb run name and sanitize it for filename
        wandb_name = wandb.run.name
        wandb_name_clean = re.sub(r'[^\w\-_.]', '_', wandb_name)
        results_file = os.path.join(output_dir, f"test_results_{wandb_name_clean}.json")
    else:
        # Use timestamp if wandb not available
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"test_results_{timestamp}.json")

    # Get model path from model configuration
    model_path = getattr(model.config, '_name_or_path', 'unknown')

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "rouge_scores": {
                "rouge1": rouge_results["rouge1"],
                "rouge2": rouge_results["rouge2"],
                "rougeL": rouge_results["rougeL"],
                "rougeLsum": rouge_results["rougeLsum"]
            },
            "test_config": {
                "model_path": model_path,
                "test_dataset_path": test_dataset_path,
                "max_new_tokens": max_new_tokens,
                "num_beams": num_beams
            },
            "summary": {
                "total_examples": len(predictions),
                "average_prediction_length": np.mean([len(pred) for pred in predictions]),
                "average_reference_length": np.mean([len(ref) for ref in references])
            },
            "wandb_run_name": wandb.run.name if WANDB_AVAILABLE and wandb.run is not None else None,
            "timestamp": datetime.datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)

    print(f"Detailed results saved to: {results_file}")

    # Save all predictions and references
    predictions_file = os.path.join(output_dir, "all_predictions.jsonl")
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for i, (input_text, pred, ref) in enumerate(zip(all_inputs, predictions, references)):
            f.write(json.dumps({
                "id": i,
                "input": input_text,
                "prediction": pred,
                "reference": ref
            }, ensure_ascii=False) + '\n')

    print(f"All predictions saved to: {predictions_file}")

    return rouge_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune a language model')
    parser.add_argument('--model', type=str, required=True,
                       choices=['viking-7b', 'gemma-2b', 'mt5', 'gemma-7b'],
                       help='Model to fine-tune')
    parser.add_argument('--train_dataset', type=str, default='/app/data/output/processed_data_train.jsonl',
                       help='Path to processed dataset')
    parser.add_argument('--val_dataset', type=str, default='/app/data/output/processed_data_val.jsonl',
                       help='Path to processed val dataset')
    parser.add_argument('--test_dataset', type=str,
                       help='Path to processed test dataset')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for the fine-tuned model')
    parser.add_argument('--hf_token', type=str,
                       help='Hugging Face authentication token for private models')

    # Hyperparameters
    parser.add_argument('--max_input_text_tokens', type=int, default=2048,
                       help='Max tokens for input to summarisation')
    parser.add_argument('--max_extra_prompt_tokens', type=int, default=40,
                       help='Max extra tokens for input prompt (the task description)')
    parser.add_argument('--max_output_summary_tokens', type=int, default=512,
                       help='Max tokens for output from summarisation')
    parser.add_argument('--max_epochs', type=int, default=10,
                       help='Maximum number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=5,
                       help='Validation batch size')
    parser.add_argument('--val_data_size', type=int, default=20,
                       help='Number of examples to use for validation')
    parser.add_argument('--val_beam_size', type=int, default=4,
                       help='Beam size for evaluation during validation')
    parser.add_argument('--val_steps', type=int, default=100,
                       help='Run validation every N steps')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Run training from checkpoint')

    args = parser.parse_args()

    MAX_INPUT_PROMPT_TOKENS = args.max_input_text_tokens + args.max_extra_prompt_tokens
    # Model mapping
    model_mapping = {
        'viking-7b': 'LumiOpen/Viking-7B',
        'gemma-2b': 'google/gemma-2b',
        'mt5': 'google/mt5-base',
        'gemma-7b': 'google/gemma-7b'
    }

    print("Arguments: ", args)

    model_name = model_mapping[args.model]

    if args.model == 'viking-7b':
        output_dir = args.output_dir or "/app/models/viking_finetuned"
    else:
        output_dir = args.output_dir or "/app/models/gemma_finetuned"

    fine_tune_model(
        model_name=model_name,
        dataset_path=args.train_dataset,
        val_dataset_path=args.val_dataset,
        test_dataset_path=args.test_dataset,
        output_dir=output_dir,
        hf_token=args.hf_token,
        max_input_text_tokens=args.max_input_text_tokens,
        max_extra_prompt_tokens=args.max_extra_prompt_tokens,
        max_input_prompt_tokens=MAX_INPUT_PROMPT_TOKENS,
        max_output_summary_tokens=args.max_output_summary_tokens,
        max_epochs=args.max_epochs,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        val_data_size=args.val_data_size,
        val_beam_size=args.val_beam_size,
        val_steps=args.val_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
