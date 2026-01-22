# finetune.py
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    TrainingArguments,
    Trainer, EarlyStoppingCallback,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from huggingface_hub import login
import pandas as pd
import json
import subprocess
import tempfile
import torch
import argparse
import os
from peft import LoraConfig, get_peft_model
torch.cuda.empty_cache()

import multiprocessing as mp
from tqdm import tqdm
import time 


class CPUEvaluationCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset_path, num_samples=20, eval_steps=100, patience=5):
        self.tokenizer = tokenizer
        self.eval_dataset_path = eval_dataset_path
        self.num_samples = num_samples  # Reduced for faster evaluation
        self.eval_steps = eval_steps
        self.patience = patience
        self.best_loss = float('inf')
        self.no_improvement_count = 0
        self.model = None
        self.evaluation_times = []

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.model = model
        print(f"CPU Evaluation Callback initialized (evaluating every {self.eval_steps} steps)")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self.model = model

        if (state.global_step % self.eval_steps == 0 and
            state.global_step > 0 and
            self.model is not None):

            self.run_cpu_evaluation(args, state, control)

    def run_cpu_evaluation(self, args, state, control):
        checkpoint_dir = tempfile.mkdtemp()
        eval_start_time = time.time()

        try:
            # Save model and tokenizer
            self.model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)

            print(f"\n--- Step {state.global_step}: Starting CPU Evaluation ---")

            # Run CPU evaluation with progress tracking
            result = subprocess.run([
                'python', 'scripts/cpu_evaluation_worker.py',
                '--model_path', checkpoint_dir,
                '--eval_data', self.eval_dataset_path,
                '--num_samples', str(self.num_samples),
                #'--num_processes', '4',  # Conservative number
                #'--sequential'  # Use sequential for more reliable progress tracking
            ], capture_output=True, text=True, timeout=600)

            eval_time = time.time() - eval_start_time
            self.evaluation_times.append(eval_time)
            avg_eval_time = sum(self.evaluation_times) / len(self.evaluation_times)

            if result.returncode == 0:
                try:
                    # Extract the loss from output
                    output_lines = result.stdout.strip().split('\n')
                    loss_line = [line for line in output_lines if 'Average loss:' in line]
                    time_line = [line for line in output_lines if 'Evaluation completed in' in line]

                    if loss_line:
                        current_loss = float(loss_line[0].split(':')[-1].strip())

                        print(f"✅ Evaluation completed in {eval_time:.2f}s (avg: {avg_eval_time:.2f}s)")
                        print(f"📊 Current loss: {current_loss:.4f}")
                        print(f"🏆 Best loss: {self.best_loss:.4f}")

                        # Check for improvement
                        if current_loss < self.best_loss:
                            self.best_loss = current_loss
                            self.no_improvement_count = 0
                            print("🎉 New best loss achieved!")
                        else:
                            self.no_improvement_count += 1
                            print(f"⏳ No improvement ({self.no_improvement_count}/{self.patience})")

                        # Trigger early stopping if no improvement
                        if self.no_improvement_count >= self.patience:
                            print("🛑 Early stopping triggered!")
                            control.should_training_stop = True
                    else:
                        print("❌ Could not parse loss from evaluation output")

                except ValueError as e:
                    print(f"❌ Error parsing loss value: {e}")
            else:
                print(f"❌ CPU evaluation failed with return code {result.returncode}")
                print(f"Stderr: {result.stderr}")

        except subprocess.TimeoutExpired:
            print("⏰ CPU evaluation timed out after 10 minutes")
        except Exception as e:
            print(f"❌ CPU evaluation error: {e}")
        finally:
            # Clean up temporary directory
            import shutil
            shutil.rmtree(checkpoint_dir, ignore_errors=True)

        print("--- Evaluation Complete ---\n")

            

def fine_tune_model(
    model_name: str,
    dataset_path: str,
    val_dataset_path: str,
    output_dir: str,
    hf_token: str = None
):
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

    # Create a pandas DataFrame
    df = pd.DataFrame(val_data)

    # Convert to Hugging Face Dataset
    val_dataset = Dataset.from_pandas(df)

    def format_example(example):
        # Format the input-output pair for the model
        text = f"### Oppgave: Oppsummer følgende tekst\n{example['input']}\n\n### Svar: {example['output']}"
        return {"text": text}

    def tokenize_function(examples):
        # Tokenize the formatted text
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
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

    cpu_eval_callback =  CPUEvaluationCallback(eval_dataset_path=val_dataset_path, tokenizer=tokenizer, num_samples=5, eval_steps=500, patience=5)

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

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        fp16=False,
        logging_steps=10,
        save_strategy="steps",        # It's often best to align save and eval strategies
        greater_is_better=False,      # A lower eval_loss is better
        save_steps=500,
        optim="adamw_torch",
        report_to="none",
        gradient_checkpointing=True,
        dataloader_pin_memory=False  # Can help with memory issues
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[cpu_eval_callback]
    )

    # Start training
    print("Starting training...")
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
