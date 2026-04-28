from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
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
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import evaluate
import numpy as np
torch.cuda.empty_cache()

# Add safe globals before loading
#torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
import torch.distributed as dist

import multiprocessing as mp
import random
import time
import datetime
from tqdm import tqdm

# Import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Weights & Biases not available. Install with: pip install wandb")


class ROUGECallback(TrainerCallback):
    """Custom callback to compute ROUGE scores during validation"""

    def __init__(self, tokenizer, val_dataset, compute_rouge_every_n_steps=100):
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.compute_rouge_every_n_steps = compute_rouge_every_n_steps
        self.rouge = evaluate.load('rouge')

    def on_evaluate(self, args, state, control, **kwargs):
        # Compute ROUGE scores during evaluation
        if state.global_step % self.compute_rouge_every_n_steps == 0:
            self._compute_rouge(**kwargs)

    def _compute_rouge(self, model, **kwargs):
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

                # Generate prediction
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
    from_checkpoint: bool,
    test_dataset_path: str = None,
    hf_token: str = None,
):
    """
    # Get rank from environment (provided by torchrun)
     = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"Process {} of {world_size} starting...")


    # Set the correct device
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_devices = list(map(int, cuda_visible.split(","))) if cuda_visible else list(range(torch.cuda.device_count()))

    if  >= len(visible_devices):
        raise RuntimeError(f"Invalid  {local_rank} — only {len(visible_devices)} visible GPUs: {visible_devices}")

    device_id = visible_devices[]
    torch.cuda.set_device(device_id)

    print(f"[Rank {}] Using device {device_id} / visible GPUs: {visible_devices}")

    # Initialize process group
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        print(f"[Rank {}] Process group initialized.")

    if dist.is_initialized():
        print(f"Rank {dist.get_rank()} initialized on device {torch.cuda.current_device()}")

    # Ensure we're using the environment-set cache directory
    cache_dir = os.environ.get('HF_HOME', os.environ.get('TRANSFORMERS_CACHE', None))
    print(f"Using cache directory: {cache_dir}")


    if  >= torch.cuda.device_count():
        raise RuntimeError(f"Invalid  {local_rank} — only {torch.cuda.device_count()} GPUs available.")

    # Only initialize wandb and HF login on rank 0
    if  == 0:
        # Initialize wandb only on main process
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

        # Login to Hugging Face only on main process
        if hf_token:
            print("Logging in to Hugging Face Hub...")
            login(token=hf_token)
    else:
        # Other processes should not log in
        if WANDB_AVAILABLE:
            os.environ["WANDB_MODE"] = "disabled"

    """
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

    # Step 1: Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Use the normalized float 4-bit type
        bnb_4bit_use_double_quant=True,  # Enable nested quantization for even lower memory usage
        bnb_4bit_compute_dtype=torch.bfloat16  # Use bfloat16 for faster computation
    )

    # Load model with device mapping for each process
    try:
        if model_name == 'google/mt5-base':
            model = MT5ForConditionalGeneration.from_pretrained(model_name)
        else:
            # Each process loads the model to its specific GPU
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="auto",  # Each process uses its own GPU
                token=hf_token if hf_token else None
            )
    except Exception as e:
        print(f"Error loading model with device_map: {e}")
        try:
            # Fallback without device_map
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                torch_dtype=torch.float16,
                token=hf_token if hf_token else None
            )
        except Exception as e2:
            print(f"Error loading model: {e2}")
            return

    # Add barrier to synchronize processes after model loading
    #if world_size > 1:
    #   torch.distributed.barrier()

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
        compute_rouge_every_n_steps=500  # Compute ROUGE every 200 steps
    )
    callbacks.append(rouge_callback)

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
        save_total_limit=10,  # Only keep 3 best checkpoints
        dataloader_pin_memory=False,
        #=local_rank,
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

    # Start training
    print("Starting training...")
    trainer.train(resume_from_checkpoint=from_checkpoint)

    # Save the best model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"Training completed. Best model saved to {output_dir}")

    # Run testing if test dataset is provided
    if test_dataset_path and os.path.exists(test_dataset_path):
        print(f"Running testing on test dataset: {test_dataset_path}")
        test_model_on_gpu(model, tokenizer, test_dataset_path, output_dir)

    # Finish wandb run
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


def test_model_on_gpu(model, tokenizer, test_dataset_path, output_dir):
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
                max_new_tokens=256,
                num_beams=4,
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

    # Save test results to file
    results_file = os.path.join(output_dir, "test_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(rouge_results, f, indent=2, ensure_ascii=False)

    print(f"Test results saved to: {results_file}")

    # Save a few examples for inspection
    examples_file = os.path.join(output_dir, "test_examples.json")
    examples_to_save = []
    for i in range(min(10, len(predictions))):
        examples_to_save.append({
            "input": formatted_test_dataset[i]['input_text'],
            "prediction": predictions[i],
            "reference": references[i]
        })

    with open(examples_file, 'w', encoding='utf-8') as f:
        json.dump(examples_to_save, f, indent=2, ensure_ascii=False)

    print(f"Example predictions saved to: {examples_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune a language model')
    parser.add_argument('--model', type=str, required=True,
                       choices=['viking-7b', 'gemma-2b', 'mt5', 'gemma-7b-it'],
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
    parser.add_argument('--from_checkpoint', type=bool, default=False,
                       help='Run training from checkpoint')

    args = parser.parse_args()

    MAX_INPUT_PROMPT_TOKENS = args.max_input_text_tokens + args.max_extra_prompt_tokens
    # Model mapping
    model_mapping = {
        'viking-7b': 'LumiOpen/Viking-7B',
        'gemma-2b': 'google/gemma-2b',
        'mt5': 'google/mt5-base',
        'gemma-7b-it': 'google/gemma-7b-it'
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
        from_checkpoint=args.from_checkpoint,
    )