import os
import torch
import sys
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from train_worker import (
    setup_distributed_training,
    setup_wandb_and_hf,
    load_model_with_quantization,
    setup_lora_training,
    create_trainer,
    run_training_epoch
)
from tokenization_utils import get_tokenizer, load_and_preprocess_datasets, tokenize_dataset, format_train_example

from test_utils import ROUGECallback, test_model_on_gpu


from transformers import TrainingArguments, EarlyStoppingCallback, DataCollatorForLanguageModeling
from peft import LoraConfig

# ADD GLOBAL WANDB AVAILABILITY FOR CONSISTENCY
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add this function to your run_training.py
import torch.distributed as dist

def fine_tune_model(
    model_name: str,
    dataset_path: str,
    val_dataset_path: str,
    output_dir: str,
    max_input_prompt_tokens: int,
    max_epochs: int,
    train_batch_size: int,
    val_batch_size: int,
    val_steps: int,
    from_checkpoint: bool,
    test_dataset_path: str = None,
    hf_token: str = None,
    **kwargs
):
    """Main function to run fine-tuning with distributed training"""
    
     # Setup distributed training
    local_rank, world_size = setup_distributed_training()
    
    # Get the device for this process
    device = torch.device(f"cuda:{local_rank}")
    print(f"[Rank {local_rank}] Using device: {device}")

    # Setup WandB and HF (only on rank 0)
    config = {
        "model_name": model_name,
        "dataset": dataset_path,
        "output_dir": output_dir,
        "max_epochs": max_epochs,
        "train_batch_size": train_batch_size,
    }
    setup_wandb_and_hf(local_rank, hf_token, config)

    # Load tokenizer
    tokenizer = get_tokenizer(model_name, hf_token)

    # Load model with quantization - PASS local_rank
    model = load_model_with_quantization(model_name, hf_token, local_rank)
    
    # EXPLICITLY MOVE MODEL TO THE CORRECT DEVICE
    model = model.to(device)
    print(f"[Rank {local_rank}] Model moved to device: {device}")

    # Wait for all processes to complete model loading
    if dist.is_initialized():
        dist.barrier()


    # Load and preprocess datasets
    train_dataset, val_dataset, test_dataset = load_and_preprocess_datasets(
        dataset_path, val_dataset_path, test_dataset_path
    )

    # Tokenize datasets
    formatted_train_dataset, tokenized_train_dataset = tokenize_dataset(
        train_dataset, tokenizer, max_input_prompt_tokens, format_train_example
    )
    
    # Also create tokenized version of validation for loss computation
    _, tokenized_val_dataset = tokenize_dataset(
        val_dataset, tokenizer, max_input_prompt_tokens, format_train_example
    )
    
    # Format validation dataset for generation (without answers)
    formatted_val_dataset_for_rouge = val_dataset.map(
        lambda example: {
            "input_text": f"### Oppgave: Oppsummer følgende tekst\n{example['input']}\n\n### Svar:",
            "input": example['input'],
            "output": example['output']
        }
    )

    # Setup LoRA training
    model = setup_lora_training(model)

    # Create callbacks
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    rouge_callback = ROUGECallback(
        tokenizer=tokenizer,
        val_dataset=formatted_val_dataset_for_rouge,
        compute_rouge_every_n_steps=500
    )
    callbacks.append(rouge_callback)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Training arguments
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
        report_to="wandb" if WANDB_AVAILABLE else "none",
        gradient_checkpointing=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_accumulation_steps=2,
        warmup_steps=100,
        weight_decay=0.01,
        save_total_limit=10,
        dataloader_pin_memory=False,
        local_rank=local_rank,
        ddp_find_unused_parameters=False,
    )

    # Create trainer
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        training_args=training_args,
        data_collator=data_collator,
        callbacks=callbacks
    )

    # Run training
    trainer = run_training_epoch(trainer, model, rouge_callback, local_rank)

    # Save model and run testing (only on rank 0)
    if local_rank == 0:
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        print(f"Training completed. Best model saved to {output_dir}")

        if test_dataset_path and os.path.exists(test_dataset_path):
            print(f"Running testing on test dataset: {test_dataset_path}")
            test_model_on_gpu(model, tokenizer, test_dataset_path, output_dir)

        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()

def main():
    """Main function that handles command line arguments"""
    parser = argparse.ArgumentParser(description='Fine-tune a language model with distributed training')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                       choices=['viking-7b', 'gemma-2b', 'mt5', 'gemma-7b'],
                       help='Model to fine-tune')
    parser.add_argument('--train_dataset', type=str, required=True,
                       help='Path to training dataset')
    parser.add_argument('--val_dataset', type=str, required=True,
                       help='Path to validation dataset')
    parser.add_argument('--test_dataset', type=str,
                       help='Path to test dataset')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for the fine-tuned model')
    parser.add_argument('--hf_token', type=str,
                       help='Hugging Face authentication token')

    # Training hyperparameters
    parser.add_argument('--max_input_text_tokens', type=int, default=2048)
    parser.add_argument('--max_extra_prompt_tokens', type=int, default=40)
    parser.add_argument('--max_output_summary_tokens', type=int, default=512)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=5)
    parser.add_argument('--val_data_size', type=int, default=20)
    parser.add_argument('--val_beam_size', type=int, default=4)
    parser.add_argument('--val_steps', type=int, default=100)
    parser.add_argument('--from_checkpoint', action='store_true',
                       help='Resume training from checkpoint')

    args = parser.parse_args()

    # Model mapping
    model_mapping = {
        'viking-7b': 'LumiOpen/Viking-7B',
        'gemma-2b': 'google/gemma-2b',
        'mt5': 'google/mt5-base',
        'gemma-7b': 'google/gemma-7b'
    }

    # Calculate max input prompt tokens
    MAX_INPUT_PROMPT_TOKENS = args.max_input_text_tokens + args.max_extra_prompt_tokens

    # Set default output directory
    if not args.output_dir:
        if args.model == 'viking-7b':
            args.output_dir = "/app/models/viking_finetuned"
        else:
            args.output_dir = "/app/models/gemma_finetuned"

    print("Starting distributed training with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Call the fine_tune_model function
    fine_tune_model(
        model_name=model_mapping[args.model],
        dataset_path=args.train_dataset,
        val_dataset_path=args.val_dataset,
        output_dir=args.output_dir,
        max_input_prompt_tokens=MAX_INPUT_PROMPT_TOKENS,
        max_epochs=args.max_epochs,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        val_steps=args.val_steps,
        from_checkpoint=args.from_checkpoint,
        test_dataset_path=args.test_dataset,
        hf_token=args.hf_token,
    )

if __name__ == "__main__":
    main()