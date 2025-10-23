"""
Unified fine-tuning script for both quantized (GTX3090) and non-quantized (GH200) training.

Usage:
  # With 4-bit quantization (GTX3090):
  python finetune.py --model gemma-2b --quantization 4bit --hf_token YOUR_TOKEN
  
  # Without quantization (GH200/Cray):
  python finetune.py --model gemma-2b --quantization none --max_steps 1200 --hf_token YOUR_TOKEN

Before running:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""

import argparse
import glob
import json
import os
import random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from datasets import Dataset
import evaluate
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
import torch
import torch.serialization

# Fix for PyTorch 2.6+ weights_only security issue
# Patch torch.load to disable weights_only for checkpoint files (we trust our own checkpoints)
_original_torch_load = torch.load

def _torch_load_with_weights_only_false(path, *args, **kwargs):
    """Wrapper around torch.load that disables weights_only for checkpoint compatibility."""
    # Only disable weights_only for checkpoint-related files
    if 'rng_state' in str(path) or 'optimizer' in str(path) or 'scheduler' in str(path):
        kwargs['weights_only'] = False
    return _original_torch_load(path, *args, **kwargs)

torch.load = _torch_load_with_weights_only_false

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Trainer,
    TrainingArguments,
)

# Optional imports for quantization
try:
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    print("Warning: BitsAndBytesConfig/prepare_model_for_kbit_training not available.")
    print("Quantization will be disabled. This is expected on ARM-based systems (e.g., GH200).")


# Default values when command-line args are not supplied
MAX_INPUT_TEXT_TOKENS = 2048  # max tokens for input to summarisation
MAX_EXTRA_PROMPT_TOKENS = 40  # max extra tokens for input prompt (the task description)
MAX_INPUT_PROMPT_TOKENS = MAX_INPUT_TEXT_TOKENS + MAX_EXTRA_PROMPT_TOKENS
MAX_OUTPUT_SUMMARY_TOKENS = 512  # max tokens for output from summarisation
MAX_EPOCHS = 30
TRAIN_BATCH_SIZE = 2
VAL_BATCH_SIZE = 10
VAL_DATA_SIZE = 20  # number of examples to use for validation
VAL_BEAM_SIZE = 4  # beam size for evaluation
VAL_STEPS = 200


class EvalDataCollator:
    """Custom data collator for evaluation that pads both input_ids and labels.
    
    Uses left-padding for input_ids (decoder-only models) and right-padding for labels.
    """
    
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        # Separate input_ids and labels
        input_ids = [f['input_ids'] for f in features]
        labels = [f['labels'] for f in features]
        
        # Pad input_ids (LEFT padding for decoder-only models)
        max_input_length = max(len(ids) for ids in input_ids)
        if self.pad_to_multiple_of:
            max_input_length = ((max_input_length + self.pad_to_multiple_of - 1) 
                               // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        padded_input_ids = []
        attention_mask = []
        for ids in input_ids:
            padding_length = max_input_length - len(ids)
            # LEFT padding: pad tokens go BEFORE the actual tokens
            padded_input_ids.append([self.tokenizer.pad_token_id] * padding_length + ids)
            attention_mask.append([0] * padding_length + [1] * len(ids))
        
        # Pad labels (RIGHT padding - standard for targets)
        max_label_length = max(len(lbl) for lbl in labels)
        if self.pad_to_multiple_of:
            max_label_length = ((max_label_length + self.pad_to_multiple_of - 1) 
                               // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        padded_labels = []
        for lbl in labels:
            padding_length = max_label_length - len(lbl)
            # RIGHT padding for labels: pad with -100 so they're ignored
            padded_labels.append(lbl + [-100] * padding_length)
        
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long),
        }


class CausalLMTrainer(Trainer):
    def __init__(self, *args, 
                 generation_max_length: Optional[int] = None,
                 generation_num_beams: Optional[int] = None,
                 eval_data_collator: Optional[Any] = None,
                 **kwargs) -> None:
        # 1. Store generation parameters
        self.generation_max_length = generation_max_length
        self.generation_num_beams = generation_num_beams
        self.eval_data_collator = eval_data_collator
        # 2. Call parent constructor
        super().__init__(*args, **kwargs)
        # 3. Store reference to tokenizer for compatibility
        self._processing_class = self.tokenizer
    
    def get_eval_dataloader(self, eval_dataset=None):
        """Override to use a different data collator for evaluation."""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        # Temporarily swap the data collator
        original_collator = self.data_collator
        if self.eval_data_collator is not None:
            self.data_collator = self.eval_data_collator
        
        # Get the dataloader using parent's method
        dataloader = super().get_eval_dataloader(eval_dataset)
        
        # Restore original collator
        self.data_collator = original_collator
        
        return dataloader

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        # 1. Compute Loss (Standard behavior)
        if prediction_loss_only:
            # For loss-only, we can't compute it properly since eval data is prompt-only
            # Just return None and skip loss calculation
            return (None, None, None)

        # If we are here, we are evaluating the model
        print('*** evaluation: prediction_step ***')
        torch.cuda.empty_cache()

        # 2. Get Input IDs and Labels
        # input_ids contains ONLY the prompt (no target answer)
        if 'input_ids' in inputs:
            input_ids = inputs["input_ids"]
        else:
            raise KeyError("input_ids not found in batch. Check your DataCollator setup.")

        # labels contains the tokenized target summary (for ROUGE)
        labels = inputs.get("labels")
        
        # Remove padding from labels (-100 will be there from DataCollator)
        # We need clean label token IDs for ROUGE
        if labels is not None:
            # Create a copy and replace -100 with pad_token_id for proper shape
            labels = labels.clone()
            labels[labels == -100] = self._processing_class.pad_token_id

        print('*** evaluation: input_ids (prompt only) ***', input_ids.shape)
        if labels is not None:
            print('*** evaluation: labels (target summary) ***', labels.shape)

        # 3. Autoregressive Generation (The Key Step)
        # Generate from the prompt only
        generated_ids = model.generate(
            input_ids=input_ids,
            use_cache=True,
            max_new_tokens=self.generation_max_length,  # Generate up to 512 new tokens
            num_beams=self.generation_num_beams,
            do_sample=False,  # Greedy/beam search
            pad_token_id=self._processing_class.pad_token_id,
            eos_token_id=self._processing_class.eos_token_id,
        )
        
        # CRITICAL: Remove the input prompt from generated_ids
        # model.generate() returns [input_ids + new_tokens], we only want the new tokens for ROUGE
        input_length = input_ids.shape[1]
        generated_ids = generated_ids[:, input_length:]
        
        print('*** evaluation: generated_ids (generated summary only) ***', generated_ids.shape)
        
        torch.cuda.empty_cache()

        # No loss calculation during evaluation (we can't compute it from prompt-only data)
        loss = None
        
        # 5. Return Results
        # The Trainer expects: (loss, predictions, labels)
        # predictions: generated summary tokens
        # labels: target summary tokens
        return (loss, generated_ids, labels)


# Load ROUGE metric once
rouge = evaluate.load("rouge")


def load_model_with_optional_quantization(
    model_name: str,
    quantization: str,
    hf_token: Optional[str] = None
):
    """Load model with optional quantization.
    
    Args:
        model_name: Model identifier (e.g., 'google/gemma-2b')
        quantization: One of 'none', '4bit', '8bit'
        hf_token: Hugging Face token for private models
    
    Returns:
        Loaded model
    """
    if model_name == 'google/mt5-base':
        return MT5ForConditionalGeneration.from_pretrained(model_name)
    
    if quantization == 'none':
        # No quantization (GH200/Cray path)
        print("Loading model without quantization (FP16)...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                token=hf_token
            )
        except Exception as e:
            print(f"Error loading model with device_map: {e}")
            print("Trying fallback without device_map...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                token=hf_token
            ).cuda()
        return model
    
    elif quantization == '4bit':
        # 4-bit quantization (GTX3090 path)
        if not QUANTIZATION_AVAILABLE:
            raise ImportError(
                "BitsAndBytesConfig not available. Install bitsandbytes for quantization support:\n"
                "  pip install bitsandbytes"
            )
        
        print("Loading model with 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token
        )
        return model
    
    elif quantization == '8bit':
        # 8-bit quantization (optional)
        if not QUANTIZATION_AVAILABLE:
            raise ImportError(
                "BitsAndBytesConfig not available. Install bitsandbytes for quantization support:\n"
                "  pip install bitsandbytes"
            )
        
        print("Loading model with 8-bit quantization...")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token
        )
        return model
    
    else:
        raise ValueError(f"Unknown quantization method: {quantization}")


def prepare_model_for_lora(model, use_quantization: bool):
    """Prepare model for LoRA training with or without quantization.
    
    Args:
        model: The model to prepare
        use_quantization: Whether the model uses quantization
    
    Returns:
        Prepared model (note: model is modified in-place)
    """
    if use_quantization:
        # Use peft's built-in function for quantized models
        if not QUANTIZATION_AVAILABLE:
            raise ImportError("prepare_model_for_kbit_training not available. Install peft.")
        
        print("Preparing quantized model for LoRA training...")
        model = prepare_model_for_kbit_training(model)
    else:
        # Manual preparation for non-quantized models
        print("Preparing non-quantized model for LoRA training...")
        
        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Enable gradients for input embeddings (critical for LoRA!)
        # Without this, the embeddings stay frozen and cause "does not require grad" errors
        if hasattr(model, 'get_input_embeddings'):
            input_embeddings = model.get_input_embeddings()
            if input_embeddings is not None:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                input_embeddings.register_forward_hook(make_inputs_require_grad)
        
        # Disable cache for gradient checkpointing
        model.config.use_cache = False
    
    return model


def fine_tune_model(
    model_name: str,
    dataset_path: str,
    val_dataset_path: str,
    output_dir: str,
    quantization: str = 'none',
    max_steps: Optional[int] = None,
    num_train_epochs: Optional[int] = None,
    hf_token: Optional[str] = None
):
    """Fine-tune a language model with LoRA.
    
    Args:
        model_name: Model identifier
        dataset_path: Path to training dataset (JSONL)
        val_dataset_path: Path to validation dataset (JSONL)
        output_dir: Directory to save the fine-tuned model
        quantization: Quantization method ('none', '4bit', '8bit')
        max_steps: Maximum training steps (overrides num_train_epochs if set)
        num_train_epochs: Number of training epochs
        hf_token: Hugging Face authentication token
    """
    
    def compute_metrics(eval_pred):
        print('*** evaluation: compute_metrics ***')
        
        preds, labels = eval_pred  # preds: generated summary ids; labels: target summary ids
        print('*** evaluation: preds ***', preds.shape)
        print('*** evaluation: labels ***', labels.shape)
        
        # Replace -100 and pad tokens so we can decode properly
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Strip/normalize for ROUGE
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        
        # Debug: print first example
        if len(decoded_preds) > 0:
            print(f'\n*** Example 1 ***')
            print(f'Prediction: {decoded_preds[0][:200]}...')
            print(f'Reference:  {decoded_labels[0][:200]}...\n')

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
    
    # For decoder-only models (like GPT, Gemma), use left padding for generation
    # This ensures the model attends to the actual prompt, not padding
    tokenizer.padding_side = 'left'

    # Load model with optional quantization
    try:
        model = load_model_with_optional_quantization(model_name, quantization, hf_token)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load and preprocess dataset
    print(f"Loading dataset from: {dataset_path}")

    # Read JSONL file manually
    train_data = []
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                train_data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading training dataset: {e}")
        return

    # Read validation JSONL file
    val_data = []
    try:
        with open(val_dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                val_data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading validation dataset: {e}")
        return

    # Create training dataset
    train_df = pd.DataFrame(train_data)
    dataset = Dataset.from_pandas(train_df)
    
    # Randomly sample VAL_DATA_SIZE examples from val_data
    val_data = random.sample(val_data, min(VAL_DATA_SIZE, len(val_data)))
    val_df = pd.DataFrame(val_data)
    val_dataset = Dataset.from_pandas(val_df)
    print(f'*** validation dataset size: {len(val_dataset)} examples ***')

    def format_example_train(example):
        # Format the input-output pair for the model (TRAINING: full text for teacher forcing)
        text = f"Oppgave: Oppsummer følgende tekst:\n\n###\n\n{example['input']}\n\n###\n\nOppsummering:\n\n###\n\n{example['output']}\n\n###\n"
        return {"text": text}

    def format_example_eval(example):
        # Format for EVALUATION: only the input prompt, not the answer
        prompt = f"Oppgave: Oppsummer følgende tekst:\n\n###\n\n{example['input']}\n\n###\n\nOppsummering:\n\n###\n\n"
        # Keep the target output separate for ROUGE calculation
        return {
            "prompt": prompt,
            "target_summary": example['output']
        }

    def tokenize_function_train(examples):
        # Tokenize the formatted text for training
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_INPUT_PROMPT_TOKENS + MAX_OUTPUT_SUMMARY_TOKENS,
            padding=True
        )

    def tokenize_function_eval(examples):
        # Tokenize ONLY the prompt (without answer) for evaluation
        tokenized_prompts = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=MAX_INPUT_PROMPT_TOKENS,
            padding=False
        )
        # Tokenize target summaries for labels
        tokenized_targets = tokenizer(
            examples["target_summary"],
            truncation=True,
            max_length=MAX_OUTPUT_SUMMARY_TOKENS,
            padding=False
        )
        # Store target token IDs as labels
        tokenized_prompts["labels"] = tokenized_targets["input_ids"]
        return tokenized_prompts

    formatted_dataset = dataset.map(format_example_train)
    tokenized_dataset = formatted_dataset.map(tokenize_function_train, batched=True)

    # Format and tokenize the VALIDATION dataset differently
    formatted_val_dataset = val_dataset.map(format_example_eval)
    tokenized_val_dataset = formatted_val_dataset.map(tokenize_function_eval, batched=True)
    
    # Data collators
    # For TRAINING: use standard causal LM collator (creates labels by shifting)
    train_data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # For EVALUATION: use custom collator that pads both input_ids and labels
    eval_data_collator = EvalDataCollator(tokenizer=tokenizer)

    # Prepare model for LoRA training
    use_quantization = (quantization != 'none')
    model = prepare_model_for_lora(model, use_quantization)

    # Define LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # depends on model architecture
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA adapters
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Determine training duration
    if max_steps is not None and max_steps > 0:
        train_epochs = None
        train_steps = max_steps
        print(f"Training for {max_steps} steps (epochs ignored)")
    else:
        train_epochs = num_train_epochs if num_train_epochs is not None else MAX_EPOCHS
        train_steps = -1
        print(f"Training for {train_epochs} epochs")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=VAL_BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,  # Reduced from 2e-5 - too high can cause instability
        num_train_epochs=train_epochs,
        max_steps=train_steps,
        fp16=False,
        logging_steps=10,
        
        # Validate + save on a schedule (needed for early stopping)
        eval_strategy="steps",
        eval_steps=VAL_STEPS,  # align eval & save cadence
        save_strategy="steps",
        save_steps=VAL_STEPS,
        save_total_limit=10,  # keep disk usage sane

        # Pick the best checkpoint and restore it at the end
        load_best_model_at_end=True,
        metric_for_best_model="rougeLsum",
        greater_is_better=True,
        
        # Numerical stability improvements
        max_grad_norm=0.5,  # More aggressive gradient clipping
        warmup_steps=500,  # Much longer warmup (was 100)
        warmup_ratio=0.0,  # Ensure warmup_steps is used
        weight_decay=0.05,  # Increased regularization (was 0.01)
        adam_epsilon=1e-8,  # Standard Adam epsilon
        adam_beta1=0.9,
        adam_beta2=0.999,
                
        optim="adamw_torch",
        report_to="none",
        gradient_checkpointing=True,
        dataloader_pin_memory=False,  # Can help with memory issues
    )
    
    # Check if we're resuming from checkpoint
    checkpoints_exist = len(glob.glob(os.path.join(output_dir, "checkpoint-*"))) > 0
    
    # Only add EarlyStoppingCallback when starting fresh (not resuming)
    # This avoids KeyError when resuming from checkpoint
    callbacks = []
    if not checkpoints_exist:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=10,  # stop if no improvement for n evals
            early_stopping_threshold=0.0  # require strictly better than best
        )
        callbacks.append(early_stopping)
        print("Adding EarlyStoppingCallback (fresh training)")
    else:
        print("Resuming from checkpoint - skipping EarlyStoppingCallback to avoid state errors")

    # Initialize Trainer
    trainer = CausalLMTrainer(
        # Generation settings (important so ROUGE is computed on model outputs)
        generation_max_length=MAX_OUTPUT_SUMMARY_TOKENS,
        generation_num_beams=VAL_BEAM_SIZE,
        eval_data_collator=eval_data_collator,  # Use separate collator for eval
        # General Trainer settings
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=train_data_collator,  # Training collator
        callbacks=callbacks
    )

    # Start training
    if checkpoints_exist:
        print("Resuming training from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("Starting training from scratch...")
        trainer.train()
    
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"Training completed. Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fine-tune a language model with optional quantization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GTX3090 with 4-bit quantization:
  python finetune.py --model gemma-2b --quantization 4bit --hf_token YOUR_TOKEN

  # GH200/Cray without quantization, fixed steps:
  python finetune.py --model gemma-2b --quantization none --max_steps 1200 --hf_token YOUR_TOKEN

  # Without quantization, train for epochs:
  python finetune.py --model gemma-2b --quantization none --num_train_epochs 3 --hf_token YOUR_TOKEN
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       choices=['viking-7b', 'gemma-2b', 'mt5', 'gemma-7b'],
                       help='Model to fine-tune')
    parser.add_argument('--quantization', type=str, default='none',
                       choices=['none', '4bit', '8bit'],
                       help='Quantization method (default: none). Use "4bit" for GTX3090, "none" for GH200.')
    parser.add_argument('--train_dataset', type=str, default='/app/data/output/processed_data_train.jsonl',
                       help='Path to training dataset (JSONL format)')
    parser.add_argument('--val_dataset', type=str, default='/app/data/output/processed_data_val.jsonl',
                       help='Path to validation dataset (JSONL format)')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for the fine-tuned model')
    parser.add_argument('--max_steps', type=int, default=None,
                       help='Maximum training steps (overrides num_train_epochs if set)')
    parser.add_argument('--num_train_epochs', type=int, default=None,
                       help=f'Number of training epochs (default: {MAX_EPOCHS}, ignored if max_steps is set)')
    parser.add_argument('--hf_token', type=str,
                       help='Hugging Face authentication token for private models')

    args = parser.parse_args()

    # Validate quantization availability
    if args.quantization != 'none' and not QUANTIZATION_AVAILABLE:
        print(f"ERROR: Quantization requested ({args.quantization}) but BitsAndBytesConfig is not available.")
        print("Please install bitsandbytes: pip install bitsandbytes")
        print("Or use --quantization none to run without quantization.")
        exit(1)

    # Model mapping
    model_mapping = {
        'viking-7b': 'LumiOpen/Viking-7B',
        'gemma-2b': 'google/gemma-2b',
        'mt5': 'google/mt5-base',
        'gemma-7b': 'google/gemma-7b'
    }

    model_name = model_mapping[args.model]

    # Set default output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif args.model == 'viking-7b':
        output_dir = "/app/models/viking_finetuned"
    else:
        output_dir = "/app/models/gemma_finetuned"

    # Run fine-tuning
    fine_tune_model(
        model_name=model_name,
        dataset_path=args.train_dataset,
        val_dataset_path=args.val_dataset,
        output_dir=output_dir,
        quantization=args.quantization,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        hf_token=args.hf_token
    )

