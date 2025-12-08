"""
Script to evaluate FSDP checkpoints from distributed training.

Note: With FULL_STATE_DICT checkpoints (now default), checkpoints are NOT distributed/sharded.
They can be loaded directly in single-GPU mode for evaluation.

Usage:
  # Set environment variables first:
  export HF_TOKEN=your_huggingface_token  # or HUGGINGFACE_TOKEN
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  
  # Evaluate a checkpoint from FSDP training:
  python evaluate_distributed_checkpoints.py \
    --model gemma-3-12b-pt \
    --checkpoint_dir training_runs/gemma-3-12b-pt-fsdp/checkpoint-100 \
    --val_dataset data/processed_data_val.jsonl
  
  # Just load for inference (no evaluation):
  python evaluate_distributed_checkpoints.py \
    --model gemma-3-12b-pt \
    --checkpoint_dir training_runs/gemma-3-12b-pt-fsdp/checkpoint-500 \
    --skip_eval
"""

import argparse
import json
import os
import random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

import evaluate
from datasets import Dataset
from huggingface_hub import login
from peft import PeftModel, LoraConfig
import torch
import torch.serialization
import wandb

# Fix for PyTorch 2.6+ weights_only security issue
_original_torch_load = torch.load

def _torch_load_with_weights_only_false(path, *args, **kwargs):
    """Wrapper around torch.load that disables weights_only for checkpoint compatibility."""
    if 'rng_state' in str(path) or 'optimizer' in str(path) or 'scheduler' in str(path):
        kwargs['weights_only'] = False
    return _original_torch_load(path, *args, **kwargs)

torch.load = _torch_load_with_weights_only_false

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Evaluation parameters
MAX_INPUT_TEXT_TOKENS = 2048
MAX_EXTRA_PROMPT_TOKENS = 40
MAX_INPUT_PROMPT_TOKENS = MAX_INPUT_TEXT_TOKENS + MAX_EXTRA_PROMPT_TOKENS
MAX_OUTPUT_SUMMARY_TOKENS = 512
VAL_BATCH_SIZE = 5
VAL_DATA_SIZE = 50
VAL_BEAM_SIZE = 4


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
            padded_input_ids.append([self.tokenizer.pad_token_id] * padding_length + ids)
            attention_mask.append([0] * padding_length + [1] * len(ids))
        
        # Pad labels (RIGHT padding)
        max_label_length = max(len(lbl) for lbl in labels)
        if self.pad_to_multiple_of:
            max_label_length = ((max_label_length + self.pad_to_multiple_of - 1) 
                               // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        padded_labels = []
        for lbl in labels:
            padding_length = max_label_length - len(lbl)
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
        self.generation_max_length = generation_max_length
        self.generation_num_beams = generation_num_beams
        self.eval_data_collator = eval_data_collator
        super().__init__(*args, **kwargs)
        self._processing_class = self.tokenizer
    
    def get_eval_dataloader(self, eval_dataset=None):
        """Override to use a different data collator for evaluation."""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        original_collator = self.data_collator
        if self.eval_data_collator is not None:
            self.data_collator = self.eval_data_collator
        
        dataloader = super().get_eval_dataloader(eval_dataset)
        self.data_collator = original_collator
        
        return dataloader

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        if prediction_loss_only:
            return (None, None, None)

        print('*** evaluation: prediction_step ***')
        torch.cuda.empty_cache()

        if 'input_ids' in inputs:
            input_ids = inputs["input_ids"]
        else:
            raise KeyError("input_ids not found in batch")

        labels = inputs.get("labels")
        
        if labels is not None:
            labels = labels.clone()
            labels[labels == -100] = self._processing_class.pad_token_id

        print('*** evaluation: input_ids (prompt only) ***', input_ids.shape)
        if labels is not None:
            print('*** evaluation: labels (target summary) ***', labels.shape)

        generated_ids = model.generate(
            input_ids=input_ids,
            use_cache=True,
            max_new_tokens=self.generation_max_length,
            num_beams=self.generation_num_beams,
            do_sample=False,
            pad_token_id=self._processing_class.pad_token_id,
            eos_token_id=self._processing_class.eos_token_id,
        )
        
        input_length = input_ids.shape[1]
        generated_ids = generated_ids[:, input_length:]
        
        print('*** evaluation: generated_ids (generated summary only) ***', generated_ids.shape)
        
        torch.cuda.empty_cache()

        loss = None
        
        return (loss, generated_ids, labels)


def load_model_and_peft_checkpoint(
    model_name: str,
    checkpoint_dir: str
):
    """Load base model and PEFT checkpoint for inference.
    
    Args:
        model_name: Base model identifier (e.g., 'google/gemma-2b')
        checkpoint_dir: Path to PEFT checkpoint directory
    
    Returns:
        Loaded model with PEFT adapters
    
    Note:
        HuggingFace token is read from HF_TOKEN or HUGGINGFACE_TOKEN environment variable
    """
    # Get HF token from environment
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    print(f"Loading base model: {model_name}")
    
    # Load base model without quantization, on single GPU
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token
    )
    
    print(f"Loading PEFT checkpoint from: {checkpoint_dir}")
    
    # Load PEFT adapter - this should work in single-GPU mode
    # PeftModel.from_pretrained handles loading the adapter weights
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_dir,
        is_trainable=False  # Set to False for inference only
    )
    
    print("Successfully loaded PEFT checkpoint!")
    model.print_trainable_parameters()
    
    return model


def evaluate_checkpoint(
    model_name: str,
    checkpoint_dir: str,
    val_dataset_path: str,
    output_dir: Optional[str] = None,
    max_input_text_tokens: int = MAX_INPUT_TEXT_TOKENS,
    max_extra_prompt_tokens: int = MAX_EXTRA_PROMPT_TOKENS,
    max_output_summary_tokens: int = MAX_OUTPUT_SUMMARY_TOKENS,
    val_batch_size: int = VAL_BATCH_SIZE,
    val_data_size: int = VAL_DATA_SIZE,
    val_beam_size: int = VAL_BEAM_SIZE
):
    """Load a PEFT checkpoint and run evaluation.
    
    Args:
        model_name: Base model identifier
        checkpoint_dir: Path to PEFT checkpoint
        val_dataset_path: Path to validation dataset (JSONL)
        output_dir: Optional directory to save evaluation results
        max_input_text_tokens: Maximum tokens for input text
        max_extra_prompt_tokens: Maximum extra tokens for input prompt
        max_output_summary_tokens: Maximum tokens for output summary
        val_batch_size: Validation batch size per device
        val_data_size: Number of examples to use for validation
        val_beam_size: Beam size for validation generation
    
    Note:
        HuggingFace token is read from HF_TOKEN or HUGGINGFACE_TOKEN environment variable
    """
    
    # Get HF token from environment
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        print("HuggingFace token found in environment")
        os.environ["HF_TOKEN"] = hf_token
    else:
        print("WARNING: No HuggingFace token found in environment")
    
    # Initialize Weights & Biases for logging evaluation results
    checkpoint_name = os.path.basename(checkpoint_dir)
    print("Initializing Weights & Biases...")
    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(
        project="lm-finetuning",  # Same project as training
        name=f"eval_{checkpoint_name}",
        tags=["evaluation", model_name.split('/')[-1]],
        config={
            "model_name": model_name,
            "checkpoint_dir": checkpoint_dir,
            "val_dataset": val_dataset_path,
            "val_batch_size": val_batch_size,
            "val_beam_size": val_beam_size,
        }
    )
    
    def compute_metrics(eval_pred):
        print('*** evaluation: compute_metrics ***')
        
        # Load ROUGE metric (lazy loading after cache paths are set)
        rouge = evaluate.load("rouge")
        
        preds, labels = eval_pred
        print('*** evaluation: preds ***', preds.shape)
        print('*** evaluation: labels ***', labels.shape)
        
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        vocab_size = tokenizer.vocab_size
        print(f'*** Vocab size: {vocab_size} ***')
        
        preds = np.clip(preds, 0, vocab_size - 1)
        labels = np.clip(labels, 0, vocab_size - 1)
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        
        # Print first 5 examples
        num_examples_to_show = min(5, len(decoded_preds))
        if num_examples_to_show > 0:
            print(f'\n*** Showing first {num_examples_to_show} examples ***')
            for i in range(num_examples_to_show):
                print(f'\n--- Example {i+1} ---')
                print(f'Prediction: {decoded_preds[i][:200]}...')
                print(f'Reference:  {decoded_labels[i][:200]}...')
            print()

        scores = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        print('*** evaluation: computed_metrics ***', scores)
        return {k: v * 100 for k, v in scores.items()}

    
    # Token is already set in environment from above
    # No need to call login() separately

    # Load tokenizer
    print(f"Loading tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token if hf_token else None
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = 'left'

    # Load model with PEFT checkpoint
    model = load_model_and_peft_checkpoint(model_name, checkpoint_dir)

    # Load validation dataset
    print(f"Loading validation dataset from: {val_dataset_path}")
    val_data = []
    with open(val_dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            val_data.append(json.loads(line))

    # Create validation dataset (matching wandb_finetune.py)
    val_df = pd.DataFrame(val_data)
    val_df = val_df[val_df['output'].notna()]  # Filter out null outputs
    val_df = val_df.sample(n=min(val_data_size, len(val_df)))  # Sample requested size
    
    # Validate data quality
    assert val_df['input'].apply(lambda x: x is not None and x != '').all()
    assert val_df['input'].notna().all()
    assert val_df['output'].apply(lambda x: x is not None and x != '').all()
    assert val_df['output'].notna().all()
    
    val_dataset = Dataset.from_pandas(val_df)
    print(f'*** validation dataset size: {len(val_dataset)} examples ***')

    def format_example_eval(example):
        prompt = f"Oppgave: Oppsummer følgende tekst:\n\n###\n\n{example['input']}\n\n###\n\nOppsummering:\n\n###\n\n"
        return {
            "prompt": prompt,
            "target_summary": example['output']
        }

    def tokenize_function_eval(examples):
        max_input_prompt_tokens = max_input_text_tokens + max_extra_prompt_tokens
        tokenized_prompts = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=max_input_prompt_tokens,
            padding=False
        )
        tokenized_targets = tokenizer(
            examples["target_summary"],
            truncation=True,
            max_length=max_output_summary_tokens,
            padding=False
        )
        tokenized_prompts["labels"] = tokenized_targets["input_ids"]
        return tokenized_prompts

    formatted_val_dataset = val_dataset.map(format_example_eval)
    tokenized_val_dataset = formatted_val_dataset.map(tokenize_function_eval, batched=True)
    
    eval_data_collator = EvalDataCollator(tokenizer=tokenizer)

    # Set up evaluation-only training args
    if output_dir is None:
        output_dir = os.path.join(checkpoint_dir, "eval_results")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=val_batch_size,
        fp16=False,
        dataloader_pin_memory=False,
        report_to="none",
    )

    # Initialize Trainer for evaluation only
    trainer = CausalLMTrainer(
        generation_max_length=max_output_summary_tokens,
        generation_num_beams=val_beam_size,
        eval_data_collator=eval_data_collator,
        model=model,
        args=training_args,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Run evaluation
    print("\n" + "=" * 70)
    print("Running evaluation on checkpoint...")
    print("=" * 70 + "\n")
    
    eval_results = trainer.evaluate()
    
    print("\n" + "=" * 70)
    print("Evaluation Results:")
    print("=" * 70)
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")
    print("=" * 70 + "\n")
    
    # Log results to Weights & Biases
    if wandb.run is not None:
        wandb_metrics = {}
        for key, value in eval_results.items():
            if 'rouge' in key.lower():
                wandb_metrics[f"checkpoint_eval/{key}"] = value
            else:
                wandb_metrics[f"checkpoint_eval/{key}"] = value
        wandb.log(wandb_metrics)
        print("Results logged to Weights & Biases")
    
    # Save results to file
    results_file = os.path.join(output_dir, "eval_results.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    # Finish W&B run
    wandb.finish()
    
    return eval_results, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Load PEFT checkpoint from distributed training for evaluation/inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Set environment first:
  export HF_TOKEN=your_huggingface_token
  
  # Evaluate a checkpoint:
  python evaluate_distributed_checkpoints.py \\
    --model gemma-3-12b-pt \\
    --checkpoint_dir training_runs/gemma-3-12b-pt-fsdp/checkpoint-100 \\
    --val_dataset /cluster/projects/nn12075k/shared/datasets/dataset_43221_examples/processed_data_val.jsonl

  # Load without evaluation (inference only):
  python evaluate_distributed_checkpoints.py \\
    --model gemma-3-12b-pt \\
    --checkpoint_dir training_runs/gemma-3-12b-pt-fsdp/checkpoint-100 \\
    --skip_eval
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       choices=['viking-7b', 'gemma-2b', 'mt5', 'gemma-7b', 'gemma-3-12b-pt'],
                       help='Base model that was fine-tuned')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Path to PEFT checkpoint directory (e.g., training_runs/gemma-3-12b-pt-fsdp/checkpoint-100)')
    parser.add_argument('--val_dataset', type=str, default=None,
                       help='Path to validation dataset (JSONL format). Required unless --skip_eval is used.')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for evaluation results (default: checkpoint_dir/eval_results)')
    # HF_TOKEN is now read from environment variable (HF_TOKEN or HUGGINGFACE_TOKEN)
    parser.add_argument('--skip_eval', action='store_true',
                       help='Skip evaluation, only load the model')
    
    # Hyperparameters (match finetune.py)
    parser.add_argument('--max_input_text_tokens', type=int, default=MAX_INPUT_TEXT_TOKENS,
                       help=f'Maximum tokens for input text (default: {MAX_INPUT_TEXT_TOKENS})')
    parser.add_argument('--max_extra_prompt_tokens', type=int, default=MAX_EXTRA_PROMPT_TOKENS,
                       help=f'Maximum extra tokens for input prompt (default: {MAX_EXTRA_PROMPT_TOKENS})')
    parser.add_argument('--max_output_summary_tokens', type=int, default=MAX_OUTPUT_SUMMARY_TOKENS,
                       help=f'Maximum tokens for output summary (default: {MAX_OUTPUT_SUMMARY_TOKENS})')
    parser.add_argument('--val_batch_size', type=int, default=VAL_BATCH_SIZE,
                       help=f'Validation batch size per device (default: {VAL_BATCH_SIZE})')
    parser.add_argument('--val_data_size', type=int, default=VAL_DATA_SIZE,
                       help=f'Number of examples to use for validation (default: {VAL_DATA_SIZE})')
    parser.add_argument('--val_beam_size', type=int, default=VAL_BEAM_SIZE,
                       help=f'Beam size for validation generation (default: {VAL_BEAM_SIZE})')

    args = parser.parse_args()
    
    # Validate arguments
    if not args.skip_eval and args.val_dataset is None:
        parser.error("--val_dataset is required when evaluation is enabled (use --skip_eval to skip evaluation)")

    # Model mapping
    model_mapping = {
        'viking-7b': 'LumiOpen/Viking-7B',
        'gemma-2b': 'google/gemma-2b',
        'mt5': 'google/mt5-base',
        'gemma-7b': 'google/gemma-7b',
        'gemma-3-12b-pt': 'google/gemma-3-12b-pt'
    }

    model_name = model_mapping[args.model]

    if args.skip_eval:
        # Just load the model
        print("Loading model without evaluation...")
        # Token is read from environment in load_model_and_peft_checkpoint
        model = load_model_and_peft_checkpoint(model_name, args.checkpoint_dir)
        print("Model loaded successfully! Ready for inference.")
    else:
        # Run evaluation
        evaluate_checkpoint(
            model_name=model_name,
            checkpoint_dir=args.checkpoint_dir,
            val_dataset_path=args.val_dataset,
            output_dir=args.output_dir,
            max_input_text_tokens=args.max_input_text_tokens,
            max_extra_prompt_tokens=args.max_extra_prompt_tokens,
            max_output_summary_tokens=args.max_output_summary_tokens,
            val_batch_size=args.val_batch_size,
            val_data_size=args.val_data_size,
            val_beam_size=args.val_beam_size
        )
