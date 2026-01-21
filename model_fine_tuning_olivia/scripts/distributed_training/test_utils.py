import os
import torch
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM, 
    MT5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
import datetime
import re
import json
import pandas as pd
from tqdm import tqdm
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import random  # ADDED FOR CONSISTENCY

# Global WANDB availability check - MOVED TO TOP LEVEL
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
        # DON'T load evaluate here - load it lazily when needed
        self._rouge = None  # Initialize as None

    @property
    def rouge(self):
        """Lazy load evaluate only when needed"""
        if self._rouge is None:
            import evaluate  # Import here to avoid circular imports
            self._rouge = evaluate.load('rouge')
        return self._rouge

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

        # Compute ROUGE scores using the lazy-loaded property
        if predictions and references:
            rouge_results = self.rouge.compute(  # This will trigger the lazy load
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
    import evaluate
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
    # Save test results to file with wandb name or timestamp
    if WANDB_AVAILABLE and wandb.run is not None:
        # Get wandb run name and sanitize it for filename
        wandb_name = wandb.run.name
        wandb_name_clean = re.sub(r'[^\w\-_.]', '_', wandb_name)
        predictions_file = os.path.join(output_dir, f"all_predictions_{wandb_name_clean}.json")
    else:
        # Use timestamp if wandb not available
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_file = os.path.join(output_dir, f"all_predictions_{timestamp}.json")

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
