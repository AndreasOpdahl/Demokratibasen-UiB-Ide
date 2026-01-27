#!/usr/bin/env python3
"""
Run NLI-based faithfulness evaluation on a subset of predictions.

This script allows you to run NLI faithfulness metrics separately on a subset
of examples from the evaluation predictions JSONL file.

Usage:
    # Run on first 100 examples from a checkpoint's predictions file:
    python run_nli_faithfulness_subset.py \
        --predictions_file models/gemma-7b/all_eval_results/checkpoint-100-inputs-refs-preds.jsonl \
        --subset_size 100 \
        --output_file models/gemma-7b/all_eval_results/checkpoint-100-nli-faithfulness.json

    # Run on all examples:
    python run_nli_faithfulness_subset.py \
        --predictions_file models/gemma-7b/all_eval_results/checkpoint-100-inputs-refs-preds.jsonl \
        --output_file models/gemma-7b/all_eval_results/checkpoint-100-nli-faithfulness.json
"""

import argparse
import json
import os
import random
from pathlib import Path

from extended_evaluation import NLIFaithfulnessGate


def load_predictions_from_jsonl(predictions_file: str, subset_size: int = None, random_seed: int = 42):
    """Load input texts and predictions from JSONL file.
    
    Args:
        predictions_file: Path to JSONL file with predictions
        subset_size: Number of examples to use (None = all)
        random_seed: Random seed for subset sampling
    
    Returns:
        Tuple of (input_texts, prediction_texts)
    """
    input_texts = []
    prediction_texts = []
    
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            input_texts.append(entry.get("input_text", ""))
            prediction_texts.append(entry.get("prediction", ""))
    
    if subset_size and subset_size < len(input_texts):
        # Sample subset
        random.seed(random_seed)
        indices = random.sample(range(len(input_texts)), subset_size)
        input_texts = [input_texts[i] for i in indices]
        prediction_texts = [prediction_texts[i] for i in indices]
        print(f"Sampled {subset_size} examples from {len(input_texts) + len(indices) - subset_size} total examples")
    else:
        print(f"Using all {len(input_texts)} examples")
    
    return input_texts, prediction_texts


def main():
    parser = argparse.ArgumentParser(
        description='Run NLI-based faithfulness evaluation on a subset of predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on first 100 examples:
  python run_nli_faithfulness_subset.py \\
    --predictions_file models/gemma-7b/all_eval_results/checkpoint-100-inputs-refs-preds.jsonl \\
    --subset_size 100

  # Run on all examples:
  python run_nli_faithfulness_subset.py \\
    --predictions_file models/gemma-7b/all_eval_results/checkpoint-100-inputs-refs-preds.jsonl

  # Specify output file:
  python run_nli_faithfulness_subset.py \\
    --predictions_file models/gemma-7b/all_eval_results/checkpoint-100-inputs-refs-preds.jsonl \\
    --subset_size 50 \\
    --output_file results/nli_checkpoint_100.json
        """
    )
    
    parser.add_argument('--predictions_file', type=str, required=True,
                       help='Path to JSONL file with predictions (from evaluation)')
    parser.add_argument('--subset_size', type=int, default=None,
                       help='Number of examples to evaluate (default: all)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output JSON file path (default: predictions_file with -nli-faithfulness.json suffix)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for subset sampling (default: 42)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.predictions_file):
        print(f"ERROR: Predictions file does not exist: {args.predictions_file}")
        return 1
    
    # Determine output file
    if args.output_file is None:
        base_path = Path(args.predictions_file)
        output_file = base_path.parent / f"{base_path.stem}-nli-faithfulness.json"
    else:
        output_file = Path(args.output_file)
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("NLI Faithfulness Evaluation (Subset)")
    print("=" * 70)
    print(f"Input file: {args.predictions_file}")
    print(f"Output file: {output_file}")
    if args.subset_size:
        print(f"Subset size: {args.subset_size} examples")
    print("=" * 70)
    print()
    
    # Load predictions
    print("Loading predictions...")
    input_texts, prediction_texts = load_predictions_from_jsonl(
        args.predictions_file,
        subset_size=args.subset_size,
        random_seed=args.random_seed
    )
    
    if len(input_texts) == 0:
        print("ERROR: No examples found in predictions file")
        return 1
    
    # Initialize NLI gate
    print("Initializing NLI model (this may take a minute on first run)...")
    gate = NLIFaithfulnessGate()
    
    # Run evaluation
    print(f"\nRunning NLI faithfulness evaluation on {len(input_texts)} examples...")
    print("This may take a while (~4.5 seconds per example)...")
    print()
    
    faithfulness_results = gate.eval_faithfulness(input_texts, prediction_texts)
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(faithfulness_results, f, indent=2, ensure_ascii=False, default=str)
    
    # Print summary
    print("\n" + "=" * 70)
    print("NLI Faithfulness Results:")
    print("=" * 70)
    print(f"Mean entailment score: {faithfulness_results['mean_entailment_score']:.4f}")
    print(f"Min entailment score: {faithfulness_results['min_entailment_score']:.4f}")
    print(f"Max contradiction score: {faithfulness_results['max_contradiction_score']:.4f}")
    print(f"Mean outlier rate: {faithfulness_results['mean_outlier_rate']:.4f}")
    print(f"Ratio passed: {faithfulness_results['ratio_passed']:.4f}")
    print("=" * 70)
    print(f"\nResults saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
