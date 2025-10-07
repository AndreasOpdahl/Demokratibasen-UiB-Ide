# evaluate_model.py
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    MT5Tokenizer,
    MT5ForConditionalGeneration
)
import torch
import json
import argparse
import pandas as pd
from rouge_score import rouge_scorer
import numpy as np
from collections import defaultdict

def load_model(model_path, model_type):
    """Load the fine-tuned model and tokenizer"""
    print(f"Loading model from {model_path}...")
    
    if model_type == "mt5":
        tokenizer = MT5Tokenizer.from_pretrained(model_path)
        model = MT5ForConditionalGeneration.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
    
    # Move to GPU if available
    #if torch.cuda.is_available():
    #    model = model.cuda()
    #    print("Model moved to GPU")
    
    return model, tokenizer

def generate_summary(model, tokenizer, text, model_type, max_new_tokens=200):
    """Generate a summary for the given text"""
    # Format the input based on model type
    if model_type == "mt5":
        # For mT5, use a simpler prompt as it's a seq2seq model
        input_text = f"summarize: {text}"
    else:
        # For causal models, use the same format as during training
        input_text = f"### Oppgave: Oppsummer følgende tekst\n{text}\n\n### Svar:"

    # Tokenize input with appropriate max_length
    # Use a larger max_length to accommodate long inputs
    inputs = tokenizer.encode(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024  # Increased from previous value
    )

    #if torch.cuda.is_available():
    #    inputs = inputs.cuda()

    # Generate summary with max_new_tokens instead of max_length
    with torch.no_grad():
        if model_type == "mt5":
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,  # Use max_new_tokens instead
                num_beams=4,
                early_stopping=True
            )
        else:
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,  # Use max_new_tokens instead
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )

    # Decode the generated summary
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # For causal models, remove the prompt from the output if it's included
    if model_type != "mt5" and "### Svar:" in summary:
        summary = summary.split("### Svar:")[-1].strip()

    return summary


def calculate_rouge_scores(scorer, text1, text2):
    """Calculate ROUGE scores between two texts"""
    return scorer.score(text1, text2)

def evaluate_model_comprehensive(model, tokenizer, test_data, model_type, num_samples=10):
    """Comprehensive evaluation with multiple ROUGE comparisons"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Initialize score trackers for all comparisons
    scores_oi = defaultdict(list)  # Output vs Input
    scores_or = defaultdict(list)  # Output vs Reference
    scores_ri = defaultdict(list)  # Reference vs Input
    
    print(f"Evaluating on {min(num_samples, len(test_data))} samples...")
    
    for i, example in enumerate(test_data[:num_samples]):
        print(f"\n--- Sample {i+1} ---")
        print("Input text snippet:", example['input'][:200] + "...")
        print("Reference summary:", example['output'])
        
        # Generate summary
        generated_summary = generate_summary(model, tokenizer, example['input'], model_type)
        print("Generated summary:", generated_summary)
        
        # Calculate ROUGE scores for all comparisons
        # 1. Output vs Input (how much of the input is captured in the output)
        oi_scores = calculate_rouge_scores(scorer, example['input'], generated_summary)
        for key in oi_scores:
            scores_oi[key].append(oi_scores[key].fmeasure)
        
        # 2. Output vs Reference (how close the output is to the reference)
        or_scores = calculate_rouge_scores(scorer, example['output'], generated_summary)
        for key in or_scores:
            scores_or[key].append(or_scores[key].fmeasure)
        
        # 3. Reference vs Input (how much of the input is captured in the reference)
        ri_scores = calculate_rouge_scores(scorer, example['input'], example['output'])
        for key in ri_scores:
            scores_ri[key].append(ri_scores[key].fmeasure)
        
        # Print individual sample scores
        print("\nROUGE Scores for this sample:")
        print("Output vs Input:")
        for key in oi_scores:
            print(f"  {key}: {oi_scores[key].fmeasure:.3f}")
        
        print("Output vs Reference:")
        for key in or_scores:
            print(f"  {key}: {or_scores[key].fmeasure:.3f}")
            
        print("Reference vs Input:")
        for key in ri_scores:
            print(f"  {key}: {ri_scores[key].fmeasure:.3f}")
    
    # Calculate average ROUGE scores for each comparison
    def calculate_averages(scores_dict):
        averages = {}
        for key in scores_dict:
            averages[key] = np.mean(scores_dict[key])
        return averages
    
    avg_oi = calculate_averages(scores_oi)
    avg_or = calculate_averages(scores_or)
    avg_ri = calculate_averages(scores_ri)
    
    # Print comprehensive results
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*60)
    
    print("\n1. OUTPUT vs INPUT (How much of the input is captured in generated summaries):")
    for key in avg_oi:
        print(f"   {key}: {avg_oi[key]:.3f}")
    
    print("\n2. OUTPUT vs REFERENCE (How close generated summaries are to reference summaries):")
    for key in avg_or:
        print(f"   {key}: {avg_or[key]:.3f}")
    
    print("\n3. REFERENCE vs INPUT (How much of the input is captured in reference summaries):")
    for key in avg_ri:
        print(f"   {key}: {avg_ri[key]:.3f}")
    
    # Calculate and print performance ratios
    print("\n4. PERFORMANCE RATIOS (Output vs Reference / Reference vs Input):")
    for key in avg_or:
        if avg_ri[key] > 0:
            ratio = avg_or[key] / avg_ri[key]
            print(f"   {key}: {ratio:.3f} ({(ratio*100):.1f}% of ideal performance)")
        else:
            print(f"   {key}: N/A (Reference vs Input score is 0)")
    
    return {
        'output_vs_input': avg_oi,
        'output_vs_reference': avg_or,
        'reference_vs_input': avg_ri
    }

def save_results_to_csv(results, filename):
    """Save evaluation results to a CSV file"""
    df_data = []
    
    for comparison_type, scores in results.items():
        for rouge_type, score in scores.items():
            df_data.append({
                'comparison': comparison_type,
                'rouge_type': rouge_type,
                'score': score
            })
    
    df = pd.DataFrame(df_data)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of a fine-tuned summarization model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the fine-tuned model')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['causal', 'mt5'],
                       help='Type of model (causal for Gemma/Viking, mt5 for mT5)')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data JSONL file')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to evaluate on')
    parser.add_argument('--output_csv', type=str,
                       help='Path to save results as CSV')
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.model_type)
    
    # Load test data
    test_data = []
    with open(args.test_data, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    # Evaluate model
    results = evaluate_model_comprehensive(
        model, tokenizer, test_data, args.model_type, args.num_samples
    )
    
    # Save results if requested
    if args.output_csv:
        save_results_to_csv(results, args.output_csv)

if __name__ == "__main__":
    main()
