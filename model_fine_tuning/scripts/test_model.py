# test_model.py
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    pipeline
)
import torch
import json
import argparse
import pandas as pd
from rouge_score import rouge_scorer
import numpy as np

def load_model(model_path, model_type):
    """Load the fine-tuned model and tokenizer"""
    print(f"Loading model from {model_path}...")
    
    if model_type == "mt5":
        tokenizer = MT5Tokenizer.from_pretrained(model_path)
        model = MT5ForConditionalGeneration.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to GPU")
    
    return model, tokenizer

def generate_summary(model, tokenizer, text, model_type, max_length=512):
    """Generate a summary for the given text"""
    # Format the input based on model type
    if model_type == "mt5":
        # For mT5, use a simpler prompt as it's a seq2seq model
        input_text = f"summarize: {text}"
    else:
        # For causal models, use the same format as during training
        input_text = f"### Oppgave: Oppsummer følgende tekst\n{text}\n\n### Svar:"
    
    # Tokenize input
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=1024)
    
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    
    # Generate summary
    with torch.no_grad():
        if model_type == "mt5":
            outputs = model.generate(
                inputs, 
                max_length=max_length, 
                num_beams=4, 
                early_stopping=True
            )
        else:
            outputs = model.generate(
                inputs, 
                max_length=max_length, 
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

def evaluate_model(model, tokenizer, test_data, model_type, num_samples=10):
    """Evaluate the model on test data"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = []
    
    print(f"Evaluating on {min(num_samples, len(test_data))} samples...")
    
    for i, example in enumerate(test_data[:num_samples]):
        print(f"\n--- Sample {i+1} ---")
        print("Input text snippet:", example['input'][:200] + "...")
        print("Reference summary:", example['output'])
        
        # Generate summary
        generated_summary = generate_summary(model, tokenizer, example['input'], model_type)
        print("Generated summary:", generated_summary)
        
        # Calculate ROUGE scores
        scores = scorer.score(example['output'], generated_summary)
        rouge_scores.append(scores)
        
        print("ROUGE scores:")
        for key in scores:
            print(f"  {key}: {scores[key].fmeasure:.3f}")
    
    # Calculate average ROUGE scores
    avg_scores = {key: {'f1': 0, 'precision': 0, 'recall': 0} for key in ['rouge1', 'rouge2', 'rougeL']}
    
    for scores in rouge_scores:
        for key in scores:
            avg_scores[key]['f1'] += scores[key].fmeasure
            avg_scores[key]['precision'] += scores[key].precision
            avg_scores[key]['recall'] += scores[key].recall
    
    for key in avg_scores:
        for metric in avg_scores[key]:
            avg_scores[key][metric] /= len(rouge_scores)
    
    print("\n=== Average ROUGE Scores ===")
    for key in avg_scores:
        print(f"{key}: F1={avg_scores[key]['f1']:.3f}, P={avg_scores[key]['precision']:.3f}, R={avg_scores[key]['recall']:.3f}")
    
    return avg_scores

def interactive_test(model, tokenizer, model_type):
    """Interactive mode for testing the model"""
    print("\n=== Interactive Mode ===")
    print("Enter text to summarize (type 'quit' to exit):")
    
    while True:
        text = input("\nInput text: ")
        if text.lower() == 'quit':
            break
        
        summary = generate_summary(model, tokenizer, text, model_type)
        print(f"\nGenerated summary: {summary}")

def main():
    parser = argparse.ArgumentParser(description='Test a fine-tuned summarization model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the fine-tuned model')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['causal', 'mt5'],
                       help='Type of model (causal for Gemma/Viking, mt5 for mT5)')
    parser.add_argument('--test_data', type=str,
                       help='Path to test data JSONL file')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to evaluate on')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.model_type)
    
    # Test data evaluation
    if args.test_data:
        # Load test data
        test_data = []
        with open(args.test_data, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line))
        
        # Evaluate model
        evaluate_model(model, tokenizer, test_data, args.model_type, args.num_samples)
    
    # Interactive mode
    if args.interactive or not args.test_data:
        interactive_test(model, tokenizer, args.model_type)

if __name__ == "__main__":
    main()
