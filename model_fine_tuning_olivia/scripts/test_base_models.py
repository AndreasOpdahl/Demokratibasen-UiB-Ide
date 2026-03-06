#!/usr/bin/env python3
"""
Test script to query base models (before fine-tuning) and save predictions.

This script tests that all chat-based models are correctly configured and can
generate responses using their proper prompt formats. It's useful for:
1. Verifying chat templates are set correctly
2. Ensuring prompt formats match expected formats
3. Testing base model behavior before fine-tuning
4. Saving predictions for comparison

Usage:
    python test_base_models.py \
        --models="normistral-7b-instruct,norskgpt-llama3-8b,llama-3.1-8b-instruct" \
        --test_dataset data/output/new_processed_data_val.jsonl \
        --output_dir base_model_predictions \
        --hf_token YOUR_TOKEN \
        --num_examples 10
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

# Disable tokenizer parallelism to avoid fork warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add scripts directory to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from model_configs import (
    get_model_config,
    get_model_config_by_hf_name,
    get_model_name_mapping,
    get_doc_type_norwegian
)
from utils.formatting import format_eval_example
from utils.dataset_loading import load_jsonl_dataset


def load_base_model(model_name: str, hf_token: Optional[str] = None, use_multi_gpu: bool = False):
    """Load base model (without fine-tuning) for testing.
    
    Args:
        model_name: HuggingFace model identifier
        hf_token: Hugging Face authentication token
        use_multi_gpu: If True, use device_map="auto" for multi-GPU
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\n{'='*70}")
    print(f"Loading base model: {model_name}")
    print(f"{'='*70}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token if hf_token else None
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Left padding for generation
    
    # Set chat template if needed (same logic as training script)
    model_config = get_model_config_by_hf_name(model_name)
    if model_config:
        template_type = model_config.prompt_config.template_type
        
        if template_type in ['mistral', 'llama2', 'llama3', 'llama3.1', 'chatml']:
            if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
                print(f"Setting chat template for {model_name} (template_type: {template_type})...")
                
                official_model_map = {
                    'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
                    'llama2': 'meta-llama/Llama-2-7b-chat-hf',
                    'llama3': 'meta-llama/Meta-Llama-3-8B-Instruct',
                    'llama3.1': 'meta-llama/Llama-3.1-8B-Instruct',
                    'chatml': 'HuggingFaceH4/zephyr-7b-beta',
                }
                
                template_set = False
                if template_type in official_model_map:
                    try:
                        from transformers import AutoTokenizer as OfficialTokenizer
                        official_model = official_model_map[template_type]
                        official_tokenizer = OfficialTokenizer.from_pretrained(
                            official_model,
                            token=hf_token if hf_token else None
                        )
                        if hasattr(official_tokenizer, 'chat_template') and official_tokenizer.chat_template:
                            tokenizer.chat_template = official_tokenizer.chat_template
                            template_set = True
                            print(f"✓ Set {template_type} chat template from official model: {official_model}")
                    except Exception as e:
                        print(f"⚠ Could not load template from {official_model_map.get(template_type, 'official model')}: {e}")
                
                # Fallback to standard formats
                if not template_set:
                    if template_type == 'mistral':
                        mistral_template = (
                            "{%- for message in messages %}"
                            "{%- if message['role'] == 'system' %}"
                            "{{ message['content'] }}"
                            "{%- elif message['role'] == 'user' %}"
                            "<s>[INST] {{ message['content'] }} [/INST]"
                            "{%- elif message['role'] == 'assistant' %}"
                            " {{ message['content'] }}</s>"
                            "{%- endif %}"
                            "{%- endfor %}"
                        )
                        tokenizer.chat_template = mistral_template
                        print(f"✓ Set Mistral chat template using standard format (fallback)")
                    elif template_type == 'llama2':
                        llama2_template = (
                            "{%- for message in messages %}"
                            "{%- if message['role'] == 'system' %}"
                            "<<SYS>>\n{{ message['content'] }}\n<</SYS>>\n\n"
                            "{%- elif message['role'] == 'user' %}"
                            "[INST] {{ message['content'] }} [/INST]"
                            "{%- elif message['role'] == 'assistant' %}"
                            " {{ message['content'] }}"
                            "{%- endif %}"
                            "{%- endfor %}"
                        )
                        tokenizer.chat_template = llama2_template
                        print(f"✓ Set Llama-2 chat template using standard format (fallback)")
                    elif template_type in ['llama3', 'llama3.1']:
                        llama3_template = (
                            "{% for message in messages %}"
                            "{% if message['role'] == 'user' %}"
                            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
                            "{% elif message['role'] == 'assistant' %}"
                            "<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
                            "{% endif %}"
                            "{% endfor %}"
                        )
                        tokenizer.chat_template = llama3_template
                        print(f"✓ Set Llama-3 chat template using standard format (fallback)")
                    elif template_type == 'chatml':
                        chatml_template = (
                            "{% for message in messages %}"
                            "{% if message['role'] == 'system' %}"
                            "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
                            "{% elif message['role'] == 'user' %}"
                            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
                            "{% elif message['role'] == 'assistant' %}"
                            "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
                            "{% endif %}"
                            "{% endfor %}"
                        )
                        tokenizer.chat_template = chatml_template
                        print(f"✓ Set ChatML chat template using standard format (fallback)")
    
    # Load model
    print("Loading model...")
    num_gpus = torch.cuda.device_count()
    
    if use_multi_gpu and num_gpus > 1:
        print(f"Using model parallelism across {num_gpus} GPUs")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token if hf_token else None,
            low_cpu_mem_usage=True,
        )
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            token=hf_token if hf_token else None,
            low_cpu_mem_usage=True,
        )
    
    model.eval()  # Set to evaluation mode
    
    print(f"✓ Model loaded successfully")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Device: {next(model.parameters()).device if hasattr(model, 'parameters') else 'unknown'}")
    
    return model, tokenizer


def test_model_generation(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    model_name: str,
    test_examples: List[Dict[str, Any]],
    max_new_tokens: int = 200,
    num_examples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Test model generation on examples.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        model_name: Model identifier
        test_examples: List of test examples
        max_new_tokens: Maximum tokens to generate
        num_examples: Number of examples to test (None = all)
    
    Returns:
        List of predictions with input, prompt, and generated text
    """
    if num_examples is not None:
        test_examples = test_examples[:num_examples]
    
    print(f"\n{'='*70}")
    print(f"Testing generation on {len(test_examples)} examples")
    print(f"{'='*70}")
    
    results = []
    model_config = get_model_config_by_hf_name(model_name)
    
    for i, example in enumerate(test_examples):
        print(f"\n--- Example {i+1}/{len(test_examples)} ---")
        
        # Format prompt using model's prompt config
        formatted_example = format_eval_example(example, model_name, tokenizer=tokenizer)
        prompt = formatted_example.get('prompt', '')
        input_text = example.get('input', '')
        reference = example.get('output', '')
        
        print(f"Input text (first 100 chars): {input_text[:100]}...")
        print(f"Prompt (first 200 chars): {prompt[:200]}...")
        
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        print("Generating...")
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy decoding for consistency
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            except Exception as e:
                print(f"⚠ Generation failed: {e}")
                results.append({
                    "example_id": i,
                    "input_text": input_text,
                    "prompt": prompt,
                    "reference": reference,
                    "prediction": f"[ERROR: {str(e)}]",
                    "error": str(e)
                })
                continue
        
        # Decode generated text
        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0][input_length:]
        
        # For Alpaca format, extract only after "Response:" marker
        # This handles cases where model might include the instruction prompt
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Check if model included "Response:" in output (Alpaca format)
        if "Response:" in generated_text:
            response_pos = generated_text.find("Response:")
            if response_pos >= 0:
                # Extract only after "Response:"
                after_response = generated_text[response_pos + len("Response:"):].strip()
                if after_response:
                    generated_text = after_response
                    print(f"  Extracted content after 'Response:' marker")
        
        # Also check if model included "Instruction:" (model copying input)
        if generated_text.strip().startswith("Instruction:"):
            print(f"  ⚠ Model output starts with 'Instruction:' - likely copying input")
            # Try to find "Response:" after the instruction
            response_pos = generated_text.find("Response:")
            if response_pos >= 0:
                after_response = generated_text[response_pos + len("Response:"):].strip()
                if after_response:
                    generated_text = after_response
                    print(f"  Extracted content after 'Response:' marker")
        
        # Clean up generated text (remove special tokens, etc.)
        generated_text = generated_text.strip()
        
        print(f"Generated (first 200 chars): {generated_text[:200]}...")
        print(f"Reference (first 200 chars): {reference[:200]}...")
        
        results.append({
            "example_id": i,
            "input_text": input_text,
            "prompt": prompt,
            "reference": reference,
            "prediction": generated_text,
        })
    
    return results


def save_predictions(results: List[Dict[str, Any]], output_dir: str, model_short_name: str):
    """Save predictions to JSONL file.
    
    Args:
        results: List of prediction results
        output_dir: Output directory
        model_short_name: Short model name for filename
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{model_short_name}_base_predictions_{timestamp}.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Saved {len(results)} predictions to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Test base models (before fine-tuning) to verify prompt formatting and generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test a single model
  python test_base_models.py \\
    --models="normistral-7b-instruct" \\
    --test_dataset data/output/new_processed_data_val.jsonl \\
    --output_dir base_model_predictions \\
    --hf_token YOUR_TOKEN \\
    --num_examples 5

  # Test multiple models
  python test_base_models.py \\
    --models="normistral-7b-instruct,norskgpt-llama3-8b,llama-3.1-8b-instruct" \\
    --test_dataset data/output/new_processed_data_val.jsonl \\
    --output_dir base_model_predictions \\
    --hf_token YOUR_TOKEN \\
    --num_examples 10 \\
    --use_multi_gpu
        """
    )
    
    parser.add_argument('--models', type=str, required=True,
                       help='Comma-separated list of model short names (e.g., "normistral-7b-instruct,norskgpt-llama3-8b")')
    parser.add_argument('--test_dataset', type=str, required=True,
                       help='Path to test dataset (JSONL format)')
    parser.add_argument('--output_dir', type=str, default='base_model_predictions',
                       help='Output directory for predictions (default: base_model_predictions)')
    parser.add_argument('--hf_token', type=str, default=None,
                       help='Hugging Face authentication token (or set HF_TOKEN env var)')
    parser.add_argument('--num_examples', type=int, default=10,
                       help='Number of examples to test per model (default: 10)')
    parser.add_argument('--max_new_tokens', type=int, default=200,
                       help='Maximum tokens to generate (default: 200)')
    parser.add_argument('--use_multi_gpu', action='store_true',
                       help='Use model parallelism across multiple GPUs')
    
    args = parser.parse_args()
    
    # Get HF token from env if not provided
    if args.hf_token is None:
        args.hf_token = os.environ.get('HF_TOKEN')
    
    # Parse models
    model_names = [m.strip() for m in args.models.split(',')]
    
    # Get model name mapping
    model_mapping = get_model_name_mapping()
    
    # Validate models
    valid_models = []
    for model_short_name in model_names:
        if model_short_name not in model_mapping:
            print(f"⚠ Warning: Unknown model '{model_short_name}', skipping")
            continue
        valid_models.append(model_short_name)
    
    if not valid_models:
        print("ERROR: No valid models found")
        return 1
    
    print("="*70)
    print("Base Model Testing")
    print("="*70)
    print(f"Models to test: {', '.join(valid_models)}")
    print(f"Test dataset: {args.test_dataset}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of examples per model: {args.num_examples}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print("="*70)
    
    # Load test dataset
    print(f"\nLoading test dataset from: {args.test_dataset}")
    test_data = load_jsonl_dataset(args.test_dataset, dataset_type="test", raise_on_error=True)
    if test_data is None:
        print("ERROR: Failed to load test dataset")
        return 1
    
    print(f"✓ Loaded {len(test_data)} examples")
    
    # Test each model
    all_results = {}
    for model_short_name in valid_models:
        print(f"\n{'='*70}")
        print(f"Testing model: {model_short_name}")
        print(f"{'='*70}")
        
        try:
            # Get HuggingFace model name
            hf_model_name = model_mapping[model_short_name]
            
            # Load model and tokenizer
            model, tokenizer = load_base_model(
                hf_model_name,
                hf_token=args.hf_token,
                use_multi_gpu=args.use_multi_gpu
            )
            
            # Test generation
            results = test_model_generation(
                model=model,
                tokenizer=tokenizer,
                model_name=hf_model_name,
                test_examples=test_data,
                max_new_tokens=args.max_new_tokens,
                num_examples=args.num_examples
            )
            
            # Save predictions
            output_file = save_predictions(results, args.output_dir, model_short_name)
            all_results[model_short_name] = {
                "results": results,
                "output_file": output_file
            }
            
            # Clean up
            del model
            del tokenizer
            torch.cuda.empty_cache()
            
            print(f"\n✓ Completed testing for {model_short_name}")
            
        except Exception as e:
            print(f"\n✗ ERROR testing {model_short_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_short_name] = {
                "error": str(e),
                "results": []
            }
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    for model_short_name, result_info in all_results.items():
        if "error" in result_info:
            print(f"  {model_short_name}: FAILED - {result_info['error']}")
        else:
            print(f"  {model_short_name}: SUCCESS - {len(result_info['results'])} predictions saved to {result_info['output_file']}")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
