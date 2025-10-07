# fixed_evaluation_worker.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import json
import argparse
from tqdm import tqdm
import os

def load_model_safely(model_path):
    """Load model with better error handling and diagnostics"""
    print(f"🔧 Attempting to load model from: {model_path}")
    
    # Check what files are available
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        return None, None
    
    files = os.listdir(model_path)
    print(f"📁 Files in model directory: {files}")
    
    try:
        # First try to load tokenizer
        print("🔄 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Check if it's a PEFT model
        if "adapter_config.json" in files:
            print("🎯 Detected PEFT/LoRA model")
            config = PeftConfig.from_pretrained(model_path)
            print(f"   Base model: {config.base_model_name_or_path}")
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.float16,
                device_map="cpu"
            )
            
            # Load PEFT model
            model = PeftModel.from_pretrained(base_model, model_path)
            print("✅ PEFT model loaded successfully")
            
        else:
            print("🎯 Detected standard model")
            # Try different loading methods
            try:
                # Try with safetensors first
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="cpu"
                )
            except:
                # Fallback to basic loading
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="cpu"
                )
            print("✅ Standard model loaded successfully")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def evaluate_model_interactive(model_path, eval_data_path, num_samples=5):
    """Interactive evaluation to debug the model"""
    print("🧪 Starting interactive evaluation...")
    
    # Load model
    model, tokenizer = load_model_safely(model_path)
    if model is None or tokenizer is None:
        print("❌ Failed to load model")
        return float('inf')
    
    # Load a few samples
    eval_data = []
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            eval_data.append(json.loads(line))
    
    print(f"📊 Evaluating {len(eval_data)} samples...")
    
    losses = []
    for i, sample in enumerate(eval_data):
        print(f"\n--- Sample {i+1} ---")
        print(f"Input preview: {sample['input'][:100]}...")
        print(f"Output preview: {sample['output'][:100]}...")
        
        try:
            # Tokenize
            inputs = tokenizer(
                sample['input'], 
                return_tensors="pt", 
                truncation=True, 
                max_length=256,
                padding=True
            )
            
            # Ensure we have labels for loss calculation
            labels = inputs["input_ids"].clone()
            
            print(f"Tokenized input shape: {inputs['input_ids'].shape}")
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss.item()
                
            print(f"✅ Loss: {loss:.4f}")
            losses.append(loss)
            
        except Exception as e:
            print(f"❌ Error in sample {i+1}: {e}")
            losses.append(float('inf'))
    
    if losses:
        avg_loss = sum(losses) / len(losses)
        print(f"\n📈 Average loss: {avg_loss:.4f}")
        return avg_loss
    else:
        return float('inf')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--eval_data", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    
    args = parser.parse_args()
    
    # Run interactive evaluation
    avg_loss = evaluate_model_interactive(args.model_path, args.eval_data, args.num_samples)
    
    return avg_loss

if __name__ == "__main__":
    main()

