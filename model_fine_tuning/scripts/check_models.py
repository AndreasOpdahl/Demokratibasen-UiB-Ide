# check_models.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel

def check_model_directory(model_path):
    """Check what's in your model directory"""
    print(f"🔍 Checking model directory: {model_path}")
    
    if not os.path.exists(model_path):
        print("❌ Directory does not exist!")
        return False
    
    print("📁 Contents:")
    for item in os.listdir(model_path):
        item_path = os.path.join(model_path, item)
        size = os.path.getsize(item_path) if os.path.isfile(item_path) else "DIR"
        print(f"   {item} ({size})")
    
    # Check if it's a valid model
    try:
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print("✅ This appears to be a PEFT model")
            config = PeftConfig.from_pretrained(model_path)
            print(f"   Base model: {config.base_model_name_or_path}")
        else:
            print("✅ This appears to be a standard model")
            # Try to load tokenizer as a test
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            print("✅ Tokenizer loads successfully")
            
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def list_all_models():
    """List all models in your models directory"""
    models_dir = "/app/models"
    if not os.path.exists(models_dir):
        print(f"❌ Models directory {models_dir} does not exist!")
        return
    
    print("📁 Available models:")
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            print(f"\n🎯 Model: {item}")
            check_model_directory(item_path)

if __name__ == "__main__":
    list_all_models()
    
    # Test specific model path
    specific_path = "/app/models/gemma_finetuned"
    if os.path.exists(specific_path):
        print(f"\n🔍 Detailed check for {specific_path}:")
        check_model_directory(specific_path)
