# benchmark_evaluation.py
import time
import subprocess
import os

def benchmark_evaluation_settings():
    """Benchmark different evaluation settings"""
    
    # First, check what models are available
    models_dir = "/app/models"
    available_models = []
    
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path):
                available_models.append(item_path)
    
    if not available_models:
        print("❌ No models found in /app/models/")
        return
    
    print("🎯 Available models:")
    for model_path in available_models:
        print(f"   {model_path}")
    
    # Use the first available model
    model_path = available_models[0]
    eval_data_path = "/app/data/output/processed_data_val.jsonl"
    
    if not os.path.exists(eval_data_path):
        print(f"❌ Evaluation data not found: {eval_data_path}")
        return
    
    settings = [
        {"processes": 1, "samples": 5, "sequential": True, "label": "Sequential (5 samples)"},
        {"processes": 2, "samples": 5, "sequential": False, "label": "2 processes (5 samples)"},
        {"processes": 1, "samples": 10, "sequential": True, "label": "Sequential (10 samples)"},
        {"processes": 2, "samples": 10, "sequential": False, "label": "2 processes (10 samples)"},
    ]
    
    results = []
    
    for setting in settings:
        print(f"\n🧪 Testing: {setting['label']}")
        
        start_time = time.time()
        
        cmd = [
            'python', 'scripts/cpu_evaluation_worker.py',
            '--model_path', model_path,
            '--eval_data', eval_data_path,
            '--num_samples', str(setting['samples']),
            '--num_processes', str(setting['processes']),
            '--max_length', '256'  # Even shorter for benchmarking
        ]
        
        if setting['sequential']:
            cmd.append('--sequential')
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                # Extract loss
                output_lines = result.stdout.strip().split('\n')
                loss_line = [line for line in output_lines if 'Average loss:' in line]
                loss = float(loss_line[0].split(':')[-1].strip()) if loss_line else float('inf')
                
                results.append({
                    'setting': setting['label'],
                    'time': elapsed_time,
                    'loss': loss,
                    'samples_per_second': setting['samples'] / elapsed_time
                })
                
                print(f"  ✅ Time: {elapsed_time:.2f}s, Loss: {loss:.4f}, Samples/s: {setting['samples']/elapsed_time:.2f}")
            else:
                print(f"  ❌ Failed: {result.stderr}")
                results.append({
                    'setting': setting['label'],
                    'time': elapsed_time,
                    'loss': float('inf'),
                    'samples_per_second': 0
                })
                
        except subprocess.TimeoutExpired:
            print("  ⏰ Timeout after 5 minutes")
            results.append({
                'setting': setting['label'],
                'time': 300,
                'loss': float('inf'),
                'samples_per_second': 0
            })
    
    # Print summary
    print("\n--- 📊 Benchmark Results ---")
    valid_results = [r for r in results if r['loss'] != float('inf')]
    if valid_results:
        for result in sorted(valid_results, key=lambda x: x['samples_per_second'], reverse=True):
            print(f"{result['setting']}: {result['samples_per_second']:.2f} samples/sec (loss: {result['loss']:.4f})")
    else:
        print("❌ No valid benchmarks completed")

if __name__ == "__main__":
    # First, diagnose the model situation
    print("🔍 Running model diagnostics...")
    os.system("python scripts/check_models.py")
    
    print("\n🚀 Starting benchmark...")
    benchmark_evaluation_settings()
