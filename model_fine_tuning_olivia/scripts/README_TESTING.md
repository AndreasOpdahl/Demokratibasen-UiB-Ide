# End-to-End Testing for Refactored Code

This document describes how to run end-to-end tests to verify that refactorings haven't broken existing functionality.

## Quick Start

### On Olivia (HPC Cluster) - Recommended

```bash
# Test with default settings (gemma-2b, minimal resources)
sbatch --account=YOUR_ACCOUNT run_test_e2e.sbatch

# Test with a specific model
sbatch --account=YOUR_ACCOUNT run_test_e2e.sbatch --model=gemma-7b

# Test with custom output directory (keeps test data)
sbatch --account=YOUR_ACCOUNT run_test_e2e.sbatch --test_dir=./test_output --keep_test_data

# Skip training test (only test utilities)
sbatch --account=YOUR_ACCOUNT run_test_e2e.sbatch --skip_training

# Skip evaluation test (only test utilities and training)
sbatch --account=YOUR_ACCOUNT run_test_e2e.sbatch --skip_evaluation

# Dry-run to validate configuration
sbatch --account=YOUR_ACCOUNT run_test_e2e.sbatch --dry-run
```

### Local Testing (if you have GPU access)

```bash
# Test with default settings (gemma-2b, minimal resources)
python scripts/test_e2e.py

# Test with a specific model
python scripts/test_e2e.py --model viking-7b

# Test with custom output directory (keeps test data)
python scripts/test_e2e.py --test_dir ./test_output --keep_test_data

# Skip training test (only test utilities)
python scripts/test_e2e.py --skip_training

# Skip evaluation test (only test utilities and training)
python scripts/test_e2e.py --skip_evaluation
```

## What Gets Tested

1. **Utility Imports and Functionality** (TEST 3)
   - Verifies all refactored utilities can be imported
   - Tests basic functionality of checkpoint utilities
   - Tests text cleaning functions

2. **Training** (TEST 1)
   - Runs training with minimal steps (5 steps)
   - Uses small dataset (10 examples)
   - Single GPU, no distributed training
   - WandB disabled
   - Verifies checkpoint creation

3. **Evaluation** (TEST 2)
   - Evaluates a checkpoint using refactored utilities
   - Uses minimal validation set (5 examples)
   - Single GPU
   - WandB disabled
   - Verifies ROUGE metrics are computed

## Test Configuration

- **Model**: `gemma-2b` by default (smallest model for quick testing)
- **Training Steps**: 5 steps
- **Dataset Size**: 10 training examples, 5 validation examples
- **Batch Size**: 2 (training), 4 (validation)
- **WandB**: Disabled (via `WANDB_DISABLED=true`)
- **GPU**: Single GPU (no DDP/FSDP)

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- All dependencies from `requirements.txt`
- Access to model checkpoints (Hugging Face Hub)
- At least 1 GPU with sufficient memory for the model
- For Olivia: SLURM account, `.env` file with `HUGGINGFACE_TOKEN` and `WANDB_API_KEY`

## Output

The test script will:
- Create temporary test datasets
- Run training and evaluation
- Display test results
- Clean up temporary files (unless `--keep_test_data` is used)

## Troubleshooting

### Model Download Issues
If you get authentication errors, you may need to set `HF_TOKEN`:
```bash
export HF_TOKEN=your_token_here
python scripts/test_e2e.py
```

### GPU Memory Issues
If you run out of GPU memory, try:
- Using a smaller model: `--model gemma-2b`
- Reducing batch size in the test script
- Using CPU (not recommended, very slow)

### Import Errors
If you get import errors, ensure you're in the correct directory:
```bash
cd model_fine_tuning_olivia
python scripts/test_e2e.py
```

## Expected Runtime

- **Utilities Test**: < 1 second
- **Training Test**: ~2-5 minutes (depending on model size)
- **Evaluation Test**: ~1-3 minutes (depending on model size)
- **Total**: ~3-8 minutes for full test suite

## Notes

- The test uses minimal resources to keep runtime low
- WandB is automatically disabled for testing
- Test datasets are automatically generated
- All refactored utilities are tested for basic functionality
