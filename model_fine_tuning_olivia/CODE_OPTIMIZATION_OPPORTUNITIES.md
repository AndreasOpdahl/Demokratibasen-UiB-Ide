# Code Optimization Opportunities

## Analysis of Redundancies in Core Scripts

This document identifies redundancies and optimization opportunities in:
- `wandb_finetune.py` (1957 lines)
- `evaluate_distributed_checkpoints_multigpu.py` (2203 lines)
- `monitor_and_evaluate_checkpoints.py` (560 lines)

---

## 1. **JSONL Dataset Loading Logic** ⚠️ HIGH PRIORITY

### Current State:
- **Duplicate code** in `wandb_finetune.py` (lines 949-998 for train, 1000-1049 for val)
- **Duplicate code** in `evaluate_distributed_checkpoints_multigpu.py` (lines 1396-1437)
- Same Git LFS pointer checking, file size validation, JSON parsing, error handling

### Proposed Solution:
Create shared utility function:
```python
# utils/dataset_loading.py
def load_jsonl_dataset(file_path: str, dataset_type: str = "dataset") -> List[Dict]:
    """
    Load JSONL dataset with Git LFS pointer detection and error handling.
    
    Args:
        file_path: Path to JSONL file
        dataset_type: Type name for error messages (e.g., "training", "validation")
    
    Returns:
        List of parsed JSON objects
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is Git LFS pointer or invalid JSON
    """
```

**Impact**: Remove ~150 lines of duplicate code, improve maintainability

---

## 2. **EvalDataCollator Class** ⚠️ HIGH PRIORITY

### Current State:
- **Identical implementation** in both `wandb_finetune.py` (lines 290-335) and `evaluate_distributed_checkpoints_multigpu.py` (lines 176-219)
- Same left-padding logic for input_ids, right-padding for labels

### Proposed Solution:
Move to shared module:
```python
# utils/data_collators.py
class EvalDataCollator:
    """Custom data collator for evaluation..."""
```

**Impact**: Remove ~60 lines of duplicate code

---

## 3. **CausalLMTrainer Class** ⚠️ MEDIUM PRIORITY

### Current State:
- Similar but **not identical** implementations:
  - `wandb_finetune.py` (lines 338-499): Has `compute_loss` override for prompt masking
  - `evaluate_distributed_checkpoints_multigpu.py` (lines 222-416): Has `_move_model_to_device` override for model parallelism
- Both have `prediction_step`, `get_eval_dataloader`, `compute_metrics` patterns

### Proposed Solution:
Create base class with common functionality, allow subclasses for specific overrides:
```python
# utils/trainers.py
class BaseCausalLMTrainer(Trainer):
    """Base trainer with common evaluation functionality"""
    
class TrainingCausalLMTrainer(BaseCausalLMTrainer):
    """For training: includes compute_loss override"""
    
class EvaluationCausalLMTrainer(BaseCausalLMTrainer):
    """For evaluation: includes device_map handling"""
```

**Impact**: Reduce duplication, improve maintainability (~200 lines)

---

## 4. **Tokenization Functions** ⚠️ HIGH PRIORITY

### Current State:
- `tokenize_function_eval` duplicated in:
  - `wandb_finetune.py` (lines 1151-1169)
  - `evaluate_distributed_checkpoints_multigpu.py` (lines 1473-1488)
- `tokenize_function_train` only in `wandb_finetune.py` (lines 1108-1149)
- Similar logic with slight variations

### Proposed Solution:
Create shared tokenization utilities:
```python
# utils/tokenization.py
def tokenize_eval_examples(
    examples: Dict,
    tokenizer: AutoTokenizer,
    max_input_prompt_tokens: int,
    max_output_summary_tokens: int
) -> Dict:
    """Tokenize evaluation examples (prompt + target)"""

def tokenize_train_examples(
    examples: Dict,
    tokenizer: AutoTokenizer,
    max_input_prompt_tokens: int,
    max_output_summary_tokens: int
) -> Dict:
    """Tokenize training examples (full text with prompt_length tracking)"""
```

**Impact**: Remove ~50 lines of duplicate code

---

## 5. **Example Formatting Functions** ⚠️ MEDIUM PRIORITY

### Current State:
- `format_example_eval` duplicated in:
  - `wandb_finetune.py` (lines 1086-1106)
  - `evaluate_distributed_checkpoints_multigpu.py` (lines 1451-1471)
- `format_example_train` only in `wandb_finetune.py` (lines 1065-1084)
- Both use `get_model_config_by_hf_name` and `get_doc_type_norwegian`

### Proposed Solution:
Move to shared module:
```python
# utils/formatting.py
def format_eval_example(example: Dict, model_name: str) -> Dict:
    """Format evaluation example with model-specific prompt"""

def format_train_example(example: Dict, model_name: str) -> Dict:
    """Format training example with model-specific prompt"""
```

**Impact**: Remove ~40 lines of duplicate code

---

## 6. **ROUGE Metrics Computation** ⚠️ MEDIUM PRIORITY

### Current State:
- `compute_metrics` function duplicated in:
  - `wandb_finetune.py` (lines 703-755)
  - `evaluate_distributed_checkpoints_multigpu.py` (lines 1207-1269)
- Same logic: load rouge, decode, clean text, compute scores, log to wandb
- Minor variations in wandb logging

### Proposed Solution:
Create shared metrics computation:
```python
# utils/metrics.py
def compute_rouge_metrics(
    eval_pred: Tuple,
    tokenizer: AutoTokenizer,
    log_to_wandb: bool = False,
    step: Optional[int] = None
) -> Dict[str, float]:
    """Compute ROUGE metrics from predictions and labels"""
```

**Impact**: Remove ~60 lines of duplicate code

---

## 7. **GPU Memory Monitoring** ⚠️ LOW PRIORITY

### Current State:
- `check_gpu_memory_utilization` in `evaluate_distributed_checkpoints_multigpu.py` (lines 863-932)
- `GPUMemoryCallback` in `wandb_finetune.py` (lines 143-244)
- Similar logic but different implementations (callback vs function)

### Proposed Solution:
Create unified GPU memory utilities:
```python
# utils/gpu_monitoring.py
def check_gpu_memory_utilization(num_gpus: Optional[int] = None) -> Dict:
    """Check GPU memory (shared function)"""

class GPUMemoryCallback(TrainerCallback):
    """Callback wrapper for training"""
```

**Impact**: Consolidate ~100 lines, improve consistency

---

## 8. **WandB Initialization and Logging** ⚠️ MEDIUM PRIORITY

### Current State:
- WandB init logic scattered across files with similar patterns
- Different config structures but same initialization pattern
- Logging patterns repeated

### Proposed Solution:
Create WandB utilities:
```python
# utils/wandb_utils.py
def init_wandb_for_training(...) -> wandb.Run:
    """Initialize WandB for training runs"""

def init_wandb_for_evaluation(...) -> wandb.Run:
    """Initialize WandB for evaluation runs"""

def log_evaluation_metrics(results: Dict, step: int, ...):
    """Log evaluation metrics consistently"""
```

**Impact**: Standardize WandB usage, reduce ~80 lines

---

## 9. **Checkpoint Path Handling** ⚠️ LOW PRIORITY

### Current State:
- Checkpoint step extraction logic duplicated:
  - `evaluate_distributed_checkpoints_multigpu.py` (lines 558-563, 964-970)
  - `monitor_and_evaluate_checkpoints.py` (lines 49-54, 309)
- Path manipulation for `all_eval_results/` directory

### Proposed Solution:
Create checkpoint utilities:
```python
# utils/checkpoint_utils.py
def extract_checkpoint_step(checkpoint_path: str) -> int:
    """Extract step number from checkpoint path"""

def get_eval_results_path(checkpoint_dir: str, model_dir: str) -> str:
    """Get evaluation results file path (new or old location)"""
```

**Impact**: Remove ~30 lines, improve consistency

---

## 10. **Prompt Logging to WandB** ⚠️ LOW PRIORITY

### Current State:
- Similar prompt example collection and logging in:
  - `wandb_finetune.py` (lines 1173-1227)
  - `evaluate_distributed_checkpoints_multigpu.py` (lines 1492-1534)

### Proposed Solution:
Create shared function:
```python
# utils/prompt_logging.py
def log_prompt_examples_to_wandb(
    dataset: Dataset,
    model_name: str,
    num_examples: int = 5
):
    """Log example prompts to WandB config"""
```

**Impact**: Remove ~50 lines of duplicate code

---

## 11. **Text Cleaning Utilities** ⚠️ LOW PRIORITY

### Current State:
- Text cleaning logic duplicated in:
  - `evaluate_distributed_checkpoints_multigpu.py` (lines 375-384, 1232-1241)
  - Similar patterns in `compute_metrics` functions

### Proposed Solution:
Create shared cleaning function:
```python
# utils/text_cleaning.py
def clean_decoded_text(text: str) -> str:
    """Remove special tokens, backslashes, normalize whitespace"""
```

**Impact**: Remove ~20 lines, improve consistency

---

## 12. **Model Loading Patterns** ⚠️ MEDIUM PRIORITY

### Current State:
- Tokenizer loading duplicated in both files
- Similar error handling patterns
- Pad token setup repeated

### Proposed Solution:
Create model loading utilities:
```python
# utils/model_loading.py
def load_tokenizer_safely(model_name: str, hf_token: Optional[str] = None) -> AutoTokenizer:
    """Load tokenizer with error handling and pad token setup"""
```

**Impact**: Remove ~30 lines, improve error handling consistency

---

## 13. **Evaluation Results File Handling** ⚠️ LOW PRIORITY

### Current State:
- Similar logic for saving/loading evaluation results in:
  - `evaluate_distributed_checkpoints_multigpu.py` (lines 1007-1164, 1985-2059)
  - `monitor_and_evaluate_checkpoints.py` (lines 62-90, 107-153)

### Proposed Solution:
Create shared utilities:
```python
# utils/eval_results.py
def load_eval_results(checkpoint_dir: str, model_dir: str) -> Optional[Dict]:
    """Load evaluation results from new or old location"""

def save_eval_results(results: Dict, checkpoint_dir: str, model_dir: str):
    """Save evaluation results to both locations"""

def update_evaluation_summary(results: Dict, checkpoint_dir: str, model_dir: str):
    """Update evaluation_summary.json"""
```

**Impact**: Remove ~100 lines, improve maintainability

---

## Summary Statistics

| Category | Duplicate Lines | Priority | Impact |
|----------|----------------|----------|--------|
| JSONL Loading | ~150 | HIGH | High |
| EvalDataCollator | ~60 | HIGH | Medium |
| CausalLMTrainer | ~200 | MEDIUM | High |
| Tokenization | ~50 | HIGH | Medium |
| Example Formatting | ~40 | MEDIUM | Medium |
| ROUGE Metrics | ~60 | MEDIUM | Medium |
| GPU Memory | ~100 | LOW | Low |
| WandB Utils | ~80 | MEDIUM | Medium |
| Checkpoint Utils | ~30 | LOW | Low |
| Prompt Logging | ~50 | LOW | Low |
| Text Cleaning | ~20 | LOW | Low |
| Model Loading | ~30 | MEDIUM | Low |
| Eval Results | ~100 | LOW | Medium |
| **TOTAL** | **~970 lines** | | **High** |

---

## Recommended Implementation Order

1. **Phase 1 (High Impact, Low Risk)**:
   - JSONL dataset loading utility
   - EvalDataCollator class
   - Tokenization functions
   - Text cleaning utilities

2. **Phase 2 (Medium Impact, Medium Risk)**:
   - Example formatting functions
   - ROUGE metrics computation
   - Model loading utilities
   - WandB utilities

3. **Phase 3 (Lower Impact, Higher Risk)**:
   - CausalLMTrainer refactoring
   - Evaluation results handling
   - Checkpoint utilities
   - GPU memory consolidation

---

## Proposed File Structure

```
model_fine_tuning_olivia/
├── scripts/
│   ├── wandb_finetune.py (reduced from 1957 to ~1400 lines)
│   ├── evaluate_distributed_checkpoints_multigpu.py (reduced from 2203 to ~1600 lines)
│   └── monitor_and_evaluate_checkpoints.py (reduced from 560 to ~450 lines)
└── utils/
    ├── __init__.py
    ├── dataset_loading.py
    ├── data_collators.py
    ├── trainers.py
    ├── tokenization.py
    ├── formatting.py
    ├── metrics.py
    ├── gpu_monitoring.py
    ├── wandb_utils.py
    ├── checkpoint_utils.py
    ├── prompt_logging.py
    ├── text_cleaning.py
    ├── model_loading.py
    └── eval_results.py
```

---

## Notes

- All changes should maintain backward compatibility
- Test thoroughly after each phase
- Keep original functionality intact
- Document all utility functions
- Consider type hints for better IDE support
