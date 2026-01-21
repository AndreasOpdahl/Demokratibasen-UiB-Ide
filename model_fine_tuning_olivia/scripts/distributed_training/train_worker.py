import os
import torch
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM, 
    MT5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from tokenization_utils import get_tokenizer, load_and_preprocess_datasets, tokenize_dataset, format_train_example, format_eval_example
import wandb

# Global WANDB availability check
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def setup_distributed_training():
    """Setup distributed training environment"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"Process {local_rank} of {world_size} starting...")

    # Set the correct device
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_devices = list(map(int, cuda_visible.split(","))) if cuda_visible else list(range(torch.cuda.device_count()))

    if local_rank >= len(visible_devices):
        raise RuntimeError(f"Invalid local_rank {local_rank} — only {len(visible_devices)} visible GPUs: {visible_devices}")

    device_id = visible_devices[local_rank]
    torch.cuda.set_device(device_id)

    print(f"[Rank {local_rank}] Using device {device_id} / visible GPUs: {visible_devices}")

    # Initialize process group
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        print(f"[Rank {local_rank}] Process group initialized.")

    if dist.is_initialized():
        print(f"Rank {dist.get_rank()} initialized on device {torch.cuda.current_device()}")

    return local_rank, world_size

def setup_wandb_and_hf(local_rank: int, hf_token: str = None, config: dict = None):
    """Setup WandB and Hugging Face login (only on rank 0)"""
    if local_rank == 0:
        # Initialize wandb only on main process
        if WANDB_AVAILABLE:
            wandb.init(
                project="text-summarization-finetuning",
                config=config
            )
        else:
            print("Weights & Biases not available. Training will proceed without logging.")

        # Login to Hugging Face only on main process
        if hf_token:
            print("Logging in to Hugging Face Hub...")
            from huggingface_hub import login
            login(token=hf_token)
    else:
        # Other processes should not log in
        if WANDB_AVAILABLE:
            os.environ["WANDB_MODE"] = "disabled"

def load_model_with_quantization(model_name: str, hf_token: str = None):
    """Load model with 4-bit quantization"""
    # Configure 4-bit quantization
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    try:
        if model_name == 'google/mt5-base':
            model = MT5ForConditionalGeneration.from_pretrained(model_name)
        else:
            # For DDP, remove device_map and handle device placement manually
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                # Remove device_map="auto" - let DDP handle device placement
                device_map=None,  # ← CRITICAL FIX
                token=hf_token if hf_token else None
            )
        return model
    except Exception as e:
        print(f"Error loading model with device_map: {e}")
        # Fallback without device_map
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            token=hf_token if hf_token else None
        )
        return model

def setup_lora_training(model, lora_config: LoraConfig = None):
    """Setup LoRA training for the model"""
    model = prepare_model_for_kbit_training(model)

    if lora_config is None:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

    model = get_peft_model(model, lora_config)
    print(model.print_trainable_parameters())
    return model

def create_trainer(model, tokenizer, train_dataset, eval_dataset, training_args, callbacks=None):
    """Create and return Trainer instance"""
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks or []
    )
    
    return trainer

def run_training_epoch(trainer, model, rouge_callback, local_rank):
    """Run training with validation and callbacks"""
    # Validate once before training (both loss and ROUGE)
    if local_rank == 0:
        print("Running validation before training...")
        
        # Compute initial loss
        initial_eval = trainer.evaluate()
        print(f"Initial validation loss: {initial_eval['eval_loss']:.4f}")

        # Compute initial ROUGE
        print("Computing initial ROUGE scores...")
        rouge_callback._compute_rouge(model=model)

        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({"eval_loss": initial_eval['eval_loss'], "step": 0})

    # Start training
    print("Starting training...")
    trainer.train()

    return trainer