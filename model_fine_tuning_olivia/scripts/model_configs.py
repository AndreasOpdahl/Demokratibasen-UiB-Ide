"""
Model and prompt configuration definitions for fine-tuning.

This module centralizes all model-specific configurations including:
- LoRA parameters
- Learning rates
- Prompt templates (chat vs plain text)
- Model name mappings
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable
from peft import LoraConfig


@dataclass
class PromptConfig:
    """Configuration for prompt formatting."""
    
    # Prompt template type
    template_type: str  # 'plain', 'llama2', 'llama3', 'mistral', 'custom'
    
    # Template strings (use {input} and {output} placeholders)
    train_template: str
    eval_template: str
    
    # Optional: custom formatting function
    format_fn: Optional[Callable] = None
    
    def format_train(self, input_text: str, output_text: str) -> str:
        """Format training example."""
        if self.format_fn:
            return self.format_fn(input_text, output_text, is_training=True)
        return self.train_template.format(input=input_text, output=output_text)
    
    def format_eval(self, input_text: str) -> str:
        """Format evaluation example."""
        if self.format_fn:
            return self.format_fn(input_text, None, is_training=False)
        return self.eval_template.format(input=input_text)


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    
    short_name: str
    hf_name: str
    lora_r: int
    lora_alpha: int
    lora_target_modules: List[str]
    prompt_config: PromptConfig
    architecture: str
    lora_dropout: float = 0.05
    learning_rate: float = 1e-5
    
    def get_lora_config(self) -> LoraConfig:
        """Get LoRA configuration for this model."""
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )


# Prompt configurations
PROMPT_PLAIN = PromptConfig(
    template_type='plain',
    train_template="Oppgave: Oppsummer følgende tekst:\n\n###\n\n{input}\n\n###\n\nOppsummering:\n\n###\n\n{output}\n\n###\n",
    eval_template="Oppgave: Oppsummer følgende tekst:\n\n###\n\n{input}\n\n###\n\nOppsummering:\n\n###\n\n",
)

PROMPT_LLAMA2 = PromptConfig(
    template_type='llama2',
    train_template="[INST] Oppsummer følgende tekst:\n\n{input}\n\nOppsummering: [/INST] {output}",
    eval_template="[INST] Oppsummer følgende tekst:\n\n{input}\n\nOppsummering: [/INST]",
)

PROMPT_LLAMA3 = PromptConfig(
    template_type='llama3',
    train_template="<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nOppsummer følgende tekst:\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>",
    eval_template="<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nOppsummer følgende tekst:\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
)

# Custom prompt for Normistral (more explicit instruction)
PROMPT_NORMISTRAL = PromptConfig(
    template_type='custom',
    train_template="Du er en ekspert på tekstoppsummering. Oppsummer følgende tekst på norsk:\n\n{input}\n\nOppsummering:\n\n{output}",
    eval_template="Du er en ekspert på tekstoppsummering. Oppsummer følgende tekst på norsk:\n\n{input}\n\nOppsummering:\n\n",
)


# Model configurations
MODEL_CONFIGS = {
    # Gemma models
    'gemma-2b': ModelConfig(
        short_name='gemma-2b',
        hf_name='google/gemma-2b',
        lora_r=8,
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj"],
        learning_rate=1e-5,
        prompt_config=PROMPT_PLAIN,
        architecture='gemma',
    ),
    'gemma-7b': ModelConfig(
        short_name='gemma-7b',
        hf_name='google/gemma-7b',
        lora_r=8,
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj"],
        learning_rate=1e-5,
        prompt_config=PROMPT_PLAIN,
        architecture='gemma',
    ),
    # Gemma-2 models (new generation) - https://huggingface.co/collections/google/gemma-2-release
    'gemma-2-9b': ModelConfig(
        short_name='gemma-2-9b',
        hf_name='google/gemma-2-9b',
        lora_r=8,
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj"],
        learning_rate=1e-5,
        prompt_config=PROMPT_PLAIN,
        architecture='gemma',
    ),
    'gemma-2-27b': ModelConfig(
        short_name='gemma-2-27b',
        hf_name='google/gemma-2-27b',
        lora_r=16,  # Increased for larger model
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj"],
        learning_rate=2e-5,  # Higher LR for larger model
        prompt_config=PROMPT_PLAIN,
        architecture='gemma',
    ),
    # Gemma-3 models - https://huggingface.co/collections/google/gemma-3-release
    'gemma-3-12b': ModelConfig(
        short_name='gemma-3-12b',
        hf_name='google/gemma-3-12b-pt',  # Pre-trained version for fine-tuning
        lora_r=16,  # Increased for larger model
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj"],
        learning_rate=2e-5,  # Higher LR for larger model
        prompt_config=PROMPT_PLAIN,
        architecture='gemma',
    ),
    'gemma-3-27b': ModelConfig(
        short_name='gemma-3-27b',
        hf_name='google/gemma-3-27b-pt',  # Pre-trained version for fine-tuning
        lora_r=16,  # Increased for larger model
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj"],
        learning_rate=2e-5,  # Higher LR for larger model
        prompt_config=PROMPT_PLAIN,
        architecture='gemma',
    ),
    
    # Viking models (Mistral-based)
    'viking-7b': ModelConfig(
        short_name='viking-7b',
        hf_name='LumiOpen/Viking-7B',
        lora_r=8,
        lora_alpha=16,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        learning_rate=1e-5,
        prompt_config=PROMPT_PLAIN,
        architecture='mistral',
    ),
    'viking-13b': ModelConfig(
        short_name='viking-13b',
        hf_name='LumiOpen/Viking-13B',
        lora_r=8,
        lora_alpha=16,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        learning_rate=2e-5,  # Higher LR for larger model
        prompt_config=PROMPT_PLAIN,
        architecture='mistral',
    ),
    'viking-33b': ModelConfig(
        short_name='viking-33b',
        hf_name='LumiOpen/Viking-33B',  # https://huggingface.co/LumiOpen/Viking-33B
        lora_r=16,  # Increased for larger model
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        learning_rate=2e-5,  # Higher LR for larger model
        prompt_config=PROMPT_PLAIN,
        architecture='mistral',
    ),
    
    # Normistral models (Mistral-based) - using custom prompt
    'normistral-7b': ModelConfig(
        short_name='normistral-7b',
        hf_name='norallm/normistral-7b-warm',
        lora_r=16,  # Increased for better adaptation
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        learning_rate=1.5e-5,
        prompt_config=PROMPT_NORMISTRAL,  # Custom prompt
        architecture='mistral',
    ),
    'normistral-11b': ModelConfig(
        short_name='normistral-11b',
        hf_name='norallm/normistral-11b-warm',
        lora_r=16,  # Increased for better adaptation
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        learning_rate=2e-5,
        prompt_config=PROMPT_NORMISTRAL,  # Custom prompt
        architecture='mistral',
    ),
    
    # Llama-based models - using chat templates
    'norskgpt-llama3-8b': ModelConfig(
        short_name='norskgpt-llama3-8b',
        hf_name='bineric/norskgpt-llama3-8b',
        lora_r=16,  # Increased for better adaptation
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        learning_rate=1e-5,
        prompt_config=PROMPT_LLAMA3,  # Llama-3 chat format
        architecture='llama',
    ),
    'llama-2-13b-chat-norwegian': ModelConfig(
        short_name='llama-2-13b-chat-norwegian',
        hf_name='ruternorway/llama-2-13b-chat-norwegian',
        lora_r=16,  # Increased for better adaptation
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        learning_rate=2e-5,
        prompt_config=PROMPT_LLAMA2,  # Llama-2 chat format
        architecture='llama',
    ),
    
    # MT5 (special case)
    'mt5': ModelConfig(
        short_name='mt5',
        hf_name='google/mt5-base',
        lora_r=8,
        lora_alpha=16,
        lora_target_modules=[],  # MT5 uses different architecture
        learning_rate=1e-5,
        prompt_config=PROMPT_PLAIN,
        architecture='mt5',
    ),
}


def get_model_config(short_name: str) -> ModelConfig:
    """Get configuration for a model by short name.
    
    Args:
        short_name: Short model name (e.g., 'gemma-7b')
    
    Returns:
        ModelConfig for the model
    
    Raises:
        ValueError: If model name is not found
    """
    if short_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {short_name}. "
            f"Available models: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[short_name]


def get_model_config_by_hf_name(hf_name: str) -> Optional[ModelConfig]:
    """Get configuration by HuggingFace model name.
    
    Args:
        hf_name: HuggingFace model identifier
    
    Returns:
        ModelConfig if found, None otherwise
    """
    for config in MODEL_CONFIGS.values():
        if config.hf_name == hf_name:
            return config
    return None


def get_model_name_mapping() -> dict:
    """Get mapping from short names to HuggingFace names.
    
    Returns:
        Dictionary mapping short_name -> hf_name
    """
    return {config.short_name: config.hf_name for config in MODEL_CONFIGS.values()}
