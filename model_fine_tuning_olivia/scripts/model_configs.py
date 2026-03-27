"""
Model and prompt configuration definitions for fine-tuning.

This module centralizes all model-specific configurations including:
- LoRA parameters
- Learning rates
- Prompt templates (chat vs plain text)
- Model name mappings
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any
from peft import LoraConfig


# Document type mapping (English -> Norwegian)
DOC_TYPE_MAPPING = {
    "case_minutes": "vedtak",
    "case_presentation": "saksforelegg",
    "meeting_minutes": "møtereferat",
    "meeting_agenda": "saksliste",
    "case_attachment": "vedlegg"
}

def get_doc_type_norwegian(doc_type: Optional[str]) -> str:
    """Get Norwegian translation of document type, default to 'tekst' if unknown."""
    if doc_type and doc_type in DOC_TYPE_MAPPING:
        return DOC_TYPE_MAPPING[doc_type]
    return "tekst"  # Default fallback


@dataclass
class PromptConfig:
    """Configuration for prompt formatting."""
    
    # Prompt template type
    template_type: str  # 'plain', 'llama2', 'llama3', 'llama3.1', 'mistral', 'chatml', 'custom'
    
    # Template strings (use {input}, {output}, and {doc_type} placeholders)
    train_template: str
    eval_template: str
    
    # Optional: custom formatting function
    format_fn: Optional[Callable] = None
    
    # Optional: tokenizer for chat template (for llama3, mistral, etc.)
    tokenizer: Optional[Any] = None
    
    def format_train(self, input_text: str, output_text: str, doc_type: Optional[str] = None, tokenizer: Optional[Any] = None) -> str:
        """Format training example.
        
        For Llama-2, Llama-3, Mistral, and ChatML models, uses tokenizer.apply_chat_template() if available.
        Otherwise falls back to manual template formatting.
        """
        # Use tokenizer's chat template if available (recommended for llama2, llama3, llama3.1, mistral, chatml)
        use_chat_template = (
            self.template_type in ['llama2', 'llama3', 'llama3.1', 'mistral', 'chatml'] and 
            (tokenizer or self.tokenizer) is not None and
            hasattr((tokenizer or self.tokenizer), 'apply_chat_template')
        )
        
        if use_chat_template:
            tok = tokenizer or self.tokenizer
            if tok is not None:  # Explicit None check for type checker
                # Check if tokenizer actually has a chat_template defined
                # Some tokenizers have apply_chat_template() but no chat_template attribute
                has_chat_template = (
                    hasattr(tok, 'chat_template') and 
                    tok.chat_template is not None
                ) or (
                    hasattr(tok, 'tokenizer') and 
                    hasattr(tok.tokenizer, 'chat_template') and 
                    tok.tokenizer.chat_template is not None
                )
                
                if has_chat_template:
                    doc_type_nor = get_doc_type_norwegian(doc_type)
                    messages = [
                        {
                            "role": "user",
                            "content": f"Du er en ekspert på tekstoppsummering. Oppsummer følgende {doc_type_nor} på norsk:\n\n{input_text}"
                        },
                        {
                            "role": "assistant",
                            "content": output_text
                        }
                    ]
                    try:
                        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    except Exception as e:
                        # Silently fall back to manual formatting if apply_chat_template fails
                        pass
                # If no chat_template, fall through to manual formatting
        
        if self.format_fn:
            return self.format_fn(input_text, output_text, is_training=True, doc_type=doc_type)
        doc_type_nor = get_doc_type_norwegian(doc_type)
        return self.train_template.format(input=input_text, output=output_text, doc_type=doc_type_nor)
    
    def format_eval(self, input_text: str, doc_type: Optional[str] = None, tokenizer: Optional[Any] = None) -> str:
        """Format evaluation example.
        
        For Llama-2, Llama-3, Mistral, and ChatML models, uses tokenizer.apply_chat_template() if available.
        Otherwise falls back to manual template formatting.
        """
        # Use tokenizer's chat template if available (recommended for llama2, llama3, llama3.1, mistral, chatml)
        use_chat_template = (
            self.template_type in ['llama2', 'llama3', 'llama3.1', 'mistral', 'chatml'] and 
            (tokenizer or self.tokenizer) is not None and
            hasattr((tokenizer or self.tokenizer), 'apply_chat_template')
        )
        
        if use_chat_template:
            tok = tokenizer or self.tokenizer
            if tok is not None:  # Explicit None check for type checker
                # Check if tokenizer actually has a chat_template defined
                # Some tokenizers have apply_chat_template() but no chat_template attribute
                has_chat_template = (
                    hasattr(tok, 'chat_template') and 
                    tok.chat_template is not None
                ) or (
                    hasattr(tok, 'tokenizer') and 
                    hasattr(tok.tokenizer, 'chat_template') and 
                    tok.tokenizer.chat_template is not None
                )
                
                if has_chat_template:
                    doc_type_nor = get_doc_type_norwegian(doc_type)
                    messages = [
                        {
                            "role": "user",
                            "content": f"Du er en ekspert på tekstoppsummering. Oppsummer følgende {doc_type_nor} på norsk:\n\n{input_text}"
                        }
                    ]
                    try:
                        # Use apply_chat_template with add_generation_prompt=True
                        # This should add the assistant prompt after [/INST] for Mistral models
                        formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        # Fix double <s> token from some chat templates (e.g. Mistral)
                        if formatted.startswith('<s><s>'):
                            formatted = '<s>' + formatted[4:]
                        return formatted
                    except Exception:
                        pass  # Fall back to manual formatting
                # If no chat_template, fall through to manual formatting
        
        if self.format_fn:
            return self.format_fn(input_text, None, is_training=False, doc_type=doc_type)
        doc_type_nor = get_doc_type_norwegian(doc_type)
        return self.eval_template.format(input=input_text, doc_type=doc_type_nor)


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
    train_batch_size: Optional[int] = None  # Default training batch size (None = use global default)
    val_batch_size: Optional[int] = None    # Default validation batch size (None = use global default)
    max_input_text_tokens: Optional[int] = None  # Model-specific input token limit (None = use global default)
    max_output_summary_tokens: Optional[int] = None  # Model-specific output token limit (None = use global default)
    
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
    train_template="Oppgave: Oppsummer følgende {doc_type}:\n\n###\n\n{input}\n\n###\n\nOppsummering:\n\n###\n\n{output}\n\n###\n",
    eval_template="Oppgave: Oppsummer følgende {doc_type}:\n\n###\n\n{input}\n\n###\n\nOppsummering:\n\n###\n\n",
)

PROMPT_LLAMA2 = PromptConfig(
    template_type='llama2',
    train_template="[INST] Oppsummer følgende {doc_type}:\n\n{input}\n\nOppsummering: [/INST] {output}",
    eval_template="[INST] Oppsummer følgende {doc_type}:\n\n{input}\n\nOppsummering: [/INST]",
)

# Llama-3 chat template format
PROMPT_LLAMA3 = PromptConfig(
    template_type='llama3',
    train_template="<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nDu er en ekspert på tekstoppsummering. Oppsummer følgende {doc_type} på norsk:\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>",
    eval_template="<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nDu er en ekspert på tekstoppsummering. Oppsummer følgende {doc_type} på norsk:\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
)

# Llama-3.1 chat template format (uses tokenizer.apply_chat_template when tokenizer is provided)
# Llama-3.1 uses the same chat template format as Llama-3, but we use a separate config for clarity
PROMPT_LLAMA3_1 = PromptConfig(
    template_type='llama3.1',
    train_template="<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nDu er en ekspert på tekstoppsummering. Oppsummer følgende {doc_type} på norsk:\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>",
    eval_template="<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nDu er en ekspert på tekstoppsummering. Oppsummer følgende {doc_type} på norsk:\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
)

# Mistral chat template for Normistral models
PROMPT_NORMISTRAL = PromptConfig(
    template_type='mistral',
    train_template="<s>[INST] Du er en ekspert på tekstoppsummering. Oppsummer følgende {doc_type} på norsk:\n\n{input} [/INST] {output}</s>",
    eval_template="<s>[INST] Du er en ekspert på tekstoppsummering. Oppsummer følgende {doc_type} på norsk:\n\n{input} [/INST]",
)

# ChatML template format (for EuroLLM and similar models - uses tokenizer.apply_chat_template)
# ChatML uses <|im_start|>role\ncontent<|im_end|>\n format
PROMPT_CHATML = PromptConfig(
    template_type='chatml',
    train_template="<|im_start|>system\nDu er en ekspert på tekstoppsummering.<|im_end|>\n<|im_start|>user\nOppsummer følgende {doc_type} på norsk:\n\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>\n",
    eval_template="<|im_start|>system\nDu er en ekspert på tekstoppsummering.<|im_end|>\n<|im_start|>user\nOppsummer følgende {doc_type} på norsk:\n\n{input}<|im_end|>\n<|im_start|>assistant\n",
)

# Alpaca instruction format (for GPT-J based models)
PROMPT_ALPACA = PromptConfig(
    template_type='alpaca',
    train_template="Instruction: Oppsummer følgende {doc_type} på norsk:\n\n{input}\n\nResponse: {output}",
    eval_template="Instruction: Oppsummer følgende {doc_type} på norsk:\n\n{input}\n\nResponse:",
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
        train_batch_size=4,
        val_batch_size=16,
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
        train_batch_size=4,
        val_batch_size=8,
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
        train_batch_size=4,
        val_batch_size=4,
    ),
    'gemma-2-27b': ModelConfig(
        short_name='gemma-2-27b',
        hf_name='google/gemma-2-27b',
        lora_r=16,  # Increased for larger model
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Expanded for better adaptation
        learning_rate=1e-5,  # Reduced from 2e-5 for stability (large models need conservative LR)
        prompt_config=PROMPT_PLAIN,
        architecture='gemma',
        train_batch_size=2,  # Very large model - smaller batch
        val_batch_size=2,
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
        train_batch_size=4,
        val_batch_size=8,
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
        train_batch_size=2,  # Very large model - smaller batch
        val_batch_size=2,
    ),
    
    # Viking models (Mistral-based) - base models, use plain format (no chat template)
    'viking-7b': ModelConfig(
        short_name='viking-7b',
        hf_name='LumiOpen/Viking-7B',
        lora_r=8,
        lora_alpha=16,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        learning_rate=1e-5,
        prompt_config=PROMPT_PLAIN,
        architecture='mistral',
        train_batch_size=4,
        val_batch_size=16,
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
        train_batch_size=4,
        val_batch_size=8,
    ),
    'viking-33b': ModelConfig(
        short_name='viking-33b',
        hf_name='LumiOpen/Viking-33B',  # https://huggingface.co/LumiOpen/Viking-33B
        lora_r=16,  # Increased for larger model
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        learning_rate=2e-5,  # Higher LR for larger model
        prompt_config=PROMPT_NORMISTRAL,
        architecture='mistral',
        train_batch_size=2,  # Very large model - smaller batch
        val_batch_size=2,
    ),
    
    # EuroLLM (Mistral-based) - uses ChatML template format
    'eurollm-9b-instruct': ModelConfig(
        short_name='eurollm-9b-instruct',
        hf_name='utter-project/EuroLLM-9B-Instruct-2512',
        lora_r=16,  # Increased for better adaptation
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        learning_rate=1.5e-5,
        prompt_config=PROMPT_CHATML,  # ChatML template format (uses tokenizer.apply_chat_template)
        architecture='mistral',
        train_batch_size=4,
        val_batch_size=8,
    ),
    
    # NorwAI (Mistral-based) - using Mistral chat template
    'norwai-mistral-7b-instruct': ModelConfig(
        short_name='norwai-mistral-7b-instruct',
        hf_name='NorwAI/NorwAI-Mistral-7B-instruct',
        lora_r=16,  # Increased for better adaptation
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        learning_rate=1.5e-5,
        prompt_config=PROMPT_NORMISTRAL,  # Mistral chat template
        architecture='mistral',
        train_batch_size=4,
        val_batch_size=16,
    ),
    
    # Normistral models (Mistral-based) - using Mistral chat template
    'normistral-7b': ModelConfig(
        short_name='normistral-7b',
        hf_name='norallm/normistral-7b-warm',
        lora_r=16,  # Increased for better adaptation
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        learning_rate=1.5e-5,
        prompt_config=PROMPT_NORMISTRAL,  # Mistral chat template
        architecture='mistral',
        train_batch_size=4,
        val_batch_size=16,
        max_output_summary_tokens=256,  # Shorter limit reduces gibberish/hallucination in generation
    ),
    'normistral-7b-instruct': ModelConfig(
        short_name='normistral-7b-instruct',
        hf_name='norallm/normistral-7b-warm-instruct',
        lora_r=16,  # Increased for better adaptation
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        learning_rate=1.5e-5,
        # This tokenizer uses a ChatML-style role template (<|im_start|> ... <|im_end|>).
        # Keep config aligned with that format for fallback/manual paths.
        prompt_config=PROMPT_CHATML,
        architecture='mistral',
        train_batch_size=4,
        val_batch_size=16,
        max_output_summary_tokens=256,  # Limit runway to reduce gibberish/hallucination
    ),
    'normistral-11b': ModelConfig(
        short_name='normistral-11b',
        hf_name='norallm/normistral-11b-warm',
        lora_r=16,  # Increased for better adaptation
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        learning_rate=2e-5,
        prompt_config=PROMPT_NORMISTRAL,  # Mistral chat template
        architecture='mistral',
        train_batch_size=4,
        val_batch_size=6,
        max_output_summary_tokens=256,  # Limit runway to reduce gibberish/hallucination
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
        train_batch_size=4,
        val_batch_size=16,
    ),
    'llama-3.1-8b-instruct': ModelConfig(
        short_name='llama-3.1-8b-instruct',
        hf_name='meta-llama/Llama-3.1-8B-Instruct',
        lora_r=16,  # Increased for better adaptation
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        learning_rate=1e-5,
        prompt_config=PROMPT_LLAMA3_1,  # Llama-3.1 chat format (uses tokenizer.apply_chat_template)
        architecture='llama',
        train_batch_size=4,
        val_batch_size=16,
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
        train_batch_size=4,
        val_batch_size=8,
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
        train_batch_size=4,
        val_batch_size=32,
    ),
    
    # GPT-J based models (GPT-NeoX architecture)
    'nb-gpt-j-6b': ModelConfig(
        short_name='nb-gpt-j-6b',
        hf_name='NbAiLab/nb-gpt-j-6B-torgersen-alpaca',
        lora_r=16,  # Increased for better adaptation
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # GPT-J uses similar attention structure
        learning_rate=1e-5,
        prompt_config=PROMPT_ALPACA,  # Alpaca instruction format
        architecture='gptj',
        train_batch_size=4,
        val_batch_size=16,
        max_input_text_tokens=1500,  # Reduced for 2048 context window (1500 + 40 prompt + 500 output = 2040)
        max_output_summary_tokens=500,  # Reduced to fit within 2048 total context
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
    """Get configuration by HuggingFace model name or short name.
    
    Args:
        hf_name: HuggingFace model identifier (e.g. 'google/gemma-2b-it') or short name (e.g. 'gemma-2b')
    
    Returns:
        ModelConfig if found, None otherwise
    """
    for config in MODEL_CONFIGS.values():
        if config.hf_name == hf_name:
            return config
    # Fallback: try short_name (e.g. when passed from checkpoint path or model identifier)
    if hf_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[hf_name]
    return None


def get_model_name_mapping() -> dict:
    """Get mapping from short names to HuggingFace names.
    
    Returns:
        Dictionary mapping short_name -> hf_name
    """
    return {config.short_name: config.hf_name for config in MODEL_CONFIGS.values()}
