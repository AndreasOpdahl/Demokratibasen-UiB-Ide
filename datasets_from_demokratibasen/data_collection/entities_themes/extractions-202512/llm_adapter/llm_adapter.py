import importlib


# Prefix → model-family mapping.  Order matters: longer/more-specific
# prefixes must come before shorter ones so that e.g. "gpt-3.5" matches
# before a hypothetical "gpt" prefix.
_MODEL_PREFIX_TO_FAMILY: list[tuple[str, str]] = [
    # GPT / OpenAI
    ("gpt-", "GPT"),
    ("o1-", "GPT"),
    ("o3-", "GPT"),
    ("o4-", "GPT"),
    ("text-davinci-", "GPT"),
    ("text-curie-", "GPT"),
    ("text-babbage-", "GPT"),
    ("text-ada-", "GPT"),
    # Gemini
    ("gemini-", "Gemini"),
    # Claude
    ("claude-", "Claude"),
    # DeepSeek
    ("deepseek-", "DeepSeek"),
    # Mistral
    ("mistral-", "Mistral"),
    ("pixtral-", "Mistral"),
    # Qwen  (covers qwen-, qwen1.5-, qwen2-, qwen2.5-, qwen3-)
    ("qwen", "Qwen"),
]


def detect_model_family(model_name: str) -> str:
    """
    Infer the model family from the model name.

    Returns one of "GPT", "Gemini", "Claude", "DeepSeek", "Mistral", "Qwen".
    Raises ValueError if the model name cannot be mapped to a known family.
    """
    lower = model_name.lower()
    for prefix, family in _MODEL_PREFIX_TO_FAMILY:
        if lower.startswith(prefix):
            return family
    known = sorted({f for _, f in _MODEL_PREFIX_TO_FAMILY})
    raise ValueError(
        f"Cannot detect model family for '{model_name}'. "
        f"Known families (by prefix): {', '.join(known)}. "
        f"If your model belongs to one of these, rename it or pass "
        f"the correct prefix."
    )


def get_factory(model_family: str):
    """
    Dynamically import and instantiate the factory class based on model_family.
    
    Args:
        model_family: The first part of the Factory class name (e.g., "OpenAI")
                     Case-insensitive - "openai", "OpenAI", "OPENAI" all work
        
    Returns:
        An instance of the factory class
        
    Raises:
        ImportError: If the factory module cannot be imported
        AttributeError: If the factory class doesn't exist
    """
    normalized = model_family.strip()
    if not normalized:
        raise ValueError("model_family cannot be empty")
    
    normalized_lower = normalized.lower()
    
    # Convert to module name (e.g., "gpt" -> "gpt_factory")
    module_name = f"llm_adapter.{normalized_lower}_factory"
    
    try:
        module = importlib.import_module(module_name)
        
        # Try to find the factory class - check multiple naming patterns
        possible_names = [
            f"{normalized}Factory",  # Try exact input first (e.g., "GPT" -> "GPTFactory")
        ]
        
        # Handle common case variations
        if normalized_lower == "gpt":
            possible_names.insert(0, "GPTFactory")  # Direct match for GPT
        elif normalized_lower == "gemini":
            possible_names.insert(0, "GeminiFactory")  # Direct match for Gemini
        elif normalized_lower == "claude":
            possible_names.insert(0, "ClaudeFactory")  # Direct match for Claude
        elif normalized_lower == "deepseek":
            possible_names.insert(0, "DeepSeekFactory")  # Direct match for DeepSeek
        elif normalized_lower == "mistral":
            possible_names.insert(0, "MistralFactory")  # Direct match for Mistral
        elif normalized_lower == "qwen":
            possible_names.insert(0, "QwenFactory")  # Direct match for Qwen
        else:
            # Try title case and capitalize
            possible_names.extend([
                f"{normalized.title()}Factory",  # "some_name" -> "Some_NameFactory"
                f"{normalized.capitalize()}Factory",  # "someName" -> "SomenameFactory"
            ])
        
        # Try each possible name
        factory_class = None
        for class_name in possible_names:
            if hasattr(module, class_name):
                factory_class = getattr(module, class_name)
                break
        
        # If still not found, try to find any class ending with "Factory"
        if factory_class is None:
            for attr_name in dir(module):
                if attr_name.endswith("Factory") and not attr_name.startswith("_"):
                    factory_class = getattr(module, attr_name)
                    break
        
        if factory_class is None:
            raise AttributeError(f"Factory class not found in module '{module_name}'. Tried: {possible_names}")
        
        return factory_class()
    except ImportError as e:
        raise ImportError(f"Could not import factory module '{module_name}': {e}")
    except AttributeError as e:
        raise AttributeError(f"Factory class not found in module '{module_name}': {e}")


class LLMAdapter:
    def __init__(self, model_factory, model_name: str):
        self.model_name = model_name
        self.model = model_factory.create(model_name)

    def generate_text(
        self, 
        prompt: str, 
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None,
        json_schema: dict = None
    ) -> str:
        return self.model.generate_text(
            prompt, 
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema
        )
