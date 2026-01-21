import os
import openai
from pathlib import Path
from dotenv import load_dotenv

# tiktoken is required for accurate token counting
try:
    import tiktoken
except ImportError:
    raise ImportError(
        "tiktoken is required but not installed. "
        "Please install it with: pip install tiktoken"
    )


# Load .env from extractions-202512 directory (same directory as the main scripts)
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)


def get_model_window_size(model_name: str) -> int:
    """
    Returns the context window size (in tokens) for popular GPT models.
    
    Args:
        model_name: Name of the GPT model
        
    Returns:
        Context window size in tokens, or None if model is not recognized
        
    Note:
        Window sizes are for the input context. The actual usable window
        may be slightly less due to system overhead and response tokens.
    """
    # Normalize model name (handle variations)
    model_lower = model_name.lower()
    
    # Context window sizes for popular GPT models
    # Updated as of 2024
    window_sizes = {
        # GPT-4.1 series (128k context)
        "gpt-4.1": 128000,
        "gpt-4.1-mini": 128000,
        
        # GPT-4o series (128k context)
        "gpt-4o": 128000,
        "gpt-4o-2024": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4o-mini-2024": 128000,
        
        # GPT-4 Turbo series (128k context)
        "gpt-4-turbo": 128000,
        "gpt-4-turbo-2024": 128000,
        "gpt-4-turbo-preview": 128000,
        "gpt-4-0125-preview": 128000,
        "gpt-4-1106-preview": 128000,
        
        # GPT-4 base (8k context)
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-0613": 8192,
        "gpt-4-32k-0613": 32768,
        
        # GPT-3.5 Turbo series (16k context for most, 4k for older)
        "gpt-3.5-turbo": 16385,
        "gpt-3.5-turbo-16k": 16385,
        "gpt-3.5-turbo-1106": 16385,
        "gpt-3.5-turbo-0125": 16385,
        "gpt-3.5-turbo-0613": 4096,
        "gpt-3.5-turbo-16k-0613": 16385,
        
        # Legacy models
        "gpt-3.5-turbo-instruct": 4096,
        "text-davinci-003": 4097,
        "text-davinci-002": 4097,
        "text-davinci-001": 2049,
        "text-curie-001": 2049,
        "text-babbage-001": 2049,
        "text-ada-001": 2049,
    }
    
    # Try exact match first
    if model_name in window_sizes:
        return window_sizes[model_name]
    
    # Try case-insensitive match
    for key, value in window_sizes.items():
        if key.lower() == model_lower:
            return value
    
    # Try partial match (e.g., "gpt-4o" matches "gpt-4o-mini")
    for key, value in window_sizes.items():
        if model_lower.startswith(key.lower()) or key.lower().startswith(model_lower):
            return value
    
    # Default fallback for unknown models (conservative estimate)
    return 4096


def estimate_tokens(text: str, model_name: str = "gpt-4") -> int:
    """
    Estimate the number of tokens in a text string.
    
    Args:
        text: The text to count tokens for
        model_name: The model name (used to select appropriate tokenizer)
        
    Returns:
        Estimated number of tokens
    """
    try:
        # Try to get encoding for the specific model
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base (used by GPT-4 and GPT-3.5-turbo)
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception as e:
        raise RuntimeError(f"Failed to count tokens with tiktoken: {e}")


class GPTModel:
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key)
        self.window_size = get_model_window_size(model_name)

    def _estimate_message_tokens(self, messages: list) -> int:
        """
        Estimate total tokens for a list of messages including formatting overhead.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Estimated total tokens including message formatting overhead
        """
        total_tokens = 0
        
        # Each message has ~4 tokens overhead for formatting (role, etc.)
        message_overhead = 4
        
        for message in messages:
            content = message.get("content", "")
            content_tokens = estimate_tokens(content, self.model_name)
            total_tokens += content_tokens + message_overhead
        
        # Additional overhead for the entire request (boundaries, etc.)
        total_tokens += 3
        
        return total_tokens

    def _truncate_prompt(self, prompt: str, system_prompt: str = None, max_tokens: int = None) -> str:
        """
        Truncate the prompt to fit within the model's window size.
        
        Args:
            prompt: The user prompt to potentially truncate
            system_prompt: The system prompt (if any)
            max_tokens: Maximum tokens for the response
            
        Returns:
            Truncated prompt that fits within the window
        """
        # Calculate response tokens needed
        response_tokens = max_tokens if max_tokens is not None else 4096
        
        # Build messages to estimate total input tokens
        test_messages = []
        if system_prompt:
            test_messages.append({"role": "system", "content": system_prompt})
        test_messages.append({"role": "user", "content": prompt})
        
        # Estimate total input tokens including formatting
        input_tokens = self._estimate_message_tokens(test_messages)
        total_tokens = input_tokens + response_tokens
        
        # If everything fits, return prompt as-is
        if total_tokens <= self.window_size:
            return prompt
        
        # Calculate how many tokens we can use for the user prompt
        # Account for: system prompt (if any), message formatting, response tokens, safety margin
        # Use a larger safety margin to account for token counting inaccuracies
        safety_margin = 200  # Extra safety margin (increased from 50)
        system_tokens = estimate_tokens(system_prompt, self.model_name) if system_prompt else 0
        system_overhead = 4 if system_prompt else 0  # Message formatting overhead
        user_overhead = 4  # User message formatting overhead
        request_overhead = 3  # Request-level overhead
        
        available_for_prompt = (
            self.window_size 
            - response_tokens 
            - system_tokens 
            - system_overhead 
            - user_overhead 
            - request_overhead 
            - safety_margin
        )
        
        # Ensure we have at least some space for the prompt
        if available_for_prompt < 100:
            # Fallback: use a conservative fraction of the window
            available_for_prompt = max(100, self.window_size // 4)
        
        # Estimate tokens in the prompt
        prompt_tokens = estimate_tokens(prompt, self.model_name)
        
        # If prompt fits, verify it actually fits with a final check
        if prompt_tokens <= available_for_prompt:
            # Double-check with actual message tokens
            test_messages_final = []
            if system_prompt:
                test_messages_final.append({"role": "system", "content": system_prompt})
            test_messages_final.append({"role": "user", "content": prompt})
            final_input_tokens = self._estimate_message_tokens(test_messages_final)
            if final_input_tokens + response_tokens <= self.window_size:
                return prompt
            # If it doesn't actually fit, continue to truncation
        
        # Truncate the prompt using tiktoken
        try:
            enc = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        
        # Encode and truncate
        encoded = enc.encode(prompt)
        truncated_encoded = encoded[:available_for_prompt]
        truncated_prompt = enc.decode(truncated_encoded)
        
        # Verify the truncated prompt actually fits
        verify_messages = []
        if system_prompt:
            verify_messages.append({"role": "system", "content": system_prompt})
        verify_messages.append({"role": "user", "content": truncated_prompt})
        verify_tokens = self._estimate_message_tokens(verify_messages)
        
        # If still too large, truncate more aggressively
        if verify_tokens + response_tokens > self.window_size:
            # Reduce by the excess amount plus more safety margin
            excess = verify_tokens + response_tokens - self.window_size
            additional_reduction = excess + 100  # Extra reduction for safety
            new_available = max(100, available_for_prompt - additional_reduction)
            truncated_encoded = encoded[:new_available]
            truncated_prompt = enc.decode(truncated_encoded)
        
        return truncated_prompt

    def generate_text(
        self, 
        prompt: str, 
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None,
        json_schema: dict = None
    ) -> str:
        # Truncate prompt if necessary to fit within window size
        truncated_prompt = self._truncate_prompt(prompt, system_prompt, max_tokens)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": truncated_prompt})
        
        kwargs = {
            "model": self.model_name,
            "messages": messages,
        }
        
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        
        # Add schema support if provided
        if json_schema:
            # Check if model supports structured outputs (gpt-4o, gpt-4-turbo, etc.)
            model_lower = self.model_name.lower()
            supports_structured = any(x in model_lower for x in ["gpt-4o", "gpt-4-turbo", "gpt-4-1106", "gpt-4-0125"])
            
            # Check if model supports json_object response format
            # Note: gpt-4 (base) doesn't support any response_format, only gpt-4-turbo and newer do
            # gpt-4.1-mini and gpt-4.1 may support json_object but not structured outputs
            supports_json_object = any(x in model_lower for x in ["gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo", "gpt-4.1"])
            
            if supports_structured:
                # Use structured outputs with full schema
                # json_schema structure from Prompt class:
                # { "name": "...", "description": "...", "schema": { "name": "...", "description": "...", "parameters": {...} } }
                # OpenAI expects: { "name": "...", "description": "...", "schema": { "type": "object", "properties": {...}, "required": [...] } }
                
                if "schema" in json_schema:
                    # Full structure from Prompt class
                    inner_schema = json_schema["schema"]
                    name = json_schema.get("name", inner_schema.get("name", "extract_case_info"))
                    description = json_schema.get("description", inner_schema.get("description", ""))
                    
                    # Extract the actual JSON schema from "parameters" or use the schema directly
                    if "parameters" in inner_schema:
                        # Old format: schema has "parameters" key
                        openai_schema = inner_schema["parameters"]
                    elif "type" in inner_schema:
                        # New format: schema is already a JSON schema
                        openai_schema = inner_schema
                    else:
                        # Fallback: use inner_schema as-is
                        openai_schema = inner_schema
                else:
                    # Just the schema object (shouldn't happen, but handle it)
                    openai_schema = json_schema
                    name = "extract_case_info"
                    description = ""
                
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "description": description,
                        "schema": openai_schema
                    }
                }
            elif supports_json_object:
                # Fallback to basic JSON object format for models that support it
                kwargs["response_format"] = {"type": "json_object"}
            # else: For models like gpt-4 (base) that don't support response_format,
            #       we rely only on prompt instructions (schema is already in system prompt)
            
        return self.client.chat.completions.create(**kwargs)


class GPTFactory:
    def create(self, model_name: str):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it in your .env file or environment variables."
            )
        return GPTModel(api_key=api_key, model_name=model_name)

