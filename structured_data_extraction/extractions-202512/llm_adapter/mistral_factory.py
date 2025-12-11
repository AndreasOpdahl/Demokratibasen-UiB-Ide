"""
Mistral factory for creating Mistral model instances.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Load .env from extractions-202512 directory (same directory as the main scripts)
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

try:
    import openai
except ImportError:
    raise ImportError(
        "openai is required but not installed. "
        "Please install it with: pip install openai"
    )

# tiktoken is required for accurate token counting
try:
    import tiktoken
except ImportError:
    raise ImportError(
        "tiktoken is required but not installed. "
        "Please install it with: pip install tiktoken"
    )


def get_model_window_size(model_name: str) -> int:
    """
    Returns the context window size (in tokens) for popular Mistral models.
    
    Args:
        model_name: Name of the Mistral model
        
    Returns:
        Context window size in tokens, or default if model is not recognized
    """
    model_lower = model_name.lower()
    
    # Context window sizes for popular Mistral models
    window_sizes = {
        # Mistral Large series
        "mistral-large": 128000,  # 128k context
        "mistral-large-latest": 128000,  # 128k context (points to latest)
        "mistral-large-2402": 128000,
        "mistral-large-2407": 128000,
        
        # Mistral Medium series (32k context)
        "mistral-medium": 32000,
        "mistral-medium-latest": 32000,
        "mistral-medium-2312": 32000,
        
        # Mistral Small series (32k context)
        "mistral-small": 32000,
        "mistral-small-latest": 32000,
        "mistral-small-2402": 32000,
        
        # Mistral Tiny series (32k context)
        "mistral-tiny": 32000,
        "mistral-tiny-2312": 32000,
        
        # Pixtral series (128k context)
        "pixtral-12b": 128000,
        "pixtral-12b-2409": 128000,
        
        # Mistral Nemo series (128k context)
        "mistral-nemo": 128000,
        "mistral-nemo-2407": 128000,
    }
    
    # Try exact match first
    if model_name in window_sizes:
        return window_sizes[model_name]
    
    # Try case-insensitive match
    for key, value in window_sizes.items():
        if key.lower() == model_lower:
            return value
    
    # Try partial match
    for key, value in window_sizes.items():
        if model_lower.startswith(key.lower()) or key.lower().startswith(model_lower):
            return value
    
    # Default fallback for unknown models
    return 32000  # Conservative default for Mistral models


def estimate_tokens(text: str, model_name: str = "mistral-medium") -> int:
    """
    Estimate the number of tokens in a text using tiktoken.
    
    Args:
        text: The text to count tokens for
        model_name: The model name (used to select appropriate tokenizer)
        
    Returns:
        Estimated number of tokens
    """
    try:
        # Mistral models use similar tokenization to GPT models
        # Fallback to cl100k_base (used by GPT-4 and GPT-3.5-turbo)
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception as e:
        raise RuntimeError(f"Failed to count tokens with tiktoken: {e}")


class MistralModel:
    """Wrapper for Mistral API models (uses OpenAI-compatible API)."""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.window_size = get_model_window_size(model_name)
        
        # Mistral uses OpenAI-compatible API
        # Base URL is https://api.mistral.ai/v1
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1"
        )
    
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
        safety_margin = 200  # Extra safety margin
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
        enc = tiktoken.get_encoding("cl100k_base")  # Mistral models use similar tokenization
        
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
    ):
        """
        Generate text using Mistral API (OpenAI-compatible).
        
        Args:
            prompt: User prompt (the document text)
            system_prompt: System prompt (instructions)
            temperature: Temperature for generation (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            Response object compatible with OpenAI's response structure
            Access content via response.choices[0].message.content
        """
        # Truncate prompt if necessary
        truncated_prompt = self._truncate_prompt(prompt, system_prompt, max_tokens)
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": truncated_prompt})
        
        # Build API parameters
        api_params = {
            "model": self.model_name,
            "messages": messages,
        }
        
        # Add temperature if provided
        if temperature is not None:
            api_params["temperature"] = temperature
        
        # Add max_tokens if provided
        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens
        
        # Add schema support if provided
        # Mistral's API has a response_format parameter that forces JSON output
        # This guarantees valid JSON, but doesn't enforce schema structure
        # The model must rely on prompt instructions to include all required fields
        if json_schema:
            api_params["response_format"] = {"type": "json_object"}
        
        # Generate content
        try:
            response = self.client.chat.completions.create(**api_params)
        except Exception as e:
            # Check for specific error types and provide clearer error messages
            error_str = str(e).lower()
            error_type = type(e).__name__.lower()
            
            # Check for insufficient balance (402) - check status code if available
            if hasattr(e, 'status_code') and e.status_code == 402:
                raise ValueError(
                    f"Insufficient balance in Mistral account. Please add credits to your account."
                ) from e
            
            # Also check error message for balance issues
            if "402" in error_str or ("insufficient" in error_str and "balance" in error_str):
                raise ValueError(
                    f"Insufficient balance in Mistral account. Please add credits to your account."
                ) from e
            
            # Check for API key issues
            if any(keyword in error_str for keyword in ["authentication", "api key", "unauthorized", "401", "403"]):
                raise ValueError(
                    f"API key authentication failed. Please check your MISTRAL_API_KEY environment variable."
                ) from e
            
            # Check if this is specifically a model-related error
            # Must have both "model" and an error indicator, and NOT be a balance/auth error
            # Check for status code 400 with invalid_model type
            is_model_error = (
                (hasattr(e, 'status_code') and e.status_code == 400 and "invalid" in error_str and "model" in error_str) or
                ("invalid model" in error_str) or
                ("invalid_model" in error_str) or
                ("model" in error_str and ("not found" in error_str or "does not exist" in error_str)) or
                ("404" in error_str and "model" in error_str) or
                error_type in ["notfounderror", "modelnotfounderror"]
            )
            
            if is_model_error:
                valid_models = [
                    "mistral-large-latest", "mistral-medium", "mistral-small",
                    "mistral-tiny", "pixtral-12b", "mistral-nemo"
                ]
                raise ValueError(
                    f"Mistral model '{self.model_name}' does not exist or you do not have access to it. "
                    f"Valid models are: {', '.join(valid_models)}"
                ) from e
            
            # For other errors, re-raise with original message to help debug
            raise
        
        # Response is already OpenAI-compatible, return as-is
        return response


def validate_mistral_model_name(model_name: str) -> bool:
    """
    Validate that a Mistral model name is recognized.
    
    Args:
        model_name: Name of the Mistral model
        
    Returns:
        True if model name is recognized, False otherwise
    """
    model_lower = model_name.lower()
    
    # List of valid Mistral model names
    valid_models = [
        # Mistral Large series
        "mistral-large",
        "mistral-large-latest",  # Points to latest version
        "mistral-large-2402",
        "mistral-large-2407",
        
        # Mistral Medium series
        "mistral-medium",
        "mistral-medium-latest",
        "mistral-medium-2312",
        
        # Mistral Small series
        "mistral-small",
        "mistral-small-latest",
        "mistral-small-2402",
        
        # Mistral Tiny series
        "mistral-tiny",
        "mistral-tiny-2312",
        
        # Pixtral series
        "pixtral-12b",
        "pixtral-12b-2409",
        
        # Mistral Nemo series
        "mistral-nemo",
        "mistral-nemo-2407",
    ]
    
    # Check exact match
    if model_name in valid_models:
        return True
    
    # Check case-insensitive match
    for valid in valid_models:
        if valid.lower() == model_lower:
            return True
    
    # Check partial match (e.g., "mistral-large-2407" matches "mistral-large")
    # Only allow if input is longer/more specific than the base model name
    for valid in valid_models:
        valid_lower = valid.lower()
        # Input must start with valid model name followed by a dash (more specific)
        if model_lower.startswith(valid_lower + "-"):
            return True
    
    return False


class MistralFactory:
    """Factory for creating Mistral model instances."""
    
    def create(self, model_name: str):
        """
        Create a Mistral model instance.
        
        Args:
            model_name: Name of the Mistral model (e.g., "mistral-large")
            
        Returns:
            MistralModel instance
            
        Raises:
            ValueError: If MISTRAL_API_KEY is not set or model name is invalid
        """
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY environment variable is not set. "
                "Please set it in your .env file or environment variables."
            )
        
        # Validate model name early
        if not validate_mistral_model_name(model_name):
            valid_models = [
                "mistral-large-latest", "mistral-medium", "mistral-small",
                "mistral-tiny", "pixtral-12b", "mistral-nemo"
            ]
            raise ValueError(
                f"Invalid Mistral model name: '{model_name}'. "
                f"Valid models are: {', '.join(valid_models)}"
            )
        
        # Try to create the model to validate it exists
        try:
            model = MistralModel(api_key=api_key, model_name=model_name)
            # Test that the model can be accessed (this will fail if model doesn't exist)
            # We can't easily test without making an API call, so we'll let it fail on first use
            # but at least we've validated the model name format
            return model
        except Exception as e:
            # If it's a model not found error, raise a clearer error
            error_str = str(e).lower()
            if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
                valid_models = [
                    "mistral-large-latest", "mistral-medium", "mistral-small",
                    "mistral-tiny", "pixtral-12b", "mistral-nemo"
                ]
                raise ValueError(
                    f"Mistral model '{model_name}' does not exist or you do not have access to it. "
                    f"Valid models are: {', '.join(valid_models)}"
                ) from e
            # Check for API key issues
            if any(keyword in error_str for keyword in ["authentication", "api key", "unauthorized", "401", "403"]):
                raise ValueError(
                    f"API key authentication failed. Please check your MISTRAL_API_KEY environment variable."
                ) from e
            raise

