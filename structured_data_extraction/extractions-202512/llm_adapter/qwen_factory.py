"""
Qwen factory for creating Qwen model instances.
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
    from openai import AuthenticationError
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
    Returns the context window size (in tokens) for popular Qwen models.
    
    Args:
        model_name: Name of the Qwen model
        
    Returns:
        Context window size in tokens, or default if model is not recognized
    """
    model_lower = model_name.lower()
    
    # Context window sizes for popular Qwen models
    window_sizes = {
        # Qwen Max series (8k context)
        "qwen-max": 8000,
        "qwen-max-longcontext": 30000,  # Extended context version
        
        # Qwen Plus series (32k context)
        "qwen-plus": 32000,
        "qwen-plus-longcontext": 32000,
        
        # Qwen Turbo series (8k context)
        "qwen-turbo": 8000,
        "qwen-turbo-longcontext": 30000,  # Extended context version
        
        # Qwen 2.5 series (128k context)
        "qwen2.5-72b-instruct": 128000,
        "qwen2.5-32b-instruct": 128000,
        "qwen2.5-14b-instruct": 128000,
        "qwen2.5-7b-instruct": 128000,
        "qwen2.5-3b-instruct": 128000,
        "qwen2.5-1.5b-instruct": 128000,
        
        # Qwen 2.0 series (128k context)
        "qwen2-72b-instruct": 128000,
        "qwen2-32b-instruct": 128000,
        "qwen2-14b-instruct": 128000,
        "qwen2-7b-instruct": 128000,
        "qwen2-1.5b-instruct": 128000,
        
        # Qwen 1.5 series (32k context)
        "qwen1.5-72b-chat": 32000,
        "qwen1.5-32b-chat": 32000,
        "qwen1.5-14b-chat": 32000,
        "qwen1.5-7b-chat": 32000,
        "qwen1.5-4b-chat": 32000,
        "qwen1.5-1.8b-chat": 32000,
        
        # Qwen 3 series
        "qwen3-8b": 128000,
        "qwen3-7b": 128000,
        "qwen3-1.8b": 128000,
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
    return 8000  # Conservative default for Qwen models


def estimate_tokens(text: str, model_name: str = "qwen-turbo") -> int:
    """
    Estimate the number of tokens in a text using tiktoken.
    
    Args:
        text: The text to count tokens for
        model_name: The model name (used to select appropriate tokenizer)
        
    Returns:
        Estimated number of tokens
    """
    try:
        # Qwen models use similar tokenization to GPT models
        # Fallback to cl100k_base (used by GPT-4 and GPT-3.5-turbo)
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception as e:
        raise RuntimeError(f"Failed to count tokens with tiktoken: {e}")


class QwenModel:
    """Wrapper for Qwen API models (uses OpenAI-compatible API via DashScope)."""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.window_size = get_model_window_size(model_name)
        
        # Qwen uses OpenAI-compatible API via Alibaba Cloud DashScope
        # Use international endpoint for non-China users
        # For China domestic users, use: https://dashscope.aliyuncs.com/compatible-mode/v1
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
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
        enc = tiktoken.get_encoding("cl100k_base")  # Qwen models use similar tokenization
        
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
        Generate text using Qwen API (OpenAI-compatible via DashScope international endpoint).
        
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
        
        # Add schema support if provided (basic JSON object format for OpenAI-compatible APIs)
        if json_schema:
            api_params["response_format"] = {"type": "json_object"}
        
        # Generate content
        try:
            response = self.client.chat.completions.create(**api_params)
        except AuthenticationError as e:
            # Catch AuthenticationError specifically (401 errors)
            raise ValueError(
                f"API key authentication failed. Please check your QWEN_API_KEY environment variable."
            ) from e
        except Exception as e:
            # Check for specific error types and provide clearer error messages
            error_str = str(e).lower()
            error_type = type(e).__name__.lower()
            
            # Check for insufficient balance (402) - check status code if available
            if hasattr(e, 'status_code') and e.status_code == 402:
                raise ValueError(
                    f"Insufficient balance in Qwen account. Please add credits to your account."
                ) from e
            
            # Also check error message for balance issues
            if "402" in error_str or ("insufficient" in error_str and "balance" in error_str):
                raise ValueError(
                    f"Insufficient balance in Qwen account. Please add credits to your account."
                ) from e
            
            # Check for API key issues in error message
            if any(keyword in error_str for keyword in ["authentication", "api key", "unauthorized", "401", "403", "incorrect api key"]):
                raise ValueError(
                    f"API key authentication failed. Please check your QWEN_API_KEY environment variable."
                ) from e
            
            # Check if this is specifically a model-related error
            # Must have both "model" and an error indicator, and NOT be a balance/auth error
            is_model_error = (
                ("model" in error_str and ("not found" in error_str or "does not exist" in error_str)) or
                ("404" in error_str and "model" in error_str) or
                error_type in ["notfounderror", "modelnotfounderror"]
            )
            
            if is_model_error:
                valid_models = [
                    "qwen-max", "qwen-plus", "qwen-turbo", "qwen2.5-72b-instruct",
                    "qwen2.5-32b-instruct", "qwen2.5-14b-instruct", "qwen2.5-7b-instruct"
                ]
                raise ValueError(
                    f"Qwen model '{self.model_name}' does not exist or you do not have access to it. "
                    f"Valid models are: {', '.join(valid_models)}"
                ) from e
            
            # For other errors, re-raise with original message to help debug
            raise
        
        # Response is already OpenAI-compatible, return as-is
        return response


def validate_qwen_model_name(model_name: str) -> bool:
    """
    Validate that a Qwen model name is recognized.
    
    Args:
        model_name: Name of the Qwen model
        
    Returns:
        True if model name is recognized, False otherwise
    """
    model_lower = model_name.lower()
    
    # List of valid Qwen model names
    valid_models = [
        # Qwen Max series
        "qwen-max",
        "qwen-max-longcontext",
        
        # Qwen Plus series
        "qwen-plus",
        "qwen-plus-longcontext",
        
        # Qwen Turbo series
        "qwen-turbo",
        "qwen-turbo-longcontext",
        
        # Qwen Flash series
        "qwen-flash",
        "qwen-flash-2025-07-28",
        
        # Qwen 2.5 series
        "qwen2.5-72b-instruct",
        "qwen2.5-32b-instruct",
        "qwen2.5-14b-instruct",
        "qwen2.5-7b-instruct",
        "qwen2.5-3b-instruct",
        "qwen2.5-1.5b-instruct",
        
        # Qwen 2.0 series
        "qwen2-72b-instruct",
        "qwen2-32b-instruct",
        "qwen2-14b-instruct",
        "qwen2-7b-instruct",
        "qwen2-1.5b-instruct",
        
        # Qwen 1.5 series
        "qwen1.5-72b-chat",
        "qwen1.5-32b-chat",
        "qwen1.5-14b-chat",
        "qwen1.5-7b-chat",
        "qwen1.5-4b-chat",
        "qwen1.5-1.8b-chat",
        
        # Qwen 3 series
        "qwen3-8b",
        "qwen3-7b",
        "qwen3-1.8b",
    ]
    
    # Check exact match
    if model_name in valid_models:
        return True
    
    # Check case-insensitive match
    for valid in valid_models:
        if valid.lower() == model_lower:
            return True
    
    # Check partial match (e.g., "qwen2.5-72b-instruct" matches "qwen2.5")
    # Only allow if input is longer/more specific than the base model name
    for valid in valid_models:
        valid_lower = valid.lower()
        # Input must start with valid model name followed by a dash or dot (more specific)
        if model_lower.startswith(valid_lower + "-") or model_lower.startswith(valid_lower + "."):
            return True
    
    return False


class QwenFactory:
    """Factory for creating Qwen model instances."""
    
    def create(self, model_name: str):
        """
        Create a Qwen model instance.
        
        Args:
            model_name: Name of the Qwen model (e.g., "qwen-turbo")
            
        Returns:
            QwenModel instance
            
        Raises:
            ValueError: If QWEN_API_KEY is not set or model name is invalid
        """
        api_key = os.getenv("QWEN_API_KEY")
        if not api_key:
            raise ValueError(
                "QWEN_API_KEY environment variable is not set. "
                "Please set it in your .env file or environment variables."
            )
        
        # Validate model name early
        if not validate_qwen_model_name(model_name):
            valid_models = [
                "qwen-max", "qwen-plus", "qwen-turbo", "qwen-flash", "qwen2.5-72b-instruct",
                "qwen2.5-32b-instruct", "qwen2.5-14b-instruct", "qwen2.5-7b-instruct"
            ]
            raise ValueError(
                f"Invalid Qwen model name: '{model_name}'. "
                f"Valid models are: {', '.join(valid_models)}"
            )
        
        # Try to create the model to validate it exists
        try:
            model = QwenModel(api_key=api_key, model_name=model_name)
            # Test that the model can be accessed (this will fail if model doesn't exist)
            # We can't easily test without making an API call, so we'll let it fail on first use
            # but at least we've validated the model name format
            return model
        except Exception as e:
            # If it's a model not found error, raise a clearer error
            error_str = str(e).lower()
            if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
                valid_models = [
                    "qwen-max", "qwen-plus", "qwen-turbo", "qwen2.5-72b-instruct",
                    "qwen2.5-32b-instruct", "qwen2.5-14b-instruct", "qwen2.5-7b-instruct"
                ]
                raise ValueError(
                    f"Qwen model '{model_name}' does not exist or you do not have access to it. "
                    f"Valid models are: {', '.join(valid_models)}"
                ) from e
            # Check for API key issues
            if any(keyword in error_str for keyword in ["authentication", "api key", "unauthorized", "401", "403"]):
                raise ValueError(
                    f"API key authentication failed. Please check your QWEN_API_KEY environment variable."
                ) from e
            raise

