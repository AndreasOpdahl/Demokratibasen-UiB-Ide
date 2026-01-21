"""
DeepSeek factory for creating DeepSeek model instances.
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


def get_model_window_size(model_name: str) -> int:
    """
    Returns the context window size (in tokens) for popular DeepSeek models.
    
    Args:
        model_name: Name of the DeepSeek model
        
    Returns:
        Context window size in tokens, or default if model is not recognized
    """
    model_lower = model_name.lower()
    
    # Context window sizes for popular DeepSeek models
    window_sizes = {
        # DeepSeek Chat series
        "deepseek-chat": 64000,  # 64k tokens
        
        # DeepSeek V2 series
        "deepseek-chat-v2": 64000,
        "deepseek-chat-v2-0324": 64000,
        
        # DeepSeek V1 series
        "deepseek-chat-v1": 32000,  # 32k tokens
        "deepseek-chat-v1-0324": 32000,
        
        # DeepSeek Coder series
        "deepseek-coder": 16000,  # 16k tokens
        "deepseek-coder-v2": 64000,  # 64k tokens
        "deepseek-coder-v2-lite": 64000,
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
    return 64000  # Conservative default for DeepSeek models


class DeepSeekModel:
    """Wrapper for DeepSeek API models (uses OpenAI-compatible API)."""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.window_size = get_model_window_size(model_name)
        
        # DeepSeek uses OpenAI-compatible API
        # Base URL is https://api.deepseek.com (OpenAI SDK will append /v1 automatically)
        # Use reasonable timeout for API calls (30s for request, 60s total)
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=(30.0, 60.0),  # 30s connect timeout, 60s read timeout
            max_retries=2  # Allow 2 retries for transient network issues
        )
    
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
        # DeepSeek models have large context windows, so truncation is rarely needed
        # But we'll implement a simple character-based truncation as a safety measure
        response_tokens = max_tokens if max_tokens is not None else 4096
        
        # Rough estimate: 1 token ≈ 4 characters for Norwegian text
        # Reserve space for response and overhead
        safety_margin = 1000  # Characters
        response_chars = response_tokens * 4
        
        system_chars = len(system_prompt) if system_prompt else 0
        available_chars = self.window_size * 4 - response_chars - system_chars - safety_margin
        
        if len(prompt) <= available_chars:
            return prompt
        
        # Truncate to fit
        return prompt[:available_chars]
    
    def generate_text(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None,
        json_schema: dict = None
    ):
        """
        Generate text using DeepSeek API (OpenAI-compatible).
        
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
        except openai.InternalServerError as e:
            # Handle 503 and other 5xx errors - don't retry, fail fast
            error_str = str(e).lower()
            if "503" in error_str or "service" in error_str and "busy" in error_str:
                raise ValueError(
                    f"DeepSeek API is currently unavailable (service too busy). "
                    f"Please try again later or use an alternative provider."
                ) from e
            raise
        except (openai.APITimeoutError, openai.APIConnectionError) as e:
            # Network/timeout errors - fail fast without retries
            raise ValueError(
                f"DeepSeek API connection failed or timed out: {e}"
            ) from e
        except Exception as e:
            # Check for specific error types and provide clearer error messages
            error_str = str(e).lower()
            error_type = type(e).__name__.lower()
            
            # Check for insufficient balance (402) - check status code if available
            # OpenAI SDK wraps errors in APIStatusError with status_code attribute
            if hasattr(e, 'status_code') and e.status_code == 402:
                raise ValueError(
                    f"Insufficient balance in DeepSeek account. Please add credits to your account."
                ) from e
            
            # Also check error message for balance issues
            if "402" in error_str or ("insufficient" in error_str and "balance" in error_str):
                raise ValueError(
                    f"Insufficient balance in DeepSeek account. Please add credits to your account."
                ) from e
            
            # Check for API key issues
            if any(keyword in error_str for keyword in ["authentication", "api key", "unauthorized", "401", "403"]):
                raise ValueError(
                    f"API key authentication failed. Please check your DEEPSEEK_API_KEY environment variable."
                ) from e
            
            # Check if this is specifically a model-related error
            # Must have both "model" and an error indicator, and NOT be a balance/auth error
            # Check for status code 400 with "Model Not Exist" message (DeepSeek uses this format)
            is_model_error = (
                (hasattr(e, 'status_code') and e.status_code == 400 and ("model not exist" in error_str or "model not found" in error_str)) or
                ("model not exist" in error_str) or
                ("model not found" in error_str) or
                ("model" in error_str and ("not found" in error_str or "does not exist" in error_str)) or
                ("404" in error_str and "model" in error_str) or
                error_type in ["notfounderror", "modelnotfounderror"]
            )
            
            if is_model_error:
                valid_models = [
                    "deepseek-chat", "deepseek-chat-v2",
                    "deepseek-coder", "deepseek-coder-v2"
                ]
                raise ValueError(
                    f"DeepSeek model '{self.model_name}' does not exist or you do not have access to it. "
                    f"Valid models are: {', '.join(valid_models)}"
                ) from e
            
            # For other errors, re-raise with original message to help debug
            raise
        
        # Response is already OpenAI-compatible, return as-is
        return response


def validate_deepseek_model_name(model_name: str) -> bool:
    """
    Validate that a DeepSeek model name is recognized.
    
    Args:
        model_name: Name of the DeepSeek model
        
    Returns:
        True if model name is recognized, False otherwise
    """
    model_lower = model_name.lower()
    
    # List of valid DeepSeek model names
    valid_models = [
        # DeepSeek Chat series
        "deepseek-chat",
        
        # DeepSeek Chat V2 series
        "deepseek-chat-v2",
        "deepseek-chat-v2-0324",
        
        # DeepSeek Chat V1 series
        "deepseek-chat-v1",
        "deepseek-chat-v1-0324",
        
        # DeepSeek Coder series
        "deepseek-coder",
        "deepseek-coder-v2",
        "deepseek-coder-v2-lite",
    ]
    
    # Check exact match
    if model_name in valid_models:
        return True
    
    # Check case-insensitive match
    for valid in valid_models:
        if valid.lower() == model_lower:
            return True
    
    # Check partial match (e.g., "deepseek-chat-v3-0324" matches "deepseek-chat-v3")
    # Only allow if input is longer/more specific than the base model name
    for valid in valid_models:
        valid_lower = valid.lower()
        # Input must start with valid model name followed by a dash (more specific)
        if model_lower.startswith(valid_lower + "-"):
            return True
    
    return False


class DeepSeekFactory:
    """Factory for creating DeepSeek model instances."""
    
    def create(self, model_name: str):
        """
        Create a DeepSeek model instance.
        
        Args:
            model_name: Name of the DeepSeek model (e.g., "deepseek-chat")
            
        Returns:
            DeepSeekModel instance
            
        Raises:
            ValueError: If DEEPSEEK_API_KEY is not set or model name is invalid
        """
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY environment variable is not set. "
                "Please set it in your .env file or environment variables."
            )
        
        # Validate model name early
        if not validate_deepseek_model_name(model_name):
            valid_models = [
                "deepseek-chat", "deepseek-chat-v2",
                "deepseek-coder", "deepseek-coder-v2"
            ]
            raise ValueError(
                f"Invalid DeepSeek model name: '{model_name}'. "
                f"Valid models are: {', '.join(valid_models)}"
            )
        
        # Try to create the model to validate it exists
        try:
            model = DeepSeekModel(api_key=api_key, model_name=model_name)
            # Test that the model can be accessed (this will fail if model doesn't exist)
            # We can't easily test without making an API call, so we'll let it fail on first use
            # but at least we've validated the model name format
            return model
        except Exception as e:
            # If it's a model not found error, raise a clearer error
            error_str = str(e).lower()
            if "model" in error_str and ("not found" in error_str or "not exist" in error_str or "does not exist" in error_str):
                valid_models = [
                    "deepseek-chat", "deepseek-chat-v2",
                    "deepseek-coder", "deepseek-coder-v2"
                ]
                raise ValueError(
                    f"DeepSeek model '{model_name}' does not exist or you do not have access to it. "
                    f"Valid models are: {', '.join(valid_models)}"
                ) from e
            # Check for API key issues
            if any(keyword in error_str for keyword in ["authentication", "api key", "unauthorized", "401", "403"]):
                raise ValueError(
                    f"API key authentication failed. Please check your DEEPSEEK_API_KEY environment variable."
                ) from e
            raise

