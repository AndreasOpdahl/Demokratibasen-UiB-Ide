"""
Claude factory for creating Claude model instances.
"""

import os
import json
import math
import re
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Load .env from extractions-202512 directory (same directory as the main scripts)
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

try:
    import anthropic
except ImportError:
    raise ImportError(
        "anthropic is required but not installed. "
        "Please install it with: pip install anthropic"
    )

# tiktoken is required for accurate token counting
try:
    import tiktoken
except ImportError:
    raise ImportError(
        "tiktoken is required but not installed. "
        "Please install it with: pip install tiktoken"
    )


# Exponential backoff configuration
MAX_RETRIES = 5
INITIAL_BACKOFF = 4.0  # Start with 4 seconds
MAX_BACKOFF = 64.0     # Maximum 64 seconds between retries
BACKOFF_MULTIPLIER = 2.0  # Double the wait time each retry


def _extract_retry_delay(error: Exception) -> Optional[float]:
    """
    Extract suggested retry delay from Claude API error response.
    
    Args:
        error: The exception that occurred
        
    Returns:
        Suggested retry delay in seconds, or None if not found
    """
    
    # Try to extract from error message: "Please retry in X.XXs" or "retry_after: X.XXs"
    error_str = str(error)
    # Try "Please retry in X.XXs" pattern
    match = re.search(r'Please retry in ([\d.]+)s', error_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, AttributeError):
            pass
    # Try "retry_after: X.XXs" or "retry after.*X.XXs" pattern
    match = re.search(r'retry[_\s]*after[:\s]+([\d.]+)s', error_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, AttributeError):
            pass
    
    # Try to extract from error object attributes (if it's a structured error)
    try:
        # Check if error has retry_after attribute (Anthropic API errors may have this)
        if hasattr(error, 'retry_after'):
            retry_after = error.retry_after
            if isinstance(retry_after, (int, float)):
                return float(retry_after)
            # If it's a timedelta or similar, try to extract seconds
            if hasattr(retry_after, 'total_seconds'):
                return retry_after.total_seconds()
        
        # Check if error has response attribute (HTTP error)
        if hasattr(error, 'response'):
            response = error.response
            # Check for Retry-After header
            if hasattr(response, 'headers'):
                retry_after_header = response.headers.get('Retry-After')
                if retry_after_header:
                    try:
                        return float(retry_after_header)
                    except (ValueError, TypeError):
                        pass
    except (AttributeError, KeyError, ValueError, TypeError):
        pass
    
    # Try to parse as JSON if error string looks like JSON
    try:
        if '{' in error_str and ('retry' in error_str.lower() or 'after' in error_str.lower()):
            # Try to extract JSON-like structure from error string
            json_match = re.search(r'\{.*\}', error_str, re.DOTALL)
            if json_match:
                error_dict = json.loads(json_match.group(0))
                # Check for retry_after in various locations
                if 'retry_after' in error_dict:
                    retry_after = error_dict['retry_after']
                    if isinstance(retry_after, (int, float)):
                        return float(retry_after)
                if 'error' in error_dict and isinstance(error_dict['error'], dict):
                    if 'retry_after' in error_dict['error']:
                        retry_after = error_dict['error']['retry_after']
                        if isinstance(retry_after, (int, float)):
                            return float(retry_after)
    except (json.JSONDecodeError, KeyError, ValueError, AttributeError):
        pass
    
    return None


def _is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is retryable with exponential backoff.
    
    Args:
        error: The exception that occurred
        
    Returns:
        True if the error is retryable, False otherwise
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    
    # Check error type first (most reliable) - similar to Gemini factory
    if any(etype in error_type for etype in [
        "ratelimiterror", "ratelimit", "overloadederror", "overloaded",
        "apierror", "internalerror", "internal_server_error",
        "serviceunavailable", "timeout", "connection"
    ]):
        return True
    
    # Rate limiting errors (429)
    if "429" in error_str or "rate limit" in error_str or "quota" in error_str:
        return True
    
    # Overloaded errors (specific Anthropic error type)
    if any(keyword in error_str for keyword in [
        "overloaded_error", "overloaded", "service overloaded",
        "too many requests", "capacity"
    ]):
        return True
    
    # Transient server errors (500, 502, 503, 504)
    if any(code in error_str for code in ["500", "502", "503", "504"]):
        return True
    
    # Network/timeout errors
    if any(keyword in error_str for keyword in [
        "timeout", "connection", "network", "unavailable", 
        "service unavailable", "internal server error", "bad gateway",
        "gateway timeout", "temporarily"
    ]):
        return True
    
    # Anthropic API specific retryable errors
    if any(keyword in error_str for keyword in [
        "rate_limit_error", "internal_server_error",
        "api_error", "server_error", "temporary"
    ]):
        return True
    
    # Check for specific Anthropic exception types (case-insensitive)
    if error_type in ["ratelimiterror", "apierror", "internalerror", "overloadederror"]:
        return True
    
    # Don't retry on permanent errors (400, 401, 403, 404)
    # BUT: Don't exclude 429 errors - check that it's not a 429 first
    # (429 is already handled above, but we want to avoid false positives on "400" in "4000" etc)
    if not ("429" in error_str or "overloaded" in error_str):
        if any(code in error_str for code in ["400", "401", "403", "404"]):
            return False
    
    # Don't retry on model validation errors
    if any(keyword in error_str for keyword in [
        "not found", "does not exist", "invalid", "bad request",
        "model_not_found", "invalid_request", "authentication_error"
    ]):
        return False
    
    # Default: don't retry if we're not sure
    return False


def get_model_window_size(model_name: str) -> int:
    """
    Returns the context window size (in tokens) for popular Claude models.
    
    Args:
        model_name: Name of the Claude model
        
    Returns:
        Context window size in tokens, or default if model is not recognized
    """
    model_lower = model_name.lower()
    
    # Context window sizes for popular Claude models
    window_sizes = {
        # Claude 3.7 Sonnet series (current)
        "claude-3-7-sonnet": 200000,  # 200k tokens
        "claude-3-7-sonnet-20250219": 200000,
        
        # Claude 3.5 Haiku series (still active, cheaper)
        "claude-3-5-haiku": 200000,
        "claude-3-5-haiku-20241022": 200000,
        
        # Claude 3 Opus series
        "claude-3-opus": 200000,
        "claude-3-opus-20240229": 200000,
        
        # Claude 3 Sonnet series
        "claude-3-sonnet": 200000,
        "claude-3-sonnet-20240229": 200000,
        
        # Claude 3 Haiku series
        "claude-3-haiku": 200000,
        "claude-3-haiku-20240307": 200000,
        
        # Claude Opus 4 series (newer models)
        "claude-opus-4": 200000,
        "claude-opus-4-20250514": 200000,
        "claude-opus-4-1": 200000,
        "claude-opus-4-1-20250820": 200000,
        "claude-opus-4-5": 200000,
        
        # Claude Sonnet 4 series (newer models)
        "claude-sonnet-4": 200000,
        "claude-sonnet-4-20250514": 200000,
        "claude-sonnet-4-5": 200000,
        
        # Claude Haiku 4 series (newer models with native structured outputs)
        "claude-haiku-4-5": 200000,
        
        # Claude 2 series (legacy)
        "claude-2": 100000,  # 100k tokens
        "claude-2.1": 200000,  # 200k tokens
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
    return 200000  # Conservative default for Claude models


class ClaudeModel:
    """Wrapper for Claude API models."""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.window_size = get_model_window_size(model_name)
        
        # Create the Anthropic client
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        try:
            # Claude models use cl100k_base encoding (same as GPT-4)
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception as e:
            # Fallback: rough estimate if tiktoken fails
            # 1 token ≈ 4 characters for Norwegian/English text
            return len(text) // 4
    
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
        # Count tokens in system prompt
        system_tokens = self._count_tokens(system_prompt) if system_prompt else 0
        
        # Determine response tokens - use reasonable default if max_tokens is None or very large
        if max_tokens is None or max_tokens >= 100000:
            # For "all" or very large values, use a reasonable default for JSON responses
            # JSON responses are typically small, but allow some headroom
            response_tokens = 8192
        else:
            response_tokens = max_tokens
        
        # Reserve larger safety margin for API overhead, formatting, message structure, etc.
        # Claude API has overhead for message formatting, system/user role markers, etc.
        # Use 40k safety margin to account for potential token counting differences between
        # tiktoken and Anthropic's actual tokenizer, plus API overhead
        # Increased from 30k to 40k because tiktoken and Anthropic's tokenizer can differ significantly
        # and API overhead (message formatting, role markers, etc.) can add significant tokens
        safety_margin = 40000  # tokens - very large margin to ensure we stay under limit
        
        # Add extra margin for system prompt token counting inaccuracies
        # System prompt token counting might be slightly off, so add 20% buffer (increased from 15%)
        system_token_buffer = int(system_tokens * 0.20) if system_tokens > 0 else 0
        
        # Calculate available tokens for the user prompt
        # Total limit: window_size
        # Used by: system_prompt + user_prompt + response + overhead
        available_tokens = self.window_size - system_tokens - system_token_buffer - response_tokens - safety_margin
        
        # Ensure available_tokens is positive and reasonable
        if available_tokens <= 0:
            # If system prompt and response already exceed window, truncate more aggressively
            # Keep at least 5000 tokens for user prompt (but this shouldn't normally happen)
            available_tokens = max(5000, self.window_size - system_tokens - response_tokens - 45000)
        
        # Be extra conservative: use only 75% of available tokens to account for
        # potential differences in token counting between tiktoken and Anthropic's tokenizer
        # Reduced from 80% to 75% to be even more aggressive and avoid edge cases
        # The difference between tiktoken and Anthropic's actual tokenizer can be significant
        # and API overhead can add more tokens than expected
        available_tokens = int(available_tokens * 0.75)
        
        # Count tokens in the prompt
        prompt_tokens = self._count_tokens(prompt)
        
        # If prompt fits, return as-is
        if prompt_tokens <= available_tokens:
            return prompt
        
        # Truncate prompt to fit within available tokens
        # Use tiktoken to truncate accurately at token boundaries
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            encoded = enc.encode(prompt)
            if len(encoded) > available_tokens:
                # available_tokens is already reduced to 80% for safety, so truncate directly
                truncated_encoded = encoded[:available_tokens]
                truncated_prompt = enc.decode(truncated_encoded)
                
                # Double-check: verify that system + truncated prompt doesn't exceed limit
                # This is a final safety check to ensure we're well under the limit
                total_estimated = system_tokens + len(truncated_encoded) + response_tokens
                # Use a larger buffer (10k) to account for API overhead and token counting differences
                if total_estimated > self.window_size - 10000:  # Leave 10k buffer
                    # If still too large, truncate even more aggressively
                    # Reserve 50k for safety (system + response + overhead + buffer)
                    max_user_tokens = self.window_size - system_tokens - response_tokens - 50000
                    if max_user_tokens > 0:
                        truncated_encoded = encoded[:max_user_tokens]
                        truncated_prompt = enc.decode(truncated_encoded)
                
                return truncated_prompt
            else:
                return prompt
        except Exception:
            # Fallback: rough character-based truncation if tiktoken fails
            # Estimate: 4 chars per token, but be more conservative (use 3.5 to account for variations)
            available_chars = int(available_tokens * 3.5)
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
        Generate text using Claude API.
        
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
        
        # Build message parameters
        message_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": truncated_prompt}],
        }
        
        # Add system prompt if provided
        if system_prompt:
            message_params["system"] = system_prompt
        
        # Add temperature if provided
        if temperature is not None:
            message_params["temperature"] = temperature
        
        # max_tokens is required by Anthropic API
        # If None or very large (meaning "all"), use a reasonable default for JSON responses
        # JSON responses are typically small, but we want enough headroom
        if max_tokens is None or max_tokens >= 100000:
            api_max_tokens = 8192  # Reasonable default for structured JSON outputs
        else:
            api_max_tokens = max_tokens
        message_params["max_tokens"] = api_max_tokens
        
        # Check if model supports native structured outputs (beta feature)
        # Available for: Claude Sonnet 4.5, Opus 4.1, Opus 4.5, Haiku 4.5
        model_lower = self.model_name.lower()
        supports_native_structured_outputs = any(x in model_lower for x in [
            "claude-sonnet-4-5", "claude-opus-4-1", "claude-opus-4-5", "claude-haiku-4-5"
        ])
        
        # Initialize use_beta_api flag (will be set to True if using native structured outputs)
        use_beta_api = False
        
        # Add schema support if provided
        if json_schema:
            # Extract schema from nested structure
            if "schema" in json_schema:
                inner_schema = json_schema["schema"]
                if "parameters" in inner_schema:
                    # Old format: schema has "parameters" key
                    claude_schema = inner_schema["parameters"]
                elif "type" in inner_schema:
                    # New format: schema is already a JSON schema
                    claude_schema = inner_schema
                else:
                    claude_schema = inner_schema
            else:
                claude_schema = json_schema
            
            # Try to use native structured outputs if model supports it
            if supports_native_structured_outputs:
                # Use native structured outputs (beta feature)
                # This is more reliable than the tool-calling workaround
                try:
                    from anthropic import transform_schema
                    message_params["output_format"] = {
                        "type": "json_schema",
                        "schema": transform_schema(claude_schema),
                    }
                    # Use beta API with structured outputs header
                    message_params["betas"] = ["structured-outputs-2025-11-13"]
                    use_beta_api = True
                except ImportError:
                    # transform_schema not available, fall back to tool-calling
                    use_beta_api = False
            
            if not use_beta_api:
                # Claude API requires property keys to match pattern '^[a-zA-Z0-9_.-]{1,64}$'
                # Normalize property keys to ensure they match
                import copy
                normalized_schema = copy.deepcopy(claude_schema)
                if "properties" in normalized_schema:
                    # Create new properties dict with normalized keys
                    new_properties = {}
                    for key, value in normalized_schema["properties"].items():
                        # Normalize key: keep only allowed characters, max 64 chars
                        normalized_key = ''.join(c for c in key if c.isalnum() or c in '_.-')[:64]
                        if normalized_key and normalized_key != key:
                            # If key was changed, warn but use normalized version
                            import warnings
                            warnings.warn(f"Claude API: Property key '{key}' normalized to '{normalized_key}'")
                        new_properties[normalized_key if normalized_key else key] = value
                    normalized_schema["properties"] = new_properties
                
                # Claude uses tools parameter for structured outputs (fallback for older models)
                message_params["tools"] = [{
                    "name": json_schema.get("name", "extract_case_info"),
                    "description": json_schema.get("description", ""),
                    "input_schema": normalized_schema
                }]
                message_params["tool_choice"] = {"type": "tool", "name": json_schema.get("name", "extract_case_info")}
        
        # Generate content with exponential backoff
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                if use_beta_api:
                    # Use beta API for native structured outputs
                    response = self.client.beta.messages.create(**message_params)
                else:
                    # Use standard API (with tool-calling if schema provided)
                    response = self.client.messages.create(**message_params)
                # Success - break out of retry loop
                break
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                error_type = type(e).__name__
                
                # Check for model not found errors (but be more specific)
                # Only treat as model not found if it's explicitly about the model
                is_model_error = (
                    any(keyword in error_str for keyword in ["404", "model_not_found", "invalid_request_error"]) and
                    ("model" in error_str or "does not exist" in error_str or "not found" in error_str) and
                    self.model_name.lower() in error_str
                )
                
                if is_model_error:
                    valid_models = [
                        "claude-3-7-sonnet", "claude-3-5-haiku-20241022", "claude-3-opus", "claude-3-sonnet", 
                        "claude-3-haiku-20240307", "claude-opus-4-20250514", "claude-opus-4-1-20250820",
                        "claude-opus-4-5", "claude-sonnet-4-20250514", "claude-sonnet-4-5", 
                        "claude-haiku-4-5", "claude-2", "claude-2.1"
                    ]
                    raise ValueError(
                        f"Claude model '{self.model_name}' does not exist or you do not have access to it. "
                        f"Valid models are: {', '.join(valid_models)}"
                    ) from e
                
                # Check for API key issues - don't retry these
                if any(keyword in error_str for keyword in ["authentication", "api key", "unauthorized", "401", "403"]):
                    raise ValueError(
                        f"API key authentication failed. Please check your ANTHROPIC_API_KEY environment variable."
                    ) from e
                
                # Check if error is retryable
                if not _is_retryable_error(e):
                    # Not retryable - raise immediately
                    raise
                
                # If this is the last attempt, raise the error
                if attempt == MAX_RETRIES - 1:
                    raise
                
                # Try to extract suggested retry delay from error
                suggested_delay = _extract_retry_delay(e)
                if suggested_delay is not None:
                    # Use suggested delay, rounded up to nearest second, but cap it at MAX_BACKOFF
                    backoff_time = min(math.ceil(suggested_delay), MAX_BACKOFF)
                else:
                    # Calculate backoff time with exponential backoff
                    backoff_time = min(INITIAL_BACKOFF * (BACKOFF_MULTIPLIER ** attempt), MAX_BACKOFF)
                
                # Log retry attempt
                delay_source = "suggested" if suggested_delay is not None else "exponential backoff"
                print(
                    f"Retryable error encountered (attempt {attempt + 1}/{MAX_RETRIES}): {type(e).__name__}: {str(e)[:100]}. "
                    f"Retrying in {backoff_time:.1f} seconds ({delay_source})...",
                    file=sys.stderr
                )
                
                time.sleep(backoff_time)
        
        # Return a response object that mimics OpenAI's response structure
        # for compatibility with extract_structured_data.py
        class ClaudeResponse:
            def __init__(self, claude_response, is_beta_structured_output=False):
                self._response = claude_response
                self._is_beta_structured_output = is_beta_structured_output
                
                # Extract text from Claude response
                if is_beta_structured_output:
                    # Native structured outputs: response.content[0].text contains the JSON string
                    if claude_response.content and len(claude_response.content) > 0:
                        self.text = claude_response.content[0].text
                    else:
                        self.text = ""
                else:
                    # Tool-calling approach: extract from tool_use blocks
                    # Claude returns content as a list of content blocks
                    # Each block can be a TextBlock or ToolUseBlock
                    # When using structured outputs (tools), the response is in tool_use blocks
                    text_parts = []
                    if claude_response.content:
                        for block in claude_response.content:
                            # Handle ToolUseBlock (structured outputs) - extract input as JSON string
                            if hasattr(block, 'type') and block.type == 'tool_use':
                                # Tool use block contains the structured output
                                if hasattr(block, 'input'):
                                    import json
                                    text_parts.append(json.dumps(block.input, ensure_ascii=False))
                                elif hasattr(block, 'content'):
                                    # Some versions use content
                                    text_parts.append(str(block.content))
                            # Handle TextBlock (has .text attribute)
                            elif hasattr(block, 'text'):
                                text_parts.append(block.text)
                            # Handle string blocks
                            elif isinstance(block, str):
                                text_parts.append(block)
                            # Handle dict-like blocks (TextBlockContent or ToolUseBlockContent)
                            elif isinstance(block, dict):
                                if block.get('type') == 'tool_use' and 'input' in block:
                                    import json
                                    text_parts.append(json.dumps(block['input'], ensure_ascii=False))
                                elif 'text' in block:
                                    text_parts.append(block['text'])
                                else:
                                    text_parts.append(str(block))
                            # Try to convert to string as fallback
                            else:
                                text_parts.append(str(block))
                    
                    self.text = "".join(text_parts) if text_parts else ""
                
                # Create a choices-like structure for compatibility
                class Choice:
                    def __init__(self, text):
                        class Message:
                            def __init__(self, content):
                                self.content = content
                        self.message = Message(text)
                
                self.choices = [Choice(self.text)]
        
        return ClaudeResponse(response, is_beta_structured_output=use_beta_api)


def normalize_claude_model_name(model_name: str) -> str:
    """
    Normalize Claude model name. The API accepts both short and versioned names,
    but some models work better with versioned names. This function preserves
    the input if it's already valid, or converts to versioned format for models
    that require it.
    
    Args:
        model_name: Name of the Claude model (may be short or versioned)
        
    Returns:
        Model name that works with the API (may be short or versioned)
    """
    model_lower = model_name.lower()
    
    # If already versioned, return as-is
    if model_name in [
        "claude-3-7-sonnet-20250219",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-opus-4-20250514",
        "claude-opus-4-1-20250820",
        "claude-sonnet-4-20250514",
        "claude-2", "claude-2.1"
    ]:
        return model_name
    
    # For models that the API accepts in short form, keep them short
    # The API accepts: claude-3-7-sonnet, claude-3-opus, claude-3-sonnet, etc.
    # Only normalize models that definitely need versioning
    if model_lower in ["claude-3-5-haiku", "claude-3-haiku", "claude-sonnet-4", "claude-opus-4", "claude-opus-4-1"]:
        # These models need versioning
        mapping = {
            "claude-3-5-haiku": "claude-3-5-haiku-20241022",
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-sonnet-4": "claude-sonnet-4-20250514",
            "claude-opus-4": "claude-opus-4-20250514",
            "claude-opus-4-1": "claude-opus-4-1-20250820",
        }
        return mapping.get(model_lower, model_name)
    
    # For other models, keep the short name (API accepts both)
    # claude-3-7-sonnet, claude-3-opus, claude-3-sonnet work as-is
    return model_name


def validate_claude_model_name(model_name: str) -> bool:
    """
    Validate that a Claude model name is recognized.
    
    Args:
        model_name: Name of the Claude model
        
    Returns:
        True if model name is recognized, False otherwise
    """
    model_lower = model_name.lower()
    
    # List of valid Claude model names
    valid_models = [
        # Claude 3.7 Sonnet series (current)
        "claude-3-7-sonnet",
        "claude-3-7-sonnet-20250219",
        
        # Claude 3.5 Haiku series (still active, cheaper)
        "claude-3-5-haiku",
        "claude-3-5-haiku-20241022",
        
        # Claude 3 Opus series
        "claude-3-opus",
        "claude-3-opus-20240229",
        
        # Claude 3 Sonnet series
        "claude-3-sonnet",
        "claude-3-sonnet-20240229",
        
        # Claude 3 Haiku series
        "claude-3-haiku",
        "claude-3-haiku-20240307",
        
        # Claude Opus 4 series (newer models)
        "claude-opus-4",
        "claude-opus-4-20250514",
        "claude-opus-4-1",
        "claude-opus-4-1-20250820",
        "claude-opus-4-5",
        
        # Claude Sonnet 4 series (newer models)
        "claude-sonnet-4",
        "claude-sonnet-4-20250514",
        "claude-sonnet-4-5",
        
        # Claude Haiku 4 series (newer models with native structured outputs)
        "claude-haiku-4-5",
        
        # Claude 2 series (legacy)
        "claude-2",
        "claude-2.1",
    ]
    
    # Check exact match
    if model_name in valid_models:
        return True
    
    # Check case-insensitive match
    for valid in valid_models:
        if valid.lower() == model_lower:
            return True
    
    # Check partial match (e.g., "claude-3-5-sonnet-20241022" matches "claude-3-5-sonnet")
    # Only allow input to start with valid model (input is longer/versioned), not the reverse
    for valid in valid_models:
        if model_lower.startswith(valid.lower() + "-"):
            return True
    
    return False


class ClaudeFactory:
    """Factory for creating Claude model instances."""
    
    def create(self, model_name: str):
        """
        Create a Claude model instance.
        
        Args:
            model_name: Name of the Claude model (e.g., "claude-3-5-sonnet")
            
        Returns:
            ClaudeModel instance
            
        Raises:
            ValueError: If ANTHROPIC_API_KEY is not set or model name is invalid
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Please set it in your .env file or environment variables."
            )
        
        # Validate model name early
        if not validate_claude_model_name(model_name):
            valid_models = [
                "claude-3-7-sonnet", "claude-3-5-haiku-20241022", "claude-3-opus", "claude-3-sonnet", 
                "claude-3-haiku-20240307", "claude-opus-4-20250514", "claude-opus-4-1-20250820",
                "claude-opus-4-5", "claude-sonnet-4-20250514", "claude-sonnet-4-5", 
                "claude-haiku-4-5", "claude-2", "claude-2.1"
            ]
            raise ValueError(
                f"Invalid Claude model name: '{model_name}'. "
                f"Valid models are: {', '.join(valid_models)}"
            )
        
        # Normalize model name to versioned format required by API
        normalized_model_name = normalize_claude_model_name(model_name)
        
        # Try to create the model to validate it exists
        try:
            model = ClaudeModel(api_key=api_key, model_name=normalized_model_name)
            # Test that the model can be accessed (this will fail if model doesn't exist)
            # We can't easily test without making an API call, so we'll let it fail on first use
            # but at least we've validated the model name format
            return model
        except Exception as e:
            # If it's a model not found error, raise a clearer error
            error_str = str(e).lower()
            if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
                # Show both short and versioned names that work
                valid_models = [
                    "claude-3-7-sonnet", "claude-3-5-haiku-20241022", "claude-3-opus", 
                    "claude-3-sonnet", "claude-3-haiku-20240307", "claude-opus-4-20250514", 
                    "claude-opus-4-1-20250820", "claude-opus-4-5", "claude-sonnet-4-20250514", 
                    "claude-sonnet-4-5", "claude-haiku-4-5", "claude-2", "claude-2.1"
                ]
                raise ValueError(
                    f"Claude model '{normalized_model_name}' does not exist or you do not have access to it. "
                    f"Valid models are: {', '.join(valid_models)}"
                ) from e
            # Check for API key issues
            if any(keyword in error_str for keyword in ["authentication", "api key", "unauthorized", "401", "403"]):
                raise ValueError(
                    f"API key authentication failed. Please check your ANTHROPIC_API_KEY environment variable."
                ) from e
            raise

