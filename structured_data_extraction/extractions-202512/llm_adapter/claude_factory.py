"""
Claude factory for creating Claude model instances.
"""

import os
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
        # Claude models have large context windows, so truncation is rarely needed
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
        
        # max_tokens is required by Anthropic API - use provided value or default to 4096
        message_params["max_tokens"] = max_tokens if max_tokens is not None else 4096
        
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
        
        # Generate content
        try:
            if use_beta_api:
                # Use beta API for native structured outputs
                response = self.client.beta.messages.create(**message_params)
            else:
                # Use standard API (with tool-calling if schema provided)
                response = self.client.messages.create(**message_params)
        except Exception as e:
            # Check for model not found errors and provide clearer error message
            error_str = str(e).lower()
            # Check for specific error types from Anthropic API
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
            # Check for API key issues
            if any(keyword in error_str for keyword in ["authentication", "api key", "unauthorized", "401", "403"]):
                raise ValueError(
                    f"API key authentication failed. Please check your ANTHROPIC_API_KEY environment variable."
                ) from e
            # For other errors (like tool/schema errors), re-raise as-is
            raise
        
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

