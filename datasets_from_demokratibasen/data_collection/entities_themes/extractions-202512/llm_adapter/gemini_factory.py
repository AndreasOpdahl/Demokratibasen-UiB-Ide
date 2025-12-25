"""
Gemini factory for creating Gemini model instances.
"""

import os
import json
import math
import random
import re
import sys
import time
import copy
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Load .env from extractions-202512 directory (same directory as the main scripts)
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

try:
    from google import genai
except ImportError:
    try:
        import google.generativeai as genai_legacy
        genai = None  # Will use legacy API
    except ImportError:
        raise ImportError(
            "google-genai or google-generativeai is required but not installed. "
            "Please install it with: pip install google-genai"
        )


# Exponential backoff configuration
MAX_RETRIES = 7  # Increased from 5 to 7 as per best practices
INITIAL_BACKOFF = 5.0  # Start with 5 seconds (increased for stronger backoff)
MAX_BACKOFF = 120.0     # Maximum 120 seconds between retries (increased from 60)
BACKOFF_MULTIPLIER = 2.5  # Multiply wait time by 2.5 each retry (increased from 2.0)
JITTER_FRACTION = 0.1  # Add ±10% random jitter to prevent thundering herd

# Client-side throttling configuration
# Add a delay between requests to avoid RPM limits
REQUEST_DELAY_MIN = 2.0  # Minimum delay between requests (seconds) - increased from 0.5
REQUEST_DELAY_MAX = 4.0  # Maximum delay between requests (seconds) - increased from 1.0


def _extract_retry_delay(error: Exception) -> Optional[float]:
    """
    Extract suggested retry delay from Gemini API error response.
    
    Args:
        error: The exception that occurred
        
    Returns:
        Suggested retry delay in seconds, or None if not found
    """
    
    # Try to extract from error message: "Please retry in X.XXs" or "retryDelay: X.XXs"
    error_str = str(error)
    # Try "Please retry in X.XXs" pattern
    match = re.search(r'Please retry in ([\d.]+)s', error_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, AttributeError):
            pass
    # Try "retryDelay: X.XXs" or "retry delay.*X.XXs" pattern
    match = re.search(r'retry\s*delay[:\s]+([\d.]+)s', error_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, AttributeError):
            pass
    
    # Try to extract from error object attributes (if it's a structured error)
    try:
        # Check if error has details attribute (Google API errors often do)
        if hasattr(error, 'details'):
            details = error.details
            if isinstance(details, (list, tuple)):
                for detail in details:
                    # Look for RetryInfo
                    if hasattr(detail, 'retry_delay'):
                        delay = detail.retry_delay
                        if hasattr(delay, 'seconds'):
                            return float(delay.seconds)
                    # Or if it's a dict-like structure
                    if isinstance(detail, dict):
                        if '@type' in detail and 'RetryInfo' in detail.get('@type', ''):
                            if 'retryDelay' in detail:
                                delay_str = detail['retryDelay']
                                # Extract seconds from "34s" format
                                match = re.search(r'(\d+(?:\.\d+)?)s?', str(delay_str))
                                if match:
                                    return float(match.group(1))
    except (AttributeError, KeyError, ValueError, TypeError):
        pass
    
    # Try to parse as JSON if error string looks like JSON
    try:
        if '{' in error_str and 'retryDelay' in error_str.lower():
            # Try to extract JSON-like structure from error string
            json_match = re.search(r'\{.*\}', error_str, re.DOTALL)
            if json_match:
                error_dict = json.loads(json_match.group(0))
                # Check details array for RetryInfo
                if 'details' in error_dict and isinstance(error_dict['details'], list):
                    for detail in error_dict['details']:
                        if isinstance(detail, dict):
                            if '@type' in detail and 'RetryInfo' in detail.get('@type', ''):
                                if 'retryDelay' in detail:
                                    delay_str = detail['retryDelay']
                                    match = re.search(r'(\d+(?:\.\d+)?)s?', str(delay_str))
                                    if match:
                                        return float(match.group(1))
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
    
    # Check error type first (most reliable)
    if "resourceexhausted" in error_type or "resource_exhausted" in error_type:
        return True
    if "clienterror" in error_type:
        # Check if it's a 429 ClientError
        if "429" in error_str:
            return True
    
    # Rate limiting errors (429) - check multiple patterns
    if "429" in error_str or "rate limit" in error_str or "quota" in error_str:
        return True
    
    # RESOURCE_EXHAUSTED errors (429 equivalent)
    if "resource_exhausted" in error_str or "resource has been exhausted" in error_str:
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
    
    # Google API specific retryable errors
    if any(keyword in error_str for keyword in [
        "resource_exhausted", "unavailable", "deadline_exceeded",
        "aborted", "internal"
    ]):
        return True
    
    # Don't retry on permanent errors (400, 401, 403, 404)
    # BUT: Don't exclude 429 errors - check that it's not a 429 first
    # (429 is already handled above, but we want to avoid false positives on "400" in "4000" etc)
    if not ("429" in error_str or "resource_exhausted" in error_str):
        if any(code in error_str for code in ["400", "401", "403", "404"]):
            return False
    
    # Don't retry on model validation errors
    if any(keyword in error_str for keyword in [
        "not found", "does not exist", "invalid", "bad request",
        "model_not_found", "invalid_request"
    ]):
        return False
    
    # Default: don't retry if we're not sure
    return False


def _sanitize_schema_for_gemini(schema: dict) -> dict:
    """
    Sanitize JSON schema for use with Gemini function declarations.
    
    The Gemini Tools / FunctionDeclaration API is stricter than general JSON Schema
    and will reject unknown fields like "additional_properties". This helper:
      - Recursively removes any "additional_properties" / "additionalProperties" keys
      - Returns a deep-copied schema so the original is not mutated.
    """
    if not isinstance(schema, dict):
        return schema

    def _sanitize(obj: dict) -> dict:
        if not isinstance(obj, dict):
            return obj
        cleaned = {}
        for k, v in obj.items():
            # Drop any additional_properties / additionalProperties fields – we enforce
            # this constraint client-side in our own validator instead.
            if k in ("additional_properties", "additionalProperties"):
                continue
            # Recurse into nested dict / list structures
            if isinstance(v, dict):
                cleaned[k] = _sanitize(v)
            elif isinstance(v, list):
                cleaned[k] = [
                    _sanitize(item) if isinstance(item, dict) else item
                    for item in v
                ]
            else:
                cleaned[k] = v
        return cleaned

    # Work on a deep copy to avoid mutating the caller's schema
    return _sanitize(copy.deepcopy(schema))


def get_model_window_size(model_name: str) -> int:
    """
    Returns the context window size (in tokens) for popular Gemini models.
    
    Args:
        model_name: Name of the Gemini model
        
    Returns:
        Context window size in tokens, or default if model is not recognized
    """
    model_lower = model_name.lower()
    
    # Context window sizes for popular Gemini models
    window_sizes = {
        # Gemini 2.5 series
        "gemini-2.5-flash": 1000000,  # 1M tokens
        "gemini-2.5-pro": 2000000,    # 2M tokens
        "models/gemini-2.5-flash": 1000000,
        "models/gemini-2.5-pro": 2000000,
        
        # Gemini 1.5 series
        "gemini-1.5-flash": 1000000,  # 1M tokens
        "gemini-1.5-pro": 2000000,    # 2M tokens
        "models/gemini-1.5-flash": 1000000,
        "models/gemini-1.5-pro": 2000000,
        
        # Gemini 1.0 series
        "gemini-pro": 32768,
        "gemini-pro-vision": 16384,
        "models/gemini-pro": 32768,
        "models/gemini-pro-vision": 16384,
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
    return 1000000  # Conservative default for Gemini models


class GeminiModel:
    """Wrapper for Gemini API models."""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.window_size = get_model_window_size(model_name)
        
        # Use the new google.genai client if available, otherwise fall back to legacy
        if genai is not None:
            # New google.genai client
            self.client = genai.Client(api_key=api_key)
            # For new API, use model name directly (without models/ prefix)
            # Remove models/ prefix if present, as the new API doesn't need it
            self.model_id = model_name.replace("models/", "") if model_name.startswith("models/") else model_name
            self.use_new_api = True
        else:
            # Legacy google.generativeai client
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=api_key)
            model_id = model_name if model_name.startswith("models/") else f"models/{model_name}"
            self.model = genai_legacy.GenerativeModel(model_id)
            self.use_new_api = False
    
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
        # Gemini models have very large context windows, so truncation is rarely needed
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
        Generate text using Gemini API.
        
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
        
        # Use new google.genai client if available
        if self.use_new_api:
            # Build config using GenerateContentConfig for new API
            try:
                from google.genai import types
                
                # Create GenerateContentConfig object
                config = types.GenerateContentConfig()
                
                if temperature is not None:
                    config.temperature = temperature
                if max_tokens is not None:
                    config.max_output_tokens = max_tokens
                
                # System instruction
                if system_prompt:
                    config.system_instruction = system_prompt
                
                # Configure safety settings to be less restrictive for structured data extraction
                # This helps avoid false positives when extracting from municipal documents
                config.safety_settings = [
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH
                    ),
                ]
                
                # Add schema support using Function Calling / Tools (more robust than response_json_schema)
                if json_schema:
                    # Extract schema and metadata from nested structure
                    tool_name = json_schema.get("name", "extract_data")
                    tool_description = json_schema.get("description", "Extract structured information from the text.")
                    
                    # Extract the actual schema
                    if "schema" in json_schema:
                        inner_schema = json_schema["schema"]
                        if "parameters" in inner_schema:
                            # Old format: schema has "parameters" key
                            function_schema = inner_schema["parameters"]
                        elif "type" in inner_schema:
                            # New format: schema is already a JSON schema
                            function_schema = inner_schema
                        else:
                            function_schema = inner_schema
                    else:
                        function_schema = json_schema

                    # Sanitize schema for Gemini tools API (remove unsupported fields
                    # like additional_properties / additionalProperties)
                    function_schema = _sanitize_schema_for_gemini(function_schema)
                    
                    # Create tool definition using Function Calling
                    tool_def = types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(
                                name=tool_name,
                                description=tool_description,
                                parameters=function_schema
                            )
                        ]
                    )
                    
                    # Add tools to config (this forces the model to use function calling)
                    config.tools = [tool_def]
                
                # Generate content using new API with exponential backoff
                last_error = None
                for attempt in range(MAX_RETRIES):
                    try:
                        response = self.client.models.generate_content(
                            model=self.model_id,
                            contents=truncated_prompt,
                            config=config
                        )
                        # Success - break out of retry loop
                        break
                    except Exception as e:
                        last_error = e
                        error_str = str(e).lower()
                        
                        # Check for model not found errors - don't retry these
                        if any(keyword in error_str for keyword in ["404", "not found", "does not exist", "model_not_found", "invalid_request_error"]):
                            valid_models = [
                                "gemini-2.5-flash", "gemini-2.5-pro",
                                "gemini-1.5-flash", "gemini-1.5-pro",
                                "gemini-pro", "gemini-pro-vision"
                            ]
                            raise ValueError(
                                f"Gemini model '{self.model_name}' does not exist or you do not have access to it. "
                                f"Valid models are: {', '.join(valid_models)}"
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
                            base_backoff = min(math.ceil(suggested_delay), MAX_BACKOFF)
                        else:
                            # Calculate backoff time with exponential backoff
                            base_backoff = min(INITIAL_BACKOFF * (BACKOFF_MULTIPLIER ** attempt), MAX_BACKOFF)
                        
                        # Add jitter to prevent thundering herd problem
                        # Jitter: ±10% random variation
                        jitter = base_backoff * JITTER_FRACTION * (2 * random.random() - 1)  # Range: -10% to +10%
                        backoff_time = max(0.1, base_backoff + jitter)  # Ensure at least 0.1 seconds
                        
                        # Log retry attempt
                        delay_source = "suggested" if suggested_delay is not None else "exponential backoff"
                        print(
                            f"Retryable error encountered (attempt {attempt + 1}/{MAX_RETRIES}): {type(e).__name__}: {str(e)[:100]}. "
                            f"Retrying in {backoff_time:.1f} seconds ({delay_source} with jitter)...",
                            file=sys.stderr
                        )
                        
                        time.sleep(backoff_time)
            except ImportError:
                # Fallback if google.genai.types is not available - use dict format
                config = {}
                
                if temperature is not None:
                    config["temperature"] = temperature
                if max_tokens is not None:
                    config["max_output_tokens"] = max_tokens
                
                # System instruction
                if system_prompt:
                    config["system_instruction"] = system_prompt
                
                # Add schema support using Function Calling / Tools (when types not available, use dict format)
                if json_schema:
                    # Extract schema and metadata from nested structure
                    tool_name = json_schema.get("name", "extract_data")
                    tool_description = json_schema.get("description", "Extract structured information from the text.")
                    
                    # Extract the actual schema
                    if "schema" in json_schema:
                        inner_schema = json_schema["schema"]
                        if "parameters" in inner_schema:
                            # Old format: schema has "parameters" key
                            function_schema = inner_schema["parameters"]
                        elif "type" in inner_schema:
                            # New format: schema is already a JSON schema
                            function_schema = inner_schema
                        else:
                            function_schema = inner_schema
                    else:
                        function_schema = json_schema

                    # Sanitize schema for Gemini tools API
                    function_schema = _sanitize_schema_for_gemini(function_schema)
                    
                    # Create tool definition using dict format (for fallback when types not available)
                    tool_def = {
                        "function_declarations": [
                            {
                                "name": tool_name,
                                "description": tool_description,
                                "parameters": function_schema
                            }
                        ]
                    }
                    
                    # Add tools to config (this forces the model to use function calling)
                    config["tools"] = [tool_def]
                
                # Generate content using new API with dict config and exponential backoff
                last_error = None
                for attempt in range(MAX_RETRIES):
                    try:
                        response = self.client.models.generate_content(
                            model=self.model_id,
                            contents=truncated_prompt,
                            config=config if config else None
                        )
                        # Success - break out of retry loop
                        break
                    except Exception as e:
                        last_error = e
                        error_str = str(e).lower()
                        
                        # Check for model not found errors - don't retry these
                        if any(keyword in error_str for keyword in ["404", "not found", "does not exist", "model_not_found", "invalid_request_error"]):
                            valid_models = [
                                "gemini-2.5-flash", "gemini-2.5-pro",
                                "gemini-1.5-flash", "gemini-1.5-pro",
                                "gemini-pro", "gemini-pro-vision"
                            ]
                            raise ValueError(
                                f"Gemini model '{self.model_name}' does not exist or you do not have access to it. "
                                f"Valid models are: {', '.join(valid_models)}"
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
                            base_backoff = min(math.ceil(suggested_delay), MAX_BACKOFF)
                        else:
                            # Calculate backoff time with exponential backoff
                            base_backoff = min(INITIAL_BACKOFF * (BACKOFF_MULTIPLIER ** attempt), MAX_BACKOFF)
                        
                        # Add jitter to prevent thundering herd problem
                        # Jitter: ±10% random variation
                        jitter = base_backoff * JITTER_FRACTION * (2 * random.random() - 1)  # Range: -10% to +10%
                        backoff_time = max(0.1, base_backoff + jitter)  # Ensure at least 0.1 seconds
                        
                        # Log retry attempt
                        delay_source = "suggested" if suggested_delay is not None else "exponential backoff"
                        print(
                            f"Retryable error encountered (attempt {attempt + 1}/{MAX_RETRIES}): {type(e).__name__}: {str(e)[:100]}. "
                            f"Retrying in {backoff_time:.1f} seconds ({delay_source} with jitter)...",
                            file=sys.stderr
                        )
                        
                        time.sleep(backoff_time)
        else:
            # Fallback for legacy google.generativeai
            generation_config = {}
            if temperature is not None:
                generation_config["temperature"] = temperature
            if max_tokens is not None:
                generation_config["max_output_tokens"] = max_tokens
            
            # For older versions, combine system and user prompts
            content_parts = []
            if system_prompt:
                content_parts.append(system_prompt)
            content_parts.append(truncated_prompt)
            
            # Generate content using legacy API with exponential backoff
            last_error = None
            for attempt in range(MAX_RETRIES):
                try:
                    response = self.model.generate_content(
                        content_parts,
                        generation_config=generation_config if generation_config else None
                    )
                    # Success - break out of retry loop
                    break
                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()
                    
                    # Check for model not found errors - don't retry these
                    if any(keyword in error_str for keyword in ["404", "not found", "does not exist", "model_not_found", "invalid_request_error"]):
                        valid_models = [
                            "gemini-2.5-flash", "gemini-2.5-pro",
                            "gemini-1.5-flash", "gemini-1.5-pro",
                            "gemini-pro", "gemini-pro-vision"
                        ]
                        raise ValueError(
                            f"Gemini model '{self.model_name}' does not exist or you do not have access to it. "
                            f"Valid models are: {', '.join(valid_models)}"
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
                        base_backoff = min(math.ceil(suggested_delay), MAX_BACKOFF)
                    else:
                        # Calculate backoff time with exponential backoff
                        base_backoff = min(INITIAL_BACKOFF * (BACKOFF_MULTIPLIER ** attempt), MAX_BACKOFF)
                    
                    # Add jitter to prevent thundering herd problem
                    # Jitter: ±10% random variation
                    jitter = base_backoff * JITTER_FRACTION * (2 * random.random() - 1)  # Range: -10% to +10%
                    backoff_time = max(0.1, base_backoff + jitter)  # Ensure at least 0.1 seconds
                    
                    # Log retry attempt
                    delay_source = "suggested" if suggested_delay is not None else "exponential backoff"
                    print(
                        f"Retryable error encountered (attempt {attempt + 1}/{MAX_RETRIES}): {type(e).__name__}: {str(e)[:100]}. "
                        f"Retrying in {backoff_time:.1f} seconds ({delay_source} with jitter)...",
                        file=sys.stderr
                    )
                    
                    time.sleep(backoff_time)
        
        # Return a response object that mimics OpenAI's response structure
        # for compatibility with extract_structured_data.py
        class GeminiResponse:
            def __init__(self, gemini_response, is_new_api=False, used_tools=False):
                self._response = gemini_response
                self._is_new_api = is_new_api
                self._used_tools = used_tools
                
                # Handle new API format (google.genai)
                if is_new_api:
                    # Check if response contains function calls (tools)
                    if used_tools:
                        # Extract data from function call
                        self.text = self._extract_function_call_data(gemini_response)
                    else:
                        # New API: response.text gives the text directly
                        try:
                            self.text = gemini_response.text
                        except AttributeError:
                            # Fallback: try to get from candidates if available
                            if hasattr(gemini_response, 'candidates') and len(gemini_response.candidates) > 0:
                                candidate = gemini_response.candidates[0]
                                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                    self.text = candidate.content.parts[0].text
                                else:
                                    raise ValueError("No text content found in new API response")
                            else:
                                raise ValueError("No text content found in new API response")
                else:
                    # Legacy API: check for safety blocks or other finish reasons
                    candidates = getattr(gemini_response, 'candidates', None)
                    if candidates is not None:
                        # Try to get length - handle both real lists and mocks
                        try:
                            candidates_len = len(candidates)
                        except (TypeError, AttributeError):
                            # Mock object or other non-iterable - skip safety check
                            candidates_len = 0
                        
                        if candidates_len > 0:
                            candidate = candidates[0]
                            finish_reason = getattr(candidate, 'finish_reason', None)
                            
                            # Finish reason 2 = SAFETY (blocked by safety filters)
                            if finish_reason == 2:  # SAFETY
                                safety_ratings = getattr(candidate, 'safety_ratings', [])
                                blocked_categories = []
                                for rating in safety_ratings:
                                    if getattr(rating, 'probability', 0) > 1:  # HIGH or MEDIUM probability
                                        category = getattr(rating, 'category', None)
                                        if category:
                                            category_name = getattr(category, 'name', 'UNKNOWN')
                                            blocked_categories.append(category_name)
                                
                                error_msg = (
                                    f"Response blocked by Gemini safety filters. "
                                    f"Finish reason: SAFETY (code {finish_reason})"
                                )
                                if blocked_categories:
                                    error_msg += f". Blocked categories: {', '.join(blocked_categories)}"
                                
                                raise ValueError(error_msg)
                            
                            # Other finish reasons that might indicate issues
                            if finish_reason == 3:  # RECITATION (content matched training data)
                                raise ValueError(
                                    "Response blocked: Content matched training data (RECITATION). "
                                    "Try rephrasing the prompt."
                                )
                            if finish_reason == 4:  # OTHER
                                raise ValueError(
                                    f"Response blocked: Unknown reason (finish_reason={finish_reason})"
                                )
                    
                    # Try to get text, handling cases where it might not be available
                    try:
                        self.text = gemini_response.text
                    except (ValueError, AttributeError) as e:
                        # If text accessor fails, check finish reason
                        candidates = getattr(gemini_response, 'candidates', None)
                        if candidates is not None:
                            try:
                                candidates_len = len(candidates)
                            except (TypeError, AttributeError):
                                candidates_len = 0
                            
                            if candidates_len > 0:
                                candidate = candidates[0]
                                finish_reason = getattr(candidate, 'finish_reason', None)
                                if finish_reason == 2:  # SAFETY
                                    raise ValueError(
                                        f"Response blocked by Gemini safety filters. "
                                        f"Finish reason: SAFETY (code {finish_reason}). "
                                        f"Original error: {e}"
                                    )
                        # Re-raise original error if we can't determine the cause
                        raise ValueError(
                            f"Failed to extract text from Gemini response. "
                            f"Original error: {e}"
                        ) from e
            
            def _extract_function_call_data(self, response):
                """
                Extract data from function call in Gemini response.
                
                Args:
                    response: Gemini API response object (can be object or dict format)
                    
                Returns:
                    JSON string containing the function call arguments
                """
                import json
                
                # Try to extract from response.candidates[0].content.parts
                # Handle both object format (new API) and dict format (fallback)
                candidates = None
                if hasattr(response, 'candidates'):
                    candidates = response.candidates
                elif isinstance(response, dict) and 'candidates' in response:
                    candidates = response['candidates']
                
                if candidates and len(candidates) > 0:
                    candidate = candidates[0]
                    
                    # Handle object format
                    if hasattr(candidate, 'content'):
                        content = candidate.content
                        parts = None
                        if hasattr(content, 'parts'):
                            parts = content.parts
                    # Handle dict format
                    elif isinstance(candidate, dict) and 'content' in candidate:
                        content = candidate['content']
                        parts = content.get('parts') if isinstance(content, dict) else None
                    else:
                        parts = None
                    
                    if parts:
                        for part in parts:
                            # Handle object format
                            if hasattr(part, 'function_call'):
                                func_call = part.function_call
                                if hasattr(func_call, 'args'):
                                    # args should be a dict - convert to JSON string
                                    return json.dumps(func_call.args, ensure_ascii=False)
                            # Handle dict format
                            elif isinstance(part, dict) and 'function_call' in part:
                                func_call = part['function_call']
                                if isinstance(func_call, dict) and 'args' in func_call:
                                    args = func_call['args']
                                    if isinstance(args, dict):
                                        return json.dumps(args, ensure_ascii=False)
                            
                            # Also check for function_response (if using two-turn)
                            # Handle object format
                            if hasattr(part, 'function_response'):
                                func_response = part.function_response
                                if hasattr(func_response, 'response'):
                                    response_data = func_response.response
                                    if isinstance(response_data, dict):
                                        return json.dumps(response_data, ensure_ascii=False)
                                    elif isinstance(response_data, str):
                                        return response_data
                            # Handle dict format
                            elif isinstance(part, dict) and 'function_response' in part:
                                func_response = part['function_response']
                                if isinstance(func_response, dict) and 'response' in func_response:
                                    response_data = func_response['response']
                                    if isinstance(response_data, dict):
                                        return json.dumps(response_data, ensure_ascii=False)
                                    elif isinstance(response_data, str):
                                        return response_data
                
                # Try alternative: check response.text if it contains JSON
                try:
                    text = response.text
                    # If text looks like JSON, return it
                    if text and (text.strip().startswith('{') or text.strip().startswith('[')):
                        return text
                except (AttributeError, ValueError):
                    pass
                
                # If all else fails, try to find JSON in any text content
                try:
                    if hasattr(response, 'candidates') and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content'):
                            # Try to get any text content
                            content_text = str(candidate.content)
                            # Look for JSON pattern
                            import re
                            json_match = re.search(r'\{.*\}', content_text, re.DOTALL)
                            if json_match:
                                return json_match.group(0)
                except Exception:
                    pass
                
                raise ValueError(
                    "Failed to extract function call data from Gemini response. "
                    "Response does not contain function call or JSON data."
                )
                
                # Create a choices-like structure for compatibility
                class Choice:
                    def __init__(self, text):
                        class Message:
                            def __init__(self, content):
                                self.content = content
                        self.message = Message(text)
                
                self.choices = [Choice(self.text)]
        
        # Client-side throttling: Add a small delay between requests to avoid RPM limits
        # This helps prevent hitting unadvertised requests-per-minute limits
        throttle_delay = random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)
        time.sleep(throttle_delay)
        
        # Determine if tools were used (if json_schema was provided)
        used_tools = json_schema is not None
        return GeminiResponse(response, is_new_api=self.use_new_api, used_tools=used_tools)


def validate_gemini_model_name(model_name: str) -> bool:
    """
    Validate that a Gemini model name is recognized.
    
    Args:
        model_name: Name of the Gemini model
        
    Returns:
        True if model name is recognized, False otherwise
    """
    model_lower = model_name.lower()
    
    # List of valid Gemini model names
    valid_models = [
        # Gemini 2.5 series
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "models/gemini-2.5-flash",
        "models/gemini-2.5-pro",
        
        # Gemini 1.5 series
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro",
        
        # Gemini 1.0 series
        "gemini-pro",
        "gemini-pro-vision",
        "models/gemini-pro",
        "models/gemini-pro-vision",
    ]
    
    # Check exact match
    if model_name in valid_models:
        return True
    
    # Check case-insensitive match
    for valid in valid_models:
        if valid.lower() == model_lower:
            return True
    
    return False


class GeminiFactory:
    """Factory for creating Gemini model instances."""
    
    def create(self, model_name: str):
        """
        Create a Gemini model instance.
        
        Args:
            model_name: Name of the Gemini model (e.g., "gemini-2.5-flash")
            
        Returns:
            GeminiModel instance
            
        Raises:
            ValueError: If GEMINI_API_KEY is not set or model name is invalid
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Please set it in your .env file or environment variables."
            )
        
        # Validate model name early
        if not validate_gemini_model_name(model_name):
            valid_models = [
                "gemini-2.5-flash", "gemini-2.5-pro",
                "gemini-1.5-flash", "gemini-1.5-pro",
                "gemini-pro", "gemini-pro-vision"
            ]
            raise ValueError(
                f"Invalid Gemini model name: '{model_name}'. "
                f"Valid models are: {', '.join(valid_models)}"
            )
        
        # Try to create the model to validate it exists
        try:
            model = GeminiModel(api_key=api_key, model_name=model_name)
            # Test that the model can be accessed (this will fail if model doesn't exist)
            # We can't easily test without making an API call, so we'll let it fail on first use
            # but at least we've validated the model name format
            return model
        except Exception as e:
            # If it's a model not found error, raise a clearer error
            error_str = str(e).lower()
            if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
                valid_models = [
                    "gemini-2.5-flash", "gemini-2.5-pro",
                    "gemini-1.5-flash", "gemini-1.5-pro",
                    "gemini-pro", "gemini-pro-vision"
                ]
                raise ValueError(
                    f"Gemini model '{model_name}' does not exist or you do not have access to it. "
                    f"Valid models are: {', '.join(valid_models)}"
                ) from e
            raise

