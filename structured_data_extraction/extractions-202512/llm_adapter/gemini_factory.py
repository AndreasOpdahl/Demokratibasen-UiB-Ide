"""
Gemini factory for creating Gemini model instances.
"""

import os
import json
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
                
                # Add schema support if provided (proper structured outputs)
                # IMPORTANT: Use response_json_schema instead of response_schema to avoid
                # client-side validation issues with additionalProperties
                if json_schema:
                    config.response_mime_type = "application/json"
                    # Extract schema from nested structure
                    if "schema" in json_schema:
                        inner_schema = json_schema["schema"]
                        if "parameters" in inner_schema:
                            # Old format: schema has "parameters" key
                            gemini_schema = inner_schema["parameters"]
                        elif "type" in inner_schema:
                            # New format: schema is already a JSON schema
                            gemini_schema = inner_schema
                        else:
                            gemini_schema = inner_schema
                    else:
                        gemini_schema = json_schema
                    
                    # Use response_json_schema to bypass client-side validation issues
                    config.response_json_schema = gemini_schema
                
                # Generate content using new API
                try:
                    response = self.client.models.generate_content(
                        model=self.model_id,
                        contents=truncated_prompt,
                        config=config
                    )
                except Exception as e:
                    # Check for model not found errors and provide clearer error message
                    error_str = str(e).lower()
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
                    raise
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
                
                # Add schema support if provided (proper structured outputs)
                # IMPORTANT: Use response_json_schema instead of response_schema to avoid
                # client-side validation issues with additionalProperties
                if json_schema:
                    config["response_mime_type"] = "application/json"
                    # Extract schema from nested structure
                    if "schema" in json_schema:
                        inner_schema = json_schema["schema"]
                        if "parameters" in inner_schema:
                            # Old format: schema has "parameters" key
                            gemini_schema = inner_schema["parameters"]
                        elif "type" in inner_schema:
                            # New format: schema is already a JSON schema
                            gemini_schema = inner_schema
                        else:
                            gemini_schema = inner_schema
                    else:
                        gemini_schema = json_schema
                    
                    # Use response_json_schema to bypass client-side validation issues
                    config["response_json_schema"] = gemini_schema
                
                # Generate content using new API with dict config
                try:
                    response = self.client.models.generate_content(
                        model=self.model_id,
                        contents=truncated_prompt,
                        config=config if config else None
                    )
                except Exception as e:
                    # Check for model not found errors and provide clearer error message
                    error_str = str(e).lower()
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
                    raise
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
            
            try:
                response = self.model.generate_content(
                    content_parts,
                    generation_config=generation_config if generation_config else None
                )
            except Exception as e:
                # Check for model not found errors and provide clearer error message
                error_str = str(e).lower()
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
                raise
        
        # Return a response object that mimics OpenAI's response structure
        # for compatibility with extract_structured_data.py
        class GeminiResponse:
            def __init__(self, gemini_response, is_new_api=False):
                self._response = gemini_response
                self._is_new_api = is_new_api
                
                # Handle new API format (google.genai)
                if is_new_api:
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
                
                # Create a choices-like structure for compatibility
                class Choice:
                    def __init__(self, text):
                        class Message:
                            def __init__(self, content):
                                self.content = content
                        self.message = Message(text)
                
                self.choices = [Choice(self.text)]
        
        return GeminiResponse(response, is_new_api=self.use_new_api)


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

