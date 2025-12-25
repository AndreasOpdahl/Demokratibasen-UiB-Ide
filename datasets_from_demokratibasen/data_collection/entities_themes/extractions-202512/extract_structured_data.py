"""
Extract structured data from case documents.

"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from llm_adapter import LLMAdapter, get_factory
from dataset_loader import DatasetLoader
from create_prompt import Prompt


# Control variables (defaults)
TEMPERATURE = 0.1
MAX_TOKENS = 4096

OUTPUT_BASE_DIR = "extracted-data"

# Bad response monitoring
BAD_RESPONSE_WINDOW = 10  # Number of documents to track in sliding window
MAX_BAD_RESPONSES = 3     # Maximum bad responses allowed in window
MAX_EXTRA_PROPERTIES_ACCEPTED = 3  # Maximum extra properties allowed in JSON response


class BadResponseMonitor:
    """
    Monitors bad responses in a sliding window.
    Tracks the last BAD_RESPONSE_WINDOW documents and counts bad responses.
    Only terminates if MAX_BAD_RESPONSES or more bad responses occur in the window.
    """
    
    def __init__(self, window_size: int = BAD_RESPONSE_WINDOW, max_bad: int = MAX_BAD_RESPONSES):
        """
        Initialize the monitor.
        
        Args:
            window_size: Size of the sliding window
            max_bad: Maximum number of bad responses allowed in the window
        """
        self.window_size = window_size
        self.max_bad = max_bad
        self.responses = []  # List of (doc_id, is_bad) tuples
    
    def record_response(self, doc_id: str, is_bad: bool):
        """
        Record a response (good or bad) for a document.
        
        Args:
            doc_id: Document identifier
            is_bad: True if the response was bad, False if good
        """
        self.responses.append((doc_id, is_bad))
        
        # Keep only the last window_size responses
        if len(self.responses) > self.window_size:
            self.responses.pop(0)
    
    def should_terminate(self) -> tuple[bool, str]:
        """
        Check if the program should terminate based on bad response count.
        
        Returns:
            Tuple of (should_terminate, message)
            - should_terminate: True if program should terminate
            - message: Explanation message
        """
        if len(self.responses) < self.window_size:
            # Not enough responses in window yet
            return False, ""
        
        # Count bad responses in current window
        bad_count = sum(1 for _, is_bad in self.responses if is_bad)
        
        if bad_count >= self.max_bad:
            bad_doc_ids = [doc_id for doc_id, is_bad in self.responses if is_bad]
            return True, (
                f"Terminating: {bad_count} bad responses in last {self.window_size} documents "
                f"(max allowed: {self.max_bad}). "
                f"Bad document IDs: {', '.join(bad_doc_ids[-bad_count:])}"
            )
        
        return False, ""
    
    def get_stats(self) -> dict:
        """
        Get statistics about the current window.
        
        Returns:
            Dictionary with statistics
        """
        if not self.responses:
            return {"total": 0, "bad": 0, "good": 0, "window_size": self.window_size}
        
        bad_count = sum(1 for _, is_bad in self.responses if is_bad)
        good_count = len(self.responses) - bad_count
        
        return {
            "total": len(self.responses),
            "bad": bad_count,
            "good": good_count,
            "window_size": self.window_size,
            "max_bad_allowed": self.max_bad
        }


def _extract_token_usage(response) -> tuple[int, int]:
    """
    Extract token usage from response object.
    
    Args:
        response: Response object from LLM API
        
    Returns:
        Tuple of (input_tokens, output_tokens)
    """
    input_tokens = 0
    output_tokens = 0
    
    # Try accessing through _response attribute first (for wrapped responses like GeminiResponse, ClaudeResponse)
    if hasattr(response, '_response'):
        return _extract_token_usage(response._response)
    
    # Try OpenAI-compatible format (GPT, DeepSeek, Mistral, Qwen)
    # Check for prompt_tokens/completion_tokens first (OpenAI format)
    if hasattr(response, 'usage'):
        usage = response.usage
        if hasattr(usage, 'prompt_tokens'):
            input_tokens = usage.prompt_tokens or 0
        if hasattr(usage, 'completion_tokens'):
            output_tokens = usage.completion_tokens or 0
        # If we got tokens, return them
        if input_tokens > 0 or output_tokens > 0:
            return (input_tokens, output_tokens)
        # Otherwise, try Claude format (input_tokens/output_tokens)
        if hasattr(usage, 'input_tokens'):
            input_tokens = usage.input_tokens or 0
        if hasattr(usage, 'output_tokens'):
            output_tokens = usage.output_tokens or 0
    
    # Try Gemini format
    elif hasattr(response, 'usage_metadata'):
        usage_metadata = response.usage_metadata
        if hasattr(usage_metadata, 'prompt_token_count'):
            input_tokens = usage_metadata.prompt_token_count or 0
        if hasattr(usage_metadata, 'candidates_token_count'):
            output_tokens = usage_metadata.candidates_token_count or 0
    
    return (input_tokens, output_tokens)


def _extract_json_from_response(response_text: str) -> dict:
    """
    Extract JSON from response text, handling markdown code blocks if present.
    
    Args:
        response_text: Raw response text that may contain JSON or markdown-wrapped JSON
        
    Returns:
        Parsed JSON as dictionary
        
    Raises:
        ValueError: If response_text is None, empty, or no valid JSON could be extracted
    """
    import re
    
    # Handle None or empty input
    if response_text is None:
        raise ValueError("Could not extract valid JSON from response: response text is None")
    
    if not isinstance(response_text, str):
        raise ValueError(f"Could not extract valid JSON from response: expected string, got {type(response_text).__name__}")
    
    if not response_text.strip():
        raise ValueError("Could not extract valid JSON from response: response text is empty")
    
    # Limit the amount of text we process to avoid slow regex on huge strings
    # Process up to 100KB of text (should be more than enough for JSON extraction)
    MAX_TEXT_LENGTH = 100000
    texts_to_search = []
    if len(response_text) > MAX_TEXT_LENGTH:
        # Try to find JSON in the first part first (most likely location)
        texts_to_search.append(response_text[:MAX_TEXT_LENGTH])
        # Also try the last part in case JSON is at the end
        texts_to_search.append(response_text[-MAX_TEXT_LENGTH:])
    else:
        texts_to_search.append(response_text)
    
    # Try to parse directly first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks (```json ... ``` or ``` ... ```)
    # Use non-greedy patterns to avoid backtracking issues
    json_patterns = [
        r'```json\s*\n(.*?)\n```',  # ```json ... ```
        r'```\s*\n(.*?)\n```',      # ``` ... ```
        r'```json\s*(.*?)```',      # ```json ... ``` (no newlines)
        r'```\s*(.*?)```',          # ``` ... ``` (no newlines)
    ]
    
    for text_to_search in texts_to_search:
        for pattern in json_patterns:
            match = re.search(pattern, text_to_search, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
    
    # Try to find JSON object boundaries { ... }
    # Use a more efficient approach: find first { and last } to avoid excessive backtracking
    for text_to_search in texts_to_search:
        # Find first opening brace
        first_brace = text_to_search.find('{')
        if first_brace == -1:
            continue
        # Find last closing brace starting from first brace
        last_brace = text_to_search.rfind('}', first_brace)
        if last_brace == -1 or last_brace <= first_brace:
            continue
        
        # Extract potential JSON
        potential_json = text_to_search[first_brace:last_brace + 1]
        try:
            return json.loads(potential_json)
        except json.JSONDecodeError:
            continue
    
    # If all else fails, raise an error with limited text preview
    preview_length = min(200, len(response_text))
    preview = response_text[:preview_length]
    if len(response_text) > preview_length:
        preview += "..."
    raise ValueError(f"Could not extract valid JSON from response (length: {len(response_text)} chars). Preview: {preview}")


def _extract_json_schema(schema_data: dict) -> dict:
    """
    Extract the actual JSON schema from schema data structure.
    
    Handles both formats:
    - Old format: {"schema": {"parameters": {...}}}
    - New format: {"schema": {"type": "object", "properties": {...}}}
    
    Args:
        schema_data: Schema data loaded from JSON file
        
    Returns:
        The actual JSON schema dictionary for validation
    """
    if "schema" in schema_data:
        inner_schema = schema_data["schema"]
        # Check if it's old format (has "parameters") or new format (has "type")
        if "parameters" in inner_schema:
            # Old format: extract the parameters schema
            return inner_schema["parameters"]
        else:
            # New format: the inner schema is the JSON schema
            return inner_schema
    else:
        # If no "schema" key, assume the whole file is the schema
        return schema_data


def _validate_json_against_schema(json_data: dict, schema: dict, allow_extra_properties: int = MAX_EXTRA_PROPERTIES_ACCEPTED) -> tuple[bool, str, dict]:
    """
    Validate JSON data against a JSON schema.
    Handles extra properties by removing them if count is acceptable.
    
    Args:
        json_data: The JSON data to validate (may be modified if extra properties are removed)
        schema: The JSON schema to validate against
        allow_extra_properties: Maximum number of extra properties to accept (default: MAX_EXTRA_PROPERTIES_ACCEPTED)
        
    Returns:
        Tuple of (is_valid, error_message, cleaned_json_data)
        - is_valid: True if valid, False otherwise
        - error_message: Empty string if valid, error description if invalid
        - cleaned_json_data: JSON data with extra properties removed (if applicable), or original if no cleaning needed
    """
    # First, try validation without modification
    try:
        import jsonschema
        jsonschema.validate(instance=json_data, schema=schema)
        return True, "", json_data
    except ImportError:
        # jsonschema not available - try basic validation
        is_valid, error_msg = _basic_schema_validation(json_data, schema)
        if is_valid:
            return True, "", json_data
        # Continue to extra properties handling below
    except Exception as e:
        # Check if error is related to extra properties
        error_msg = str(e)
        if hasattr(e, 'message'):
            error_msg = e.message
        
        # Check if this is an "additional properties" error
        if "additional properties" in error_msg.lower() or "extra properties" in error_msg.lower():
            # Try to extract and handle extra properties
            return _handle_extra_properties(json_data, schema, allow_extra_properties, error_msg)
        
        # Other validation errors - return as-is
        if hasattr(e, 'path'):
            path_str = '.'.join(str(p) for p in e.path) if e.path else ''
            if path_str:
                error_msg = f"{error_msg} (path: {path_str})"
        return False, f"Schema validation error: {error_msg}", json_data
    
    # If basic validation failed, try handling extra properties
    # Check if schema has additionalProperties restriction
    if schema.get("additionalProperties") is False:
        allowed_fields = set(schema.get("properties", {}).keys())
        actual_fields = set(json_data.keys())
        extra_fields = actual_fields - allowed_fields
        
        if extra_fields:
            return _handle_extra_properties(json_data, schema, allow_extra_properties, f"Extra fields: {', '.join(extra_fields)}")
    
    # Final validation attempt after potential cleaning
    return _basic_schema_validation_with_cleaning(json_data, schema, allow_extra_properties)


def _handle_extra_properties(json_data: dict, schema: dict, max_extra: int, original_error: str) -> tuple[bool, str, dict]:
    """
    Handle extra properties by removing them if count is acceptable.
    
    Args:
        json_data: The JSON data (will be copied, not modified)
        schema: The JSON schema
        max_extra: Maximum number of extra properties to allow
        original_error: Original error message
        
    Returns:
        Tuple of (is_valid, error_message, cleaned_json_data)
    """
    allowed_fields = set(schema.get("properties", {}).keys())
    actual_fields = set(json_data.keys())
    extra_fields = actual_fields - allowed_fields
    
    if len(extra_fields) > max_extra:
        return False, (
            f"Schema validation error: {len(extra_fields)} extra properties found "
            f"(max allowed: {max_extra}). Extra properties: {', '.join(sorted(extra_fields))}. "
            f"Original error: {original_error}"
        ), json_data
    
    # Remove extra properties and re-validate
    cleaned_data = {k: v for k, v in json_data.items() if k in allowed_fields}
    
    # Re-validate cleaned data
    try:
        import jsonschema
        jsonschema.validate(instance=cleaned_data, schema=schema)
        return True, f"Warning: Removed {len(extra_fields)} extra properties: {', '.join(sorted(extra_fields))}", cleaned_data
    except ImportError:
        # jsonschema not available - try basic validation
        is_valid, error_msg = _basic_schema_validation(cleaned_data, schema)
        if is_valid:
            return True, f"Warning: Removed {len(extra_fields)} extra properties: {', '.join(sorted(extra_fields))}", cleaned_data
        return False, f"Schema validation error after removing extra properties: {error_msg}", cleaned_data
    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'message'):
            error_msg = e.message
        return False, f"Schema validation error after removing extra properties: {error_msg}", cleaned_data


def _basic_schema_validation_with_cleaning(json_data: dict, schema: dict, max_extra: int) -> tuple[bool, str, dict]:
    """
    Basic validation with extra property handling.
    
    Args:
        json_data: The JSON data to validate
        schema: The JSON schema
        max_extra: Maximum extra properties to allow
        
    Returns:
        Tuple of (is_valid, error_message, cleaned_json_data)
    """
    if schema.get("additionalProperties") is False:
        allowed_fields = set(schema.get("properties", {}).keys())
        actual_fields = set(json_data.keys())
        extra_fields = actual_fields - allowed_fields
        
        if extra_fields:
            if len(extra_fields) > max_extra:
                return False, f"Extra fields not allowed: {', '.join(extra_fields)} ({len(extra_fields)} > {max_extra})", json_data
            
            # Remove extra properties
            cleaned_data = {k: v for k, v in json_data.items() if k in allowed_fields}
            # Re-validate
            is_valid, error_msg = _basic_schema_validation(cleaned_data, schema)
            if is_valid:
                return True, f"Warning: Removed {len(extra_fields)} extra properties: {', '.join(sorted(extra_fields))}", cleaned_data
            return False, f"Validation failed after removing extra properties: {error_msg}", cleaned_data
    
    # No extra properties issue, validate normally
    is_valid, error_msg = _basic_schema_validation(json_data, schema)
    return is_valid, error_msg, json_data


def _basic_schema_validation(json_data: dict, schema: dict) -> tuple[bool, str]:
    """
    Basic validation without jsonschema library.
    Only checks required fields and basic types.
    
    Args:
        json_data: The JSON data to validate
        schema: The JSON schema to validate against
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(json_data, dict):
        return False, "Data is not a dictionary/object"
    
    # Check required fields
    required = schema.get("required", [])
    for field in required:
        if field not in json_data:
            return False, f"Missing required field: {field}"
    
    # Check additionalProperties
    if schema.get("additionalProperties") is False:
        allowed_fields = set(schema.get("properties", {}).keys())
        actual_fields = set(json_data.keys())
        extra_fields = actual_fields - allowed_fields
        if extra_fields:
            return False, f"Extra fields not allowed: {', '.join(extra_fields)}"
    
    # Basic type checking for required fields
    properties = schema.get("properties", {})
    for field, value in json_data.items():
        if field in properties:
            field_schema = properties[field]
            expected_type = field_schema.get("type")
            if expected_type == "string" and not isinstance(value, str):
                return False, f"Field '{field}' should be string, got {type(value).__name__}"
            elif expected_type == "array" and not isinstance(value, list):
                return False, f"Field '{field}' should be array, got {type(value).__name__}"
            elif expected_type == "array":
                # Check array items
                items_schema = field_schema.get("items", {})
                if items_schema.get("type") == "string":
                    if not all(isinstance(item, str) for item in value):
                        return False, f"Field '{field}' array items should be strings"
                # Check minItems/maxItems
                min_items = field_schema.get("minItems")
                max_items = field_schema.get("maxItems")
                if min_items is not None and len(value) < min_items:
                    return False, f"Field '{field}' should have at least {min_items} items, got {len(value)}"
                if max_items is not None and len(value) > max_items:
                    return False, f"Field '{field}' should have at most {max_items} items, got {len(value)}"
    
    return True, ""




def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract structured data from case documents using LLM"
    )
    parser.add_argument(
        "--model-family",
        type=str,
        default="GPT",
        help="Model family (first part of Factory class name, e.g., 'GPT', 'Gemini', 'Claude', 'DeepSeek', 'Mistral', 'Qwen', default: GPT)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help=(
            "Model name (default: gpt-4o-mini for GPT, gemini-2.5-flash for Gemini, claude-3-5-haiku-20241022 for Claude, deepseek-chat for DeepSeek, mistral-medium-latest for Mistral, qwen-flash for Qwen). "
            "Examples: "
            "GPT: gpt-4o-mini (recommended, supports structured outputs), gpt-4.1-mini, gpt-4o, gpt-4-turbo, gpt-3.5-turbo; "
            "Gemini: gemini-2.5-flash, gemini-2.5-pro, gemini-1.5-flash, gemini-1.5-pro, gemini-pro; "
            "Claude: claude-3-5-haiku-20241022, claude-3-7-sonnet, claude-3-opus, claude-opus-4-20250514, claude-sonnet-4-20250514; "
                   "DeepSeek: deepseek-chat, deepseek-chat-v2, deepseek-coder, deepseek-coder-v2; "
                   "Mistral: mistral-medium-latest, mistral-large-latest, mistral-small, mistral-tiny, pixtral-12b, mistral-nemo; "
                   "Qwen: qwen-flash, qwen-plus, qwen-max, qwen-turbo, qwen2.5-72b-instruct, qwen2.5-32b-instruct, qwen2.5-14b-instruct, qwen2.5-7b-instruct"
        )
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Maximum number of documents to process (default: None, process all)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset-202510",
        choices=["dataset-202505", "dataset-202510"],
        help="Dataset name to use (default: dataset-202510)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="structured-data-202512",
        help="Prompt name to use (default: structured-data-202512)"
    )
    parser.add_argument(
        "--max-tokens",
        type=str,
        default=None,
        help="Maximum tokens to send to the model. Use 'all' or number >= 100000 for 'all-tokens', or a number like '4096' (default: 4096)"
    )
    parser.add_argument(
        "--ignore-bad-responses",
        action="store_true",
        help=(
            "If set, never abort due to too many bad responses in the recent window. "
            "Overrides the usual limit of max 3 bad responses in 10 (or 10 in 20 for some models)."
        )
    )
    
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    model_family = args.model_family
    model = args.model
    max_documents = args.max_documents
    dataset_name = args.dataset
    prompt_name = args.prompt
    ignore_bad_responses = args.ignore_bad_responses
    
    # Parse max_tokens argument
    if args.max_tokens is None:
        max_tokens = MAX_TOKENS
    elif args.max_tokens.lower() == "all":
        max_tokens = None  # None means "all tokens"
    else:
        try:
            max_tokens = int(args.max_tokens)
            if max_tokens >= 100000:
                max_tokens = None  # Treat very large values as "all tokens"
        except ValueError:
            print(f"Error: --max-tokens must be 'all' or a number, got '{args.max_tokens}'", file=sys.stderr)
            sys.exit(1)
    
    # Set default model based on model_family if not explicitly provided ("gpt-4o-mini" is the default)
    if model_family.lower() == "gemini" and args.model == "gpt-4o-mini":
        model = "gemini-2.5-flash"
    elif model_family.lower() == "claude" and args.model == "gpt-4o-mini":
        model = "claude-3-5-haiku-20241022"
    elif model_family.lower() == "deepseek" and args.model == "gpt-4o-mini":
        model = "deepseek-chat"
    elif model_family.lower() == "mistral" and args.model == "gpt-4o-mini":
        model = "mistral-medium-latest"
    elif model_family.lower() == "qwen" and args.model == "gpt-4o-mini":
        model = "qwen-flash"
    
    # Get the factory based on model_family
    try:
        factory = get_factory(model_family)
    except (ImportError, AttributeError) as e:
        print(f"Error: Invalid model family '{model_family}': {e}", file=sys.stderr)
        print(f"Valid model families: GPT, Gemini, Claude, DeepSeek, Mistral, Qwen", file=sys.stderr)
        sys.exit(1)
        return  # Safety return in case sys.exit is mocked
    
    # Initialize components
    try:
        llm_adapter = LLMAdapter(factory, model)
    except ValueError as e:
        # Model validation error - abort immediately
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
        return  # Safety return in case sys.exit is mocked
    except Exception as e:
        # Other errors (API key, network, etc.) - also abort
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ["model", "not found", "does not exist", "404"]):
            print(f"Error: Model '{model}' does not exist or you do not have access to it.", file=sys.stderr)
            if model_family.lower() == "gemini":
                print(f"Valid Gemini models: gemini-2.5-flash, gemini-2.5-pro, gemini-1.5-flash, gemini-1.5-pro, gemini-pro", file=sys.stderr)
            elif model_family.lower() == "gpt":
                print(f"Valid GPT models: gpt-4o-mini (recommended), gpt-4.1-mini, gpt-4o, gpt-4-turbo, gpt-3.5-turbo, etc.", file=sys.stderr)
            elif model_family.lower() == "claude":
                print(f"Valid Claude models: claude-3-5-haiku-20241022, claude-3-7-sonnet, claude-3-opus, claude-opus-4-20250514, claude-sonnet-4-20250514", file=sys.stderr)
            elif model_family.lower() == "deepseek":
                print(f"Valid DeepSeek models: deepseek-chat, deepseek-chat-v2, deepseek-coder, deepseek-coder-v2", file=sys.stderr)
            elif model_family.lower() == "mistral":
                print(f"Valid Mistral models: mistral-medium-latest, mistral-large-latest, mistral-small, mistral-tiny, pixtral-12b, mistral-nemo", file=sys.stderr)
            elif model_family.lower() == "qwen":
                print(f"Valid Qwen models: qwen-flash, qwen-plus, qwen-max, qwen-turbo, qwen2.5-72b-instruct, qwen2.5-32b-instruct, qwen2.5-14b-instruct, qwen2.5-7b-instruct", file=sys.stderr)
            sys.exit(1)
            return  # Safety return in case sys.exit is mocked
        raise
    
    text_loader = DatasetLoader(dataset_name)  # DatasetLoader finds root automatically
    
    # Initialize prompt creator with specified prompt
    prompt_creator = Prompt(prompt_name)
    
    # Determine task name: <dataset_name>-<max_tokens>-<prompt_name>
    # max_tokens format: "2048-tokens", "1024-tokens", or "all-tokens" if None or very large
    # Note: max_tokens here reflects how many tokens we request to be sent to the model,
    # but the model may limit this further based on its context window size.
    if max_tokens is None or max_tokens >= 100000:
        max_tokens_str = "all-tokens"
    else:
        max_tokens_str = f"{max_tokens}-tokens"
    
    task_name = f"{dataset_name}-{max_tokens_str}-{prompt_name}"
    
    # Create output directory: extracted-data/<task-name>/<model-name>
    output_dir = Path(__file__).parent / OUTPUT_BASE_DIR / task_name / model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track skipped documents (already processed)
    skipped_count = 0
    processed_count = 0  # Successfully processed
    attempt_count = 0    # All prompting attempts (successful + failed)
    
    # Track token usage
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Initialize bad response monitor
    # Use different thresholds for some model families (more lenient)
    model_family_lower = model_family.lower()
    if model_family_lower in ["mistral", "claude", "qwen"]:
        # Allow up to 10 bad responses in the last 20 documents
        monitor_window_size = 20
        monitor_max_bad = 10
    else:
        monitor_window_size = BAD_RESPONSE_WINDOW
        monitor_max_bad = MAX_BAD_RESPONSES
    
    bad_response_monitor = BadResponseMonitor(
        window_size=monitor_window_size,
        max_bad=monitor_max_bad
    )
    
    # Status tracking for periodic output
    status_interval = 100  # Print status every N successful documents
    last_status_time = time.time()
    last_status_input_tokens = 0
    last_status_output_tokens = 0
    start_time = time.time()
    
    # Print configuration at start of processing
    max_tokens_display = "all" if max_tokens is None or max_tokens >= 100000 else str(max_tokens)
    print("=" * 80, file=sys.stderr)
    print("Starter prosessering", file=sys.stderr)
    print("-" * 80, file=sys.stderr)
    print(f"Modell: {model}", file=sys.stderr)
    print(f"Dataset: {dataset_name}", file=sys.stderr)
    print(f"Max tokens: {max_tokens_display}", file=sys.stderr)
    print(f"Prompt: {prompt_name}", file=sys.stderr)
    print(f"Output folder: {output_dir.relative_to(Path(__file__).parent)}", file=sys.stderr)
    print("=" * 80 + "\n", file=sys.stderr)
    
    # Main loop
    for doc_id, kommune_nummer, kommune_navn, text in text_loader():
        # Check max_documents limit - count all attempts, not just successful ones
        if max_documents is not None and attempt_count >= max_documents:
            break
        # Check if output file already exists
        output_file = output_dir / f"{doc_id}.json"
        if output_file.exists():
            skipped_count += 1
            continue
        
        # Count this as an attempt (will be processed or fail)
        attempt_count += 1
        
        try:
            # Determine if model only supports basic json_object (not full structured outputs)
            # Models that only support json_object need schema in prompt
            model_family_lower = model_family.lower()
            supports_only_json_object = model_family_lower in ["mistral", "deepseek", "qwen"]
            
            # Prepare prompt with kommune_navn inserted (text, not number)
            # Include schema in prompt for models that don't support structured outputs
            prompt = prompt_creator.get_prompt(kommune_navn, include_schema=supports_only_json_object)
            
            # Prepare document text (user input)
            document_text = prompt_creator.get_document_text(text)
            
            # Get response from LLM adapter with schema
            response = llm_adapter.generate_text(
                prompt=document_text,
                system_prompt=prompt,
                temperature=TEMPERATURE,
                max_tokens=max_tokens,
                json_schema=prompt_creator.SCHEMA
            )
            
            # Extract token usage
            input_tokens, output_tokens = _extract_token_usage(response)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            
            # Parse JSON from response
            # Handle both GPT and Gemini response formats
            if hasattr(response, 'choices') and len(response.choices) > 0:
                response_content = response.choices[0].message.content
            elif hasattr(response, 'text'):
                response_content = response.text
            else:
                raise ValueError(f"Unexpected response format: {type(response)}")
            
            # Extract JSON from response (may contain markdown code blocks)
            extracted_data = _extract_json_from_response(response_content)
            
            # Validate extracted data against schema (may clean extra properties)
            json_schema = _extract_json_schema(prompt_creator.SCHEMA)
            is_valid, error_message, cleaned_data = _validate_json_against_schema(extracted_data, json_schema)
            
            if not is_valid:
                # Record bad response
                bad_response_monitor.record_response(doc_id, is_bad=True)
                
                # Check if we should terminate
                should_terminate, terminate_msg = bad_response_monitor.should_terminate()
                if should_terminate and not ignore_bad_responses:
                    print(f"Error: Schema validation failed for document {doc_id}", file=sys.stderr)
                    print(f"  {error_message}", file=sys.stderr)
                    print(f"Aborting: {terminate_msg}", file=sys.stderr)
                    sys.exit(1)
                
                raise ValueError(
                    f"Extracted data does not conform to schema: {error_message}. "
                    f"Document ID: {doc_id}. "
                    f"Extracted data: {json.dumps(extracted_data, ensure_ascii=False, indent=2)[:500]}..."
                )
            
            # If validation succeeded but there was a warning (extra properties removed), print it
            if error_message and "Warning:" in error_message:
                print(f"Advarsel for {doc_id}: {error_message}", file=sys.stderr)
                # Use cleaned data instead of original
                extracted_data = cleaned_data
            
            # Record successful response
            bad_response_monitor.record_response(doc_id, is_bad=False)
            
            # Create output record
            output_record = {
                "dokument_id": doc_id,
                "kommune_nummer": kommune_nummer,  # Four-digit number from input
                "kommune_navn": kommune_navn,      # Text name used in prompt
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": model,
                "temperature": TEMPERATURE,
                "max_tokens": max_tokens,
                "response": extracted_data
            }
            
            # Write output to individual file
            with output_file.open("w", encoding="utf-8") as fout:
                json.dump(output_record, fout, ensure_ascii=False, indent=2)
            
            processed_count += 1
            print(f"Suksess: {doc_id}")
            
            # Print status every status_interval successful documents
            if processed_count % status_interval == 0:
                current_time = time.time()
                time_since_last = current_time - last_status_time
                time_since_start = current_time - start_time
                
                input_tokens_since_last = total_input_tokens - last_status_input_tokens
                output_tokens_since_last = total_output_tokens - last_status_output_tokens
                
                avg_time_per_doc = time_since_last / status_interval
                avg_input_tokens_per_doc = input_tokens_since_last / status_interval
                avg_output_tokens_per_doc = output_tokens_since_last / status_interval
                
                # Format max_tokens for display
                max_tokens_display = "all" if max_tokens is None or max_tokens >= 100000 else str(max_tokens)
                
                print("\n" + "=" * 80, file=sys.stderr)
                print(f"Status etter {processed_count} vellykkede dokumenter", file=sys.stderr)
                print("-" * 80, file=sys.stderr)
                print(f"Modell: {model}", file=sys.stderr)
                print(f"Dataset: {dataset_name}", file=sys.stderr)
                print(f"Max tokens: {max_tokens_display}", file=sys.stderr)
                print(f"Prompt: {prompt_name}", file=sys.stderr)
                print("-" * 80, file=sys.stderr)
                print(f"Tid siden forrige status: {time_since_last:.1f} sekunder", file=sys.stderr)
                print(f"Input tokens siden forrige: {input_tokens_since_last:,}", file=sys.stderr)
                print(f"Output tokens siden forrige: {output_tokens_since_last:,}", file=sys.stderr)
                print(f"Gjennomsnittlig tid per dokument: {avg_time_per_doc:.2f} sekunder", file=sys.stderr)
                print(f"Gjennomsnittlig input tokens per dokument: {avg_input_tokens_per_doc:.1f}", file=sys.stderr)
                print(f"Gjennomsnittlig output tokens per dokument: {avg_output_tokens_per_doc:.1f}", file=sys.stderr)
                print(f"Total tid: {time_since_start:.1f} sekunder", file=sys.stderr)
                print(f"Totale input tokens: {total_input_tokens:,}", file=sys.stderr)
                print(f"Totale output tokens: {total_output_tokens:,}", file=sys.stderr)
                print("=" * 80 + "\n", file=sys.stderr)
                
                # Update tracking variables for next interval
                last_status_time = current_time
                last_status_input_tokens = total_input_tokens
                last_status_output_tokens = total_output_tokens
            
        except ValueError as e:
            # Check if this is a schema validation error
            error_str = str(e).lower()
            if "does not conform to schema" in error_str or "schema validation error" in error_str:
                # Bad response already recorded in try block above
                # Termination check also already done above
                print(f"Error: Schema validation failed for document {doc_id}", file=sys.stderr)
                print(f"  {e}", file=sys.stderr)
                # Don't abort here - let the monitor decide
                # Continue to next iteration (already recorded as bad response)
                time.sleep(2)
                continue
            
            # Check if this is a model-not-found error - abort early since all documents will fail
            is_model_error = (
                ("404" in error_str and "model" in error_str) or
                ("model" in error_str and ("does not exist" in error_str or "not exist" in error_str or "not found" in error_str)) or
                ("error code: 404" in error_str and "model" in error_str)
            )
            
            if is_model_error:
                print(f"Error: {e}", file=sys.stderr)
                print(f"Aborting: Model error will affect all documents.", file=sys.stderr)
                sys.exit(1)
            
            # Check if this is a timeout/connection error - if multiple consecutive failures, suggest aborting
            if "timed out" in error_str or "connection failed" in error_str:
                print(f"Hoppet over {doc_id} → {e}", file=sys.stderr)
                # If this is the 3rd consecutive timeout, warn user
                if attempt_count >= 3 and processed_count == 0:
                    print(f"  Advarsel: {attempt_count} påfølgende timeout-feil. API-en kan være utilgjengelig.", file=sys.stderr)
                    print(f"  Vurder å prøve en annen modell eller sjekke nettverksforbindelsen.", file=sys.stderr)
            # Check if this is a safety filter error from Gemini
            elif "safety" in error_str or "blocked by gemini" in error_str:
                print(f"Hoppet over {doc_id} → {e}", file=sys.stderr)
                print(f"  Dokument-ID: {doc_id}", file=sys.stderr)
                print(f"  Dokument-tekst (første 500 tegn): {text[:500]}...", file=sys.stderr)
                if len(text) > 500:
                    print(f"  (Totalt {len(text)} tegn)", file=sys.stderr)
            else:
                print(f"Hoppet over {doc_id} → {e}", file=sys.stderr)
            
            # Record bad response for non-schema errors too
            bad_response_monitor.record_response(doc_id, is_bad=True)
            
            # Check if we should terminate
            should_terminate, terminate_msg = bad_response_monitor.should_terminate()
            if should_terminate and not ignore_bad_responses:
                print(f"Aborting: {terminate_msg}", file=sys.stderr)
                sys.exit(1)
            
            # Attempt was made but failed - already counted in attempt_count
            time.sleep(2)
            continue
        except Exception as e:
            # Check if this is a model-not-found error - abort early since all documents will fail
            error_str = str(e).lower()
            is_model_error = (
                ("404" in error_str and "model" in error_str) or
                ("model" in error_str and ("does not exist" in error_str or "not exist" in error_str or "not found" in error_str)) or
                ("error code: 404" in error_str and "model" in error_str)
            )
            
            if is_model_error:
                print(f"Error: {e}", file=sys.stderr)
                print(f"Aborting: Model error will affect all documents.", file=sys.stderr)
                sys.exit(1)
            
            print(f"Hoppet over {doc_id} → {e}", file=sys.stderr)
            
            # Record bad response
            bad_response_monitor.record_response(doc_id, is_bad=True)
            
            # Check if we should terminate
            should_terminate, terminate_msg = bad_response_monitor.should_terminate()
            if should_terminate and not ignore_bad_responses:
                print(f"Aborting: {terminate_msg}", file=sys.stderr)
                sys.exit(1)
            
            # Attempt was made but failed - already counted in attempt_count
            time.sleep(2)
            continue
    
    # Report skipped documents
    if skipped_count > 0:
        print(f"\nHoppet over {skipped_count} eksisterende dokumenter.", file=sys.stderr)
    
    # Write prompt_schema.json and prompt.txt to output_dir
    schema_file = output_dir / "prompt_schema.json"
    with schema_file.open("w", encoding="utf-8") as fout:
        json.dump(prompt_creator.SCHEMA, fout, ensure_ascii=False, indent=2)
    
    prompt_file = output_dir / "prompt.txt"
    with prompt_file.open("w", encoding="utf-8") as fout:
        fout.write(prompt_creator.PROMPT_TEMPLATE)
    
    print(f"\nFerdig. Prosesserte {processed_count} dokumenter (av {attempt_count} forsøk). Data lagret i {output_dir.relative_to(Path.cwd())}")
    
    # Print token usage statistics
    print(f"\nToken-usage:")
    print(f"  INPUT tokens:  {total_input_tokens:,}")
    print(f"  OUTPUT tokens: {total_output_tokens:,}")
    print(f"  TOTAL tokens:  {total_input_tokens + total_output_tokens:,}")
    
    if processed_count > 0:
        mean_input_tokens = total_input_tokens / processed_count
        mean_output_tokens = total_output_tokens / processed_count
        print(f"\nPer dokument (gjennomsnitt):")
        print(f"  Dokumenter prosessert: {processed_count}")
        print(f"  INPUT tokens/dokument:  {mean_input_tokens:,.1f}")
        print(f"  OUTPUT tokens/dokument: {mean_output_tokens:,.1f}")
        print(f"  TOTAL tokens/dokument:  {(mean_input_tokens + mean_output_tokens):,.1f}")


if __name__ == "__main__":
    main()
