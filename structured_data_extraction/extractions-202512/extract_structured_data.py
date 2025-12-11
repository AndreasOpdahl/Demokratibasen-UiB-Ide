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
    """
    import re
    
    # Try to parse directly first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks (```json ... ``` or ``` ... ```)
    json_patterns = [
        r'```json\s*\n(.*?)\n```',  # ```json ... ```
        r'```\s*\n(.*?)\n```',      # ``` ... ```
        r'```json\s*(.*?)```',      # ```json ... ``` (no newlines)
        r'```\s*(.*?)```',          # ``` ... ``` (no newlines)
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    
    # Try to find JSON object boundaries { ... }
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # If all else fails, raise an error
    raise ValueError(f"Could not extract valid JSON from response: {response_text[:200]}...")


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


def _validate_json_against_schema(json_data: dict, schema: dict) -> tuple[bool, str]:
    """
    Validate JSON data against a JSON schema.
    
    Args:
        json_data: The JSON data to validate
        schema: The JSON schema to validate against
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if valid, False otherwise
        - error_message: Empty string if valid, error description if invalid
    """
    try:
        import jsonschema
        jsonschema.validate(instance=json_data, schema=schema)
        return True, ""
    except ImportError:
        # jsonschema not available - do basic validation
        return _basic_schema_validation(json_data, schema)
    except Exception as e:
        # Handle jsonschema.ValidationError and jsonschema.SchemaError
        error_msg = str(e)
        if hasattr(e, 'message'):
            error_msg = e.message
        if hasattr(e, 'path'):
            path_str = '.'.join(str(p) for p in e.path) if e.path else ''
            if path_str:
                error_msg = f"{error_msg} (path: {path_str})"
        return False, f"Schema validation error: {error_msg}"


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
        default="extraction-202512",
        help="Prompt name to use (default: extraction-202512)"
    )
    parser.add_argument(
        "--max-tokens",
        type=str,
        default=None,
        help="Maximum tokens to send to the model. Use 'all' or number >= 100000 for 'all-tokens', or a number like '4096' (default: 4096)"
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
            
            # Validate extracted data against schema
            json_schema = _extract_json_schema(prompt_creator.SCHEMA)
            is_valid, error_message = _validate_json_against_schema(extracted_data, json_schema)
            
            if not is_valid:
                raise ValueError(
                    f"Extracted data does not conform to schema: {error_message}. "
                    f"Document ID: {doc_id}. "
                    f"Extracted data: {json.dumps(extracted_data, ensure_ascii=False, indent=2)[:500]}..."
                )
            
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
            
        except ValueError as e:
            # Check if this is a schema validation error - abort immediately
            error_str = str(e).lower()
            if "does not conform to schema" in error_str or "schema validation error" in error_str:
                print(f"Error: Schema validation failed for document {doc_id}", file=sys.stderr)
                print(f"  {e}", file=sys.stderr)
                print(f"Aborting: Model output does not conform to schema. No results saved.", file=sys.stderr)
                sys.exit(1)
            
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
