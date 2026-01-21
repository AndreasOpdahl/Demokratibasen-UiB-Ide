"""
Helper functions for validating JSON responses against JSON schemas in integration tests.
"""
import json
from pathlib import Path
from typing import Dict, Any


def load_schema(prompt_name: str = "extraction-202512") -> Dict[str, Any]:
    """
    Load schema from a schema file.
    
    Args:
        prompt_name: Name of the prompt/schema to load (default: "extraction-202512")
        
    Returns:
        Schema data dictionary
    """
    schema_file = Path(__file__).parent.parent / "create_prompt" / f"{prompt_name}-schema.json"
    with open(schema_file, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_json_schema(schema_data: Dict[str, Any]) -> Dict[str, Any]:
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


def validate_json_against_schema(json_data: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, str]:
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
        return _basic_validation(json_data, schema)
    except jsonschema.ValidationError as e:
        return False, f"Schema validation error: {e.message} (path: {'.'.join(str(p) for p in e.path)})"
    except jsonschema.SchemaError as e:
        return False, f"Schema error: {e.message}"


def _basic_validation(json_data: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, str]:
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

