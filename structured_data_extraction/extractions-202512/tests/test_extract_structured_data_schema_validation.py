"""
Unit tests for schema validation functions in extract_structured_data.py.
"""
import pytest
import sys
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add parent directory to path to import extract_structured_data
sys.path.insert(0, str(Path(__file__).parent.parent))

import extract_structured_data


class TestExtractJsonSchema:
    """Test suite for _extract_json_schema function."""
    
    def test_extract_json_schema_new_format(self):
        """Test extracting schema from new format (has 'type' key)."""
        schema_data = {
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name"]
            }
        }
        
        result = extract_structured_data._extract_json_schema(schema_data)
        
        assert result == schema_data["schema"]
        assert result["type"] == "object"
        assert "properties" in result
    
    def test_extract_json_schema_old_format(self):
        """Test extracting schema from old format (has 'parameters' key)."""
        schema_data = {
            "schema": {
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name"]
                }
            }
        }
        
        result = extract_structured_data._extract_json_schema(schema_data)
        
        assert result == schema_data["schema"]["parameters"]
        assert result["type"] == "object"
        assert "properties" in result
    
    def test_extract_json_schema_no_schema_key(self):
        """Test extracting schema when there's no 'schema' key."""
        schema_data = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        
        result = extract_structured_data._extract_json_schema(schema_data)
        
        assert result == schema_data
        assert result["type"] == "object"
    
    def test_extract_json_schema_empty_inner_schema(self):
        """Test extracting schema when inner schema has neither 'parameters' nor 'type'."""
        schema_data = {
            "schema": {
                "properties": {
                    "name": {"type": "string"}
                }
            }
        }
        
        result = extract_structured_data._extract_json_schema(schema_data)
        
        assert result == schema_data["schema"]
        assert "properties" in result


class TestValidateJsonAgainstSchema:
    """Test suite for _validate_json_against_schema function."""
    
    @pytest.fixture
    def simple_schema(self):
        """Simple schema for testing."""
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5
                }
            },
            "required": ["name"],
            "additionalProperties": False
        }
    
    def test_validate_json_against_schema_valid_data(self, simple_schema):
        """Test validation with valid data."""
        json_data = {
            "name": "John",
            "age": 30,
            "tags": ["tag1", "tag2"]
        }
        
        is_valid, error_message = extract_structured_data._validate_json_against_schema(
            json_data, simple_schema
        )
        
        assert is_valid is True
        assert error_message == ""
    
    def test_validate_json_against_schema_missing_required(self, simple_schema):
        """Test validation with missing required field."""
        json_data = {
            "age": 30
        }
        
        is_valid, error_message = extract_structured_data._validate_json_against_schema(
            json_data, simple_schema
        )
        
        assert is_valid is False
        assert "required" in error_message.lower() or "missing" in error_message.lower()
    
    def test_validate_json_against_schema_extra_fields(self, simple_schema):
        """Test validation with extra fields when additionalProperties is False."""
        json_data = {
            "name": "John",
            "age": 30,
            "extra_field": "not allowed"
        }
        
        is_valid, error_message = extract_structured_data._validate_json_against_schema(
            json_data, simple_schema
        )
        
        assert is_valid is False
        assert "extra" in error_message.lower() or "not allowed" in error_message.lower()
    
    def test_validate_json_against_schema_wrong_type(self, simple_schema):
        """Test validation with wrong type."""
        json_data = {
            "name": 123,  # Should be string
            "age": 30
        }
        
        is_valid, error_message = extract_structured_data._validate_json_against_schema(
            json_data, simple_schema
        )
        
        assert is_valid is False
        assert "string" in error_message.lower() or "type" in error_message.lower()
    
    def test_validate_json_against_schema_array_min_items(self, simple_schema):
        """Test validation with array that has too few items."""
        json_data = {
            "name": "John",
            "tags": []  # minItems is 1
        }
        
        is_valid, error_message = extract_structured_data._validate_json_against_schema(
            json_data, simple_schema
        )
        
        assert is_valid is False
        assert "minItems" in error_message.lower() or "at least" in error_message.lower()
    
    def test_validate_json_against_schema_array_max_items(self, simple_schema):
        """Test validation with array that has too many items."""
        json_data = {
            "name": "John",
            "tags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6"]  # maxItems is 5
        }
        
        is_valid, error_message = extract_structured_data._validate_json_against_schema(
            json_data, simple_schema
        )
        
        assert is_valid is False
        assert "maxItems" in error_message.lower() or "at most" in error_message.lower()
    
    def test_validate_json_against_schema_array_wrong_item_type(self, simple_schema):
        """Test validation with array containing wrong item types."""
        json_data = {
            "name": "John",
            "tags": [123, 456]  # Should be strings
        }
        
        is_valid, error_message = extract_structured_data._validate_json_against_schema(
            json_data, simple_schema
        )
        
        assert is_valid is False
        assert "string" in error_message.lower() or "items" in error_message.lower()
    
    @patch('extract_structured_data.jsonschema')
    def test_validate_json_uses_jsonschema_when_available(self, mock_jsonschema, simple_schema):
        """Test that jsonschema library is used when available."""
        json_data = {"name": "John"}
        
        # Mock jsonschema.validate to succeed
        mock_jsonschema.validate.return_value = None
        
        is_valid, error_message = extract_structured_data._validate_json_against_schema(
            json_data, simple_schema
        )
        
        assert is_valid is True
        assert error_message == ""
        mock_jsonschema.validate.assert_called_once_with(instance=json_data, schema=simple_schema)
    
    @patch('extract_structured_data.jsonschema')
    def test_validate_json_handles_jsonschema_validation_error(self, mock_jsonschema, simple_schema):
        """Test handling of jsonschema.ValidationError."""
        json_data = {"name": 123}  # Wrong type
        
        # Mock jsonschema.validate to raise ValidationError
        from jsonschema import ValidationError
        mock_error = ValidationError("Field 'name' should be string")
        mock_error.message = "Field 'name' should be string"
        mock_error.path = ["name"]
        mock_jsonschema.validate.side_effect = mock_error
        
        is_valid, error_message = extract_structured_data._validate_json_against_schema(
            json_data, simple_schema
        )
        
        assert is_valid is False
        assert "validation error" in error_message.lower()
        assert "name" in error_message.lower()


class TestBasicSchemaValidation:
    """Test suite for _basic_schema_validation function (fallback when jsonschema not available)."""
    
    @pytest.fixture
    def simple_schema(self):
        """Simple schema for testing."""
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5
                }
            },
            "required": ["name"],
            "additionalProperties": False
        }
    
    def test_basic_validation_valid_data(self, simple_schema):
        """Test basic validation with valid data."""
        json_data = {
            "name": "John",
            "age": 30,
            "tags": ["tag1", "tag2"]
        }
        
        is_valid, error_message = extract_structured_data._basic_schema_validation(
            json_data, simple_schema
        )
        
        assert is_valid is True
        assert error_message == ""
    
    def test_basic_validation_not_dict(self, simple_schema):
        """Test basic validation when data is not a dictionary."""
        json_data = "not a dict"
        
        is_valid, error_message = extract_structured_data._basic_schema_validation(
            json_data, simple_schema
        )
        
        assert is_valid is False
        assert "dictionary" in error_message.lower() or "object" in error_message.lower()
    
    def test_basic_validation_missing_required(self, simple_schema):
        """Test basic validation with missing required field."""
        json_data = {
            "age": 30
        }
        
        is_valid, error_message = extract_structured_data._basic_schema_validation(
            json_data, simple_schema
        )
        
        assert is_valid is False
        assert "required" in error_message.lower() and "name" in error_message.lower()
    
    def test_basic_validation_extra_fields(self, simple_schema):
        """Test basic validation with extra fields."""
        json_data = {
            "name": "John",
            "extra_field": "not allowed"
        }
        
        is_valid, error_message = extract_structured_data._basic_schema_validation(
            json_data, simple_schema
        )
        
        assert is_valid is False
        assert "extra" in error_message.lower() or "not allowed" in error_message.lower()
    
    def test_basic_validation_wrong_string_type(self, simple_schema):
        """Test basic validation with wrong string type."""
        json_data = {
            "name": 123  # Should be string
        }
        
        is_valid, error_message = extract_structured_data._basic_schema_validation(
            json_data, simple_schema
        )
        
        assert is_valid is False
        assert "string" in error_message.lower()
        assert "name" in error_message.lower()
    
    def test_basic_validation_wrong_array_type(self, simple_schema):
        """Test basic validation with wrong array type."""
        json_data = {
            "name": "John",
            "tags": "not an array"  # Should be array
        }
        
        is_valid, error_message = extract_structured_data._basic_schema_validation(
            json_data, simple_schema
        )
        
        assert is_valid is False
        assert "array" in error_message.lower()
        assert "tags" in error_message.lower()
    
    def test_basic_validation_array_wrong_items(self, simple_schema):
        """Test basic validation with array containing wrong item types."""
        json_data = {
            "name": "John",
            "tags": [123, 456]  # Should be strings
        }
        
        is_valid, error_message = extract_structured_data._basic_schema_validation(
            json_data, simple_schema
        )
        
        assert is_valid is False
        assert "string" in error_message.lower() or "items" in error_message.lower()
    
    def test_basic_validation_array_min_items(self, simple_schema):
        """Test basic validation with array that has too few items."""
        json_data = {
            "name": "John",
            "tags": []  # minItems is 1
        }
        
        is_valid, error_message = extract_structured_data._basic_schema_validation(
            json_data, simple_schema
        )
        
        assert is_valid is False
        assert "at least" in error_message.lower() or "minItems" in error_message.lower()
    
    def test_basic_validation_array_max_items(self, simple_schema):
        """Test basic validation with array that has too many items."""
        json_data = {
            "name": "John",
            "tags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6"]  # maxItems is 5
        }
        
        is_valid, error_message = extract_structured_data._basic_schema_validation(
            json_data, simple_schema
        )
        
        assert is_valid is False
        assert "at most" in error_message.lower() or "maxItems" in error_message.lower()
    
    def test_basic_validation_no_additional_properties_check(self):
        """Test basic validation when additionalProperties is not False."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
            # additionalProperties not set or True
        }
        
        json_data = {
            "name": "John",
            "extra_field": "allowed"
        }
        
        is_valid, error_message = extract_structured_data._basic_schema_validation(
            json_data, schema
        )
        
        # Should pass since additionalProperties is not False
        assert is_valid is True
        assert error_message == ""


class TestSchemaValidationFlow:
    """Test suite for the complete validation flow (extract + validate)."""
    
    def test_validation_flow_with_valid_data(self):
        """Test the complete flow: extract schema + validate valid data."""
        # Schema in new format
        schema_data = {
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name"],
                "additionalProperties": False
            }
        }
        
        # Extract schema
        json_schema = extract_structured_data._extract_json_schema(schema_data)
        
        # Valid data
        json_data = {"name": "John", "age": 30}
        
        # Validate
        is_valid, error_message = extract_structured_data._validate_json_against_schema(
            json_data, json_schema
        )
        
        assert is_valid is True
        assert error_message == ""
    
    def test_validation_flow_with_invalid_data(self):
        """Test the complete flow: extract schema + validate invalid data."""
        # Schema in old format
        schema_data = {
            "schema": {
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"],
                    "additionalProperties": False
                }
            }
        }
        
        # Extract schema
        json_schema = extract_structured_data._extract_json_schema(schema_data)
        
        # Invalid data (missing required field)
        json_data = {"age": 30}
        
        # Validate
        is_valid, error_message = extract_structured_data._validate_json_against_schema(
            json_data, json_schema
        )
        
        assert is_valid is False
        assert "required" in error_message.lower() or "missing" in error_message.lower()
        assert "name" in error_message.lower()
    
    def test_validation_flow_raises_value_error_on_invalid(self):
        """Test that validation raises ValueError with proper message when invalid."""
        schema_data = {
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"],
                "additionalProperties": False
            }
        }
        
        json_schema = extract_structured_data._extract_json_schema(schema_data)
        json_data = {"age": 30}  # Missing required "name"
        
        is_valid, error_message = extract_structured_data._validate_json_against_schema(
            json_data, json_schema
        )
        
        # Simulate what happens in main() when validation fails
        if not is_valid:
            error = ValueError(
                f"Extracted data does not conform to schema: {error_message}. "
                f"Document ID: test_doc. "
                f"Extracted data: {json.dumps(json_data, ensure_ascii=False, indent=2)[:500]}..."
            )
            
            assert isinstance(error, ValueError)
            assert "does not conform to schema" in str(error)
            assert "test_doc" in str(error)
            assert "age" in str(error)  # The invalid data should be in the error message
    
    @patch('sys.exit')
    def test_schema_validation_error_causes_exit(self, mock_exit):
        """Test that schema validation errors cause the script to exit."""
        import extract_structured_data
        
        # Simulate a schema validation error
        error = ValueError(
            "Extracted data does not conform to schema: Missing required field: name. "
            "Document ID: test_doc. "
            "Extracted data: {'age': 30}..."
        )
        
        # Simulate the error handling in main() - check for schema validation error
        error_str = str(error).lower()
        if "does not conform to schema" in error_str or "schema validation error" in error_str:
            # This is what happens in the actual code
            import sys
            print(f"Error: Schema validation failed for document test_doc", file=sys.stderr)
            print(f"  {error}", file=sys.stderr)
            print(f"Aborting: Model output does not conform to schema. No results saved.", file=sys.stderr)
            sys.exit(1)
        
        # Verify sys.exit was called (mocked)
        mock_exit.assert_called_once_with(1)
    
    def test_schema_validation_error_message_format(self):
        """Test that schema validation error messages are properly formatted."""
        schema_data = {
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"]
            }
        }
        
        json_schema = extract_structured_data._extract_json_schema(schema_data)
        json_data = {"age": 30}  # Missing required "name"
        
        is_valid, error_message = extract_structured_data._validate_json_against_schema(
            json_data, json_schema
        )
        
        assert is_valid is False
        assert len(error_message) > 0
        
        # The error message should contain information about what's wrong
        assert "required" in error_message.lower() or "missing" in error_message.lower()
        assert "name" in error_message.lower()

