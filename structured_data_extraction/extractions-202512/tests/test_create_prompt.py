"""
Tests for create_prompt.py Prompt class.
"""
import pytest
from create_prompt import Prompt


class TestPrompt:
    """Test suite for Prompt class."""
    
    def test_get_prompt_replaces_placeholder(self):
        """Test that get_prompt replaces <<kommune_navn>> with the provided name."""
        prompt = Prompt()
        kommune_navn = "Bergen"
        
        result = prompt.get_prompt(kommune_navn)
        
        assert "<<kommune_navn>>" not in result
        assert kommune_navn in result
        assert "Du er assisterende saksredaktør i Bergen" in result
    
    def test_get_prompt_with_different_kommuner(self):
        """Test that get_prompt works with different kommune names."""
        prompt = Prompt()
        
        test_cases = [
            "Bergen",
            "Tromsø",
            "Oslo",
            "Stavanger",
            "Trondheim",
        ]
        
        for kommune_navn in test_cases:
            result = prompt.get_prompt(kommune_navn)
            assert "<<kommune_navn>>" not in result
            assert kommune_navn in result
            assert f"Du er assisterende saksredaktør i {kommune_navn}" in result
    
    def test_get_prompt_with_empty_string(self):
        """Test that get_prompt handles empty kommune name."""
        prompt = Prompt()
        kommune_navn = ""
        
        result = prompt.get_prompt(kommune_navn)
        
        assert "<<kommune_navn>>" not in result
        assert "Du er assisterende saksredaktør i " in result
    
    def test_get_prompt_preserves_template_structure(self):
        """Test that get_prompt preserves the rest of the template."""
        prompt = Prompt()
        kommune_navn = "Bergen"
        
        result = prompt.get_prompt(kommune_navn)
        
        # Check that key parts of the template are preserved
        assert "hjelpe lokaljournalister" in result
        assert "JSON objekt" in result
        assert "hva_saken_gjelder" in result
        assert "tema" in result
        assert "viktige_personer" in result
    
    def test_get_document_text_returns_text_as_is(self):
        """Test that get_document_text returns the text unchanged."""
        prompt = Prompt()
        text = "This is a test document text."
        
        result = prompt.get_document_text(text)
        
        assert result == text
        assert result is text  # Should return the same object
    
    def test_get_document_text_with_empty_string(self):
        """Test that get_document_text handles empty text."""
        prompt = Prompt()
        text = ""
        
        result = prompt.get_document_text(text)
        
        assert result == ""
    
    def test_get_document_text_with_multiline_text(self):
        """Test that get_document_text preserves multiline text."""
        prompt = Prompt()
        text = "Line 1\nLine 2\nLine 3"
        
        result = prompt.get_document_text(text)
        
        assert result == text
        assert "\n" in result
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result
    
    def test_get_document_text_with_special_characters(self):
        """Test that get_document_text preserves special characters."""
        prompt = Prompt()
        text = "Test with special chars: æøå, é, ñ, 中文, 🎉"
        
        result = prompt.get_document_text(text)
        
        assert result == text
        assert "æøå" in result
        assert "é" in result
        assert "ñ" in result
        assert "中文" in result
        assert "🎉" in result
    
    def test_get_document_text_with_long_text(self):
        """Test that get_document_text handles long text."""
        prompt = Prompt()
        text = "A" * 10000  # 10k characters
        
        result = prompt.get_document_text(text)
        
        assert result == text
        assert len(result) == 10000
    
    def test_prompt_template_is_accessible(self):
        """Test that PROMPT_TEMPLATE is accessible as a class attribute."""
        prompt = Prompt()
        
        assert hasattr(prompt, 'PROMPT_TEMPLATE')
        assert isinstance(prompt.PROMPT_TEMPLATE, str)
        assert len(prompt.PROMPT_TEMPLATE) > 0
        assert "<<kommune_navn>>" in prompt.PROMPT_TEMPLATE
    
    def test_schema_is_accessible(self):
        """Test that SCHEMA is accessible as a class attribute."""
        prompt = Prompt()
        
        assert hasattr(prompt, 'SCHEMA')
        assert isinstance(prompt.SCHEMA, dict)
        assert "name" in prompt.SCHEMA
        assert "parameters" in prompt.SCHEMA
        assert prompt.SCHEMA["name"] == "extract_case_info"
    
    def test_schema_has_required_fields(self):
        """Test that SCHEMA contains the expected required fields."""
        prompt = Prompt()
        
        assert "parameters" in prompt.SCHEMA
        assert "required" in prompt.SCHEMA["parameters"]
        required_fields = prompt.SCHEMA["parameters"]["required"]
        
        assert "hva_saken_gjelder" in required_fields
        assert "tema" in required_fields
    
    def test_schema_has_expected_properties(self):
        """Test that SCHEMA contains all expected properties."""
        prompt = Prompt()
        
        properties = prompt.SCHEMA["parameters"]["properties"]
        expected_properties = [
            "hva_saken_gjelder",
            "foreslått_vedtak",
            "forventede_konsekvenser",
            "viktige_hendelser",
            "viktige_tidspunkter",
            "viktige_personer",
            "viktige_organisasjoner",
            "viktige_steder",
            "tema",
        ]
        
        for prop in expected_properties:
            assert prop in properties, f"Property '{prop}' not found in schema"
    
    def test_multiple_instances_share_same_template(self):
        """Test that multiple Prompt instances share the same template."""
        prompt1 = Prompt()
        prompt2 = Prompt()
        
        assert prompt1.PROMPT_TEMPLATE == prompt2.PROMPT_TEMPLATE
        assert prompt1.SCHEMA == prompt2.SCHEMA
    
    def test_get_prompt_replaces_all_occurrences(self):
        """Test that get_prompt replaces all occurrences of the placeholder."""
        # First, check if there are multiple occurrences in the template
        prompt = Prompt()
        kommune_navn = "Bergen"
        
        result = prompt.get_prompt(kommune_navn)
        
        # Count occurrences of the kommune name
        count = result.count(kommune_navn)
        # Should be at least 1 (the replacement)
        assert count >= 1
        # Should not have any placeholders left
        assert result.count("<<kommune_navn>>") == 0

    def test_prompt_name_default(self):
        """Test that default prompt name is 'initial-extraction-202509'."""
        prompt = Prompt()
        
        assert prompt.prompt_name == "initial-extraction-202509"

    def test_prompt_name_explicit(self):
        """Test that prompt name can be explicitly set."""
        prompt = Prompt("initial-extraction-202509")
        
        assert prompt.prompt_name == "initial-extraction-202509"

    def test_prompt_loads_from_named_files(self):
        """Test that prompt loads prompt and schema from named files."""
        prompt = Prompt("initial-extraction-202509")
        
        # Should have loaded the prompt template
        assert hasattr(prompt, 'PROMPT_TEMPLATE')
        assert isinstance(prompt.PROMPT_TEMPLATE, str)
        assert len(prompt.PROMPT_TEMPLATE) > 0
        
        # Should have loaded the schema
        assert hasattr(prompt, 'SCHEMA')
        assert isinstance(prompt.SCHEMA, dict)
        assert "name" in prompt.SCHEMA
        assert "parameters" in prompt.SCHEMA

    def test_prompt_name_stored(self):
        """Test that prompt_name is stored as instance attribute."""
        prompt = Prompt("initial-extraction-202509")
        
        assert hasattr(prompt, 'prompt_name')
        assert prompt.prompt_name == "initial-extraction-202509"

    def test_prompt_with_missing_prompt_file(self):
        """Test that Prompt raises FileNotFoundError when prompt file is missing."""
        with pytest.raises(FileNotFoundError, match="Prompt file not found"):
            Prompt("nonexistent-prompt")

    def test_prompt_with_missing_schema_file(self, tmp_path, monkeypatch):
        """Test that Prompt raises FileNotFoundError when schema file is missing."""
        # Create a temporary prompt file but not schema file
        from pathlib import Path
        base_path = Path(__file__).parent.parent / "create_prompt"
        
        # Create a mock prompt file
        prompt_file = base_path / "test-prompt-prompt.txt"
        try:
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write("Test template <<kommune_navn>>")
            
            # Now try to create Prompt - should fail because schema file doesn't exist
            with pytest.raises(FileNotFoundError, match="Schema file not found"):
                Prompt("test-prompt")
        finally:
            # Clean up
            if prompt_file.exists():
                prompt_file.unlink()

    def test_multiple_instances_with_same_name(self):
        """Test that multiple instances with the same prompt name load the same data."""
        prompt1 = Prompt("initial-extraction-202509")
        prompt2 = Prompt("initial-extraction-202509")
        
        assert prompt1.prompt_name == prompt2.prompt_name
        assert prompt1.PROMPT_TEMPLATE == prompt2.PROMPT_TEMPLATE
        assert prompt1.SCHEMA == prompt2.SCHEMA

    def test_different_prompt_names_load_different_data(self, tmp_path):
        """Test that different prompt names can load different data (if files exist)."""
        import json
        from pathlib import Path
        
        # Create a second prompt set for testing
        base_path = Path(__file__).parent.parent / "create_prompt"
        test_prompt_name = "test-prompt-202510"
        
        # Create test files
        prompt_file = base_path / f"{test_prompt_name}-prompt.txt"
        schema_file = base_path / f"{test_prompt_name}-schema.json"
        
        try:
            # Write test prompt
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write("Test prompt for <<kommune_navn>>")
            
            # Write test schema
            with open(schema_file, "w", encoding="utf-8") as f:
                json.dump({
                    "name": "test_extract",
                    "description": "Test schema",
                    "schema": {
                        "name": "test_extract",
                        "description": "Test schema",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "test_field": {"type": "string"}
                            },
                            "required": ["test_field"]
                        }
                    }
                }, f)
            
            # Create prompts with different names
            prompt1 = Prompt("initial-extraction-202509")
            prompt2 = Prompt(test_prompt_name)
            
            # They should have different data
            assert prompt1.prompt_name != prompt2.prompt_name
            assert prompt1.PROMPT_TEMPLATE != prompt2.PROMPT_TEMPLATE
            assert prompt1.SCHEMA != prompt2.SCHEMA
            
            # Verify test prompt has correct data
            assert "Test prompt" in prompt2.PROMPT_TEMPLATE
            assert prompt2.SCHEMA["name"] == "test_extract"
            
        finally:
            # Clean up test files
            if prompt_file.exists():
                prompt_file.unlink()
            if schema_file.exists():
                schema_file.unlink()

    def test_prompt_file_format(self):
        """Test that prompt file has correct text format."""
        from pathlib import Path
        
        base_path = Path(__file__).parent.parent / "create_prompt"
        prompt_file = base_path / "initial-extraction-202509-prompt.txt"
        
        assert prompt_file.exists(), "Prompt file should exist"
        
        with open(prompt_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        assert isinstance(content, str), "Prompt file should contain text"
        assert len(content) > 0, "Prompt file should not be empty"
        assert "<<kommune_navn>>" in content, "Prompt file should contain kommune_navn placeholder"

    def test_schema_file_format(self):
        """Test that schema file has correct JSON format."""
        import json
        from pathlib import Path
        
        base_path = Path(__file__).parent.parent / "create_prompt"
        schema_file = base_path / "initial-extraction-202509-schema.json"
        
        assert schema_file.exists(), "Schema file should exist"
        
        with open(schema_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert "schema" in data, "Schema file should have 'schema' key"
        assert isinstance(data["schema"], dict), "Schema should be a dictionary"
        assert "parameters" in data["schema"], "Schema should have 'parameters'"

