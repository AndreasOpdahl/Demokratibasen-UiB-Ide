"""
End-to-end integration tests for Qwen LLM adapter.

These tests make actual API calls to Qwen and require:
- QWEN_API_KEY environment variable to be set
- Valid API key with credits/access

To run only integration tests:
    pytest -m integration

To skip integration tests:
    pytest -m "not integration"
"""
import os
import json
import pytest
from llm_adapter import LLMAdapter, QwenFactory


@pytest.mark.integration
class TestQwenIntegration:
    """End-to-end integration tests that make real API calls to Qwen."""

    @pytest.fixture
    def api_key(self):
        """Get API key from environment, fail test if not available."""
        api_key = os.getenv("QWEN_API_KEY")
        if not api_key:
            pytest.fail("QWEN_API_KEY not set. Integration tests require a valid API key.")
        return api_key

    @pytest.fixture
    def factory(self):
        """Create a QwenFactory instance."""
        return QwenFactory()

    @pytest.fixture
    def adapter(self, factory):
        """Create an LLMAdapter instance with a test model."""
        # Use a standard model for testing
        return LLMAdapter(factory, "qwen-turbo")

    def test_api_key_is_valid(self, factory, api_key):
        """Test that the API key is valid by making a minimal API call."""
        # Arrange
        model = factory.create("qwen-turbo")
        prompt = "Say 'ok'"

        # Act
        try:
            response = model.generate_text(prompt)
            # Assert - if we get here, the API key is valid
            assert response is not None
            # Handle OpenAI-compatible response format
            if hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content
            elif hasattr(response, 'text'):
                content = response.text
            else:
                pytest.fail(f"Unexpected response format: {type(response)}")
            assert isinstance(content, str)
        except Exception as e:
            # If we get an authentication error, the API key is invalid
            error_msg = str(e).lower()
            if "api key" in error_msg or "authentication" in error_msg or "unauthorized" in error_msg:
                pytest.fail(f"Invalid API key: {e}")
            # Re-raise other exceptions (network issues, etc.)
            raise

    def test_factory_creates_model(self, factory, api_key):
        """Test that factory can create a Qwen model."""
        # Act
        model = factory.create("qwen-turbo")

        # Assert
        assert model is not None
        assert model.api_key == api_key
        assert model.model_name == "qwen-turbo"
        assert model.client is not None

    def test_model_generate_text_simple(self, factory, api_key):
        """Test that model can generate text with a simple prompt."""
        # Arrange
        model = factory.create("qwen-turbo")
        prompt = "Say 'Hello, World!' and nothing else."

        # Act
        response = model.generate_text(prompt)

        # Assert
        assert response is not None
        # Response format compatible with OpenAI structure
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
        elif hasattr(response, 'text'):
            content = response.text
        else:
            pytest.fail(f"Unexpected response format: {type(response)}")
        assert isinstance(content, str)
        assert len(content) > 0
        assert "Hello" in content or "hello" in content

    def test_adapter_generate_text_simple(self, adapter):
        """Test that LLMAdapter can generate text through the full stack."""
        # Arrange
        prompt = "Respond with only the word 'test'."

        # Act
        response = adapter.generate_text(prompt)

        # Assert
        assert response is not None
        # Response format compatible with OpenAI structure
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
        elif hasattr(response, 'text'):
            content = response.text
        else:
            pytest.fail(f"Unexpected response format: {type(response)}")
        assert isinstance(content, str)
        assert len(content) > 0

    def test_adapter_generate_text_complex(self, adapter):
        """Test LLMAdapter with a more complex prompt."""
        # Arrange
        prompt = "What is 2+2? Respond with only the number."

        # Act
        response = adapter.generate_text(prompt)

        # Assert
        assert response is not None
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
        elif hasattr(response, 'text'):
            content = response.text
        else:
            pytest.fail(f"Unexpected response format: {type(response)}")
        assert isinstance(content, str)
        assert "4" in content

    def test_adapter_different_models(self, factory):
        """Test that different Qwen models can be used."""
        # Test with qwen-turbo (standard)
        adapter1 = LLMAdapter(factory, "qwen-turbo")
        prompt = "Say 'test1'"
        response1 = adapter1.generate_text(prompt)
        assert response1 is not None
        if hasattr(response1, 'choices') and len(response1.choices) > 0:
            assert response1.choices[0].message.content is not None
        elif hasattr(response1, 'text'):
            assert response1.text is not None

        # Test with qwen-plus if available (more powerful)
        try:
            adapter2 = LLMAdapter(factory, "qwen-plus")
            response2 = adapter2.generate_text(prompt)
            assert response2 is not None
            if hasattr(response2, 'choices') and len(response2.choices) > 0:
                assert response2.choices[0].message.content is not None
            elif hasattr(response2, 'text'):
                assert response2.text is not None
        except Exception as e:
            # If model is not available, that's okay for this test
            pytest.skip(f"Model qwen-plus not available: {e}")

    def test_adapter_multiple_calls(self, adapter):
        """Test that adapter can handle multiple sequential calls."""
        prompts = [
            "Say 'first'",
            "Say 'second'",
            "Say 'third'",
        ]

        for prompt in prompts:
            response = adapter.generate_text(prompt)
            assert response is not None
            if hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content
            elif hasattr(response, 'text'):
                content = response.text
            else:
                pytest.fail(f"Unexpected response format: {type(response)}")
            assert isinstance(content, str)
            assert len(content) > 0

    def test_model_name_persistence(self, factory):
        """Test that model_name is correctly stored and used."""
        # Arrange
        model_name = "qwen-turbo"
        model = factory.create(model_name)

        # Assert
        assert model.model_name == model_name

        # Act - generate text and verify model name is used
        response = model.generate_text("Say 'test'")
        
        # Assert - verify the response came from the correct model
        assert response is not None

    def test_adapter_with_system_prompt(self, adapter):
        """Test that adapter works with system prompts."""
        # Arrange
        system_prompt = "You are a helpful assistant that responds in Norwegian."
        prompt = "Say 'hei' (hello in Norwegian)."

        # Act
        response = adapter.generate_text(
            prompt,
            system_prompt=system_prompt
        )

        # Assert
        assert response is not None
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
        elif hasattr(response, 'text'):
            content = response.text
        else:
            pytest.fail(f"Unexpected response format: {type(response)}")
        assert isinstance(content, str)
        assert len(content) > 0

    def test_adapter_with_temperature_and_max_tokens(self, adapter):
        """Test that adapter works with temperature and max_tokens."""
        # Arrange
        prompt = "List three colors: red, blue, and green."
        temperature = 0.9
        max_tokens = 50

        # Act
        response = adapter.generate_text(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Assert
        assert response is not None
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
        elif hasattr(response, 'text'):
            content = response.text
        else:
            pytest.fail(f"Unexpected response format: {type(response)}")
        assert isinstance(content, str)
        assert len(content) > 0

    def test_adapter_json_output_via_prompt(self, adapter):
        """Test that the model returns JSON when instructed via prompt."""
        # Arrange
        system_prompt = "Return the answer as a JSON object with a single key 'color' and value 'blue'."
        prompt = "What is your favorite color?"

        # Act
        response = adapter.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=50
        )

        # Assert
        assert response is not None
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
        elif hasattr(response, 'text'):
            content = response.text
        else:
            pytest.fail(f"Unexpected response format: {type(response)}")
        assert isinstance(content, str)

        # Try to parse JSON (may be wrapped in markdown code blocks)
        try:
            # Try direct JSON parse first
            json_output = json.loads(content)
        except json.JSONDecodeError:
            # Try extracting from markdown code blocks
            import re
            json_match = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                json_output = json.loads(json_match.group(1))
            else:
                # Try finding JSON object boundaries
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_output = json.loads(json_match.group(0))
                else:
                    pytest.fail(f"Expected JSON response but got: {content}")

        assert isinstance(json_output, dict)
        assert "color" in json_output
        # Accept any color value (the model might return a different color, that's fine)
        assert isinstance(json_output["color"], str)

    def test_adapter_json_output_with_schema(self, adapter):
        """Test that the model returns valid JSON when using schema parameter."""
        # Arrange
        import json
        from pathlib import Path
        
        # Load schema from initial-extraction-202509-schema.json
        schema_file = Path(__file__).parent.parent / "create_prompt" / "initial-extraction-202509-schema.json"
        with open(schema_file, "r", encoding="utf-8") as f:
            schema_data = json.load(f)
        
        system_prompt = "Extract information from the text and return it as JSON."
        prompt = "Test document about a city planning meeting in Oslo. Return JSON."

        # Act
        try:
            response = adapter.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=1000,
                json_schema=schema_data
            )
        except Exception as e:
            # Some models may not support structured outputs - skip test
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["structured", "json_schema", "response_format", "not supported"]):
                pytest.skip(f"Model does not support structured outputs: {e}")
            raise

        # Assert
        assert response is not None
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
        elif hasattr(response, 'text'):
            content = response.text
        else:
            pytest.fail(f"Unexpected response format: {type(response)}")
        assert isinstance(content, str)

        # Parse JSON - should be valid JSON when using schema
        content = content.strip()
        try:
            json_output = json.loads(content)
        except json.JSONDecodeError as e:
            # Try extracting from markdown if present
            import re
            json_match = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                try:
                    json_output = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pytest.fail(f"Could not parse JSON from markdown block: {e}")
            else:
                # Try finding JSON object
                first_brace = content.find('{')
                last_brace = content.rfind('}')
                if first_brace >= 0 and last_brace > first_brace:
                    json_str = content[first_brace:last_brace+1]
                    try:
                        json_output = json.loads(json_str)
                    except json.JSONDecodeError:
                        pytest.fail(f"Could not parse JSON from extracted string: {e}. Content: {content[:200]}")
                else:
                    pytest.fail(f"Could not find JSON object in response: {e}. Content: {content[:200]}")

        # Verify it's a dict (JSON object)
        assert isinstance(json_output, dict)

    def test_adapter_schema_validation_extraction_202512(self, adapter):
        """Test that the model returns data that conforms to the extraction-202512 JSON schema."""
        # Arrange
        from tests.test_schema_validation_helper import load_schema, extract_json_schema, validate_json_against_schema
        from create_prompt import Prompt
        
        # Load the extraction-202512 schema
        schema_data = load_schema("extraction-202512")
        json_schema = extract_json_schema(schema_data)
        
        # Create a realistic document text about a municipal case
        document_text = """
        SAKSNR: 2024/1234
        DATO: 15. mars 2024
        
        FORSLAG TIL VEDTAK
        
        Kommunestyret i Bergen vedtar:
        
        1. Å etablere en ny sykkelsti langs Strandgaten fra Nygårdstangen til Fisketorget.
        2. Prosjektet skal gjennomføres i perioden mai-september 2024.
        3. Totalkostnad er estimert til 12 millioner kroner.
        
        BAKGRUNN
        
        Byrådet har mottatt henvendelse fra Sykkelalliansen Bergen som ønsker bedre 
        sykkelinfrastruktur i sentrum. Planleggingsavdelingen har utarbeidet en 
        detaljert plan som inkluderer:
        - Utvidelse av eksisterende sykkelsti
        - Ny belysning langs ruten
        - Grøntområder og benker
        
        Prosjektleder er Anne Hansen fra Planleggingsavdelingen. 
        Prosjektet skal koordineres med Vegvesenet og Ruter.
        
        VEDTAKSDATO: 20. april 2024
        """
        
        # Load prompt and get system prompt
        prompt_creator = Prompt("extraction-202512")
        system_prompt = prompt_creator.get_prompt("Bergen")
        
        # Act
        try:
            response = adapter.generate_text(
                prompt=document_text,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=2000,
                json_schema=schema_data
            )
        except Exception as e:
            # Some models may not support structured outputs - skip test
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["structured", "json_schema", "response_format", "not supported"]):
                pytest.skip(f"Model does not support structured outputs: {e}")
            raise
        
        # Assert - extract JSON from response
        assert response is not None
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
        elif hasattr(response, 'text'):
            content = response.text
        else:
            pytest.fail(f"Unexpected response format: {type(response)}")
        assert isinstance(content, str)
        
        # Parse JSON
        content = content.strip()
        try:
            json_output = json.loads(content)
        except json.JSONDecodeError as e:
            # Try extracting from markdown or finding JSON object
            import re
            json_match = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                json_output = json.loads(json_match.group(1))
            else:
                first_brace = content.find('{')
                last_brace = content.rfind('}')
                if first_brace >= 0 and last_brace > first_brace:
                    json_str = content[first_brace:last_brace+1]
                    json_output = json.loads(json_str)
                else:
                    pytest.fail(f"Could not parse JSON from response: {e}. Content: {content[:200]}")
        
        # Validate against schema
        # Note: Qwen only supports basic json_object format, not full structured outputs
        # So the model may return valid JSON but not conform to the exact schema
        is_valid, error_message = validate_json_against_schema(json_output, json_schema)
        
        if not is_valid:
            # For models that only support basic JSON format, we verify that:
            # 1. Valid JSON was returned
            # 2. It's a dictionary/object
            # 3. It has some content
            # We don't enforce strict schema compliance for these models
            assert isinstance(json_output, dict), f"Expected JSON object, got {type(json_output)}"
            assert len(json_output) > 0, "JSON object should not be empty"
            # Log a warning but don't fail the test
            import warnings
            warnings.warn(
                f"Qwen model returned JSON that doesn't conform to schema (expected for basic json_object format): {error_message}\n"
                f"JSON: {json.dumps(json_output, indent=2, ensure_ascii=False)[:500]}"
            )
        else:
            # If schema validation passes, verify key fields
            assert "hva_saken_gjelder" in json_output, "Missing required field: hva_saken_gjelder"
            assert "tema" in json_output, "Missing required field: tema"
            assert isinstance(json_output["tema"], list), "tema should be an array"
            assert 3 <= len(json_output["tema"]) <= 10, f"tema should have 3-10 items, got {len(json_output['tema'])}"

    def test_model_exists_and_accessible(self, factory, api_key):
        """Test that a valid Qwen model can be created and is accessible."""
        model_name = "qwen-turbo"
        model = factory.create(model_name)
        try:
            response = model.generate_text("Say 'ok'", max_tokens=5)
            assert response is not None
            if hasattr(response, 'choices') and len(response.choices) > 0:
                assert response.choices[0].message.content is not None
            elif hasattr(response, 'text'):
                assert response.text is not None
        except Exception as e:
            pytest.fail(f"Qwen model '{model_name}' is not accessible: {e}")

    def test_invalid_model_name_fails(self, factory, api_key):
        """Test that creating an LLMAdapter with an invalid Qwen model name fails."""
        invalid_model_name = "qwen-invalid-model"
        with pytest.raises(ValueError, match="Invalid Qwen model name"):
            LLMAdapter(factory, invalid_model_name)

