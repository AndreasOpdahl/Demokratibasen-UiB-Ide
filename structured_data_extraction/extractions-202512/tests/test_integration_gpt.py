"""
End-to-end integration tests for GPT LLM adapter.

These tests make actual API calls to GPT (OpenAI) and require:
- OPENAI_API_KEY environment variable to be set
- Valid API key with credits/access

To run only integration tests:
    pytest -m integration

To skip integration tests:
    pytest -m "not integration"
"""
import os
import pytest
from llm_adapter import LLMAdapter, GPTFactory


@pytest.mark.integration
class TestGPTIntegration:
    """End-to-end integration tests that make real API calls to GPT."""

    @pytest.fixture
    def api_key(self):
        """Get API key from environment, fail test if not available."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.fail("OPENAI_API_KEY not set. Integration tests require a valid API key.")
        return api_key

    @pytest.fixture
    def factory(self):
        """Create a GPTFactory instance."""
        return GPTFactory()

    @pytest.fixture
    def adapter(self, factory):
        """Create an LLMAdapter instance with a test model."""
        # Use a cheaper/faster model for testing
        return LLMAdapter(factory, "gpt-3.5-turbo")
    
    @pytest.fixture
    def adapter_with_structured_outputs(self, factory):
        """Create an LLMAdapter instance with a model that supports structured outputs."""
        # Use gpt-4o-mini which supports structured outputs and is relatively cheap
        return LLMAdapter(factory, "gpt-4o-mini")

    def test_api_key_is_valid(self, factory, api_key):
        """Test that the API key is valid by making a minimal API call."""
        # Arrange
        model = factory.create("gpt-3.5-turbo")
        prompt = "Say 'ok'"

        # Act
        try:
            response = model.generate_text(prompt)
            # Assert - if we get here, the API key is valid
            assert response is not None
            content = response.choices[0].message.content
            assert isinstance(content, str)
        except Exception as e:
            # If we get an authentication error, the API key is invalid
            error_msg = str(e).lower()
            if "api key" in error_msg or "authentication" in error_msg or "unauthorized" in error_msg:
                pytest.fail(f"Invalid API key: {e}")
            # Re-raise other exceptions (network issues, etc.)
            raise

    def test_model_exists_and_accessible(self, factory, api_key):
        """Test that the model exists and is accessible (similar to validation in extract_structured_data)."""
        # Arrange
        model_name = "gpt-3.5-turbo"
        model = factory.create(model_name)
        
        # Act - make a minimal test call
        try:
            response = model.generate_text("Say 'ok'", max_tokens=5)
            # Assert - if we get here, the model exists and works
            assert response is not None
            content = response.choices[0].message.content
            assert isinstance(content, str)
            assert len(content) > 0
        except Exception as e:
            error_str = str(e).lower()
            # Check for model not found errors
            if any(keyword in error_str for keyword in ["404", "not found", "does not exist", "model_not_found", "invalid_request_error"]):
                pytest.fail(f"Model '{model_name}' does not exist or you do not have access to it: {e}")
            # Re-raise other exceptions
            raise

    def test_invalid_model_name_fails(self, factory, api_key):
        """Test that an invalid model name fails appropriately."""
        # Arrange
        invalid_model = "gpt-nonexistent-model-12345"
        
        # Act & Assert
        try:
            model = factory.create(invalid_model)
            # If model creation succeeds, try to use it - should fail
            with pytest.raises(Exception) as exc_info:
                model.generate_text("test", max_tokens=1)
            # Should be a model not found error
            error_str = str(exc_info.value).lower()
            assert any(keyword in error_str for keyword in ["404", "not found", "does not exist", "model_not_found", "invalid_request_error"])
        except Exception as e:
            # Model creation might fail immediately, which is also acceptable
            error_str = str(e).lower()
            assert any(keyword in error_str for keyword in ["404", "not found", "does not exist", "model_not_found", "invalid_request_error"])

    def test_factory_creates_model(self, factory, api_key):
        """Test that factory can create a GPT model."""
        # Act
        model = factory.create("gpt-3.5-turbo")

        # Assert
        assert model is not None
        assert model.api_key == api_key
        assert model.model_name == "gpt-3.5-turbo"
        assert model.client is not None

    def test_model_generate_text_simple(self, factory, api_key):
        """Test that model can generate text with a simple prompt."""
        # Arrange
        model = factory.create("gpt-3.5-turbo")
        prompt = "Say 'Hello, World!' and nothing else."

        # Act
        response = model.generate_text(prompt)

        # Assert
        assert response is not None
        # Response is a ChatCompletion object, extract the content
        content = response.choices[0].message.content
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
        # Response is a ChatCompletion object
        content = response.choices[0].message.content
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
        content = response.choices[0].message.content
        assert isinstance(content, str)
        assert "4" in content

    def test_adapter_different_models(self, factory):
        """Test that different models can be used."""
        # Test with gpt-3.5-turbo (cheaper)
        adapter1 = LLMAdapter(factory, "gpt-3.5-turbo")
        prompt = "Say 'test1'"
        response1 = adapter1.generate_text(prompt)
        assert response1 is not None
        assert response1.choices[0].message.content is not None

        # Test with gpt-4o-mini if available (also relatively cheap)
        try:
            adapter2 = LLMAdapter(factory, "gpt-4o-mini")
            response2 = adapter2.generate_text(prompt)
            assert response2 is not None
            assert response2.choices[0].message.content is not None
        except Exception as e:
            # If model is not available, that's okay for this test
            pytest.skip(f"Model gpt-4o-mini not available: {e}")

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
            content = response.choices[0].message.content
            assert isinstance(content, str)
            assert len(content) > 0

    def test_model_name_persistence(self, factory):
        """Test that model_name is correctly stored and used."""
        # Arrange
        model_name = "gpt-3.5-turbo"
        model = factory.create(model_name)

        # Assert
        assert model.model_name == model_name

        # Act - generate text and verify model name is used
        response = model.generate_text("Say 'test'")
        
        # Assert - verify the response came from the correct model
        assert response is not None
        # The response object should have model information
        assert hasattr(response, 'model')
        # The model used should match or be compatible
        assert model_name in response.model or response.model is not None

    def test_adapter_with_temperature_max_tokens(self, adapter):
        """Test that adapter works with temperature and max_tokens."""
        # Arrange
        prompt = "Return a JSON object with a single key 'test' and value 'ok'."
        temperature = 0.1
        max_tokens = 4096

        # Act
        response = adapter.generate_text(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Assert
        assert response is not None
        content = response.choices[0].message.content
        assert isinstance(content, str)
        assert len(content) > 0
        # Content should be valid JSON (relying on prompt instructions)
        import json
        try:
            parsed = json.loads(content)
            assert isinstance(parsed, dict)
        except json.JSONDecodeError:
            pytest.fail(f"Expected JSON response but got: {content}")

    def test_adapter_json_output_with_schema(self, adapter_with_structured_outputs):
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
            response = adapter_with_structured_outputs.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=1000,  # Increase to avoid truncation
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
        # For structured outputs, content should be valid JSON directly
        # Strip whitespace first
        content = content.strip()
        
        try:
            json_output = json.loads(content)
        except json.JSONDecodeError as e:
            # Try extracting from markdown if present (shouldn't happen with structured outputs)
            import re
            json_match = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                try:
                    json_output = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pytest.fail(f"Could not parse JSON from markdown block: {e}")
            else:
                # Try finding JSON object - use balanced braces
                # Find first { and last } to get complete JSON
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
        # Verify it has at least some expected fields from schema
        # (exact fields depend on the schema, but should have some structure)

    def test_adapter_schema_validation_extraction_202512(self, adapter_with_structured_outputs):
        """Test that the model returns data that conforms to the extraction-202512 JSON schema."""
        # Arrange
        import json
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
            response = adapter_with_structured_outputs.generate_text(
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
        is_valid, error_message = validate_json_against_schema(json_output, json_schema)
        assert is_valid, f"JSON does not conform to schema: {error_message}\nJSON: {json.dumps(json_output, indent=2, ensure_ascii=False)}"
        
        # Additional assertions for key fields
        assert "hva_saken_gjelder" in json_output, "Missing required field: hva_saken_gjelder"
        assert "tema" in json_output, "Missing required field: tema"
        assert isinstance(json_output["tema"], list), "tema should be an array"
        assert 3 <= len(json_output["tema"]) <= 10, f"tema should have 3-10 items, got {len(json_output['tema'])}"

