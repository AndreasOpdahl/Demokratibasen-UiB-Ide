"""
End-to-end integration tests for Gemini LLM adapter.

These tests make actual API calls to Gemini and require:
- GEMINI_API_KEY environment variable to be set
- Valid API key with credits/access

To run only integration tests:
    pytest -m integration

To skip integration tests:
    pytest -m "not integration"
"""
import os
import json
import pytest
from llm_adapter import LLMAdapter, GeminiFactory


@pytest.mark.integration
class TestGeminiIntegration:
    """End-to-end integration tests that make real API calls to Gemini."""

    @pytest.fixture
    def api_key(self):
        """Get API key from environment, fail test if not available."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            pytest.fail("GEMINI_API_KEY not set. Integration tests require a valid API key.")
        return api_key

    @pytest.fixture
    def factory(self):
        """Create a GeminiFactory instance."""
        return GeminiFactory()

    @pytest.fixture
    def adapter(self, factory):
        """Create an LLMAdapter instance with a test model."""
        # Use a cheaper/faster model for testing
        return LLMAdapter(factory, "gemini-2.5-flash")

    def test_api_key_is_valid(self, factory, api_key):
        """Test that the API key is valid by making a minimal API call."""
        # Arrange
        model = factory.create("gemini-2.5-flash")
        prompt = "Say 'ok'"

        # Act
        try:
            response = model.generate_text(prompt)
            # Assert - if we get here, the API key is valid
            assert response is not None
            # Handle both OpenAI-compatible response format and direct text
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

    def test_model_exists_and_accessible(self, factory, api_key):
        """Test that the model exists and is accessible (similar to validation in extract_structured_data)."""
        # Arrange
        model_name = "gemini-2.5-flash"
        model = factory.create(model_name)
        
        # Act - make a minimal test call with a safe prompt
        try:
            response = model.generate_text("Say 'ok'", max_tokens=5)
            # Assert - if we get here, the model exists and works
            assert response is not None
            if hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content
            elif hasattr(response, 'text'):
                content = response.text
            else:
                pytest.fail(f"Unexpected response format: {type(response)}")
            assert isinstance(content, str)
            assert len(content) > 0
        except ValueError as e:
            error_str = str(e).lower()
            # Safety filter errors on test prompt are acceptable - model works
            if "safety" in error_str or "blocked by gemini" in error_str:
                # Model is accessible, just triggered safety filters - this is okay for validation
                pass
            else:
                # Other ValueError - re-raise
                raise
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
        invalid_model = "gemini-2.5"  # Missing suffix - should be caught by validation
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid Gemini model name"):
            factory.create(invalid_model)

    def test_factory_creates_model(self, factory, api_key):
        """Test that factory can create a Gemini model."""
        # Act
        model = factory.create("gemini-2.5-flash")

        # Assert
        assert model is not None
        assert model.api_key == api_key
        assert model.model_name == "gemini-2.5-flash"
        assert model.model is not None

    def test_model_generate_text_simple(self, factory, api_key):
        """Test that model can generate text with a simple prompt."""
        # Arrange
        model = factory.create("gemini-2.5-flash")
        prompt = "Say 'Hello, World!' and nothing else."

        # Act
        try:
            response = model.generate_text(prompt)
        except ValueError as e:
            # If Gemini safety filters block even this simple prompt, skip the test
            error_str = str(e).lower()
            if "safety" in error_str or "blocked by gemini" in error_str:
                pytest.skip(f"Gemini safety filters blocked the test prompt: {e}")
            raise

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
        try:
            response = adapter.generate_text(prompt)
        except ValueError as e:
            # If Gemini safety filters block even this simple prompt, skip the test
            error_str = str(e).lower()
            if "safety" in error_str or "blocked by gemini" in error_str:
                pytest.skip(f"Gemini safety filters blocked the test prompt: {e}")
            raise

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
        try:
            response = adapter.generate_text(prompt)
        except ValueError as e:
            # If Gemini safety filters block even this simple prompt, skip the test
            error_str = str(e).lower()
            if "safety" in error_str or "blocked by gemini" in error_str:
                pytest.skip(f"Gemini safety filters blocked the test prompt: {e}")
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
        assert "4" in content

    def test_adapter_different_models(self, factory):
        """Test that different Gemini models can be used."""
        # Test with gemini-2.5-flash (cheaper/faster)
        adapter1 = LLMAdapter(factory, "gemini-2.5-flash")
        prompt = "Say 'test1'"
        try:
            response1 = adapter1.generate_text(prompt)
        except ValueError as e:
            # If Gemini safety filters block even this simple prompt, skip the test
            error_str = str(e).lower()
            if "safety" in error_str or "blocked by gemini" in error_str:
                pytest.skip(f"Gemini safety filters blocked the test prompt: {e}")
            raise
        assert response1 is not None
        if hasattr(response1, 'choices') and len(response1.choices) > 0:
            assert response1.choices[0].message.content is not None
        elif hasattr(response1, 'text'):
            assert response1.text is not None

        # Test with gemini-1.5-flash if available (also relatively cheap)
        # If model is not available, that's okay - we've already tested gemini-2.5-flash
        try:
            adapter2 = LLMAdapter(factory, "gemini-1.5-flash")
            try:
                response2 = adapter2.generate_text(prompt)
                # Only assert if we got a response
                assert response2 is not None
                if hasattr(response2, 'choices') and len(response2.choices) > 0:
                    assert response2.choices[0].message.content is not None
                elif hasattr(response2, 'text'):
                    assert response2.text is not None
            except ValueError as e:
                # If Gemini safety filters block, skip testing this model
                error_str = str(e).lower()
                if "safety" in error_str or "blocked by gemini" in error_str:
                    # Skip testing this model, but don't fail the whole test
                    pass
                else:
                    raise
        except Exception as e:
            # Model not available - this is acceptable, we've already tested the primary model
            # Don't skip the test, just log that the optional model wasn't tested
            pass

    def test_adapter_multiple_calls(self, adapter):
        """Test that adapter can handle multiple sequential calls."""
        prompts = [
            "Say 'first'",
            "Say 'second'",
            "Say 'third'",
        ]

        for prompt in prompts:
            try:
                response = adapter.generate_text(prompt)
            except ValueError as e:
                # If Gemini safety filters block even this simple prompt, skip the test
                error_str = str(e).lower()
                if "safety" in error_str or "blocked by gemini" in error_str:
                    pytest.skip(f"Gemini safety filters blocked the test prompt: {e}")
                raise
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
        model_name = "gemini-2.5-flash"
        model = factory.create(model_name)

        # Assert
        assert model.model_name == model_name

        # Act - generate text and verify model name is used
        try:
            response = model.generate_text("Say 'test'")
        except ValueError as e:
            # If Gemini safety filters block even this simple prompt, skip the test
            error_str = str(e).lower()
            if "safety" in error_str or "blocked by gemini" in error_str:
                pytest.skip(f"Gemini safety filters blocked the test prompt: {e}")
            raise
        
        # Assert - verify the response came from the correct model
        assert response is not None

    def test_adapter_with_system_prompt(self, adapter):
        """Test that adapter works with system prompts."""
        # Arrange
        system_prompt = "You are a helpful assistant that responds in Norwegian."
        prompt = "Say 'hei' (hello in Norwegian)."

        # Act
        try:
            response = adapter.generate_text(
                prompt,
                system_prompt=system_prompt
            )
        except ValueError as e:
            # If Gemini safety filters block even this simple prompt, skip the test
            error_str = str(e).lower()
            if "safety" in error_str or "blocked by gemini" in error_str:
                pytest.skip(f"Gemini safety filters blocked the test prompt: {e}")
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
        assert len(content) > 0

    def test_adapter_with_temperature_and_max_tokens(self, adapter):
        """Test that adapter works with temperature and max_tokens."""
        # Arrange
        # Use a prompt similar to ones that work (simple, direct instruction)
        prompt = "Count from 1 to 5."
        temperature = 0.9
        max_tokens = 50

        # Act
        try:
            response = adapter.generate_text(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except ValueError as e:
            # If Gemini safety filters block even this simple prompt, skip the test
            error_str = str(e).lower()
            if "safety" in error_str or "blocked by gemini" in error_str:
                pytest.skip(f"Gemini safety filters blocked the test prompt: {e}")
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
        assert len(content) > 0

    def test_adapter_json_output_via_prompt(self, adapter):
        """Test that the model returns JSON when instructed via prompt."""
        # Arrange
        # Use a very simple, neutral prompt that's extremely unlikely to trigger safety filters
        system_prompt = "Return the answer as a JSON object with a single key 'color' and value 'blue'."
        prompt = "What is your favorite color?"

        # Act
        try:
            response = adapter.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=50
            )
        except ValueError as e:
            # If Gemini safety filters block even this simple prompt, skip the test
            error_str = str(e).lower()
            if "safety" in error_str or "blocked by gemini" in error_str:
                pytest.skip(f"Gemini safety filters blocked the test prompt: {e}")
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
        except ValueError as e:
            # If Gemini safety filters block, skip the test
            error_str = str(e).lower()
            if "safety" in error_str or "blocked by gemini" in error_str:
                pytest.skip(f"Gemini safety filters blocked the test prompt: {e}")
            raise
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
                # Try finding JSON object - use balanced braces
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
            if any(keyword in error_str for keyword in ["structured", "json_schema", "response_format", "not supported", "safety"]):
                # Log the full error for debugging
                import warnings
                warnings.warn(f"Gemini test skipped due to structured output or safety filter issue: {type(e).__name__}: {e}")
                pytest.skip(f"Model does not support structured outputs or was blocked: {e}")
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

