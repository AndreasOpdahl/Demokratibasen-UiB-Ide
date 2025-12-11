"""
Tests for LLMAdapter class.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch

from llm_adapter import LLMAdapter, get_factory


class TestLLMAdapter:
    """Test suite for LLMAdapter class."""

    def test_init_creates_model_from_factory(self):
        """Test that LLMAdapter initializes with model from factory."""
        # Arrange
        mock_factory = Mock()
        mock_model = Mock()
        mock_factory.create.return_value = mock_model
        model_name = "test-model"

        # Act
        adapter = LLMAdapter(mock_factory, model_name)

        # Assert
        assert adapter.model_name == model_name
        assert adapter.model == mock_model
        mock_factory.create.assert_called_once_with(model_name)

    def test_init_stores_model_name(self):
        """Test that model_name is stored correctly."""
        # Arrange
        mock_factory = Mock()
        mock_factory.create.return_value = Mock()
        model_name = "gpt-4"

        # Act
        adapter = LLMAdapter(mock_factory, model_name)

        # Assert
        assert adapter.model_name == model_name

    def test_generate_text_delegates_to_model(self):
        """Test that generate_text delegates to the underlying model."""
        # Arrange
        mock_factory = Mock()
        mock_model = Mock()
        mock_model.generate_text.return_value = "Generated response"
        mock_factory.create.return_value = mock_model
        adapter = LLMAdapter(mock_factory, "test-model")
        prompt = "Test prompt"

        # Act
        result = adapter.generate_text(prompt)

        # Assert
        assert result == "Generated response"
        mock_model.generate_text.assert_called_once_with(
            prompt, 
            system_prompt=None,
            temperature=None,
            max_tokens=None,
            json_schema=None
        )

    def test_generate_text_passes_prompt_correctly(self):
        """Test that the prompt is passed correctly to the model."""
        # Arrange
        mock_factory = Mock()
        mock_model = Mock()
        mock_model.generate_text.return_value = "Response"
        mock_factory.create.return_value = mock_model
        adapter = LLMAdapter(mock_factory, "test-model")
        prompt = "What is the meaning of life?"

        # Act
        adapter.generate_text(prompt)

        # Assert
        mock_model.generate_text.assert_called_once_with(
            "What is the meaning of life?", 
            system_prompt=None,
            temperature=None,
            max_tokens=None,
            json_schema=None
        )

    def test_generate_text_with_different_prompts(self):
        """Test generate_text with various prompt formats."""
        # Arrange
        mock_factory = Mock()
        mock_model = Mock()
        mock_factory.create.return_value = mock_model
        adapter = LLMAdapter(mock_factory, "test-model")

        test_cases = [
            "Simple prompt",
            "Multi-line\nprompt",
            "Prompt with special chars: !@#$%",
            "",
        ]

        for prompt in test_cases:
            mock_model.generate_text.return_value = f"Response to: {prompt}"
            # Act
            result = adapter.generate_text(prompt)
            # Assert
            assert result == f"Response to: {prompt}"
            mock_model.generate_text.assert_called_with(
                prompt, 
                system_prompt=None,
            temperature=None,
            max_tokens=None,
            json_schema=None
        )

    def test_different_model_names(self):
        """Test that different model names are handled correctly."""
        # Arrange
        mock_factory = Mock()
        mock_factory.create.return_value = Mock()

        model_names = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "custom-model"]

        for model_name in model_names:
            # Act
            adapter = LLMAdapter(mock_factory, model_name)
            # Assert
            assert adapter.model_name == model_name
            mock_factory.create.assert_called_with(model_name)

    def test_generate_text_with_system_prompt(self):
        """Test that generate_text passes system_prompt to the model."""
        # Arrange
        mock_factory = Mock()
        mock_model = Mock()
        mock_model.generate_text.return_value = "Generated response"
        mock_factory.create.return_value = mock_model
        adapter = LLMAdapter(mock_factory, "test-model")
        prompt = "Test prompt"
        system_prompt = "You are a helpful assistant."

        # Act
        result = adapter.generate_text(prompt, system_prompt=system_prompt)

        # Assert
        assert result == "Generated response"
        mock_model.generate_text.assert_called_once_with(
            prompt, 
            system_prompt=system_prompt,
            temperature=None,
            max_tokens=None,
            json_schema=None
        )

    def test_generate_text_without_system_prompt(self):
        """Test that generate_text works without system_prompt."""
        # Arrange
        mock_factory = Mock()
        mock_model = Mock()
        mock_model.generate_text.return_value = "Generated response"
        mock_factory.create.return_value = mock_model
        adapter = LLMAdapter(mock_factory, "test-model")
        prompt = "Test prompt"

        # Act
        result = adapter.generate_text(prompt)

        # Assert
        assert result == "Generated response"
        mock_model.generate_text.assert_called_once_with(
            prompt, 
            system_prompt=None,
            temperature=None,
            max_tokens=None,
            json_schema=None
        )

    def test_generate_text_system_prompt_passed_correctly(self):
        """Test that system_prompt is passed correctly to the model."""
        # Arrange
        mock_factory = Mock()
        mock_model = Mock()
        mock_model.generate_text.return_value = "Response"
        mock_factory.create.return_value = mock_model
        adapter = LLMAdapter(mock_factory, "test-model")
        prompt = "What is 2+2?"
        system_prompt = "You are a math tutor."

        # Act
        adapter.generate_text(prompt, system_prompt=system_prompt)

        # Assert
        mock_model.generate_text.assert_called_once_with(
            prompt, 
            system_prompt=system_prompt,
            temperature=None,
            max_tokens=None,
            json_schema=None
        )

    def test_generate_text_with_different_system_prompts(self):
        """Test generate_text with various system prompts."""
        # Arrange
        mock_factory = Mock()
        mock_model = Mock()
        mock_factory.create.return_value = mock_model
        adapter = LLMAdapter(mock_factory, "test-model")
        prompt = "Answer the question."

        system_prompts = [
            "You are a helpful assistant.",
            "You are a technical expert.",
            "You are a creative writer.",
            "",
        ]

        for system_prompt in system_prompts:
            mock_model.generate_text.return_value = f"Response with system: {system_prompt}"
            # Act
            result = adapter.generate_text(prompt, system_prompt=system_prompt)
            # Assert
            assert result == f"Response with system: {system_prompt}"
            mock_model.generate_text.assert_called_with(
                prompt, 
                system_prompt=system_prompt,
            temperature=None,
            max_tokens=None,
            json_schema=None
        )


    def test_generate_text_with_temperature(self):
        """Test that generate_text passes temperature to the model."""
        # Arrange
        mock_factory = Mock()
        mock_model = Mock()
        mock_model.generate_text.return_value = "Generated response"
        mock_factory.create.return_value = mock_model
        adapter = LLMAdapter(mock_factory, "test-model")
        prompt = "Test prompt"
        temperature = 0.1

        # Act
        result = adapter.generate_text(prompt, temperature=temperature)

        # Assert
        assert result == "Generated response"
        mock_model.generate_text.assert_called_once_with(
            prompt,
            system_prompt=None,
            temperature=temperature,
            max_tokens=None,
            json_schema=None
        )

    def test_generate_text_with_max_tokens(self):
        """Test that generate_text passes max_tokens to the model."""
        # Arrange
        mock_factory = Mock()
        mock_model = Mock()
        mock_model.generate_text.return_value = "Generated response"
        mock_factory.create.return_value = mock_model
        adapter = LLMAdapter(mock_factory, "test-model")
        prompt = "Test prompt"
        max_tokens = 4096

        # Act
        result = adapter.generate_text(prompt, max_tokens=max_tokens)

        # Assert
        assert result == "Generated response"
        mock_model.generate_text.assert_called_once_with(
            prompt,
            system_prompt=None,
            temperature=None,
            max_tokens=max_tokens,
            json_schema=None
        )

    def test_generate_text_with_all_parameters(self):
        """Test that generate_text passes all optional parameters to the model."""
        # Arrange
        mock_factory = Mock()
        mock_model = Mock()
        mock_model.generate_text.return_value = "Generated response"
        mock_factory.create.return_value = mock_model
        adapter = LLMAdapter(mock_factory, "test-model")
        prompt = "Test prompt"
        system_prompt = "You are a helpful assistant."
        temperature = 0.1
        max_tokens = 4096

        # Act
        result = adapter.generate_text(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Assert
        assert result == "Generated response"
        mock_model.generate_text.assert_called_once_with(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=None
        )

    def test_generate_text_with_json_schema(self):
        """Test that generate_text passes json_schema to the model."""
        # Arrange
        mock_factory = Mock()
        mock_model = Mock()
        mock_model.generate_text.return_value = "Generated response"
        mock_factory.create.return_value = mock_model
        adapter = LLMAdapter(mock_factory, "test-model")
        prompt = "Test prompt"
        json_schema = {"type": "object", "properties": {}}

        # Act
        result = adapter.generate_text(prompt, json_schema=json_schema)

        # Assert
        assert result == "Generated response"
        mock_model.generate_text.assert_called_once_with(
            prompt,
            system_prompt=None,
            temperature=None,
            max_tokens=None,
            json_schema=json_schema
        )


class TestGetFactory:
    """Test suite for get_factory function."""

    def test_get_factory_with_gpt(self):
        """Test that get_factory returns GPTFactory for 'GPT'."""
        # Arrange
        model_family = "GPT"

        # Act
        factory = get_factory(model_family)

        # Assert
        from llm_adapter.gpt_factory import GPTFactory
        assert isinstance(factory, GPTFactory)

    def test_get_factory_with_lowercase_gpt(self):
        """Test that get_factory works with lowercase model family."""
        # Arrange
        model_family = "gpt"

        # Act
        factory = get_factory(model_family)

        # Assert
        from llm_adapter.gpt_factory import GPTFactory
        assert isinstance(factory, GPTFactory)

    def test_get_factory_with_invalid_model_family(self):
        """Test that get_factory raises ImportError for invalid model family."""
        # Arrange
        model_family = "InvalidFactory"

        # Act & Assert
        with pytest.raises(ImportError, match="Could not import factory module"):
            get_factory(model_family)

    def test_get_factory_with_missing_class(self):
        """Test that get_factory raises AttributeError when class doesn't exist."""
        # Arrange
        model_family = "NonExistent"

        # Act & Assert
        with pytest.raises((ImportError, AttributeError)):
            get_factory(model_family)

    @patch('llm_adapter.llm_adapter.importlib.import_module')
    def test_get_factory_import_error(self, mock_import):
        """Test that get_factory handles ImportError correctly."""
        # Arrange
        mock_import.side_effect = ImportError("Module not found")
        model_family = "TestFactory"

        # Act & Assert
        with pytest.raises(ImportError, match="Could not import factory module"):
            get_factory(model_family)

    @patch('llm_adapter.llm_adapter.importlib.import_module')
    def test_get_factory_attribute_error(self, mock_import):
        """Test that get_factory handles AttributeError correctly."""
        # Arrange
        mock_module = Mock(spec=[])  # Module with no attributes
        mock_import.return_value = mock_module
        model_family = "TestFactory"

        # Act & Assert
        with pytest.raises(AttributeError, match="Factory class.*not found"):
            get_factory(model_family)

    def test_get_factory_rejects_empty_model_family(self):
        """Test that get_factory rejects empty model family."""
        with pytest.raises(ValueError, match="model_family cannot be empty"):
            get_factory("")
        
        with pytest.raises(ValueError, match="model_family cannot be empty"):
            get_factory("   ")  # Whitespace only

    def test_get_factory_with_qwen(self):
        """Test that get_factory returns QwenFactory for 'Qwen'."""
        # Arrange
        model_family = "Qwen"

        # Act
        factory = get_factory(model_family)

        # Assert
        from llm_adapter.qwen_factory import QwenFactory
        assert isinstance(factory, QwenFactory)

    def test_get_factory_with_lowercase_qwen(self):
        """Test that get_factory works with lowercase model family."""
        # Arrange
        model_family = "qwen"

        # Act
        factory = get_factory(model_family)

        # Assert
        from llm_adapter.qwen_factory import QwenFactory
        assert isinstance(factory, QwenFactory)

    def test_get_factory_with_mistral(self):
        """Test that get_factory returns MistralFactory for 'Mistral'."""
        # Arrange
        model_family = "Mistral"

        # Act
        factory = get_factory(model_family)

        # Assert
        from llm_adapter.mistral_factory import MistralFactory
        assert isinstance(factory, MistralFactory)

    def test_get_factory_with_lowercase_mistral(self):
        """Test that get_factory works with lowercase model family."""
        # Arrange
        model_family = "mistral"

        # Act
        factory = get_factory(model_family)

        # Assert
        from llm_adapter.mistral_factory import MistralFactory
        assert isinstance(factory, MistralFactory)
