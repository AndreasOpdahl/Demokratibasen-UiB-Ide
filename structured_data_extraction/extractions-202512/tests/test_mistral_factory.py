"""
Tests for MistralFactory and MistralModel classes.
"""
import os
import pytest
from unittest.mock import Mock, MagicMock, patch, call

from llm_adapter.mistral_factory import MistralFactory, MistralModel


class TestMistralModel:
    """Test suite for MistralModel class."""

    @patch("llm_adapter.mistral_factory.openai.OpenAI")
    def test_init_creates_openai_client_with_mistral_base_url(self, mock_openai_class):
        """Test that MistralModel initializes with OpenAI client and Mistral base URL."""
        # Arrange
        api_key = "test-api-key"
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Act
        model = MistralModel(api_key, "mistral-large")

        # Assert
        assert model.api_key == api_key
        assert model.model_name == "mistral-large"
        assert model.client == mock_client
        mock_openai_class.assert_called_once_with(api_key=api_key, base_url="https://api.mistral.ai/v1")

    @patch("llm_adapter.mistral_factory.openai.OpenAI")
    def test_init_stores_api_key(self, mock_openai_class):
        """Test that API key is stored correctly."""
        # Arrange
        api_key = "sk-mistral-test123456"
        mock_openai_class.return_value = Mock()

        # Act
        model = MistralModel(api_key, "mistral-large")

        # Assert
        assert model.api_key == api_key
        assert model.model_name == "mistral-large"

    @patch("llm_adapter.mistral_factory.openai.OpenAI")
    def test_generate_text_calls_client_chat_completions(self, mock_openai_class):
        """Test that generate_text calls the OpenAI client correctly."""
        # Arrange
        api_key = "test-api-key"
        mock_client = Mock()
        mock_chat = Mock()
        mock_completions = Mock()
        mock_create = Mock()
        
        mock_client.chat = mock_chat
        mock_chat.completions = mock_completions
        mock_completions.create = mock_create
        mock_create.return_value = "Generated response"
        
        mock_openai_class.return_value = mock_client
        model = MistralModel(api_key, "mistral-large")
        prompt = "Test prompt"

        # Act
        result = model.generate_text(prompt)

        # Assert
        assert result == "Generated response"
        mock_create.assert_called_once_with(
            model="mistral-large",
            messages=[{"role": "user", "content": prompt}]
        )

    @patch("llm_adapter.mistral_factory.openai.OpenAI")
    def test_generate_text_with_system_prompt(self, mock_openai_class):
        """Test that generate_text includes system prompt in messages when provided."""
        # Arrange
        api_key = "test-api-key"
        mock_client = Mock()
        mock_chat = Mock()
        mock_completions = Mock()
        mock_create = Mock()
        
        mock_client.chat = mock_chat
        mock_chat.completions = mock_completions
        mock_completions.create = mock_create
        mock_create.return_value = "Generated response"
        
        mock_openai_class.return_value = mock_client
        model = MistralModel(api_key, "mistral-large")
        prompt = "Test prompt"
        system_prompt = "You are a helpful assistant."

        # Act
        result = model.generate_text(prompt, system_prompt=system_prompt)

        # Assert
        assert result == "Generated response"
        mock_create.assert_called_once_with(
            model="mistral-large",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

    @patch("llm_adapter.mistral_factory.openai.OpenAI")
    def test_generate_text_with_temperature_and_max_tokens(self, mock_openai_class):
        """Test that generate_text includes temperature and max_tokens when provided."""
        # Arrange
        api_key = "test-api-key"
        mock_client = Mock()
        mock_chat = Mock()
        mock_completions = Mock()
        mock_create = Mock()
        
        mock_client.chat = mock_chat
        mock_chat.completions = mock_completions
        mock_completions.create = mock_create
        mock_create.return_value = "Generated response"
        
        mock_openai_class.return_value = mock_client
        model = MistralModel(api_key, "mistral-large")
        prompt = "Another message."
        temperature = 0.5
        max_tokens = 100

        # Act
        result = model.generate_text(prompt, temperature=temperature, max_tokens=max_tokens)

        # Assert
        assert result == "Generated response"
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert call_args.kwargs["model"] == "mistral-large"
        assert call_args.kwargs["temperature"] == temperature
        assert call_args.kwargs["max_tokens"] == max_tokens
        assert call_args.kwargs["messages"][0]["content"] == prompt

    @patch("llm_adapter.mistral_factory.openai.OpenAI")
    def test_generate_text_all_parameters(self, mock_openai_class):
        """Test that generate_text includes all optional parameters when provided."""
        # Arrange
        api_key = "test-api-key"
        mock_client = Mock()
        mock_chat = Mock()
        mock_completions = Mock()
        mock_create = Mock()
        
        mock_client.chat = mock_chat
        mock_chat.completions = mock_completions
        mock_completions.create = mock_create
        mock_create.return_value = "Generated response"
        
        mock_openai_class.return_value = mock_client
        model = MistralModel(api_key, "mistral-medium")
        prompt = "Full test message."
        system_prompt = "Full system message."
        temperature = 0.7
        max_tokens = 200

        # Act
        result = model.generate_text(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Assert
        assert result == "Generated response"
        mock_create.assert_called_once_with(
            model="mistral-medium",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )


class TestMistralFactory:
    """Test suite for MistralFactory class."""

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test-api-key"})
    @patch("llm_adapter.mistral_factory.MistralModel")
    def test_create_returns_mistral_model(self, mock_model_class):
        """Test that create returns a MistralModel instance."""
        # Arrange
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        factory = MistralFactory()
        model_name = "mistral-large"

        # Act
        result = factory.create(model_name)

        # Assert
        assert result == mock_model_instance
        mock_model_class.assert_called_once_with(
            api_key="test-api-key",
            model_name="mistral-large"
        )

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "sk-mistral-test123456"})
    @patch("llm_adapter.mistral_factory.MistralModel")
    def test_create_uses_environment_api_key(self, mock_model_class):
        """Test that create uses MISTRAL_API_KEY from environment."""
        # Arrange
        mock_model_class.return_value = Mock()
        factory = MistralFactory()

        # Act
        factory.create("mistral-large")

        # Assert
        mock_model_class.assert_called_once_with(api_key="sk-mistral-test123456", model_name="mistral-large")

    @patch.dict(os.environ, {}, clear=True)
    @patch("llm_adapter.mistral_factory.MistralModel")
    def test_create_with_no_api_key_in_env(self, mock_model_class):
        """Test that create raises ValueError when API key is missing."""
        # Arrange
        mock_model_class.return_value = Mock()
        factory = MistralFactory()

        # Act & Assert
        with pytest.raises(ValueError, match="MISTRAL_API_KEY environment variable is not set"):
            factory.create("mistral-large")
        
        # Should not create model when API key is missing
        mock_model_class.assert_not_called()

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"})
    @patch("llm_adapter.mistral_factory.MistralModel")
    def test_create_passes_model_name_to_model(self, mock_model_class):
        """Test that create passes model_name to MistralModel."""
        # Arrange
        mock_model_class.return_value = Mock()
        factory = MistralFactory()
        model_name = "mistral-medium"

        # Act
        factory.create(model_name)

        # Assert
        mock_model_class.assert_called_once_with(api_key="test-key", model_name="mistral-medium")

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"})
    @patch("llm_adapter.mistral_factory.MistralModel")
    def test_create_with_different_model_names(self, mock_model_class):
        """Test create with various model names."""
        # Arrange
        mock_model_class.return_value = Mock()
        factory = MistralFactory()

        model_names = ["mistral-large", "mistral-medium", "mistral-small", "mistral-tiny"]

        for model_name in model_names:
            # Act
            factory.create(model_name)
            # Assert
            # All should use the same API key from env, but different model names
            mock_model_class.assert_called_with(api_key="test-key", model_name=model_name)
            mock_model_class.reset_mock() # Reset mock for next iteration

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "key1"})
    @patch("llm_adapter.mistral_factory.MistralModel")
    def test_create_returns_new_instance_each_time(self, mock_model_class):
        """Test that create returns a new model instance each time."""
        # Arrange
        mock_instance1 = Mock()
        mock_instance2 = Mock()
        mock_model_class.side_effect = [mock_instance1, mock_instance2]
        factory = MistralFactory()

        # Act
        result1 = factory.create("mistral-large")
        result2 = factory.create("mistral-large")

        # Assert
        assert result1 == mock_instance1
        assert result2 == mock_instance2
        assert result1 != result2
        assert mock_model_class.call_count == 2


class TestMistralModelWindowSize:
    """Test suite for Mistral model window size detection."""

    def test_get_model_window_size_mistral_large(self):
        """Test window size for mistral-large."""
        from llm_adapter.mistral_factory import get_model_window_size
        
        assert get_model_window_size("mistral-large") == 128000
        assert get_model_window_size("mistral-large-latest") == 128000

    def test_get_model_window_size_mistral_medium(self):
        """Test window size for mistral-medium."""
        from llm_adapter.mistral_factory import get_model_window_size
        assert get_model_window_size("mistral-medium") == 32000
        assert get_model_window_size("mistral-medium-latest") == 32000

    def test_get_model_window_size_mistral_small(self):
        """Test window size for mistral-small."""
        from llm_adapter.mistral_factory import get_model_window_size
        assert get_model_window_size("mistral-small") == 32000

    def test_get_model_window_size_pixtral(self):
        """Test window size for pixtral-12b."""
        from llm_adapter.mistral_factory import get_model_window_size
        assert get_model_window_size("pixtral-12b") == 128000

    def test_get_model_window_size_unknown_model(self):
        """Test default window size for an unknown model."""
        from llm_adapter.mistral_factory import get_model_window_size
        assert get_model_window_size("unknown-mistral-model") == 32000


class TestMistralModelNameValidation:
    """Test suite for Mistral model name validation."""

    def test_validate_mistral_model_name_valid(self):
        """Test valid Mistral model names."""
        from llm_adapter.mistral_factory import validate_mistral_model_name
        
        valid_models = [
            "mistral-large",
            "mistral-large-latest",
            "mistral-large-3",
            "mistral-medium",
            "mistral-small",
            "mistral-tiny",
            "pixtral-12b",
            "mistral-nemo",
        ]
        for model_name in valid_models:
            assert validate_mistral_model_name(model_name) is True

    def test_validate_mistral_model_name_invalid(self):
        """Test invalid Mistral model names."""
        from llm_adapter.mistral_factory import validate_mistral_model_name
        
        invalid_models = [
            "mistral-invalid",
            "mistral-unknown",
            "gpt-4",
            "",
            "mistral",  # Too short, needs a suffix
        ]
        for model_name in invalid_models:
            assert validate_mistral_model_name(model_name) is False

    def test_validate_mistral_model_name_case_insensitivity(self):
        """Test case-insensitivity for model names."""
        from llm_adapter.mistral_factory import validate_mistral_model_name
        
        assert validate_mistral_model_name("MISTRAL-LARGE") is True
        assert validate_mistral_model_name("Mistral-Large") is True

