"""
Tests for DeepSeekFactory and DeepSeekModel classes.
"""
import os
import pytest
from unittest.mock import Mock, MagicMock, patch, call

from llm_adapter.deepseek_factory import DeepSeekFactory, DeepSeekModel


class TestDeepSeekModel:
    """Test suite for DeepSeekModel class."""

    @patch("llm_adapter.deepseek_factory.openai.OpenAI")
    def test_init_creates_openai_client_with_deepseek_base_url(self, mock_openai_class):
        """Test that DeepSeekModel initializes with OpenAI client configured for DeepSeek."""
        # Arrange
        api_key = "test-api-key"
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Act
        model = DeepSeekModel(api_key, "deepseek-chat")

        # Assert
        assert model.api_key == api_key
        assert model.model_name == "deepseek-chat"
        assert model.client == mock_client
        mock_openai_class.assert_called_once_with(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=(30.0, 60.0),  # 30s connect timeout, 60s read timeout
            max_retries=2  # Allow 2 retries for transient network issues
        )

    @patch("llm_adapter.deepseek_factory.openai.OpenAI")
    def test_init_stores_api_key(self, mock_openai_class):
        """Test that API key is stored correctly."""
        # Arrange
        api_key = "sk-test123456"
        mock_openai_class.return_value = Mock()

        # Act
        model = DeepSeekModel(api_key, "deepseek-chat")

        # Assert
        assert model.api_key == api_key
        assert model.model_name == "deepseek-chat"

    @patch("llm_adapter.deepseek_factory.openai.OpenAI")
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
        
        # Create a mock response object (OpenAI-compatible)
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Generated response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response
        
        mock_openai_class.return_value = mock_client
        model = DeepSeekModel(api_key, "deepseek-chat")
        prompt = "Test prompt"

        # Act
        result = model.generate_text(prompt)

        # Assert
        assert result is not None
        assert result.choices[0].message.content == "Generated response"
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert call_args.kwargs["model"] == "deepseek-chat"
        assert call_args.kwargs["messages"][0]["role"] == "user"
        assert call_args.kwargs["messages"][0]["content"] == prompt

    @patch("llm_adapter.deepseek_factory.openai.OpenAI")
    def test_generate_text_with_system_prompt(self, mock_openai_class):
        """Test that generate_text includes a system prompt when provided."""
        # Arrange
        api_key = "test-api-key"
        mock_client = Mock()
        mock_chat = Mock()
        mock_completions = Mock()
        mock_create = Mock()
        
        mock_client.chat = mock_chat
        mock_chat.completions = mock_completions
        mock_completions.create = mock_create
        
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Generated response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response
        
        mock_openai_class.return_value = mock_client
        model = DeepSeekModel(api_key, "deepseek-chat")
        prompt = "User message."
        system_prompt = "System message."

        # Act
        result = model.generate_text(prompt, system_prompt=system_prompt)

        # Assert
        assert result.choices[0].message.content == "Generated response"
        call_args = mock_create.call_args
        assert len(call_args.kwargs["messages"]) == 2
        assert call_args.kwargs["messages"][0]["role"] == "system"
        assert call_args.kwargs["messages"][0]["content"] == system_prompt
        assert call_args.kwargs["messages"][1]["role"] == "user"
        assert call_args.kwargs["messages"][1]["content"] == prompt

    @patch("llm_adapter.deepseek_factory.openai.OpenAI")
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
        
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Generated response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response
        
        mock_openai_class.return_value = mock_client
        model = DeepSeekModel(api_key, "deepseek-chat")
        prompt = "Another message."
        temperature = 0.5
        max_tokens = 100

        # Act
        result = model.generate_text(prompt, temperature=temperature, max_tokens=max_tokens)

        # Assert
        assert result.choices[0].message.content == "Generated response"
        call_args = mock_create.call_args
        assert call_args.kwargs["temperature"] == temperature
        assert call_args.kwargs["max_tokens"] == max_tokens

    @patch("llm_adapter.deepseek_factory.openai.OpenAI")
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
        
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Generated response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response
        
        mock_openai_class.return_value = mock_client
        model = DeepSeekModel(api_key, "deepseek-chat-v3")
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
        assert result.choices[0].message.content == "Generated response"
        call_args = mock_create.call_args
        assert call_args.kwargs["messages"][0]["content"] == system_prompt
        assert call_args.kwargs["temperature"] == temperature
        assert call_args.kwargs["max_tokens"] == max_tokens


class TestDeepSeekFactory:
    """Test suite for DeepSeekFactory class."""

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-api-key"})
    @patch("llm_adapter.deepseek_factory.DeepSeekModel")
    def test_create_returns_deepseek_model(self, mock_model_class):
        """Test that factory creates a DeepSeekModel instance."""
        # Arrange
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        factory = DeepSeekFactory()

        # Act
        result = factory.create("deepseek-chat")

        # Assert
        assert result == mock_model_instance
        mock_model_class.assert_called_once_with(
            api_key="test-api-key",
            model_name="deepseek-chat"
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_create_raises_error_when_api_key_missing(self):
        """Test that factory raises ValueError when API key is missing."""
        # Arrange
        factory = DeepSeekFactory()

        # Act & Assert
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            factory.create("deepseek-chat")

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-api-key"})
    def test_create_raises_error_for_invalid_model_name(self):
        """Test that factory raises ValueError for invalid model names."""
        # Arrange
        factory = DeepSeekFactory()

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid DeepSeek model name"):
            factory.create("invalid-deepseek-model")

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-api-key"})
    @patch("llm_adapter.deepseek_factory.DeepSeekModel")
    def test_create_validates_model_name(self, mock_model_class):
        """Test that factory validates model name before creating model."""
        # Arrange
        factory = DeepSeekFactory()

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid DeepSeek model name"):
            factory.create("deepseek-invalid")

        # Verify DeepSeekModel was not called
        mock_model_class.assert_not_called()

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-api-key"})
    @patch("llm_adapter.deepseek_factory.DeepSeekModel")
    def test_create_with_different_valid_models(self, mock_model_class):
        """Test that factory can create different valid DeepSeek models."""
        # Arrange
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        factory = DeepSeekFactory()

        valid_models = [
            "deepseek-chat",
            "deepseek-chat-v3",
            "deepseek-chat-v2",
            "deepseek-coder",
            "deepseek-coder-v2",
        ]

        for model_name in valid_models:
            # Act
            result = factory.create(model_name)

            # Assert
            assert result == mock_model_instance
            mock_model_class.assert_called_with(
                api_key="test-api-key",
                model_name=model_name
            )
            mock_model_class.reset_mock()


class TestDeepSeekModelWindowSize:
    """Test suite for DeepSeek model window size detection."""

    def test_get_model_window_size_deepseek_chat_v3(self):
        """Test window size for DeepSeek Chat V3."""
        from llm_adapter.deepseek_factory import get_model_window_size
        
        assert get_model_window_size("deepseek-chat") == 64000
        assert get_model_window_size("deepseek-chat-v3") == 64000
        assert get_model_window_size("deepseek-chat-v3-0324") == 64000

    def test_get_model_window_size_deepseek_chat_v2(self):
        """Test window size for DeepSeek Chat V2."""
        from llm_adapter.deepseek_factory import get_model_window_size
        
        assert get_model_window_size("deepseek-chat-v2") == 64000
        assert get_model_window_size("deepseek-chat-v2-0324") == 64000

    def test_get_model_window_size_deepseek_coder(self):
        """Test window size for DeepSeek Coder."""
        from llm_adapter.deepseek_factory import get_model_window_size
        
        assert get_model_window_size("deepseek-coder") == 16000
        assert get_model_window_size("deepseek-coder-v2") == 64000

    def test_get_model_window_size_default(self):
        """Test default window size for unknown models."""
        from llm_adapter.deepseek_factory import get_model_window_size
        
        assert get_model_window_size("unknown-deepseek-model") == 64000


class TestValidateDeepSeekModelName:
    """Test suite for DeepSeek model name validation."""

    def test_validate_deepseek_model_name_valid_models(self):
        """Test validation with valid DeepSeek model names."""
        from llm_adapter.deepseek_factory import validate_deepseek_model_name
        
        valid_models = [
            "deepseek-chat",
            "deepseek-chat-v3",
            "deepseek-chat-v2",
            "deepseek-chat-v1",
            "deepseek-coder",
            "deepseek-coder-v2",
        ]
        
        for model_name in valid_models:
            assert validate_deepseek_model_name(model_name) is True

    def test_validate_deepseek_model_name_invalid_models(self):
        """Test validation with invalid DeepSeek model names."""
        from llm_adapter.deepseek_factory import validate_deepseek_model_name
        
        invalid_models = [
            "deepseek-invalid",
            "gpt-4",
            "claude-3",
            "deepseek",
        ]
        
        for model_name in invalid_models:
            assert validate_deepseek_model_name(model_name) is False

    def test_validate_deepseek_model_name_case_insensitive(self):
        """Test that validation is case-insensitive."""
        from llm_adapter.deepseek_factory import validate_deepseek_model_name
        
        assert validate_deepseek_model_name("DEEPSEEK-CHAT") is True
        assert validate_deepseek_model_name("DeepSeek-Chat") is True

