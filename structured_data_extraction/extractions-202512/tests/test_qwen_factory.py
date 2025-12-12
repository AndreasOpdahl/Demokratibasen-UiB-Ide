"""
Tests for QwenFactory and QwenModel classes.
"""
import os
import pytest
from unittest.mock import Mock, MagicMock, patch, call

from llm_adapter.qwen_factory import QwenFactory, QwenModel


class TestQwenModel:
    """Test suite for QwenModel class."""

    @patch("llm_adapter.qwen_factory.openai.OpenAI")
    def test_init_creates_openai_client_with_qwen_base_url(self, mock_openai_class):
        """Test that QwenModel initializes with OpenAI client and Qwen base URL."""
        # Arrange
        api_key = "test-api-key"
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Act
        model = QwenModel(api_key, "qwen-turbo")

        # Assert
        assert model.api_key == api_key
        assert model.model_name == "qwen-turbo"
        assert model.client == mock_client
        mock_openai_class.assert_called_once_with(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )

    @patch("llm_adapter.qwen_factory.openai.OpenAI")
    def test_init_stores_api_key(self, mock_openai_class):
        """Test that API key is stored correctly."""
        # Arrange
        api_key = "sk-qwen-test123456"
        mock_openai_class.return_value = Mock()

        # Act
        model = QwenModel(api_key, "qwen-turbo")

        # Assert
        assert model.api_key == api_key
        assert model.model_name == "qwen-turbo"

    @patch("llm_adapter.qwen_factory.openai.OpenAI")
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
        model = QwenModel(api_key, "qwen-turbo")
        prompt = "Test prompt"

        # Act
        result = model.generate_text(prompt)

        # Assert
        assert result == "Generated response"
        mock_create.assert_called_once_with(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

    @patch("llm_adapter.qwen_factory.openai.OpenAI")
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
        model = QwenModel(api_key, "qwen-turbo")
        prompt = "Test prompt"
        system_prompt = "You are a helpful assistant."

        # Act
        result = model.generate_text(prompt, system_prompt=system_prompt)

        # Assert
        assert result == "Generated response"
        mock_create.assert_called_once_with(
            model="qwen-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

    @patch("llm_adapter.qwen_factory.openai.OpenAI")
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
        model = QwenModel(api_key, "qwen-turbo")
        prompt = "Another message."
        temperature = 0.5
        max_tokens = 100

        # Act
        result = model.generate_text(prompt, temperature=temperature, max_tokens=max_tokens)

        # Assert
        assert result == "Generated response"
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert call_args.kwargs["model"] == "qwen-turbo"
        assert call_args.kwargs["temperature"] == temperature
        assert call_args.kwargs["max_tokens"] == max_tokens
        assert call_args.kwargs["messages"][0]["content"] == prompt

    @patch("llm_adapter.qwen_factory.openai.OpenAI")
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
        model = QwenModel(api_key, "qwen-plus")
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
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )


class TestQwenFactory:
    """Test suite for QwenFactory class."""

    @patch.dict(os.environ, {"QWEN_API_KEY": "test-api-key"})
    @patch("llm_adapter.qwen_factory.QwenModel")
    def test_create_returns_qwen_model(self, mock_model_class):
        """Test that create returns a QwenModel instance."""
        # Arrange
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        factory = QwenFactory()
        model_name = "qwen-turbo"

        # Act
        result = factory.create(model_name)

        # Assert
        assert result == mock_model_instance
        mock_model_class.assert_called_once_with(
            api_key="test-api-key",
            model_name="qwen-turbo"
        )

    @patch.dict(os.environ, {"QWEN_API_KEY": "sk-qwen-test123456"})
    @patch("llm_adapter.qwen_factory.QwenModel")
    def test_create_uses_environment_api_key(self, mock_model_class):
        """Test that create uses QWEN_API_KEY from environment."""
        # Arrange
        mock_model_class.return_value = Mock()
        factory = QwenFactory()

        # Act
        factory.create("qwen-turbo")

        # Assert
        mock_model_class.assert_called_once_with(api_key="sk-qwen-test123456", model_name="qwen-turbo")

    @patch.dict(os.environ, {}, clear=True)
    @patch("llm_adapter.qwen_factory.QwenModel")
    def test_create_with_no_api_key_in_env(self, mock_model_class):
        """Test that create raises ValueError when API key is missing."""
        # Arrange
        mock_model_class.return_value = Mock()
        factory = QwenFactory()

        # Act & Assert
        with pytest.raises(ValueError, match="QWEN_API_KEY environment variable is not set"):
            factory.create("qwen-turbo")
        
        # Should not create model when API key is missing
        mock_model_class.assert_not_called()

    @patch.dict(os.environ, {"QWEN_API_KEY": "test-key"})
    @patch("llm_adapter.qwen_factory.QwenModel")
    def test_create_passes_model_name_to_model(self, mock_model_class):
        """Test that create passes model_name to QwenModel."""
        # Arrange
        mock_model_class.return_value = Mock()
        factory = QwenFactory()
        model_name = "qwen-plus"

        # Act
        factory.create(model_name)

        # Assert
        mock_model_class.assert_called_once_with(api_key="test-key", model_name="qwen-plus")

    @patch.dict(os.environ, {"QWEN_API_KEY": "test-key"})
    @patch("llm_adapter.qwen_factory.QwenModel")
    def test_create_with_different_model_names(self, mock_model_class):
        """Test create with various model names."""
        # Arrange
        mock_model_class.return_value = Mock()
        factory = QwenFactory()

        model_names = ["qwen-turbo", "qwen-plus", "qwen-max", "qwen2.5-7b-instruct"]

        for model_name in model_names:
            # Act
            factory.create(model_name)
            # Assert
            # All should use the same API key from env, but different model names
            mock_model_class.assert_called_with(api_key="test-key", model_name=model_name)
            mock_model_class.reset_mock() # Reset mock for next iteration

    @patch.dict(os.environ, {"QWEN_API_KEY": "key1"})
    @patch("llm_adapter.qwen_factory.QwenModel")
    def test_create_returns_new_instance_each_time(self, mock_model_class):
        """Test that create returns a new model instance each time."""
        # Arrange
        mock_instance1 = Mock()
        mock_instance2 = Mock()
        mock_model_class.side_effect = [mock_instance1, mock_instance2]
        factory = QwenFactory()

        # Act
        result1 = factory.create("qwen-turbo")
        result2 = factory.create("qwen-turbo")

        # Assert
        assert result1 == mock_instance1
        assert result2 == mock_instance2
        assert result1 != result2
        assert mock_model_class.call_count == 2


class TestQwenModelWindowSize:
    """Test suite for Qwen model window size detection."""

    def test_get_model_window_size_qwen_turbo(self):
        """Test window size for qwen-turbo."""
        from llm_adapter.qwen_factory import get_model_window_size
        
        assert get_model_window_size("qwen-turbo") == 8000
        assert get_model_window_size("qwen-turbo-longcontext") == 30000

    def test_get_model_window_size_qwen_plus(self):
        """Test window size for qwen-plus."""
        from llm_adapter.qwen_factory import get_model_window_size
        assert get_model_window_size("qwen-plus") == 32000
        assert get_model_window_size("qwen-plus-longcontext") == 32000

    def test_get_model_window_size_qwen_max(self):
        """Test window size for qwen-max."""
        from llm_adapter.qwen_factory import get_model_window_size
        assert get_model_window_size("qwen-max") == 8000

    def test_get_model_window_size_qwen2_5(self):
        """Test window size for qwen2.5 models."""
        from llm_adapter.qwen_factory import get_model_window_size
        assert get_model_window_size("qwen2.5-72b-instruct") == 128000
        assert get_model_window_size("qwen2.5-7b-instruct") == 128000

    def test_get_model_window_size_unknown_model(self):
        """Test default window size for an unknown model."""
        from llm_adapter.qwen_factory import get_model_window_size
        assert get_model_window_size("unknown-qwen-model") == 8000


class TestQwenModelNameValidation:
    """Test suite for Qwen model name validation."""

    def test_validate_qwen_model_name_valid(self):
        """Test valid Qwen model names."""
        from llm_adapter.qwen_factory import validate_qwen_model_name
        
        valid_models = [
            "qwen-turbo",
            "qwen-plus",
            "qwen-max",
            "qwen2.5-72b-instruct",
            "qwen2.5-7b-instruct",
            "qwen2-7b-instruct",
            "qwen1.5-7b-chat",
            "qwen3-8b",
        ]
        for model_name in valid_models:
            assert validate_qwen_model_name(model_name) is True

    def test_validate_qwen_model_name_invalid(self):
        """Test invalid Qwen model names."""
        from llm_adapter.qwen_factory import validate_qwen_model_name
        
        invalid_models = [
            "qwen-invalid",
            "qwen-unknown",
            "gpt-4",
            "",
            "qwen",  # Too short, needs a suffix
        ]
        for model_name in invalid_models:
            assert validate_qwen_model_name(model_name) is False

    def test_validate_qwen_model_name_case_insensitivity(self):
        """Test case-insensitivity for model names."""
        from llm_adapter.qwen_factory import validate_qwen_model_name
        
        assert validate_qwen_model_name("QWEN-TURBO") is True
        assert validate_qwen_model_name("Qwen-Turbo") is True

