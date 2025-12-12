"""
Tests for ClaudeFactory and ClaudeModel classes.
"""
import os
import pytest
from unittest.mock import Mock, MagicMock, patch, call

from llm_adapter.claude_factory import ClaudeFactory, ClaudeModel


class TestClaudeModel:
    """Test suite for ClaudeModel class."""

    @patch("llm_adapter.claude_factory.anthropic.Anthropic")
    def test_init_creates_anthropic_client(self, mock_anthropic_class):
        """Test that ClaudeModel initializes with Anthropic client."""
        # Arrange
        api_key = "test-api-key"
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Act
        model = ClaudeModel(api_key, "claude-3-7-sonnet")

        # Assert
        assert model.api_key == api_key
        assert model.model_name == "claude-3-7-sonnet"
        assert model.client == mock_client
        mock_anthropic_class.assert_called_once_with(api_key=api_key)

    @patch("llm_adapter.claude_factory.anthropic.Anthropic")
    def test_init_stores_api_key(self, mock_anthropic_class):
        """Test that API key is stored correctly."""
        # Arrange
        api_key = "sk-ant-test123456"
        mock_anthropic_class.return_value = Mock()

        # Act
        model = ClaudeModel(api_key, "claude-3-7-sonnet")

        # Assert
        assert model.api_key == api_key
        assert model.model_name == "claude-3-7-sonnet"

    @patch("llm_adapter.claude_factory.anthropic.Anthropic")
    def test_generate_text_calls_client_messages_create(self, mock_anthropic_class):
        """Test that generate_text calls the Anthropic client correctly."""
        # Arrange
        api_key = "test-api-key"
        mock_client = Mock()
        mock_messages = Mock()
        mock_create = Mock()
        
        mock_client.messages = mock_messages
        mock_messages.create = mock_create
        
        # Create a mock response object
        mock_response = Mock()
        mock_content_block = Mock()
        mock_content_block.text = "Generated response"
        mock_response.content = [mock_content_block]
        mock_create.return_value = mock_response
        
        mock_anthropic_class.return_value = mock_client
        model = ClaudeModel(api_key, "claude-3-7-sonnet")
        prompt = "Test prompt"

        # Act
        result = model.generate_text(prompt)

        # Assert
        assert result is not None
        assert result.text == "Generated response"
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "Generated response"
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert call_args.kwargs["model"] == "claude-3-7-sonnet"
        assert call_args.kwargs["messages"][0]["role"] == "user"
        assert call_args.kwargs["messages"][0]["content"] == prompt

    @patch("llm_adapter.claude_factory.anthropic.Anthropic")
    def test_generate_text_with_system_prompt(self, mock_anthropic_class):
        """Test that generate_text includes a system prompt when provided."""
        # Arrange
        api_key = "test-api-key"
        mock_client = Mock()
        mock_messages = Mock()
        mock_create = Mock()
        
        mock_client.messages = mock_messages
        mock_messages.create = mock_create
        
        mock_response = Mock()
        mock_content_block = Mock()
        mock_content_block.text = "Generated response"
        mock_response.content = [mock_content_block]
        mock_create.return_value = mock_response
        
        mock_anthropic_class.return_value = mock_client
        model = ClaudeModel(api_key, "claude-3-7-sonnet")
        prompt = "User message."
        system_prompt = "System message."

        # Act
        result = model.generate_text(prompt, system_prompt=system_prompt)

        # Assert
        assert result.text == "Generated response"
        call_args = mock_create.call_args
        assert call_args.kwargs["system"] == system_prompt

    @patch("llm_adapter.claude_factory.anthropic.Anthropic")
    def test_generate_text_with_temperature_and_max_tokens(self, mock_anthropic_class):
        """Test that generate_text includes temperature and max_tokens when provided."""
        # Arrange
        api_key = "test-api-key"
        mock_client = Mock()
        mock_messages = Mock()
        mock_create = Mock()
        
        mock_client.messages = mock_messages
        mock_messages.create = mock_create
        
        mock_response = Mock()
        mock_content_block = Mock()
        mock_content_block.text = "Generated response"
        mock_response.content = [mock_content_block]
        mock_create.return_value = mock_response
        
        mock_anthropic_class.return_value = mock_client
        model = ClaudeModel(api_key, "claude-3-7-sonnet")
        prompt = "Another message."
        temperature = 0.5
        max_tokens = 100

        # Act
        result = model.generate_text(prompt, temperature=temperature, max_tokens=max_tokens)

        # Assert
        assert result.text == "Generated response"
        call_args = mock_create.call_args
        assert call_args.kwargs["temperature"] == temperature
        assert call_args.kwargs["max_tokens"] == max_tokens

    @patch("llm_adapter.claude_factory.anthropic.Anthropic")
    def test_generate_text_all_parameters(self, mock_anthropic_class):
        """Test that generate_text includes all optional parameters when provided."""
        # Arrange
        api_key = "test-api-key"
        mock_client = Mock()
        mock_messages = Mock()
        mock_create = Mock()
        
        mock_client.messages = mock_messages
        mock_messages.create = mock_create
        
        mock_response = Mock()
        mock_content_block = Mock()
        mock_content_block.text = "Generated response"
        mock_response.content = [mock_content_block]
        mock_create.return_value = mock_response
        
        mock_anthropic_class.return_value = mock_client
        model = ClaudeModel(api_key, "claude-3-opus")
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
        assert result.text == "Generated response"
        call_args = mock_create.call_args
        assert call_args.kwargs["system"] == system_prompt
        assert call_args.kwargs["temperature"] == temperature
        assert call_args.kwargs["max_tokens"] == max_tokens


class TestClaudeFactory:
    """Test suite for ClaudeFactory class."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    @patch("llm_adapter.claude_factory.ClaudeModel")
    def test_create_returns_claude_model(self, mock_model_class):
        """Test that factory creates a ClaudeModel instance."""
        # Arrange
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        factory = ClaudeFactory()

        # Act
        result = factory.create("claude-3-7-sonnet")

        # Assert
        assert result == mock_model_instance
        # Factory keeps short names as-is (API accepts them)
        mock_model_class.assert_called_once_with(
            api_key="test-api-key",
            model_name="claude-3-7-sonnet"
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_create_raises_error_when_api_key_missing(self):
        """Test that factory raises ValueError when API key is missing."""
        # Arrange
        factory = ClaudeFactory()

        # Act & Assert
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            factory.create("claude-3-7-sonnet")

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    def test_create_raises_error_for_invalid_model_name(self):
        """Test that factory raises ValueError for invalid model names."""
        # Arrange
        factory = ClaudeFactory()

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid Claude model name"):
            factory.create("invalid-claude-model")

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    @patch("llm_adapter.claude_factory.ClaudeModel")
    def test_create_validates_model_name(self, mock_model_class):
        """Test that factory validates model name before creating model."""
        # Arrange
        factory = ClaudeFactory()

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid Claude model name"):
            factory.create("claude-invalid")

        # Verify ClaudeModel was not called
        mock_model_class.assert_not_called()

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"})
    @patch("llm_adapter.claude_factory.ClaudeModel")
    def test_create_with_different_valid_models(self, mock_model_class):
        """Test that factory can create different valid Claude models."""
        # Arrange
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        factory = ClaudeFactory()

        # Test with both short and versioned model names
        # Factory keeps short names as-is (API accepts them), only normalizes models that need it
        test_cases = [
            ("claude-3-7-sonnet", "claude-3-7-sonnet"),  # Kept as-is
            ("claude-3-opus", "claude-3-opus"),  # Kept as-is
            ("claude-3-sonnet", "claude-3-sonnet"),  # Kept as-is
            ("claude-3-haiku", "claude-3-haiku-20240307"),  # Normalized (needs versioning)
            ("claude-sonnet-4-20250514", "claude-sonnet-4-20250514"),  # Already versioned
        ]

        for input_name, expected_normalized in test_cases:
            # Act
            result = factory.create(input_name)

            # Assert
            assert result == mock_model_instance
            mock_model_class.assert_called_with(
                api_key="test-api-key",
                model_name=expected_normalized
            )
            mock_model_class.reset_mock()


class TestClaudeModelWindowSize:
    """Test suite for Claude model window size detection."""

    def test_get_model_window_size_claude_3_7_sonnet(self):
        """Test window size for Claude 3.7 Sonnet."""
        from llm_adapter.claude_factory import get_model_window_size
        
        assert get_model_window_size("claude-3-7-sonnet") == 200000
        assert get_model_window_size("claude-3-7-sonnet-20250219") == 200000

    def test_get_model_window_size_claude_3_opus(self):
        """Test window size for Claude 3 Opus."""
        from llm_adapter.claude_factory import get_model_window_size
        
        assert get_model_window_size("claude-3-opus") == 200000
        assert get_model_window_size("claude-3-opus-20240229") == 200000

    def test_get_model_window_size_claude_sonnet_4(self):
        """Test window size for Claude Sonnet 4."""
        from llm_adapter.claude_factory import get_model_window_size
        
        assert get_model_window_size("claude-sonnet-4-20250514") == 200000

    def test_get_model_window_size_default(self):
        """Test default window size for unknown models."""
        from llm_adapter.claude_factory import get_model_window_size
        
        assert get_model_window_size("unknown-claude-model") == 200000


class TestValidateClaudeModelName:
    """Test suite for Claude model name validation."""

    def test_validate_claude_model_name_valid_models(self):
        """Test validation with valid Claude model names."""
        from llm_adapter.claude_factory import validate_claude_model_name
        
        valid_models = [
            "claude-3-7-sonnet",
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
            "claude-sonnet-4-20250514",
            "claude-2",
            "claude-2.1",
        ]
        
        for model_name in valid_models:
            assert validate_claude_model_name(model_name) is True

    def test_validate_claude_model_name_invalid_models(self):
        """Test validation with invalid Claude model names."""
        from llm_adapter.claude_factory import validate_claude_model_name
        
        invalid_models = [
            "claude-invalid",
            "gpt-4",
            "gemini-pro",
            "claude-3",
            "claude-3-5",
        ]
        
        for model_name in invalid_models:
            assert validate_claude_model_name(model_name) is False

    def test_validate_claude_model_name_case_insensitive(self):
        """Test that validation is case-insensitive."""
        from llm_adapter.claude_factory import validate_claude_model_name
        
        assert validate_claude_model_name("CLAUDE-3-7-SONNET") is True
        assert validate_claude_model_name("Claude-3-7-Sonnet") is True

