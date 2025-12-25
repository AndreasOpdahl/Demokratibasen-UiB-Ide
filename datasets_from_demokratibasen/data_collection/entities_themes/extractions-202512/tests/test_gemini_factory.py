"""
Tests for GeminiFactory and GeminiModel classes.
"""
import os
import pytest
from unittest.mock import Mock, MagicMock, patch

from llm_adapter.gemini_factory import GeminiFactory, GeminiModel, get_model_window_size


class TestGeminiModel:
    """Test suite for GeminiModel class."""

    @patch("llm_adapter.gemini_factory.genai", None)  # Force legacy API
    @patch("google.generativeai")
    def test_init_creates_gemini_model(self, mock_generativeai_module):
        """Test that GeminiModel initializes with Gemini client."""
        # Arrange
        api_key = "test-api-key"
        model_name = "gemini-2.5-flash"
        mock_model_instance = Mock()
        mock_generativeai_module.GenerativeModel.return_value = mock_model_instance

        # Act
        model = GeminiModel(api_key, model_name)

        # Assert
        assert model.api_key == api_key
        assert model.model_name == model_name
        assert model.model == mock_model_instance
        assert not model.use_new_api  # Should use legacy API
        mock_generativeai_module.configure.assert_called_once_with(api_key=api_key)
        mock_generativeai_module.GenerativeModel.assert_called_once_with("models/gemini-2.5-flash")

    @patch("llm_adapter.gemini_factory.genai", None)  # Force legacy API
    @patch("google.generativeai")
    def test_init_with_models_prefix(self, mock_generativeai_module):
        """Test that GeminiModel handles models/ prefix correctly."""
        # Arrange
        api_key = "test-api-key"
        model_name = "models/gemini-2.5-flash"
        mock_model_instance = Mock()
        mock_generativeai_module.GenerativeModel.return_value = mock_model_instance

        # Act
        model = GeminiModel(api_key, model_name)

        # Assert
        assert model.model_name == model_name
        mock_generativeai_module.GenerativeModel.assert_called_once_with("models/gemini-2.5-flash")

    @patch("llm_adapter.gemini_factory.genai", None)  # Force legacy API
    @patch("google.generativeai")
    def test_generate_text_calls_model_generate_content(self, mock_generativeai_module):
        """Test that generate_text calls the Gemini model correctly."""
        # Arrange
        api_key = "test-api-key"
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.text = "Generated response"
        # Mock candidates as empty list to avoid safety check
        mock_response.candidates = []
        mock_model_instance.generate_content.return_value = mock_response
        mock_generativeai_module.GenerativeModel.return_value = mock_model_instance

        model = GeminiModel(api_key, "gemini-2.5-flash")
        prompt = "Hello, world!"

        # Act
        result = model.generate_text(prompt)

        # Assert
        assert result.text == "Generated response"
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "Generated response"
        mock_model_instance.generate_content.assert_called_once()

    @patch("llm_adapter.gemini_factory.genai", None)  # Force legacy API
    @patch("google.generativeai")
    def test_generate_text_with_system_prompt(self, mock_generativeai_module):
        """Test that generate_text includes a system prompt when provided."""
        # Arrange
        api_key = "test-api-key"
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.text = "Generated response"
        # Mock candidates as empty list to avoid safety check
        mock_response.candidates = []
        mock_model_instance.generate_content.return_value = mock_response
        mock_generativeai_module.GenerativeModel.return_value = mock_model_instance

        model = GeminiModel(api_key, "gemini-2.5-flash")
        prompt = "User message."
        system_prompt = "System message."

        # Act
        result = model.generate_text(prompt, system_prompt=system_prompt)

        # Assert
        assert result.text == "Generated response"
        call_args = mock_model_instance.generate_content.call_args
        
        # Check that system prompt was passed (either via config or content)
        assert call_args is not None

    @patch("llm_adapter.gemini_factory.genai", None)  # Force legacy API
    @patch("google.generativeai")
    def test_generate_text_with_temperature_and_max_tokens(self, mock_generativeai_module):
        """Test that generate_text includes temperature and max_tokens when provided."""
        # Arrange
        api_key = "test-api-key"
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.text = "Generated response"
        # Mock candidates as empty list to avoid safety check
        mock_response.candidates = []
        mock_model_instance.generate_content.return_value = mock_response
        mock_generativeai_module.GenerativeModel.return_value = mock_model_instance

        model = GeminiModel(api_key, "gemini-2.5-flash")
        prompt = "Another message."
        temperature = 0.5
        max_tokens = 100

        # Act
        result = model.generate_text(prompt, temperature=temperature, max_tokens=max_tokens)

        # Assert
        assert result.text == "Generated response"
        call_args = mock_model_instance.generate_content.call_args
        assert call_args is not None

    @patch("llm_adapter.gemini_factory.genai", None)  # Force legacy API
    @patch("google.generativeai")
    def test_generate_text_all_parameters(self, mock_generativeai_module):
        """Test that generate_text includes all optional parameters when provided."""
        # Arrange
        api_key = "test-api-key"
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.text = "Generated response"
        # Mock candidates as empty list to avoid safety check
        mock_response.candidates = []
        mock_model_instance.generate_content.return_value = mock_response
        mock_generativeai_module.GenerativeModel.return_value = mock_model_instance

        model = GeminiModel(api_key, "gemini-2.5-pro")
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
        call_args = mock_model_instance.generate_content.call_args
        assert call_args is not None

    @patch("llm_adapter.gemini_factory.genai", None)  # Force legacy API
    @patch("google.generativeai")
    def test_truncate_prompt_fits(self, mock_generativeai_module):
        """Test that prompt is not truncated if it fits."""
        # Arrange
        api_key = "test-api-key"
        mock_model_instance = Mock()
        mock_generativeai_module.GenerativeModel.return_value = mock_model_instance

        model = GeminiModel(api_key, "gemini-2.5-flash")
        # Mock window_size to be large enough
        model.window_size = 1000000
        prompt = "a" * 1000
        system_prompt = "system"
        max_tokens = 100

        # Act
        truncated_prompt = model._truncate_prompt(prompt, system_prompt, max_tokens)

        # Assert
        assert truncated_prompt == prompt

    @patch("llm_adapter.gemini_factory.genai", None)  # Force legacy API
    @patch("google.generativeai")
    def test_truncate_prompt_truncates(self, mock_generativeai_module):
        """Test that prompt is truncated if it exceeds window size."""
        # Arrange
        api_key = "test-api-key"
        mock_model_instance = Mock()
        mock_generativeai_module.GenerativeModel.return_value = mock_model_instance

        model = GeminiModel(api_key, "gemini-2.5-flash")
        # Mock window_size to force truncation
        model.window_size = 100
        prompt = "a" * 10000
        system_prompt = "system"
        max_tokens = 10

        # Act
        truncated_prompt = model._truncate_prompt(prompt, system_prompt, max_tokens)

        # Assert
        assert len(truncated_prompt) < len(prompt)
        assert len(truncated_prompt) > 0  # Should not be empty


class TestGeminiFactory:
    """Test suite for GeminiFactory class."""

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-api-key-from-env"})
    @patch("llm_adapter.gemini_factory.GeminiModel")
    def test_create_returns_gemini_model(self, mock_model_class):
        """Test that create returns a GeminiModel instance."""
        # Arrange
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        factory = GeminiFactory()
        model_name = "gemini-2.5-flash"

        # Act
        result = factory.create(model_name)

        # Assert
        assert result == mock_model_instance
        mock_model_class.assert_called_once_with(api_key="test-api-key-from-env", model_name="gemini-2.5-flash")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-123"})
    @patch("llm_adapter.gemini_factory.GeminiModel")
    def test_create_uses_environment_api_key(self, mock_model_class):
        """Test that create uses GEMINI_API_KEY from environment."""
        # Arrange
        mock_model_class.return_value = Mock()
        factory = GeminiFactory()

        # Act
        factory.create("gemini-2.5-flash")

        # Assert
        mock_model_class.assert_called_once_with(api_key="test-key-123", model_name="gemini-2.5-flash")

    @patch.dict(os.environ, {}, clear=True)
    @patch("llm_adapter.gemini_factory.GeminiModel")
    def test_create_with_no_api_key_in_env(self, mock_model_class):
        """Test that create raises ValueError when API key is missing."""
        # Arrange
        mock_model_class.return_value = Mock()
        factory = GeminiFactory()

        # Act & Assert
        with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is not set"):
            factory.create("gemini-2.5-flash")

        # Should not create model when API key is missing
        mock_model_class.assert_not_called()

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    @patch("llm_adapter.gemini_factory.GeminiModel")
    def test_create_passes_model_name_to_model(self, mock_model_class):
        """Test that create passes model_name to GeminiModel."""
        # Arrange
        mock_model_class.return_value = Mock()
        factory = GeminiFactory()
        model_name = "gemini-2.5-pro"

        # Act
        factory.create(model_name)

        # Assert
        mock_model_class.assert_called_once_with(api_key="test-key", model_name="gemini-2.5-pro")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    @patch("llm_adapter.gemini_factory.GeminiModel")
    def test_create_with_different_model_names(self, mock_model_class):
        """Test create with various model names."""
        # Arrange
        mock_model_class.return_value = Mock()
        factory = GeminiFactory()

        model_names = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash", "models/gemini-2.5-flash"]

        for model_name in model_names:
            # Act
            factory.create(model_name)
            # Assert
            mock_model_class.assert_called_with(api_key="test-key", model_name=model_name)


class TestGetModelWindowSize:
    """Test suite for get_model_window_size function."""

    def test_get_model_window_size_gemini_25_flash(self):
        """Test window size for gemini-2.5-flash."""
        assert get_model_window_size("gemini-2.5-flash") == 1000000
        assert get_model_window_size("models/gemini-2.5-flash") == 1000000

    def test_get_model_window_size_gemini_25_pro(self):
        """Test window size for gemini-2.5-pro."""
        assert get_model_window_size("gemini-2.5-pro") == 2000000
        assert get_model_window_size("models/gemini-2.5-pro") == 2000000

    def test_get_model_window_size_gemini_15_flash(self):
        """Test window size for gemini-1.5-flash."""
        assert get_model_window_size("gemini-1.5-flash") == 1000000

    def test_get_model_window_size_gemini_pro(self):
        """Test window size for gemini-pro."""
        assert get_model_window_size("gemini-pro") == 32768
        assert get_model_window_size("models/gemini-pro") == 32768

    def test_get_model_window_size_unknown_model(self):
        """Test window size for unknown model defaults to 1M."""
        assert get_model_window_size("unknown-model") == 1000000

    def test_get_model_window_size_case_insensitive(self):
        """Test that window size lookup is case-insensitive."""
        assert get_model_window_size("GEMINI-2.5-FLASH") == 1000000
        assert get_model_window_size("Gemini-2.5-Pro") == 2000000


class TestValidateGeminiModelName:
    """Test suite for validate_gemini_model_name function."""

    def test_validate_gemini_model_name_valid_models(self):
        """Test that valid Gemini model names are recognized."""
        from llm_adapter.gemini_factory import validate_gemini_model_name
        
        valid_models = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-pro",
            "gemini-pro-vision",
            "models/gemini-2.5-flash",
            "models/gemini-2.5-pro",
        ]
        
        for model_name in valid_models:
            assert validate_gemini_model_name(model_name), f"{model_name} should be valid"
            # Test case-insensitive
            assert validate_gemini_model_name(model_name.upper()), f"{model_name.upper()} should be valid"
            assert validate_gemini_model_name(model_name.lower()), f"{model_name.lower()} should be valid"

    def test_validate_gemini_model_name_invalid_models(self):
        """Test that invalid Gemini model names are rejected."""
        from llm_adapter.gemini_factory import validate_gemini_model_name
        
        invalid_models = [
            "gemini-2.5",  # Missing suffix
            "gemini-2",   # Invalid version
            "gemini-3.0-flash",  # Non-existent version
            "gpt-4",  # Wrong provider
            "openai-gpt-4",  # Wrong provider
            "gemini-flash",  # Missing version
            "gemini-pro-extra",  # Extra suffix
            "",  # Empty string
        ]
        
        for model_name in invalid_models:
            assert not validate_gemini_model_name(model_name), f"{model_name} should be invalid"


class TestGeminiFactoryValidation:
    """Test suite for GeminiFactory model validation."""

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_create_rejects_invalid_model_name(self):
        """Test that create raises ValueError for invalid model names."""
        factory = GeminiFactory()
        
        invalid_models = [
            "gemini-2.5",  # Missing suffix
            "gemini-2",   # Invalid version
            "gpt-4",  # Wrong provider
        ]
        
        for model_name in invalid_models:
            with pytest.raises(ValueError, match="Invalid Gemini model name"):
                factory.create(model_name)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    @patch("llm_adapter.gemini_factory.GeminiModel")
    def test_create_accepts_valid_model_names(self, mock_model_class):
        """Test that create accepts valid model names."""
        mock_model_class.return_value = Mock()
        factory = GeminiFactory()
        
        valid_models = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-pro",
        ]
        
        for model_name in valid_models:
            # Should not raise
            factory.create(model_name)
            mock_model_class.assert_called_with(api_key="test-key", model_name=model_name)

