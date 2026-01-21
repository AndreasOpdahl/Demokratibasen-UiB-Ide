"""
Tests for GPTFactory and GPTModel classes.
"""
import os
import pytest
from unittest.mock import Mock, MagicMock, patch, call

from llm_adapter.gpt_factory import GPTFactory, GPTModel


class TestGPTModel:
    """Test suite for GPTModel class."""

    @patch("llm_adapter.gpt_factory.openai.OpenAI")
    def test_init_creates_openai_client(self, mock_openai_class):
        """Test that GPTModel initializes with OpenAI client."""
        # Arrange
        api_key = "test-api-key"
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Act
        model = GPTModel(api_key, "gpt-4")

        # Assert
        assert model.api_key == api_key
        assert model.model_name == "gpt-4"
        assert model.client == mock_client
        mock_openai_class.assert_called_once_with(api_key=api_key)

    @patch("llm_adapter.gpt_factory.openai.OpenAI")
    def test_init_stores_api_key(self, mock_openai_class):
        """Test that API key is stored correctly."""
        # Arrange
        api_key = "sk-test123456"
        mock_openai_class.return_value = Mock()

        # Act
        model = GPTModel(api_key, "gpt-4")

        # Assert
        assert model.api_key == api_key
        assert model.model_name == "gpt-4"

    @patch("llm_adapter.gpt_factory.openai.OpenAI")
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
        model = GPTModel(api_key, "gpt-4")
        prompt = "Test prompt"

        # Act
        result = model.generate_text(prompt)

        # Assert
        assert result == "Generated response"
        mock_create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

    @patch("llm_adapter.gpt_factory.openai.OpenAI")
    def test_generate_text_with_different_prompts(self, mock_openai_class):
        """Test generate_text with various prompts."""
        # Arrange
        api_key = "test-api-key"
        mock_client = Mock()
        mock_chat = Mock()
        mock_completions = Mock()
        mock_create = Mock()
        
        mock_client.chat = mock_chat
        mock_chat.completions = mock_completions
        mock_completions.create = mock_create
        
        mock_openai_class.return_value = mock_client
        model = GPTModel(api_key, "gpt-3.5-turbo")

        test_prompts = [
            "Simple prompt",
            "Multi-line\nprompt",
            "Prompt with special chars: !@#$%",
        ]

        for prompt in test_prompts:
            mock_create.return_value = f"Response to: {prompt}"
            # Act
            result = model.generate_text(prompt)
            # Assert
            assert result == f"Response to: {prompt}"
            mock_create.assert_called_with(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )

    @patch("llm_adapter.gpt_factory.openai.OpenAI")
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
        model = GPTModel(api_key, "gpt-4")
        prompt = "Test prompt"
        system_prompt = "You are a helpful assistant."

        # Act
        result = model.generate_text(prompt, system_prompt=system_prompt)

        # Assert
        assert result == "Generated response"
        mock_create.assert_called_once_with(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

    @patch("llm_adapter.gpt_factory.openai.OpenAI")
    def test_generate_text_without_system_prompt(self, mock_openai_class):
        """Test that generate_text works without system prompt."""
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
        model = GPTModel(api_key, "gpt-4")
        prompt = "Test prompt"

        # Act
        result = model.generate_text(prompt)

        # Assert
        assert result == "Generated response"
        mock_create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

    @patch("llm_adapter.gpt_factory.openai.OpenAI")
    def test_generate_text_with_empty_system_prompt(self, mock_openai_class):
        """Test that generate_text handles empty system prompt correctly."""
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
        model = GPTModel(api_key, "gpt-4")
        prompt = "Test prompt"
        system_prompt = ""

        # Act
        result = model.generate_text(prompt, system_prompt=system_prompt)

        # Assert
        assert result == "Generated response"
        # Empty string is falsy, so it should not be included
        mock_create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

    @patch("llm_adapter.gpt_factory.openai.OpenAI")
    def test_generate_text_with_temperature(self, mock_openai_class):
        """Test that generate_text includes temperature when provided."""
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
        model = GPTModel(api_key, "gpt-4")
        prompt = "Test prompt"
        temperature = 0.1

        # Act
        result = model.generate_text(prompt, temperature=temperature)

        # Assert
        assert result == "Generated response"
        mock_create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )

    @patch("llm_adapter.gpt_factory.openai.OpenAI")
    def test_generate_text_with_max_tokens(self, mock_openai_class):
        """Test that generate_text includes max_tokens when provided."""
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
        model = GPTModel(api_key, "gpt-4")
        prompt = "Test prompt"
        max_tokens = 4096

        # Act
        result = model.generate_text(prompt, max_tokens=max_tokens)

        # Assert
        assert result == "Generated response"
        mock_create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )

    @patch("llm_adapter.gpt_factory.openai.OpenAI")
    def test_generate_text_with_all_parameters(self, mock_openai_class):
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
        model = GPTModel(api_key, "gpt-4")
        prompt = "Test prompt"
        system_prompt = "You are a helpful assistant."
        temperature = 0.1
        max_tokens = 4096

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
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

    @patch("llm_adapter.gpt_factory.openai.OpenAI")
    def test_generate_text_with_json_schema_structured_outputs(self, mock_openai_class):
        """Test that generate_text includes structured outputs schema for newer models."""
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
        model = GPTModel(api_key, "gpt-4o")
        prompt = "Test prompt"
        json_schema = {
            "name": "extract_case_info",
            "description": "Test schema",
            "schema": {
                "type": "object",
                "properties": {
                    "hva_saken_gjelder": {"type": "string"}
                }
            }
        }

        # Act
        result = model.generate_text(prompt, json_schema=json_schema)

        # Assert
        assert result == "Generated response"
        call_args = mock_create.call_args
        assert call_args is not None
        kwargs = call_args.kwargs
        assert "response_format" in kwargs
        assert kwargs["response_format"]["type"] == "json_schema"
        assert "json_schema" in kwargs["response_format"]

    @patch("llm_adapter.gpt_factory.openai.OpenAI")
    def test_generate_text_with_json_schema_fallback(self, mock_openai_class):
        """Test that generate_text uses json_object fallback for older models."""
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
        model = GPTModel(api_key, "gpt-3.5-turbo")
        prompt = "Test prompt"
        json_schema = {"type": "object", "properties": {}}

        # Act
        result = model.generate_text(prompt, json_schema=json_schema)

        # Assert
        assert result == "Generated response"
        call_args = mock_create.call_args
        assert call_args is not None
        kwargs = call_args.kwargs
        assert "response_format" in kwargs
        assert kwargs["response_format"]["type"] == "json_object"


class TestGPTFactory:
    """Test suite for GPTFactory class."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key-from-env"})
    @patch("llm_adapter.gpt_factory.GPTModel")
    def test_create_returns_gpt_model(self, mock_model_class):
        """Test that create returns a GPTModel instance."""
        # Arrange
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        factory = GPTFactory()
        model_name = "gpt-4"

        # Act
        result = factory.create(model_name)

        # Assert
        assert result == mock_model_instance
        mock_model_class.assert_called_once_with(api_key="test-api-key-from-env", model_name="gpt-4")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456"})
    @patch("llm_adapter.gpt_factory.GPTModel")
    def test_create_uses_environment_api_key(self, mock_model_class):
        """Test that create uses OPENAI_API_KEY from environment."""
        # Arrange
        mock_model_class.return_value = Mock()
        factory = GPTFactory()

        # Act
        factory.create("gpt-4")

        # Assert
        mock_model_class.assert_called_once_with(api_key="sk-test123456", model_name="gpt-4")

    @patch.dict(os.environ, {}, clear=True)
    @patch("llm_adapter.gpt_factory.GPTModel")
    def test_create_with_no_api_key_in_env(self, mock_model_class):
        """Test that create raises ValueError when API key is missing."""
        # Arrange
        mock_model_class.return_value = Mock()
        factory = GPTFactory()

        # Act & Assert
        with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is not set"):
            factory.create("gpt-4")
        
        # Should not create model when API key is missing
        mock_model_class.assert_not_called()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("llm_adapter.gpt_factory.GPTModel")
    def test_create_passes_model_name_to_model(self, mock_model_class):
        """Test that create passes model_name to GPTModel."""
        # Arrange
        mock_model_class.return_value = Mock()
        factory = GPTFactory()
        model_name = "gpt-4"

        # Act
        factory.create(model_name)

        # Assert
        mock_model_class.assert_called_once_with(api_key="test-key", model_name="gpt-4")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("llm_adapter.gpt_factory.GPTModel")
    def test_create_with_different_model_names(self, mock_model_class):
        """Test create with various model names."""
        # Arrange
        mock_model_class.return_value = Mock()
        factory = GPTFactory()

        model_names = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "custom-model"]

        for model_name in model_names:
            # Act
            factory.create(model_name)
            # Assert
            # All should use the same API key from env, but different model names
            mock_model_class.assert_called_with(api_key="test-key", model_name=model_name)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "key1"})
    @patch("llm_adapter.gpt_factory.GPTModel")
    def test_create_returns_new_instance_each_time(self, mock_model_class):
        """Test that create returns a new model instance each time."""
        # Arrange
        mock_instance1 = Mock()
        mock_instance2 = Mock()
        mock_model_class.side_effect = [mock_instance1, mock_instance2]
        factory = GPTFactory()

        # Act
        result1 = factory.create("gpt-4")
        result2 = factory.create("gpt-4")

        # Assert
        assert result1 == mock_instance1
        assert result2 == mock_instance2
        assert result1 != result2
        assert mock_model_class.call_count == 2

