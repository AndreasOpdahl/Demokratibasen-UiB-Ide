"""
Tests for extract_structured_data.py validation of model families and model names.
"""
import pytest
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add parent directory to path to import extract_structured_data
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_adapter import get_factory
from llm_adapter.gemini_factory import GeminiFactory


class TestModelFamilyValidation:
    """Test suite for model family validation in extract_structured_data."""

    def test_get_factory_rejects_invalid_family(self):
        """Test that get_factory rejects invalid model families."""
        invalid_families = [
            "InvalidFactory",
            "NonExistent",
            "Unknown",
            "Anthropic",  # Not implemented (no anthropic_factory module)
        ]
        
        for family in invalid_families:
            with pytest.raises((ImportError, AttributeError)):
                get_factory(family)

    def test_get_factory_accepts_valid_families(self):
        """Test that get_factory accepts valid model families."""
        valid_families = [
            "GPT",
            "gpt",
            "GPT",
            "Gemini",
            "gemini",
            "GEMINI",
            "Claude",
            "claude",
            "CLAUDE",
            "Mistral",
            "mistral",
            "MISTRAL",
            "Qwen",
            "qwen",
            "QWEN",
        ]
        
        for family in valid_families:
            factory = get_factory(family)
            assert factory is not None


class TestGeminiModelValidation:
    """Test suite for Gemini model name validation."""

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    def test_gemini_factory_rejects_invalid_model_names(self):
        """Test that GeminiFactory rejects invalid model names."""
        factory = GeminiFactory()
        
        invalid_models = [
            "gemini-2.5",  # Missing suffix (the error from user's issue)
            "gemini-2",   # Invalid version
            "gemini-3.0-flash",  # Non-existent version
            "gpt-4",  # Wrong provider
            "gemini-flash",  # Missing version
        ]
        
        for model_name in invalid_models:
            with pytest.raises(ValueError, match="Invalid Gemini model name"):
                factory.create(model_name)

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    @patch("llm_adapter.gemini_factory.GeminiModel")
    def test_gemini_factory_accepts_valid_model_names(self, mock_model_class):
        """Test that GeminiFactory accepts valid model names."""
        mock_model_class.return_value = MagicMock()
        factory = GeminiFactory()
        
        valid_models = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-pro",
            "gemini-pro-vision",
        ]
        
        for model_name in valid_models:
            # Should not raise
            result = factory.create(model_name)
            assert result is not None


class TestEarlyValidation:
    """Test suite for early validation in extract_structured_data main function."""

    @patch("sys.exit")
    @patch("extract_structured_data.get_factory")
    @patch("extract_structured_data.DatasetLoader")
    def test_main_exits_on_invalid_model_family(self, mock_loader, mock_get_factory, mock_exit):
        """Test that main exits early when model family is invalid."""
        import extract_structured_data
        
        # Mock sys.exit to raise SystemExit so execution stops
        def mock_exit_side_effect(code):
            raise SystemExit(code)
        mock_exit.side_effect = mock_exit_side_effect
        
        # Mock get_factory to raise ImportError
        mock_get_factory.side_effect = ImportError("Could not import factory module")
        
        # Mock parse_arguments
        with patch("extract_structured_data.parse_arguments") as mock_parse:
            mock_parse.return_value = MagicMock(
                model_family="InvalidFactory",
                model="gpt-4",
                max_documents=None,
                dataset="dataset-202510"
            )
            
            # Call main - should raise SystemExit
            with pytest.raises(SystemExit) as exc_info:
                extract_structured_data.main()
            
            # Verify exit code is 1
            assert exc_info.value.code == 1
            
            # Verify sys.exit was called
            mock_exit.assert_called_once_with(1)
            
            # Verify DatasetLoader was not called (early exit)
            mock_loader.assert_not_called()

    @patch("sys.exit")
    @patch("extract_structured_data.get_factory")
    @patch("extract_structured_data.DatasetLoader")
    def test_main_exits_on_invalid_model_name(self, mock_loader, mock_get_factory, mock_exit):
        """Test that main exits early when model name is invalid."""
        import extract_structured_data
        
        # Mock sys.exit to raise SystemExit so execution stops
        def mock_exit_side_effect(code):
            raise SystemExit(code)
        mock_exit.side_effect = mock_exit_side_effect
        
        # Mock get_factory to return a factory
        mock_factory = MagicMock()
        mock_get_factory.return_value = mock_factory
        
        # Mock factory.create to raise ValueError for invalid model
        mock_factory.create.side_effect = ValueError("Invalid Gemini model name: 'gemini-2.5'")
        
        # Mock parse_arguments
        with patch("extract_structured_data.parse_arguments") as mock_parse:
            mock_parse.return_value = MagicMock(
                model_family="Gemini",
                model="gemini-2.5",  # Invalid - missing suffix
                max_documents=None,
                dataset="dataset-202510"
            )
            
            # Call main - should raise SystemExit
            with pytest.raises(SystemExit) as exc_info:
                extract_structured_data.main()
            
            # Verify exit code is 1
            assert exc_info.value.code == 1
            
            # Verify sys.exit was called
            mock_exit.assert_called_once_with(1)
            
            # Verify DatasetLoader was not called (early exit)
            mock_loader.assert_not_called()

    @patch("sys.exit")
    @patch("extract_structured_data.get_factory")
    @patch("extract_structured_data.DatasetLoader")
    def test_main_exits_on_invalid_model_name_validation(self, mock_loader, mock_get_factory, mock_exit):
        """Test that main exits early when model name validation fails (e.g., invalid Gemini model name)."""
        import extract_structured_data
        
        # Mock sys.exit to raise SystemExit so execution stops
        def mock_exit_side_effect(code):
            raise SystemExit(code)
        mock_exit.side_effect = mock_exit_side_effect
        
        # Mock get_factory to return a factory
        mock_factory = MagicMock()
        mock_get_factory.return_value = mock_factory
        
        # Mock factory.create to raise ValueError for invalid model name
        # This simulates validation error (e.g., "gemini-2.5" without suffix)
        mock_factory.create.side_effect = ValueError("Invalid Gemini model name: 'gemini-2.5'")
        
        # Mock parse_arguments
        with patch("extract_structured_data.parse_arguments") as mock_parse:
            mock_parse.return_value = MagicMock(
                model_family="Gemini",
                model="gemini-2.5",  # Invalid model - missing suffix
                max_documents=None,
                dataset="dataset-202510"
            )
            
            # Call main - should raise SystemExit
            with pytest.raises(SystemExit) as exc_info:
                extract_structured_data.main()
            
            # Verify exit code is 1
            assert exc_info.value.code == 1
            
            # Verify sys.exit was called
            mock_exit.assert_called_once_with(1)
            
            # Verify DatasetLoader was not called (early exit)
            mock_loader.assert_not_called()

