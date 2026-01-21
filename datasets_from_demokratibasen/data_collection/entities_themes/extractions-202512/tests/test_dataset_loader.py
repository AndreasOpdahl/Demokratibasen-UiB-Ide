"""
Tests for DatasetLoader class and kommunenavn function.
"""
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from dataset_loader import (
    DatasetLoader, 
    kommunenavn, 
    KOMMUNENAVN, 
    DatasetAdapter202505,
    register_dataset,
    set_root_path,
)


class TestKommunenavn:
    """Test suite for kommunenavn function."""

    def test_known_kommune_bergen(self):
        """Test that known kommune ID returns correct name."""
        assert kommunenavn(4601) == "Bergen"

    def test_known_kommune_tromso(self):
        """Test that known kommune ID returns correct name."""
        assert kommunenavn(5501) == "Tromsø"

    def test_known_kommune_lyngen(self):
        """Test that known kommune ID returns correct name."""
        assert kommunenavn(5536) == "Lyngen"

    def test_unknown_kommune(self):
        """Test that unknown kommune ID returns default string."""
        assert kommunenavn(9999) == "en norsk kommune"

    def test_kommune_as_string(self):
        """Test that string kommune ID is converted and looked up."""
        assert kommunenavn("4601") == "Bergen"
        assert kommunenavn("5501") == "Tromsø"

    def test_kommune_as_float(self):
        """Test that float kommune ID is converted and looked up."""
        assert kommunenavn(4601.0) == "Bergen"

    def test_invalid_kommune_type(self):
        """Test that invalid types return default string."""
        assert kommunenavn("invalid") == "en norsk kommune"
        assert kommunenavn(None) == "en norsk kommune"
        assert kommunenavn([]) == "en norsk kommune"

    def test_kommune_zero(self):
        """Test that zero returns default string."""
        assert kommunenavn(0) == "en norsk kommune"

    def test_kommune_negative(self):
        """Test that negative number returns default string."""
        assert kommunenavn(-1) == "en norsk kommune"


def _create_test_dataset(tmp_path, data, dataset_name="test-dataset"):
    """Helper function to create and register a test dataset."""
    file_path = tmp_path / f"{dataset_name}.jsonl"
    with open(file_path, 'w', encoding='utf-8') as f:
        for doc in data:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    register_dataset(
        dataset_name,
        str(file_path.name),
        DatasetAdapter202505
    )
    set_root_path(tmp_path)
    return dataset_name, file_path


class TestDatasetLoader:
    """Test suite for DatasetLoader class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset data."""
        return [
            {
                "dok_id": "doc1",
                "kommune": 4601,
                "tekst": "This is a test document from Bergen."
            },
            {
                "dok_id": "doc2",
                "kommune": 5501,
                "tekst": "This is a test document from Tromsø."
            },
            {
                "dok_id": "doc3",
                "kommune": 5536,
                "tekst": "This is a test document from Lyngen."
            }
        ]

    @pytest.fixture
    def test_dataset_name(self, sample_data, tmp_path):
        """Register a test dataset and return its name."""
        dataset_name, _ = _create_test_dataset(tmp_path, sample_data, "test-dataset-temp")
        yield dataset_name
        # Cleanup: remove test dataset from registry
        from dataset_loader import DATASET_REGISTRY
        if dataset_name in DATASET_REGISTRY:
            del DATASET_REGISTRY[dataset_name]

    def test_init_with_existing_dataset(self, test_dataset_name):
        """Test that DatasetLoader initializes with existing dataset."""
        loader = DatasetLoader(test_dataset_name)
        assert loader.dataset_name == test_dataset_name
        assert loader.file_path.exists()

    def test_init_with_nonexistent_dataset(self):
        """Test that DatasetLoader raises ValueError for unknown dataset."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            DatasetLoader("nonexistent-dataset")

    def test_load_dataset_basic(self, test_dataset_name):
        """Test loading a basic dataset."""
        loader = DatasetLoader(test_dataset_name)
        
        results = list(loader())
        assert len(results) == 3
        assert results[0] == ("doc1", 4601, "Bergen", "This is a test document from Bergen.")
        assert results[1] == ("doc2", 5501, "Tromsø", "This is a test document from Tromsø.")
        assert results[2] == ("doc3", 5536, "Lyngen", "This is a test document from Lyngen.")

    def test_load_dataset_with_missing_dok_id(self, tmp_path):
        """Test loading dataset with missing dok_id field."""
        data = [
            {
                "kommune": 4601,
                "tekst": "Document without ID"
            }
        ]
        file_path = tmp_path / "test.jsonl"
        with open(file_path, 'w', encoding='utf-8') as f:
            for doc in data:
                f.write(json.dumps(doc) + '\n')
        
        dataset_name = "test-missing-id"
        register_dataset(dataset_name, str(file_path.name), DatasetAdapter202505)
        set_root_path(tmp_path)
        try:
            loader = DatasetLoader(dataset_name)
            results = list(loader())
            assert len(results) == 1
            assert results[0][0] == ""  # Empty dok_id
            assert results[0][1] == 4601  # kommune_nummer
            assert results[0][2] == "Bergen"  # kommune_navn
            assert results[0][3] == "Document without ID"
        finally:
            from dataset_loader import DATASET_REGISTRY
            if dataset_name in DATASET_REGISTRY:
                del DATASET_REGISTRY[dataset_name]

    def test_load_dataset_with_missing_kommune(self, tmp_path):
        """Test loading dataset with missing kommune field."""
        data = [
            {
                "dok_id": "doc1",
                "tekst": "Document without kommune"
            }
        ]
        dataset_name, _ = _create_test_dataset(tmp_path, data, "test-missing-kommune")
        try:
            loader = DatasetLoader(dataset_name)
            results = list(loader())
            assert len(results) == 1
            assert results[0][0] == "doc1"
            assert results[0][1] is None  # kommune_nummer is None when missing
            assert results[0][2] == "en norsk kommune"  # kommune_navn default
            assert results[0][3] == "Document without kommune"
        finally:
            from dataset_loader import DATASET_REGISTRY
            if dataset_name in DATASET_REGISTRY:
                del DATASET_REGISTRY[dataset_name]

    def test_load_dataset_prefers_tekst_over_text(self, tmp_path):
        """Test that 'tekst' field is preferred over 'text' field."""
        data = [
            {
                "dok_id": "doc1",
                "kommune": 4601,
                "tekst": "Norwegian tekst",
                "text": "English text"
            }
        ]
        dataset_name, _ = _create_test_dataset(tmp_path, data, "test-prefers-tekst")
        try:
            loader = DatasetLoader(dataset_name)
            results = list(loader())
            assert results[0][3] == "Norwegian tekst"
        finally:
            from dataset_loader import DATASET_REGISTRY
            if dataset_name in DATASET_REGISTRY:
                del DATASET_REGISTRY[dataset_name]

    def test_load_dataset_uses_text_when_tekst_missing(self, tmp_path):
        """Test that 'text' field is used when 'tekst' is missing (backward compatibility)."""
        data = [
            {
                "dok_id": "doc1",
                "kommune": 4601,
                "text": "English text"
            }
        ]
        dataset_name, _ = _create_test_dataset(tmp_path, data, "test-uses-text")
        try:
            loader = DatasetLoader(dataset_name)
            results = list(loader())
            assert results[0][3] == "English text"
        finally:
            from dataset_loader import DATASET_REGISTRY
            if dataset_name in DATASET_REGISTRY:
                del DATASET_REGISTRY[dataset_name]

    def test_load_dataset_skips_empty_text(self, tmp_path, capsys):
        """Test that documents with empty text are skipped with warning."""
        data = [
            {
                "dok_id": "doc1",
                "kommune": 4601,
                "tekst": "Valid text"
            },
            {
                "dok_id": "doc2",
                "kommune": 4601,
                "tekst": ""
            },
            {
                "dok_id": "doc3",
                "kommune": 4601,
                "tekst": "   "  # Only whitespace
            },
            {
                "dok_id": "doc4",
                "kommune": 4601,
                "tekst": "Another valid text"
            }
        ]
        dataset_name, _ = _create_test_dataset(tmp_path, data, "test-empty-text")
        try:
            loader = DatasetLoader(dataset_name)
            results = list(loader())
            assert len(results) == 2
            assert results[0][0] == "doc1"
            assert results[1][0] == "doc4"
            
            # Should have printed warnings for filtered documents
            captured = capsys.readouterr()
            assert "Warning: Skipping document doc2: empty or whitespace-only text" in captured.err
            assert "Warning: Skipping document doc3: empty or whitespace-only text" in captured.err
        finally:
            from dataset_loader import DATASET_REGISTRY
            if dataset_name in DATASET_REGISTRY:
                del DATASET_REGISTRY[dataset_name]

    def test_load_dataset_skips_missing_text_and_tekst(self, tmp_path):
        """Test that documents with neither text nor tekst are skipped."""
        data = [
            {
                "dok_id": "doc1",
                "kommune": 4601,
                "tekst": "Valid text"
            },
            {
                "dok_id": "doc2",
                "kommune": 4601
            }
        ]
        dataset_name, _ = _create_test_dataset(tmp_path, data, "test-missing-text")
        try:
            loader = DatasetLoader(dataset_name)
            results = list(loader())
            assert len(results) == 1
            assert results[0][0] == "doc1"
        finally:
            from dataset_loader import DATASET_REGISTRY
            if dataset_name in DATASET_REGISTRY:
                del DATASET_REGISTRY[dataset_name]

    def test_load_dataset_with_unknown_kommune(self, tmp_path):
        """Test loading dataset with unknown kommune ID."""
        data = [
            {
                "dok_id": "doc1",
                "kommune": 9999,
                "tekst": "Document from unknown kommune"
            }
        ]
        dataset_name, _ = _create_test_dataset(tmp_path, data, "test-unknown-kommune")
        try:
            loader = DatasetLoader(dataset_name)
            results = list(loader())
            assert results[0][1] == 9999  # kommune_nummer
            assert results[0][2] == "en norsk kommune"  # kommune_navn default
        finally:
            from dataset_loader import DATASET_REGISTRY
            if dataset_name in DATASET_REGISTRY:
                del DATASET_REGISTRY[dataset_name]

    def test_load_dataset_with_string_kommune(self, tmp_path):
        """Test loading dataset with kommune_nummer as string."""
        data = [
            {
                "dok_id": "doc1",
                "kommune": "4601",  # kommune_nummer as string in input
                "tekst": "Document with string kommune"
            }
        ]
        dataset_name, _ = _create_test_dataset(tmp_path, data, "test-string-kommune")
        try:
            loader = DatasetLoader(dataset_name)
            results = list(loader())
            # results structure: (doc_id, kommune_nummer, kommune_navn, text)
            assert results[0][1] == 4601  # kommune_nummer should be converted to int
            assert results[0][2] == "Bergen"  # kommune_navn should be translated
        finally:
            from dataset_loader import DATASET_REGISTRY
            if dataset_name in DATASET_REGISTRY:
                del DATASET_REGISTRY[dataset_name]

    def test_call_iterates_through_dataset(self, test_dataset_name):
        """Test that __call__ iterates through all documents."""
        loader = DatasetLoader(test_dataset_name)
        
        results = list(loader())
        
        assert len(results) == 3
        assert results[0] == ("doc1", 4601, "Bergen", "This is a test document from Bergen.")
        assert results[1] == ("doc2", 5501, "Tromsø", "This is a test document from Tromsø.")
        assert results[2] == ("doc3", 5536, "Lyngen", "This is a test document from Lyngen.")

    def test_call_raises_stopiteration(self, test_dataset_name):
        """Test that __call__ raises StopIteration when exhausted."""
        loader = DatasetLoader(test_dataset_name)
        
        # Create generator and consume all items
        gen = loader()
        list(gen)
        
        # Next call on exhausted generator should raise StopIteration
        with pytest.raises(StopIteration):
            next(gen)

    def test_call_with_empty_dataset(self, tmp_path):
        """Test that __call__ works with empty dataset."""
        data = []
        dataset_name, _ = _create_test_dataset(tmp_path, data, "test-empty")
        try:
            loader = DatasetLoader(dataset_name)
            results = list(loader())
            assert len(results) == 0
        finally:
            from dataset_loader import DATASET_REGISTRY
            if dataset_name in DATASET_REGISTRY:
                del DATASET_REGISTRY[dataset_name]

    def test_load_dataset_utf8_encoding(self, tmp_path):
        """Test that dataset handles UTF-8 characters correctly."""
        data = [
            {
                "dok_id": "doc1",
                "kommune": 4601,
                "tekst": "Test with special chars: æøå ÆØÅ"
            }
        ]
        dataset_name, _ = _create_test_dataset(tmp_path, data, "test-utf8")
        try:
            loader = DatasetLoader(dataset_name)
            results = list(loader())
            assert results[0][3] == "Test with special chars: æøå ÆØÅ"
        finally:
            from dataset_loader import DATASET_REGISTRY
            if dataset_name in DATASET_REGISTRY:
                del DATASET_REGISTRY[dataset_name]

    def test_dataset_loaded_on_init(self, test_dataset_name):
        """Test that dataset is automatically loaded when DatasetLoader is initialized."""
        loader = DatasetLoader(test_dataset_name)
        
        # Dataset should be loaded automatically and accessible via iterator
        results = list(loader())
        assert len(results) == 3
        assert results[0] == ("doc1", 4601, "Bergen", "This is a test document from Bergen.")
        assert results[1] == ("doc2", 5501, "Tromsø", "This is a test document from Tromsø.")
        assert results[2] == ("doc3", 5536, "Lyngen", "This is a test document from Lyngen.")

    def test_empty_jsonl_file(self, tmp_path):
        """Test loading an empty JSONL file."""
        file_path = tmp_path / "test.jsonl"
        with open(file_path, 'w', encoding='utf-8') as f:
            # Empty JSONL file (no lines)
            pass
        
        dataset_name = "test-empty-file"
        register_dataset(dataset_name, str(file_path.name), DatasetAdapter202505)
        set_root_path(tmp_path)
        try:
            loader = DatasetLoader(dataset_name)
            results = list(loader())
            assert len(results) == 0
        finally:
            from dataset_loader import DATASET_REGISTRY
            if dataset_name in DATASET_REGISTRY:
                del DATASET_REGISTRY[dataset_name]

    def test_load_dataset_handles_invalid_json_lines(self, tmp_path, capsys):
        """Test that invalid JSON lines are skipped with a warning."""
        file_path = tmp_path / "test.jsonl"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('{"dok_id": "doc1", "kommune": 4601, "tekst": "Valid document"}\n')
            f.write('invalid json line\n')
            f.write('{"dok_id": "doc2", "kommune": 5501, "tekst": "Another valid document"}\n')
            f.write('{"incomplete": json\n')
        
        dataset_name = "test-invalid-json"
        register_dataset(dataset_name, str(file_path.name), DatasetAdapter202505)
        set_root_path(tmp_path)
        try:
            loader = DatasetLoader(dataset_name)
            # Should load the 2 valid documents
            results = list(loader())
            assert len(results) == 2
            assert results[0][0] == "doc1"
            assert results[1][0] == "doc2"
            
            # Should have printed warnings for invalid lines
            captured = capsys.readouterr()
            assert "Warning: Skipping invalid JSON line" in captured.err
        finally:
            from dataset_loader import DATASET_REGISTRY
            if dataset_name in DATASET_REGISTRY:
                del DATASET_REGISTRY[dataset_name]

    def test_load_dataset_handles_empty_lines(self, tmp_path):
        """Test that empty lines in JSONL file are skipped."""
        file_path = tmp_path / "test.jsonl"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('{"dok_id": "doc1", "kommune": 4601, "tekst": "First document"}\n')
            f.write('\n')
            f.write('  \n')  # Line with only whitespace
            f.write('{"dok_id": "doc2", "kommune": 5501, "tekst": "Second document"}\n')
        
        dataset_name = "test-empty-lines"
        register_dataset(dataset_name, str(file_path.name), DatasetAdapter202505)
        set_root_path(tmp_path)
        try:
            loader = DatasetLoader(dataset_name)
            # Should load the 2 valid documents, skipping empty lines
            results = list(loader())
            assert len(results) == 2
            assert results[0][0] == "doc1"
            assert results[1][0] == "doc2"
        finally:
            from dataset_loader import DATASET_REGISTRY
            if dataset_name in DATASET_REGISTRY:
                del DATASET_REGISTRY[dataset_name]

    def test_load_dataset_filters_short_text(self, tmp_path, capsys):
        """Test that documents with text shorter than 10 characters are filtered out with warning."""
        file_path = tmp_path / "test.jsonl"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('{"dok_id": "doc1", "kommune": 4601, "tekst": "Valid long text document"}\n')
            f.write('{"dok_id": "doc2", "kommune": 4601, "tekst": "Short"}\n')  # 5 chars
            f.write('{"dok_id": "doc3", "kommune": 4601, "tekst": "123456789"}\n')  # 9 chars
            f.write('{"dok_id": "doc4", "kommune": 4601, "tekst": "Another valid document"}\n')
        
        dataset_name = "test-short-text"
        register_dataset(dataset_name, str(file_path.name), DatasetAdapter202505)
        set_root_path(tmp_path)
        try:
            loader = DatasetLoader(dataset_name)
            # Should only load documents with text >= 10 characters
            results = list(loader())
            assert len(results) == 2
            assert results[0][0] == "doc1"
            assert results[1][0] == "doc4"
            
            # Should have printed warnings for filtered documents
            captured = capsys.readouterr()
            assert "Warning: Skipping document doc2: text too short" in captured.err
            assert "Warning: Skipping document doc3: text too short" in captured.err
        finally:
            from dataset_loader import DATASET_REGISTRY
            if dataset_name in DATASET_REGISTRY:
                del DATASET_REGISTRY[dataset_name]

    def test_load_dataset_filters_no_alphanumeric(self, tmp_path, capsys):
        """Test that documents with no alphanumeric content are filtered out with warning."""
        file_path = tmp_path / "test.jsonl"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('{"dok_id": "doc1", "kommune": 4601, "tekst": "Valid document with text"}\n')
            f.write('{"dok_id": "doc2", "kommune": 4601, "tekst": "!@#$%^&*()!@"}\n')  # No alphanumeric, >= 10 chars
            f.write('{"dok_id": "doc3", "kommune": 4601, "tekst": "   -   -   "}\n')  # No alphanumeric, >= 10 chars
            f.write('{"dok_id": "doc4", "kommune": 4601, "tekst": "Another valid document"}\n')
        
        dataset_name = "test-no-alphanumeric"
        register_dataset(dataset_name, str(file_path.name), DatasetAdapter202505)
        set_root_path(tmp_path)
        try:
            loader = DatasetLoader(dataset_name)
            # Should only load documents with alphanumeric content
            results = list(loader())
            assert len(results) == 2
            assert results[0][0] == "doc1"
            assert results[1][0] == "doc4"
            
            # Should have printed warnings for filtered documents
            captured = capsys.readouterr()
            assert "Warning: Skipping document doc2: text contains no alphanumeric characters" in captured.err
            assert "Warning: Skipping document doc3: text contains no alphanumeric characters" in captured.err
        finally:
            from dataset_loader import DATASET_REGISTRY
            if dataset_name in DATASET_REGISTRY:
                del DATASET_REGISTRY[dataset_name]

    def test_load_dataset_backward_compatibility_with_dokument_id(self, tmp_path):
        """Test that loader still works with old field name 'dokument_id' for backward compatibility."""
        data = [
            {
                "dokument_id": "doc1",  # Old field name
                "kommune": 4601,
                "tekst": "Document with old field name"
            }
        ]
        dataset_name, _ = _create_test_dataset(tmp_path, data, "test-backward-compat")
        try:
            loader = DatasetLoader(dataset_name)
            results = list(loader())
            assert len(results) == 1
            assert results[0][0] == "doc1"  # Should still work with old field name
            assert results[0][1] == 4601  # kommune_nummer
            assert results[0][2] == "Bergen"  # kommune_navn
            assert results[0][3] == "Document with old field name"
        finally:
            from dataset_loader import DATASET_REGISTRY
            if dataset_name in DATASET_REGISTRY:
                del DATASET_REGISTRY[dataset_name]


class TestDatasetAdapter202505:
    """Test suite for DatasetAdapter202505 class."""
    
    def test_get_dok_id_with_new_field_name(self):
        """Test that dok_id is extracted from new field name."""
        doc = {"dok_id": "test123"}
        assert DatasetAdapter202505.get_dok_id(doc) == "test123"
    
    def test_get_dok_id_with_old_field_name(self):
        """Test that dok_id is extracted from old field name (backward compatibility)."""
        doc = {"dokument_id": "test456"}
        assert DatasetAdapter202505.get_dok_id(doc) == "test456"
    
    def test_get_dok_id_prefers_new_over_old(self):
        """Test that new field name is preferred over old field name."""
        doc = {"dok_id": "new", "dokument_id": "old"}
        assert DatasetAdapter202505.get_dok_id(doc) == "new"
    
    def test_get_dok_id_missing_returns_empty_string(self):
        """Test that missing dok_id returns empty string."""
        doc = {}
        assert DatasetAdapter202505.get_dok_id(doc) == ""
    
    def test_get_tekst_with_new_field_name(self):
        """Test that tekst is extracted from new field name."""
        doc = {"tekst": "Norwegian text"}
        assert DatasetAdapter202505.get_tekst(doc) == "Norwegian text"
    
    def test_get_tekst_with_old_field_name(self):
        """Test that tekst is extracted from old field name (backward compatibility)."""
        doc = {"text": "English text"}
        assert DatasetAdapter202505.get_tekst(doc) == "English text"
    
    def test_get_tekst_prefers_tekst_over_text(self):
        """Test that 'tekst' is preferred over 'text'."""
        doc = {"tekst": "Norwegian", "text": "English"}
        assert DatasetAdapter202505.get_tekst(doc) == "Norwegian"
    
    def test_get_kommune_nummer_extracts_number(self):
        """Test that kommune number is extracted correctly."""
        doc = {"kommune": 4601}
        assert DatasetAdapter202505.get_kommune_nummer(doc) == 4601
    
    def test_get_kommune_nummer_missing_returns_none(self):
        """Test that missing kommune returns None."""
        doc = {}
        assert DatasetAdapter202505.get_kommune_nummer(doc) is None
    
    def test_get_kommune_navn_translates_number_to_name(self):
        """Test that kommune number is translated to kommune name."""
        doc = {"kommune": 4601}
        assert DatasetAdapter202505.get_kommune_navn(doc) == "Bergen"
    
    def test_get_kommune_navn_unknown_returns_default(self):
        """Test that unknown kommune number returns default."""
        doc = {"kommune": 9999}
        assert DatasetAdapter202505.get_kommune_navn(doc) == "en norsk kommune"
    
    def test_get_url_extracts_url(self):
        """Test that URL is extracted if present."""
        doc = {"url": "https://example.com"}
        assert DatasetAdapter202505.get_url(doc) == "https://example.com"
    
    def test_get_url_missing_returns_none(self):
        """Test that missing URL returns None."""
        doc = {}
        assert DatasetAdapter202505.get_url(doc) is None
    
    def test_get_dok_type_with_new_field_name(self):
        """Test that dok_type is extracted from new field name."""
        doc = {"dok_type": "meeting_minutes"}
        assert DatasetAdapter202505.get_dok_type(doc) == "meeting_minutes"
    
    def test_get_dok_type_with_old_field_name(self):
        """Test that dok_type is extracted from old field name (backward compatibility)."""
        doc = {"doc_type": "meeting_minutes"}
        assert DatasetAdapter202505.get_dok_type(doc) == "meeting_minutes"
    
    def test_get_dok_tittel_with_new_field_name(self):
        """Test that dok_tittel is extracted from new field name."""
        doc = {"dok_tittel": "Test Title"}
        assert DatasetAdapter202505.get_dok_tittel(doc) == "Test Title"
    
    def test_get_dok_tittel_with_old_field_name(self):
        """Test that dok_tittel is extracted from old field name (backward compatibility)."""
        doc = {"tittel": "Test Title"}
        assert DatasetAdapter202505.get_dok_tittel(doc) == "Test Title"
    
    def test_normalize_complete_document(self):
        """Test that normalize extracts all fields correctly."""
        doc = {
            "dok_id": "test123",
            "kommune": 4601,
            "tekst": "Test content",
            "url": "https://example.com",
            "dok_type": "meeting_minutes",
            "dok_tittel": "Test Title"
        }
        normalized = DatasetAdapter202505.normalize(doc)
        
        assert normalized["dok_id"] == "test123"
        assert normalized["kommune_nummer"] == 4601
        assert normalized["kommune_navn"] == "Bergen"
        assert normalized["tekst"] == "Test content"
        assert normalized["url"] == "https://example.com"
        assert normalized["dok_type"] == "meeting_minutes"
        assert normalized["dok_tittel"] == "Test Title"
    
    def test_normalize_with_old_field_names(self):
        """Test that normalize works with old field names (backward compatibility)."""
        doc = {
            "dokument_id": "test456",
            "kommune": 5501,
            "text": "English content",
            "doc_type": "meeting_minutes",
            "tittel": "Old Title"
        }
        normalized = DatasetAdapter202505.normalize(doc)
        
        assert normalized["dok_id"] == "test456"
        assert normalized["kommune_nummer"] == 5501
        assert normalized["kommune_navn"] == "Tromsø"
        assert normalized["tekst"] == "English content"
        assert normalized["dok_type"] == "meeting_minutes"
        assert normalized["dok_tittel"] == "Old Title"
    
    def test_normalize_with_mixed_field_names(self):
        """Test that normalize handles mixed old and new field names."""
        doc = {
            "dok_id": "test789",  # New
            "kommune": 5536,
            "text": "English",  # Old
            "dok_type": "meeting_minutes",  # New
            "tittel": "Mixed Title"  # Old
        }
        normalized = DatasetAdapter202505.normalize(doc)
        
        assert normalized["dok_id"] == "test789"
        assert normalized["kommune_nummer"] == 5536
        assert normalized["kommune_navn"] == "Lyngen"
        assert normalized["tekst"] == "English"
        assert normalized["dok_type"] == "meeting_minutes"
        assert normalized["dok_tittel"] == "Mixed Title"

