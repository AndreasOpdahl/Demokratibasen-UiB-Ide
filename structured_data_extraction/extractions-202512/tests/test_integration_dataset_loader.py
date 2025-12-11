"""
Integration tests for DatasetLoader that load actual input data files.

These tests require:
- Actual data files to exist in the expected locations
- Files to be in the correct format (JSON array)

To run only integration tests:
    pytest -m integration

To skip integration tests:
    pytest -m "not integration"
"""
import json
import pytest
from pathlib import Path
from dataset_loader import DatasetLoader, set_root_path

# Number of documents to validate in integration tests
TEST_DOCS = 5


@pytest.mark.integration
class TestDatasetLoaderIntegration:
    """Integration tests that load actual data files."""

    @pytest.fixture
    def repo_root(self):
        """Get the repo root path."""
        current_file = Path(__file__)
        repo_root = current_file.resolve().parent.parent.parent.parent  # repo root
        return repo_root

    def test_actual_data_iteration_dataset_202505(self, repo_root):
        """Test iterating through first TEST_DOCS documents from dataset-202505 and validate structure."""
        set_root_path(repo_root)
        loader = DatasetLoader("dataset-202505")

        # Iterate through first TEST_DOCS documents and validate structure
        # Note: Quality filtering (length, alphanumeric content) is handled by DatasetLoader
        known_kommunes = {"Bergen", "Tromsø", "Lyngen", "en norsk kommune"}
        count = 0
        kommune_names = set()
        
        for doc_id, kommune_nummer, kommune_navn, text in loader():
            if count >= TEST_DOCS:
                break
            
            count += 1
            # Basic structure checks
            assert doc_id is not None
            assert kommune_navn is not None
            assert text is not None
            assert len(text) > 0
            
            # Type checks
            assert isinstance(doc_id, str), f"dok_id should be string, got {type(doc_id)}"
            assert isinstance(kommune_navn, str), f"kommune_navn should be string, got {type(kommune_navn)}"
            assert isinstance(text, str), f"text should be string, got {type(text)}"
            assert text.strip(), "text should not be only whitespace"
            
            # Kommune validation
            assert kommune_navn in known_kommunes, \
                f"Kommune '{kommune_navn}' is not in valid set: {known_kommunes}"
            kommune_names.add(kommune_navn)

        # Should have validated TEST_DOCS documents (or fewer if dataset is smaller)
        assert count > 0, f"Iterator should return at least one document (checked {count} of {TEST_DOCS} requested)"
        assert count <= TEST_DOCS, f"Should have validated at most {TEST_DOCS} documents, but validated {count}"

    def test_actual_data_has_dok_ids_dataset_202505(self, repo_root):
        """Test that first TEST_DOCS documents from dataset-202505 have dok_ids."""
        set_root_path(repo_root)
        loader = DatasetLoader("dataset-202505")

        # Check first TEST_DOCS documents
        dok_ids = []
        count = 0
        for doc_id, _, _, _ in loader():
            if count >= TEST_DOCS:
                break
            dok_ids.append(doc_id)
            count += 1
        
        non_empty_ids = [doc_id for doc_id in dok_ids if doc_id]
        
        assert len(non_empty_ids) > 0, \
            f"At least some of the first {TEST_DOCS} documents should have non-empty dok_id"

    def test_actual_data_file_format_dataset_202505(self, repo_root):
        """Test that dataset-202505 file can be parsed as JSONL."""
        set_root_path(repo_root)
        from dataset_loader import get_dataset_path
        data_file_path = get_dataset_path("dataset-202505", repo_root)
        
        if not data_file_path.exists():
            pytest.skip(f"Data file not found: {data_file_path}")

        # Try to parse the file as JSONL (one JSON object per line)
        line_count = 0
        with open(data_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    assert isinstance(doc, dict), \
                        f"Line {line_num} should be a JSON object, got {type(doc)}"
                    line_count += 1
                except json.JSONDecodeError as e:
                    pytest.fail(f"Line {line_num} is not valid JSON: {e}")
        
        assert line_count > 0, "File should contain at least one valid JSON line"

    def test_actual_data_iteration_dataset_202510(self, repo_root):
        """Test iterating through first TEST_DOCS documents from dataset-202510 and validate structure."""
        set_root_path(repo_root)
        loader = DatasetLoader("dataset-202510")
        
        # Verify correct adapter is being used
        from dataset_loader import DatasetAdapter202510
        assert loader.adapter_class == DatasetAdapter202510, \
            f"Expected DatasetAdapter202510, got {loader.adapter_class.__name__}"

        # Iterate through first TEST_DOCS documents and validate structure
        known_kommunes = {"Bergen", "Tromsø", "Lyngen", "en norsk kommune"}
        count = 0
        kommune_names = set()
        kommune_numbers = set()
        
        for doc_id, kommune_nummer, kommune_navn, text in loader():
            if count >= TEST_DOCS:
                break
            
            count += 1
            # Basic structure checks
            assert doc_id is not None
            assert kommune_navn is not None
            assert text is not None
            assert len(text) > 0
            
            # Type checks
            assert isinstance(doc_id, str), f"dok_id should be string, got {type(doc_id)}"
            assert isinstance(kommune_nummer, (int, type(None))), \
                f"kommune_nummer should be int or None, got {type(kommune_nummer)}"
            assert isinstance(kommune_navn, str), f"kommune_navn should be string, got {type(kommune_navn)}"
            assert isinstance(text, str), f"text should be string, got {type(text)}"
            assert text.strip(), "text should not be only whitespace"
            
            # Verify kommune_nummer is extracted (should be int, not None for valid documents)
            if kommune_nummer is not None:
                assert isinstance(kommune_nummer, int), \
                    f"kommune_nummer should be int when not None, got {type(kommune_nummer)}"
                assert 1000 <= kommune_nummer <= 9999, \
                    f"kommune_nummer should be 4 digits, got {kommune_nummer}"
                kommune_numbers.add(kommune_nummer)
            
            # Kommune validation (may include more kommunes than 202505)
            kommune_names.add(kommune_navn)

        # Should have validated TEST_DOCS documents (or fewer if dataset is smaller)
        assert count > 0, f"Iterator should return at least one document (checked {count} of {TEST_DOCS} requested)"
        assert count <= TEST_DOCS, f"Should have validated at most {TEST_DOCS} documents, but validated {count}"
        
        # Should have extracted kommune numbers
        assert len(kommune_numbers) > 0, \
            f"Expected at least some of the first {TEST_DOCS} documents to have kommune_nummer extracted, but got: {kommune_numbers}"


