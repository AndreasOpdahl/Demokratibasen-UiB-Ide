"""
Integration tests for extract_structured_data.py.

These tests make actual API calls to GPT and require:
- OPENAI_API_KEY environment variable to be set
- Valid API key with credits/access
- Actual data files to exist in the expected locations

To run only integration tests:
    pytest -m integration

To skip integration tests:
    pytest -m "not integration"
"""
import json
import os
import pytest
import shutil
from pathlib import Path
from unittest.mock import patch

from llm_adapter import LLMAdapter, GPTFactory
from dataset_loader import DatasetLoader, set_root_path
from create_prompt import Prompt


@pytest.mark.integration
class TestExtractStructuredDataIntegration:
    """Integration tests for extract_structured_data that process actual documents."""

    @pytest.fixture
    def api_key(self):
        """Get API key from environment, fail test if not available."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.fail("OPENAI_API_KEY not set. Integration tests require a valid API key.")
        return api_key

    @pytest.fixture
    def repo_root(self):
        """Get the repo root path."""
        current_file = Path(__file__)
        repo_root = current_file.resolve().parent.parent.parent.parent  # repo root
        return repo_root

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create a temporary output directory for tests."""
        output_dir = tmp_path / "extracted-data" / "debug-outputs" / "gpt-3.5-turbo"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @pytest.fixture
    def llm_adapter(self, api_key):
        """Create an LLMAdapter instance with a test model."""
        # Use a cheaper/faster model for testing
        return LLMAdapter(GPTFactory(), "gpt-3.5-turbo")

    def test_extract_first_few_documents(self, repo_root, temp_output_dir, llm_adapter):
        """Test that we can extract structured data from the first few documents."""
        # Arrange
        set_root_path(repo_root)
        text_loader = DatasetLoader("dataset-202510")
        max_documents = 3
        processed_count = 0
        skipped_count = 0
        
        # Control variables (matching extract_structured_data.py)
        TEMPERATURE = 0.1
        MAX_TOKENS = 4096
        
        # Use Prompt class with default prompt name
        prompt_creator = Prompt("initial-extraction-202509")
        
        # Process first few documents
        for doc_id, kommune_nummer, kommune_navn, text in text_loader():
            if processed_count >= max_documents:
                break
            
            # Check if output file already exists
            output_file = temp_output_dir / f"{doc_id}.json"
            if output_file.exists():
                skipped_count += 1
                continue
            
            try:
                # Prepare prompt with kommune_navn inserted (text, not number)
                prompt = prompt_creator.get_prompt(kommune_navn)
                
                # Prepare document text (user input)
                document_text = prompt_creator.get_document_text(text)
                
                # Get response from LLM adapter
                response = llm_adapter.generate_text(
                    prompt=document_text,
                    system_prompt=prompt,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS
                )
                
                # Parse JSON from response
                response_content = response.choices[0].message.content
                extracted_data = json.loads(response_content)
                
                # Verify extracted data has required fields
                assert "hva_saken_gjelder" in extracted_data, "Missing required field 'hva_saken_gjelder'"
                assert "tema" in extracted_data, "Missing required field 'tema'"
                assert isinstance(extracted_data["tema"], list), "tema should be a list"
                
                # Create output record
                output_record = {
                    "dokument_id": doc_id,
                    "kommune_nummer": kommune_nummer,  # Four-digit number from input
                    "kommune_navn": kommune_navn,      # Text name used in prompt
                    "model": "gpt-3.5-turbo",
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                    "prompt": prompt,
                    "response": extracted_data
                }
                
                # Write output to individual file
                with output_file.open("w", encoding="utf-8") as fout:
                    json.dump(output_record, fout, ensure_ascii=False, indent=2)
                
                processed_count += 1
                
            except Exception as e:
                pytest.fail(f"Failed to process document {doc_id}: {e}")
        
        # Assert that we processed the expected number of documents
        assert processed_count == max_documents, f"Expected to process {max_documents} documents, but processed {processed_count}"
        
        # Verify output files were created
        output_files = list(temp_output_dir.glob("*.json"))
        assert len(output_files) == max_documents, f"Expected {max_documents} output files, but found {len(output_files)}"
        
        # Verify structure of output files
        for output_file in output_files:
            with output_file.open("r", encoding="utf-8") as fin:
                record = json.load(fin)
                
                # Verify required fields in output record
                assert "dokument_id" in record
                assert "kommune_navn" in record
                assert "kommune_nummer" in record
                assert "model" in record
                assert "temperature" in record
                assert "max_tokens" in record
                assert "prompt" in record
                assert "response" in record
                
                # Verify response structure
                response = record["response"]
                assert "hva_saken_gjelder" in response
                assert "tema" in response

    def test_skip_existing_documents(self, repo_root, temp_output_dir, llm_adapter):
        """Test that existing documents are skipped correctly."""
        # Arrange
        set_root_path(repo_root)
        text_loader = DatasetLoader("dataset-202510")
        TEMPERATURE = 0.1
        MAX_TOKENS = 4096
        
        # Use Prompt class with default prompt name
        prompt_creator = Prompt("initial-extraction-202509")
        
        # Process first document
        doc_id = None
        for doc_id, kommune_nummer, kommune_navn, text in text_loader():
            prompt = prompt_creator.get_prompt(kommune_navn)
            document_text = prompt_creator.get_document_text(text)
            response = llm_adapter.generate_text(
                prompt=document_text,
                system_prompt=prompt,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            response_content = response.choices[0].message.content
            extracted_data = json.loads(response_content)
            
            output_file = temp_output_dir / f"{doc_id}.json"
            output_record = {
                "dokument_id": doc_id,
                "kommune_nummer": kommune_nummer,
                "kommune_navn": kommune_navn,
                "response": extracted_data
            }
            with output_file.open("w", encoding="utf-8") as fout:
                json.dump(output_record, fout, ensure_ascii=False, indent=2)
            break
        
        if doc_id is None:
            pytest.skip("No documents available in dataset")
        
        # Verify file exists
        assert (temp_output_dir / f"{doc_id}.json").exists()
        
        # Process again - should skip existing document
        text_loader2 = DatasetLoader("dataset-202510")
        skipped_count = 0
        processed_count = 0
        
        # Iterate through documents and check that the one we created is skipped
        for doc_id2, kommune_nummer2, kommune_navn2, text2 in text_loader2():
            output_file = temp_output_dir / f"{doc_id2}.json"
            if output_file.exists():
                skipped_count += 1
                # If this is the document we created, verify it was skipped correctly
                if doc_id2 == doc_id:
                    # This is the document we created - it should be skipped
                    continue
                # Otherwise, it's a different document that already exists
                continue
            
            # File doesn't exist - this is a new document
            # Only check the first few documents to see if our created document appears
            if doc_id2 == doc_id:
                # This should never happen - the file should exist
                pytest.fail(f"Document {doc_id} should have been skipped (file should exist) but was not found")
            
            # This is a different document that doesn't exist yet
            # Don't process it, just count it for verification
            processed_count += 1
            # Only check first 5 documents to avoid processing too many
            if processed_count >= 5:
                break
        
        # Assert that the document we created was skipped
        # We should have encountered it and skipped it
        assert skipped_count >= 1, f"Expected at least 1 skipped document (the one we created), but got {skipped_count}"
        # The specific document we created should have been skipped
        # Verify the file still exists
        assert (temp_output_dir / f"{doc_id}.json").exists(), f"File for document {doc_id} should still exist"

    def test_extract_with_named_prompt(self, repo_root, temp_output_dir, llm_adapter):
        """Test that extraction works with a named prompt (initial-extraction-202509)."""
        # Arrange
        set_root_path(repo_root)
        text_loader = DatasetLoader("dataset-202510")
        TEMPERATURE = 0.1
        MAX_TOKENS = 4096
        
        # Use Prompt class with explicit prompt name
        prompt_creator = Prompt("initial-extraction-202509")
        
        # Verify prompt name is set correctly
        assert prompt_creator.prompt_name == "initial-extraction-202509"
        
        # Process first document
        doc_id = None
        kommune_navn = None
        for doc_id, kommune_nummer, kommune_navn, text in text_loader():
            break
        
        if doc_id is None:
            pytest.skip("No documents available in dataset")
        
        # Check if output file already exists
        output_file = temp_output_dir / f"{doc_id}.json"
        if output_file.exists():
            output_file.unlink()  # Remove if exists for clean test
        
        try:
            # Prepare prompt with kommune_navn inserted
            prompt = prompt_creator.get_prompt(kommune_navn)
            
            # Verify prompt contains kommune name
            assert kommune_navn in prompt
            assert "<<kommune_navn>>" not in prompt
            
            # Prepare document text (user input)
            document_text = prompt_creator.get_document_text(text)
            
            # Get response from LLM adapter
            response = llm_adapter.generate_text(
                prompt=document_text,
                system_prompt=prompt,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            
            # Parse JSON from response
            response_content = response.choices[0].message.content
            extracted_data = json.loads(response_content)
            
            # Verify extracted data has required fields
            assert "hva_saken_gjelder" in extracted_data, "Missing required field 'hva_saken_gjelder'"
            assert "tema" in extracted_data, "Missing required field 'tema'"
            assert isinstance(extracted_data["tema"], list), "tema should be a list"
            
            # Verify schema matches what we expect
            assert "parameters" in prompt_creator.SCHEMA
            assert "required" in prompt_creator.SCHEMA["parameters"]
            required_fields = prompt_creator.SCHEMA["parameters"]["required"]
            assert "hva_saken_gjelder" in required_fields
            assert "tema" in required_fields
            
        except Exception as e:
            pytest.fail(f"Failed to process document with named prompt: {e}")
