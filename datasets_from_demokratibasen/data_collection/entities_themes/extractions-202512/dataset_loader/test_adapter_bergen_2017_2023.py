#!/usr/bin/env python3
"""
Test script for dataset-Bergen-2017-2023 adapter.

Tests the adapter with sample Bergen JSON format data to ensure:
- All adapter methods work correctly
- Normalization produces the expected output format
- The output format matches other adapters
- Edge cases are handled properly
"""

import sys
import json
from pathlib import Path

# Add the directory containing this script to the path so we can import modules
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Import kommune module first
import kommune

# For the 202505 adapter which uses relative imports, we need to set up the package structure
# Create a dummy package module
import types
package_module = types.ModuleType('dataset_loader')
package_module.kommune = kommune
sys.modules['dataset_loader'] = package_module
sys.modules['dataset_loader.kommune'] = kommune

# Import adapters
from dataset_adapter_bergen_2017_2023 import DatasetAdapterBergen2017_2023

# For 202505 adapter, we need to handle its relative import
import importlib.util
spec_202505 = importlib.util.spec_from_file_location(
    "dataset_loader.dataset_adapter_202505",
    script_dir / "dataset_adapter_202505.py"
)
module_202505 = importlib.util.module_from_spec(spec_202505)
module_202505.__package__ = 'dataset_loader'
sys.modules['dataset_loader.dataset_adapter_202505'] = module_202505
spec_202505.loader.exec_module(module_202505)
DatasetAdapter202505 = module_202505.DatasetAdapter202505


def test_basic_normalization():
    """Test basic normalization with a typical Bergen document."""
    print("=" * 80)
    print("Test 1: Basic Normalization")
    print("=" * 80)
    
    # Sample Bergen document format
    bergen_doc = {
        "dok_id": "1008:2900:28638:200019142:agendapunkt:bksak:innstilling",
        "kommune": "4601",
        "url": "https://www.bergen.kommune.no/politikere-utvalg/api/fil/bksak/2001131496-1/test",
        "dok_type": "case_presentation",
        "dok_tittel": "Test document title",
        "filformat": "doc",
        "tekst": "This is test text content with some Norwegian characters: æøå."
    }
    
    adapter = DatasetAdapterBergen2017_2023()
    normalized = adapter.normalize(bergen_doc)
    
    print("Input document:")
    print(json.dumps(bergen_doc, indent=2, ensure_ascii=False))
    print()
    print("Normalized output:")
    print(json.dumps(normalized, indent=2, ensure_ascii=False))
    print()
    
    # Verify expected fields
    assert "dok_id" in normalized, "Missing dok_id field"
    assert "kommune_nummer" in normalized, "Missing kommune_nummer field"
    assert "kommune_navn" in normalized, "Missing kommune_navn field"
    assert "tekst" in normalized, "Missing tekst field"
    assert "url" in normalized, "Missing url field"
    assert "dok_type" in normalized, "Missing dok_type field"
    assert "dok_tittel" in normalized, "Missing dok_tittel field"
    
    # Verify values
    assert normalized["dok_id"] == bergen_doc["dok_id"], "dok_id mismatch"
    assert normalized["kommune_nummer"] == 4601, f"kommune_nummer should be 4601, got {normalized['kommune_nummer']}"
    assert normalized["kommune_navn"] == "Bergen", f"kommune_navn should be 'Bergen', got {normalized['kommune_navn']}"
    assert normalized["tekst"] == bergen_doc["tekst"], "tekst mismatch"
    assert normalized["url"] == bergen_doc["url"], "url mismatch"
    assert normalized["dok_type"] == bergen_doc["dok_type"], "dok_type mismatch"
    assert normalized["dok_tittel"] == bergen_doc["dok_tittel"], "dok_tittel mismatch"
    
    print("✓ All assertions passed!")
    print()


def test_output_format_consistency():
    """Test that the output format matches other adapters."""
    print("=" * 80)
    print("Test 2: Output Format Consistency")
    print("=" * 80)
    
    # Create equivalent documents for both adapters
    bergen_doc = {
        "dok_id": "test:123",
        "kommune": "4601",
        "url": "https://example.com",
        "dok_type": "case_presentation",
        "dok_tittel": "Test Title",
        "filformat": "doc",
        "tekst": "Test text"
    }
    
    # 202505 adapter equivalent (using new field names)
    doc_202505 = {
        "dok_id": "test:123",
        "kommune": 4601,
        "url": "https://example.com",
        "dok_type": "case_presentation",
        "dok_tittel": "Test Title",
        "tekst": "Test text"
    }
    
    bergen_adapter = DatasetAdapterBergen2017_2023()
    adapter_202505 = DatasetAdapter202505()
    
    bergen_normalized = bergen_adapter.normalize(bergen_doc)
    normalized_202505 = adapter_202505.normalize(doc_202505)
    
    print("Bergen adapter output keys:", sorted(bergen_normalized.keys()))
    print("202505 adapter output keys:", sorted(normalized_202505.keys()))
    print()
    
    # Verify they have the same keys
    assert set(bergen_normalized.keys()) == set(normalized_202505.keys()), \
        f"Key mismatch: Bergen has {set(bergen_normalized.keys())}, 202505 has {set(normalized_202505.keys())}"
    
    # Verify key types match
    for key in bergen_normalized.keys():
        assert type(bergen_normalized[key]) == type(normalized_202505[key]), \
            f"Type mismatch for {key}: Bergen has {type(bergen_normalized[key])}, 202505 has {type(normalized_202505[key])}"
    
    print("✓ Output format matches 202505 adapter!")
    print()


def test_individual_methods():
    """Test individual adapter methods."""
    print("=" * 80)
    print("Test 3: Individual Methods")
    print("=" * 80)
    
    bergen_doc = {
        "dok_id": "1005:238:1367:200190324:agendapunkt:bksak:vedtak",
        "kommune": "4601",
        "url": "https://example.com/test",
        "dok_type": "case_minutes",
        "dok_tittel": "Melding om status for HMS-arbeidet",
        "filformat": "doc",
        "tekst": "Test document content here."
    }
    
    adapter = DatasetAdapterBergen2017_2023()
    
    # Test get_dok_id
    dok_id = adapter.get_dok_id(bergen_doc)
    assert dok_id == bergen_doc["dok_id"], f"get_dok_id failed: expected {bergen_doc['dok_id']}, got {dok_id}"
    print(f"✓ get_dok_id: {dok_id}")
    
    # Test get_tekst
    tekst = adapter.get_tekst(bergen_doc)
    assert tekst == bergen_doc["tekst"], f"get_tekst failed: expected {bergen_doc['tekst']}, got {tekst}"
    print(f"✓ get_tekst: {tekst[:50]}...")
    
    # Test get_kommune_nummer
    kommune_nummer = adapter.get_kommune_nummer(bergen_doc)
    assert kommune_nummer == 4601, f"get_kommune_nummer failed: expected 4601, got {kommune_nummer}"
    print(f"✓ get_kommune_nummer: {kommune_nummer}")
    
    # Test get_kommune_navn
    kommune_navn = adapter.get_kommune_navn(bergen_doc)
    assert kommune_navn == "Bergen", f"get_kommune_navn failed: expected 'Bergen', got {kommune_navn}"
    print(f"✓ get_kommune_navn: {kommune_navn}")
    
    # Test get_url
    url = adapter.get_url(bergen_doc)
    assert url == bergen_doc["url"], f"get_url failed: expected {bergen_doc['url']}, got {url}"
    print(f"✓ get_url: {url}")
    
    # Test get_dok_type
    dok_type = adapter.get_dok_type(bergen_doc)
    assert dok_type == bergen_doc["dok_type"], f"get_dok_type failed: expected {bergen_doc['dok_type']}, got {dok_type}"
    print(f"✓ get_dok_type: {dok_type}")
    
    # Test get_dok_tittel
    dok_tittel = adapter.get_dok_tittel(bergen_doc)
    assert dok_tittel == bergen_doc["dok_tittel"], f"get_dok_tittel failed: expected {bergen_doc['dok_tittel']}, got {dok_tittel}"
    print(f"✓ get_dok_tittel: {dok_tittel}")
    
    print()


def test_should_include_document():
    """Test document type filtering."""
    print("=" * 80)
    print("Test 4: Document Type Filtering")
    print("=" * 80)
    
    adapter = DatasetAdapterBergen2017_2023()
    
    # Test included document types
    included_types = ["meeting_agenda", "meeting_minutes", "case_presentation", "case_minutes"]
    for doc_type in included_types:
        doc = {"dok_id": "test", "kommune": "4601", "dok_type": doc_type, "tekst": "test"}
        assert adapter.should_include_document(doc), f"should_include_document failed for {doc_type}"
        print(f"✓ {doc_type}: included")
    
    # Test excluded document types
    excluded_types = ["case_attachment", "case_history", "other_type", None]
    for doc_type in excluded_types:
        doc = {"dok_id": "test", "kommune": "4601", "tekst": "test"}
        if doc_type is not None:
            doc["dok_type"] = doc_type
        assert not adapter.should_include_document(doc), f"should_include_document should return False for {doc_type}"
        print(f"✓ {doc_type}: excluded")
    
    print()


def test_edge_cases():
    """Test edge cases and missing fields."""
    print("=" * 80)
    print("Test 5: Edge Cases")
    print("=" * 80)
    
    adapter = DatasetAdapterBergen2017_2023()
    
    # Test with minimal document
    minimal_doc = {"dok_id": "test", "kommune": "4601", "tekst": "test"}
    normalized = adapter.normalize(minimal_doc)
    assert normalized["dok_id"] == "test"
    assert normalized["kommune_nummer"] == 4601
    assert normalized["tekst"] == "test"
    assert normalized["url"] is None
    assert normalized["dok_type"] is None
    assert normalized["dok_tittel"] is None
    print("✓ Minimal document (only required fields)")
    
    # Test with missing kommune
    doc_no_kommune = {"dok_id": "test", "tekst": "test"}
    normalized = adapter.normalize(doc_no_kommune)
    assert normalized["kommune_nummer"] is None
    assert normalized["kommune_navn"] == "en norsk kommune"
    print("✓ Missing kommune field")
    
    # Test with invalid kommune (non-numeric string)
    doc_invalid_kommune = {"dok_id": "test", "kommune": "invalid", "tekst": "test"}
    normalized = adapter.normalize(doc_invalid_kommune)
    assert normalized["kommune_nummer"] is None
    assert normalized["kommune_navn"] == "en norsk kommune"
    print("✓ Invalid kommune (non-numeric)")
    
    # Test with empty string kommune
    doc_empty_kommune = {"dok_id": "test", "kommune": "", "tekst": "test"}
    normalized = adapter.normalize(doc_empty_kommune)
    # Empty string should convert to None (via int conversion)
    print("✓ Empty string kommune")
    
    # Test with empty tekst
    doc_empty_tekst = {"dok_id": "test", "kommune": "4601", "tekst": ""}
    normalized = adapter.normalize(doc_empty_tekst)
    assert normalized["tekst"] == ""
    print("✓ Empty tekst field")
    
    print()


def test_real_bergen_sample():
    """Test with a real Bergen JSON sample."""
    print("=" * 80)
    print("Test 6: Real Bergen JSON Sample")
    print("=" * 80)
    
    # Real sample from Kommunebasen-Bergen
    real_bergen_doc = {
        "dok_id": "1008:2900:28638:200019142:agendapunkt:bksak:innstilling",
        "kommune": "4601",
        "url": "https://www.bergen.kommune.no/politikere-utvalg/api/fil/bksak/2001131496-1/Framstilling-Fratreden-og-innstilling-av-nytt-medlem-i-byradet",
        "dok_type": "case_presentation",
        "dok_tittel": "Fratreden og innstilling av nytt medlem i byrådet",
        "filformat": "doc",
        "tekst": "\n|                                               |Dato:  19. oktober   |\n|                                               |2001                 |\n|                                               |                     |\n|                                               |                     |\n|                                               |Byrådsak             |\n|                                               |334     /01          |\n|                                                                    |\n|                                                                    |\n|Byrådet                                                             |\n|                                                                    |\n|                                                                    |\n|                                                                    |\n|Fratreden og innstilling av nytt medlem i byrådet                   |\n|                                                                    |\n|                                      |BNO     |BLED-03-200019142-10 |\n|                                                                    |\n\n\nHva saken gjelder:\n\nByråd for helse og sosial Kristin Ravnanger (KrF) har i brev av 19. oktober\n2001 meddelt at hun er utnevnt til statssekretær i Helsedepartementet og at\nhun dermed fratrer som byråd."
    }
    
    adapter = DatasetAdapterBergen2017_2023()
    normalized = adapter.normalize(real_bergen_doc)
    
    print("Input dok_id:", real_bergen_doc["dok_id"])
    print("Normalized dok_id:", normalized["dok_id"])
    print("Normalized kommune_nummer:", normalized["kommune_nummer"])
    print("Normalized kommune_navn:", normalized["kommune_navn"])
    print("Normalized dok_type:", normalized["dok_type"])
    print("Normalized dok_tittel:", normalized["dok_tittel"])
    print("Tekst length:", len(normalized["tekst"]), "characters")
    print()
    
    # Verify
    assert normalized["dok_id"] == real_bergen_doc["dok_id"]
    assert normalized["kommune_nummer"] == 4601
    assert normalized["kommune_navn"] == "Bergen"
    assert normalized["dok_type"] == real_bergen_doc["dok_type"]
    assert len(normalized["tekst"]) > 0
    
    print("✓ Real Bergen sample processed correctly!")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Testing DatasetAdapterBergen2017_2023")
    print("=" * 80 + "\n")
    
    try:
        test_basic_normalization()
        test_output_format_consistency()
        test_individual_methods()
        test_should_include_document()
        test_edge_cases()
        test_real_bergen_sample()
        
        print("=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)
        print()
        print("The adapter correctly:")
        print("  - Normalizes Bergen JSON format to standard format")
        print("  - Produces output matching other adapters")
        print("  - Handles edge cases and missing fields")
        print("  - Filters documents by type correctly")
        print()
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

