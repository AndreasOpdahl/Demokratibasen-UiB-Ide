"""
Unit tests for summarise_extracted_data.py
"""
import pytest
from summarise_extracted_data import (
    is_acronym_or_name,
    normalize_for_comparison,
    count_string_occurrences,
    analyze_document_group
)


class TestIsAcronymOrName:
    """Test suite for is_acronym_or_name function."""
    
    def test_acronyms_all_uppercase(self):
        """Test that all-uppercase strings (min 2 chars) are recognized as acronyms."""
        assert is_acronym_or_name("NATO") is True
        assert is_acronym_or_name("USA") is True
        assert is_acronym_or_name("EU") is True
        assert is_acronym_or_name("UN") is True
        assert is_acronym_or_name("WWW") is True
    
    def test_acronyms_single_char(self):
        """Test that single character strings are not acronyms."""
        assert is_acronym_or_name("A") is False
        assert is_acronym_or_name("Z") is False
    
    def test_names_with_multiple_words(self):
        """Test that names with multiple words are recognized."""
        assert is_acronym_or_name("Ola Nordmann") is True
        assert is_acronym_or_name("John Doe") is True
        assert is_acronym_or_name("Anne Lise") is True
    
    def test_names_with_hyphen(self):
        """Test that names with hyphens are recognized."""
        assert is_acronym_or_name("Anne-Lise") is True
        assert is_acronym_or_name("Ola-Bjørn") is True
    
    def test_regular_words_not_names(self):
        """Test that regular words are not recognized as names."""
        assert is_acronym_or_name("motorferdsel") is False
        assert is_acronym_or_name("Motorferdsel") is False
        assert is_acronym_or_name("utmark") is False
        assert is_acronym_or_name("snøskuter") is False
    
    def test_empty_or_short_strings(self):
        """Test that empty or very short strings are not acronyms/names."""
        assert is_acronym_or_name("") is False
        assert is_acronym_or_name("A") is False
    
    def test_mixed_case_not_acronym(self):
        """Test that mixed case strings are not acronyms."""
        assert is_acronym_or_name("Nato") is False
        assert is_acronym_or_name("UsA") is False


class TestNormalizeForComparison:
    """Test suite for normalize_for_comparison function."""
    
    def test_normalize_regular_text_lowercase(self):
        """Test that regular text is lowercased."""
        assert normalize_for_comparison("Motorferdsel") == "motorferdsel"
        # "MOTORFERDSEL" is treated as an acronym (all uppercase, >= 2 chars), so casing is preserved
        assert normalize_for_comparison("MOTORFERDSEL") == "MOTORFERDSEL"
        assert normalize_for_comparison("motorferdsel") == "motorferdsel"
    
    def test_normalize_acronyms_preserved(self):
        """Test that acronyms preserve original casing for exact matching."""
        assert normalize_for_comparison("NATO") == "NATO"
        assert normalize_for_comparison("USA") == "USA"
        assert normalize_for_comparison("EU") == "EU"
    
    def test_normalize_names_preserved(self):
        """Test that names preserve original casing for exact matching."""
        assert normalize_for_comparison("Ola Nordmann") == "Ola Nordmann"
        assert normalize_for_comparison("Anne-Lise") == "Anne-Lise"
    
    def test_normalize_for_substring_lowercase_all(self):
        """Test that for_substring=True lowercases everything, including acronyms/names."""
        assert normalize_for_comparison("NATO", for_substring=True) == "nato"
        assert normalize_for_comparison("Ola Nordmann", for_substring=True) == "ola nordmann"
        assert normalize_for_comparison("Motorferdsel", for_substring=True) == "motorferdsel"
        assert normalize_for_comparison("MOTORFERDSEL", for_substring=True) == "motorferdsel"
    
    def test_normalize_mixed_case(self):
        """Test normalization of mixed case strings."""
        assert normalize_for_comparison("MotorFerdsel") == "motorferdsel"
        assert normalize_for_comparison("MotorFerdsel", for_substring=True) == "motorferdsel"


class TestCountStringOccurrences:
    """Test suite for count_string_occurrences function."""
    
    def test_simple_string_count(self):
        """Test basic string counting."""
        all_lists = [
            ["A", "B"],
            ["A", "C"],
            ["A"]
        ]
        counts = count_string_occurrences(all_lists)
        
        assert counts["A"]["string_count"] == 3
        assert counts["B"]["string_count"] == 1
        assert counts["C"]["string_count"] == 1
    
    def test_substr_count_basic(self):
        """Test basic substring counting."""
        all_lists = [
            ["Motorferdsel", "Motorferdsel i utmark"],
            ["Motorferdsel", "Motorferdsel i utmark"],
            ["Motorferdsel"],
            ["Motorferdsel"]
        ]
        counts = count_string_occurrences(all_lists)
        
        # "Motorferdsel" appears 4 times
        assert counts["Motorferdsel"]["string_count"] == 4
        # "Motorferdsel i utmark" appears 2 times
        assert counts["Motorferdsel i utmark"]["string_count"] == 2
        # "Motorferdsel" should have substr_count = 2 (since "Motorferdsel i utmark" appears 2 times and contains "Motorferdsel")
        assert counts["Motorferdsel"]["substr_count"] == 2
        # "Motorferdsel i utmark" has no superstrings, so substr_count = 0
        assert counts["Motorferdsel i utmark"]["substr_count"] == 0
    
    def test_substr_count_multiple_substrings(self):
        """Test substring counting with multiple superstrings."""
        all_lists = [
            ["A", "A B", "A B C"],
            ["A", "A B"],
            ["A"]
        ]
        counts = count_string_occurrences(all_lists)
        
        assert counts["A"]["string_count"] == 3
        assert counts["A B"]["string_count"] == 2
        assert counts["A B C"]["string_count"] == 1
        
        # "A" is contained in "A B" (2 times) and "A B C" (1 time) = 3 total
        assert counts["A"]["substr_count"] == 3
        # "A B" is contained in "A B C" (1 time)
        assert counts["A B"]["substr_count"] == 1
        # "A B C" has no superstrings
        assert counts["A B C"]["substr_count"] == 0
    
    def test_case_insensitive_string_count(self):
        """Test that string_count is case-insensitive (except acronyms/names)."""
        all_lists = [
            ["Motorferdsel", "MOTORFERDSEL"],
            ["motorferdsel"]
        ]
        counts = count_string_occurrences(all_lists)
        
        # "Motorferdsel" and "motorferdsel" are counted together (case-insensitive)
        # "MOTORFERDSEL" is treated as an acronym, so it's separate
        assert counts["Motorferdsel"]["string_count"] == 2  # "Motorferdsel" and "motorferdsel"
        assert counts["MOTORFERDSEL"]["string_count"] == 1  # "MOTORFERDSEL" (acronym)
        assert counts["motorferdsel"]["string_count"] == 2  # Same as "Motorferdsel"
    
    def test_case_insensitive_substr_count(self):
        """Test that substr_count uses case-insensitive substring matching."""
        all_lists = [
            ["Motorferdsel", "MOTORFERDSEL i utmark"],
            ["motorferdsel", "Motorferdsel i utmark"]
        ]
        counts = count_string_occurrences(all_lists)
        
        # "Motorferdsel i utmark" variants should count "Motorferdsel" variants
        assert counts["Motorferdsel i utmark"]["substr_count"] >= 2
        assert counts["MOTORFERDSEL i utmark"]["substr_count"] >= 2
    
    def test_acronyms_substring_counting(self):
        """Test substring counting with acronyms."""
        all_lists = [
            ["NATO", "NATO membership"],
            ["NATO", "NATO membership"],
            ["NATO"]
        ]
        counts = count_string_occurrences(all_lists)
        
        assert counts["NATO"]["string_count"] == 3
        assert counts["NATO membership"]["string_count"] == 2
        # "NATO" is contained in "NATO membership" (2 times)
        assert counts["NATO"]["substr_count"] == 2
        # "NATO membership" has no superstrings
        assert counts["NATO membership"]["substr_count"] == 0
    
    def test_names_substring_counting(self):
        """Test substring counting with names."""
        all_lists = [
            ["Ola Nordmann", "Ola Nordmann said"],
            ["Ola Nordmann"]
        ]
        counts = count_string_occurrences(all_lists)
        
        assert counts["Ola Nordmann"]["string_count"] == 2
        assert counts["Ola Nordmann said"]["string_count"] == 1
        # "Ola Nordmann" is contained in "Ola Nordmann said" (1 time)
        assert counts["Ola Nordmann"]["substr_count"] == 1
        # "Ola Nordmann said" has no superstrings
        assert counts["Ola Nordmann said"]["substr_count"] == 0
    
    def test_no_substrings(self):
        """Test that strings with no substrings have substr_count=0."""
        all_lists = [
            ["A"],
            ["B"],
            ["C"]
        ]
        counts = count_string_occurrences(all_lists)
        
        assert counts["A"]["substr_count"] == 0
        assert counts["B"]["substr_count"] == 0
        assert counts["C"]["substr_count"] == 0
    
    def test_empty_lists(self):
        """Test handling of empty lists."""
        all_lists = [
            [],
            ["A"],
            []
        ]
        counts = count_string_occurrences(all_lists)
        
        assert counts["A"]["string_count"] == 1
        assert counts["A"]["substr_count"] == 0
    
    def test_complex_substring_scenario(self):
        """Test complex scenario with multiple overlapping substrings."""
        all_lists = [
            ["Motorferdsel", "Motorferdsel i utmark", "Snøskuter"],
            ["Motorferdsel", "Motorferdsel i utmark", "Snøskuter i utmark"],
            ["Motorferdsel", "Snøskuter"],
            ["Motorferdsel"]
        ]
        counts = count_string_occurrences(all_lists)
        
        assert counts["Motorferdsel"]["string_count"] == 4
        assert counts["Motorferdsel i utmark"]["string_count"] == 2
        assert counts["Snøskuter"]["string_count"] == 2
        assert counts["Snøskuter i utmark"]["string_count"] == 1
        
        # "Motorferdsel" is contained in "Motorferdsel i utmark" (2 times)
        assert counts["Motorferdsel"]["substr_count"] == 2
        # "Motorferdsel i utmark" has no superstrings
        assert counts["Motorferdsel i utmark"]["substr_count"] == 0
        # "Snøskuter" is contained in "Snøskuter i utmark" (1 time)
        assert counts["Snøskuter"]["substr_count"] == 1
        # "Snøskuter i utmark" has no superstrings
        assert counts["Snøskuter i utmark"]["substr_count"] == 0


class TestAnalyzeDocumentGroup:
    """Test suite for analyze_document_group function."""
    
    def test_models_list_included(self):
        """Test that models list is included in analysis."""
        doc_data = {
            "model1": {"response": {"hva_saken_gjelder": "Test"}},
            "model2": {"response": {"hva_saken_gjelder": "Test2"}}
        }
        schema = {"hva_saken_gjelder": {"type": "string"}}
        
        analysis = analyze_document_group("test-doc", doc_data, schema)
        
        assert "models" in analysis
        assert isinstance(analysis["models"], list)
        assert analysis["models"] == ["model1", "model2"]
    
    def test_models_list_sorted(self):
        """Test that models list is sorted alphabetically."""
        doc_data = {
            "model_z": {"response": {"hva_saken_gjelder": "Test"}},
            "model_a": {"response": {"hva_saken_gjelder": "Test2"}},
            "model_m": {"response": {"hva_saken_gjelder": "Test3"}}
        }
        schema = {"hva_saken_gjelder": {"type": "string"}}
        
        analysis = analyze_document_group("test-doc", doc_data, schema)
        
        assert analysis["models"] == ["model_a", "model_m", "model_z"]
    
    def test_properties_as_list(self):
        """Test that properties are returned as lists in model order."""
        doc_data = {
            "model1": {"response": {"hva_saken_gjelder": "Description1"}},
            "model2": {"response": {"hva_saken_gjelder": "Description2"}},
            "model3": {"response": {"hva_saken_gjelder": "Description3"}}
        }
        schema = {"hva_saken_gjelder": {"type": "string"}}
        
        analysis = analyze_document_group("test-doc", doc_data, schema)
        
        assert "properties" in analysis
        assert "hva_saken_gjelder" in analysis["properties"]
        assert isinstance(analysis["properties"]["hva_saken_gjelder"], list)
        assert len(analysis["properties"]["hva_saken_gjelder"]) == 3
        # Should be in sorted model order
        assert analysis["properties"]["hva_saken_gjelder"][0] == ["Description1"]  # model1
        assert analysis["properties"]["hva_saken_gjelder"][1] == ["Description2"]  # model2
        assert analysis["properties"]["hva_saken_gjelder"][2] == ["Description3"]  # model3
    
    def test_counts_as_list_of_lists(self):
        """Test that counts are returned as list of [string, string_count, substr_count]."""
        doc_data = {
            "model1": {"response": {"tema": ["A", "B"]}},
            "model2": {"response": {"tema": ["A", "C"]}},
            "model3": {"response": {"tema": ["A"]}}
        }
        schema = {"tema": {"type": "array"}}
        
        analysis = analyze_document_group("test-doc", doc_data, schema)
        
        assert "counts" in analysis
        assert "tema" in analysis["counts"]
        assert isinstance(analysis["counts"]["tema"], list)
        
        # Check format: list of [string, string_count, substr_count]
        for item in analysis["counts"]["tema"]:
            assert isinstance(item, list)
            assert len(item) == 3
            assert isinstance(item[0], str)  # string
            assert isinstance(item[1], int)    # string_count
            assert isinstance(item[2], int)    # substr_count
    
    def test_counts_content(self):
        """Test that counts contain correct values."""
        doc_data = {
            "model1": {"response": {"tema": ["Motorferdsel", "Motorferdsel i utmark"]}},
            "model2": {"response": {"tema": ["Motorferdsel", "Motorferdsel i utmark"]}},
            "model3": {"response": {"tema": ["Motorferdsel"]}},
            "model4": {"response": {"tema": ["Motorferdsel"]}}
        }
        schema = {"tema": {"type": "array"}}
        
        analysis = analyze_document_group("test-doc", doc_data, schema)
        
        # Find "Motorferdsel" and "Motorferdsel i utmark" in counts
        tema_counts = {item[0]: (item[1], item[2]) for item in analysis["counts"]["tema"]}
        
        assert "Motorferdsel" in tema_counts
        assert "Motorferdsel i utmark" in tema_counts
        
        motorferdsel_count, motorferdsel_substr = tema_counts["Motorferdsel"]
        utmark_count, utmark_substr = tema_counts["Motorferdsel i utmark"]
        
        assert motorferdsel_count == 4  # Appears 4 times
        assert utmark_count == 2  # Appears 2 times
        assert motorferdsel_substr == 2  # Contained in "Motorferdsel i utmark" which appears 2 times
        assert utmark_substr == 0  # Has no superstrings
    
    def test_properties_order_matches_models(self):
        """Test that property values are in the same order as models list."""
        doc_data = {
            "model_c": {"response": {"hva_saken_gjelder": "C"}},
            "model_a": {"response": {"hva_saken_gjelder": "A"}},
            "model_b": {"response": {"hva_saken_gjelder": "B"}}
        }
        schema = {"hva_saken_gjelder": {"type": "string"}}
        
        analysis = analyze_document_group("test-doc", doc_data, schema)
        
        # Models should be sorted
        assert analysis["models"] == ["model_a", "model_b", "model_c"]
        # Properties should match that order
        assert analysis["properties"]["hva_saken_gjelder"] == [["A"], ["B"], ["C"]]
    
    def test_empty_response_handling(self):
        """Test handling of missing or empty responses."""
        doc_data = {
            "model1": {"response": {"hva_saken_gjelder": "Description1"}},
            "model2": {"response": {}},  # Missing hva_saken_gjelder
            "model3": {"response": {"hva_saken_gjelder": None}}  # None value
        }
        schema = {"hva_saken_gjelder": {"type": "string"}}
        
        analysis = analyze_document_group("test-doc", doc_data, schema)
        
        assert len(analysis["properties"]["hva_saken_gjelder"]) == 3
        assert analysis["properties"]["hva_saken_gjelder"][0] == ["Description1"]  # model1
        assert analysis["properties"]["hva_saken_gjelder"][1] == []  # model2 (missing)
        assert analysis["properties"]["hva_saken_gjelder"][2] == []  # model3 (None)
    
    def test_array_properties(self):
        """Test handling of array properties."""
        doc_data = {
            "model1": {"response": {"tema": ["A", "B"]}},
            "model2": {"response": {"tema": ["A", "C"]}}
        }
        schema = {"tema": {"type": "array"}}
        
        analysis = analyze_document_group("test-doc", doc_data, schema)
        
        assert analysis["properties"]["tema"][0] == ["A", "B"]
        assert analysis["properties"]["tema"][1] == ["A", "C"]
    
    def test_all_count_fields_included(self):
        """Test that all COUNT_FIELDS are included in counts."""
        from summarise_extracted_data import COUNT_FIELDS
        
        doc_data = {
            "model1": {"response": {
                "tema": ["A"],
                "viktige_hendelser": ["Event1"],
                "viktige_tidspunkter": ["2025-01-01"],
                "viktige_personer": ["Person1"],
                "viktige_organisasjoner": ["Org1"],
                "viktige_steder": ["Place1"]
            }}
        }
        schema = {
            "tema": {"type": "array"},
            "viktige_hendelser": {"type": "array"},
            "viktige_tidspunkter": {"type": "array"},
            "viktige_personer": {"type": "array"},
            "viktige_organisasjoner": {"type": "array"},
            "viktige_steder": {"type": "array"}
        }
        
        analysis = analyze_document_group("test-doc", doc_data, schema)
        
        for field in COUNT_FIELDS:
            assert field in analysis["counts"]
            assert isinstance(analysis["counts"][field], list)
    
    def test_dokument_id_included(self):
        """Test that dokument_id is included in analysis."""
        doc_data = {
            "model1": {"response": {"hva_saken_gjelder": "Test"}}
        }
        schema = {"hva_saken_gjelder": {"type": "string"}}
        
        analysis = analyze_document_group("test-doc-123", doc_data, schema)
        
        assert analysis["dokument_id"] == "test-doc-123"

