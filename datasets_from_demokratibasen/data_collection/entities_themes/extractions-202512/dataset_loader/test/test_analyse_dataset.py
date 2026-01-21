import json
import sys
from pathlib import Path

# Set up path for imports
test_dir = Path(__file__).parent
dataset_loader_dir = test_dir.parent  # dataset_loader/
parent_dir = dataset_loader_dir.parent  # parent of dataset_loader/

# Add parent directory to path so dataset_loader can be found as a package
sys.path.insert(0, str(parent_dir))

# Import from the package
from dataset_loader.dataset_registry import register_dataset
from dataset_loader.analyse_dataset import analyze_dataset
from dataset_loader.dataset_adapter_202505 import DatasetAdapter202505
from dataset_loader.dataset_adapter_202510 import DatasetAdapter202510


def write_jsonl(path: Path, rows: list[dict]):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def test_analyze_dataset_202505_filters_and_counts(tmp_path: Path):
    # Create a small 202505-style dataset
    rows = [
        {
            "dokument_id": "A1",
            "kommune": 4601,
            "tekst": "Meeting agenda about budget 2025.",
            "doc_type": "meeting_agenda",
            "tittel": "Agenda",
        },
        {
            "dokument_id": "A2",
            "kommune": 4601,
            "tekst": "Short",  # too short -> filtered
            "doc_type": "meeting_minutes",
            "tittel": "Minutes",
        },
        {
            "dokument_id": "A3",
            "kommune": 1103,
            "tekst": "Case minutes for city planning.",
            "doc_type": "case_minutes",
            "tittel": "Case",
        },
        {
            "dokument_id": "A4",
            "kommune": 1103,
            "tekst": "Other type which should be excluded by adapter.",
            "doc_type": "other_type",  # excluded by adapter filter
            "tittel": "Other",
        },
    ]
    data_path = tmp_path / "dataset_202505.jsonl"
    write_jsonl(data_path, rows)

    # Register dataset pointing to this temp file
    register_dataset("test-202505", str(data_path.name), DatasetAdapter202505)

    result = analyze_dataset("test-202505", root=tmp_path)

    # Raw has all valid JSON lines
    assert result.total_documents_raw == 4
    # Filtered excludes: A2 (too short) and A4 (adapter should_include_document False)
    assert result.total_documents_filtered == 2

    # Type counting: filtered includes A1 (meeting_agenda), A3 (case_minutes)
    assert result.documents_with_type_filtered == 2
    assert result.documents_without_type_filtered == 0
    assert result.type_counter_filtered.get("meeting_agenda", 0) == 1
    assert result.type_counter_filtered.get("case_minutes", 0) == 1

    # Raw type counts include all types
    assert result.documents_with_type_raw == 4
    assert result.type_counter_raw.get("other_type", 0) == 1

    # Kommune distribution (filtered)
    assert result.kommune_counter.get(4601, 0) == 1
    assert result.kommune_counter.get(1103, 0) == 1


def test_analyze_dataset_202510_passes_all_and_counts(tmp_path: Path):
    # Create a small 202510-style dataset (processed_data.jsonl)
    rows = [
        {
            "input": "Some meaningful document text here (enough length).",
            "metadata": {
                "dokument_id": "B1",
                "kommune": 301,
                "doc_type": "meeting_minutes",
                "tittel": "Minutes title",
            },
            "output": "N/A",
        },
        {
            "input": "Another document with enough content to be included.",
            "metadata": {
                "dokument_id": "B2",
                "kommune": 5001,
                "doc_type": "case_presentation",
                "tittel": "Case title",
            },
            "output": "N/A",
        },
        {
            "input": "   ",  # whitespace -> filtered by loader
            "metadata": {
                "dokument_id": "B3",
                "kommune": 5001,
                "doc_type": "meeting_agenda",
                "tittel": "Agenda title",
            },
            "output": "N/A",
        },
    ]
    data_path = tmp_path / "dataset_202510.jsonl"
    write_jsonl(data_path, rows)

    register_dataset("test-202510", str(data_path.name), DatasetAdapter202510)

    result = analyze_dataset("test-202510", root=tmp_path)

    # Raw includes all three
    assert result.total_documents_raw == 3
    # Filtered excludes the whitespace-only one
    assert result.total_documents_filtered == 2

    # Filtered type counts
    assert result.documents_with_type_filtered == 2
    assert result.type_counter_filtered.get("meeting_minutes", 0) == 1
    assert result.type_counter_filtered.get("case_presentation", 0) == 1

    # Raw type counts include all three
    assert result.documents_with_type_raw == 3
    assert result.type_counter_raw.get("meeting_agenda", 0) == 1

