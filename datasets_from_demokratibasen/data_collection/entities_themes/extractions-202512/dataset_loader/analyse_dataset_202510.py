#!/usr/bin/env python3
"""
Test script for dataset-202510 adapter.

Analyzes the dataset to answer:
- How many documents does the dataset contain?
- Why are doc_types represented?
- How many documents of each type?
"""

import sys
import json
from collections import Counter
from pathlib import Path

# Add the directory containing this script to the path so we can import dataset_loader
sys.path.insert(0, str(Path(__file__).parent))

from dataset_loader import DatasetLoader, DatasetAdapter202510


def main():
    """Analyze the dataset-202510 dataset."""
    print("=" * 80)
    print("Dataset-202510 Analysis")
    print("=" * 80)
    print()
    
    # Initialize the dataset loader
    try:
        loader = DatasetLoader("dataset-202510")
        print(f"Dataset file: {loader.file_path}")
        print(f"Adapter class: {loader.adapter_class.__name__}")
        print()
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Statistics - track both filtered and raw documents
    total_documents_filtered = 0
    total_documents_raw = 0
    documents_with_type_filtered = 0
    documents_without_type_filtered = 0
    documents_with_type_raw = 0
    documents_without_type_raw = 0
    type_counter_filtered = Counter()
    type_counter_raw = Counter()
    kommune_counter = Counter()
    
    # Also collect sample documents to understand the data structure
    sample_docs = []
    sample_docs_by_type = {}
    
    # Create a set of document IDs that pass filtering
    filtered_doc_ids = set()
    
    print("Processing documents (first pass - filtered)...")
    print()
    
    adapter = DatasetAdapter202510()
    
    try:
        # First pass: count filtered documents and track their IDs
        for doc_id, kommune_nummer, kommune_navn, text in loader():
            total_documents_filtered += 1
            filtered_doc_ids.add(doc_id)
            
            # Track kommune distribution
            if kommune_nummer is not None:
                kommune_counter[kommune_nummer] += 1
            
            # Collect a few samples
            if len(sample_docs) < 5:
                sample_docs.append({
                    "doc_id": doc_id,
                    "kommune_nummer": kommune_nummer,
                    "kommune_navn": kommune_navn,
                    "text_length": len(text)
                })
        
        # Second pass: analyze document types from raw data
        print("Analyzing document types (second pass - raw data)...")
        print()
        
        with open(loader.file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    doc = json.loads(line)
                    normalized = adapter.normalize(doc)
                    doc_id = normalized.get("dok_id")
                    total_documents_raw += 1
                    
                    doc_type = normalized.get("dok_type")
                    
                    # Count in raw data
                    if doc_type:
                        documents_with_type_raw += 1
                        type_counter_raw[doc_type] += 1
                    else:
                        documents_without_type_raw += 1
                    
                    # Count in filtered data (only if this document passed filtering)
                    if doc_id in filtered_doc_ids:
                        if doc_type:
                            documents_with_type_filtered += 1
                            type_counter_filtered[doc_type] += 1
                        else:
                            documents_without_type_filtered += 1
                        
                        # Collect samples by type (only from filtered documents)
                        if doc_type and doc_type not in sample_docs_by_type and len(sample_docs_by_type) < 10:
                            sample_docs_by_type[doc_type] = {
                                "doc_id": doc_id,
                                "dok_type": doc_type,
                                "dok_tittel": normalized.get("dok_tittel"),
                                "kommune_navn": normalized.get("kommune_navn")
                            }
                        
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}", file=sys.stderr)
                    continue
        
        # Print results
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        print()
        
        print(f"Total documents in raw file: {total_documents_raw:,}")
        print(f"Total documents (after filtering): {total_documents_filtered:,}")
        if total_documents_raw > 0:
            print(f"Documents filtered out: {total_documents_raw - total_documents_filtered:,}")
        print()
        
        print("FILTERED DOCUMENTS (used in processing):")
        print(f"  Documents with dok_type: {documents_with_type_filtered:,}")
        print(f"  Documents without dok_type: {documents_without_type_filtered:,}")
        print()
        
        print("RAW DOCUMENTS (all in file):")
        print(f"  Documents with dok_type: {documents_with_type_raw:,}")
        print(f"  Documents without dok_type: {documents_without_type_raw:,}")
        print()
        
        print("=" * 80)
        print("DOCUMENT TYPES DISTRIBUTION (FILTERED DOCUMENTS)")
        print("=" * 80)
        print()
        
        if type_counter_filtered:
            for doc_type, count in type_counter_filtered.most_common():
                percentage = (count / documents_with_type_filtered * 100) if documents_with_type_filtered > 0 else 0
                print(f"  {doc_type}: {count:,} documents ({percentage:.1f}%)")
        else:
            print("  No document types found.")
        print()
        
        print("=" * 80)
        print("WHY ARE DOC_TYPES REPRESENTED?")
        print("=" * 80)
        print()
        print("Document types (dok_type/doc_type) categorize documents by their purpose or content type.")
        print("This allows for:")
        print("  - Filtering documents by type during processing")
        print("  - Type-specific analysis and statistics")
        print("  - Different processing strategies for different document types")
        print("  - Quality control and validation per document type")
        print()
        
        if sample_docs_by_type:
            print("=" * 80)
            print("SAMPLE DOCUMENTS BY TYPE")
            print("=" * 80)
            print()
            for doc_type, sample in sample_docs_by_type.items():
                print(f"Type: {doc_type}")
                print(f"  Document ID: {sample['doc_id']}")
                if sample.get('dok_tittel'):
                    title = sample['dok_tittel']
                    if len(title) > 80:
                        title = title[:80] + "..."
                    print(f"  Title: {title}")
                print(f"  Kommune: {sample['kommune_navn']}")
                print()
        
        print("=" * 80)
        print("KOMMUNE DISTRIBUTION (top 10)")
        print("=" * 80)
        print()
        for kommune_nummer, count in kommune_counter.most_common(10):
            kommune_navn = adapter.get_kommune_navn({"metadata": {"kommune": kommune_nummer}})
            print(f"  {kommune_nummer} ({kommune_navn}): {count:,} documents")
        print()
        
    except Exception as e:
        print(f"Error processing dataset: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
