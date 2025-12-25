#!/usr/bin/env python3
"""
Test script for dataset-202505 adapter.

Analyzes the dataset to answer:
- How many documents does the dataset contain?
- Why are doc_types represented?
- How many documents of each type?
"""

import sys
from collections import Counter
from pathlib import Path

# Add the directory containing this script to the path so we can import dataset_loader
sys.path.insert(0, str(Path(__file__).parent))

from dataset_loader import DatasetLoader, DatasetAdapter202505


def main():
    """Analyze the dataset-202505 dataset."""
    print("=" * 80)
    print("Dataset-202505 Analysis")
    print("=" * 80)
    print()
    
    # Initialize the dataset loader
    try:
        loader = DatasetLoader("dataset-202505")
        print(f"Dataset file: {loader.file_path}")
        print(f"Adapter class: {loader.adapter_class.__name__}")
        print()
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Statistics
    total_documents = 0
    documents_with_type = 0
    documents_without_type = 0
    type_counter = Counter()
    kommune_counter = Counter()
    
    # Also collect sample documents to understand the data structure
    sample_docs = []
    sample_docs_by_type = {}
    
    print("Processing documents...")
    print()
    
    try:
        for doc_id, kommune_nummer, kommune_navn, text in loader():
            total_documents += 1
            
            # Get document type using the adapter
            adapter = DatasetAdapter202505()
            # We need to re-read the document to get the type
            # Since loader() yields processed data, we need to read raw data
            # For now, let's track what we can from the yielded data
            
            # Track kommune distribution
            if kommune_nummer is not None:
                kommune_counter[kommune_nummer] += 1
            
            # For type information, we need to read the raw document
            # Let's do a second pass to get type information
            # But first, let's collect basic stats
            
            # Collect a few samples
            if len(sample_docs) < 5:
                sample_docs.append({
                    "doc_id": doc_id,
                    "kommune_nummer": kommune_nummer,
                    "kommune_navn": kommune_navn,
                    "text_length": len(text)
                })
        
        # Now do a second pass to get document types from raw data
        print("Analyzing document types...")
        print()
        
        with open(loader.file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    import json
                    doc = json.loads(line)
                    normalized = adapter.normalize(doc)
                    
                    doc_type = normalized.get("dok_type")
                    
                    if doc_type:
                        documents_with_type += 1
                        type_counter[doc_type] += 1
                        
                        # Collect samples by type
                        if doc_type not in sample_docs_by_type and len(sample_docs_by_type) < 10:
                            sample_docs_by_type[doc_type] = {
                                "doc_id": normalized.get("dok_id"),
                                "dok_type": doc_type,
                                "dok_tittel": normalized.get("dok_tittel"),
                                "kommune_navn": normalized.get("kommune_navn")
                            }
                    else:
                        documents_without_type += 1
                        
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
                    print(f"  Title: {sample['dok_tittel'][:80]}...")
                print(f"  Kommune: {sample['kommune_navn']}")
                print()
        
        print("=" * 80)
        print("KOMMUNE DISTRIBUTION (top 10)")
        print("=" * 80)
        print()
        for kommune_nummer, count in kommune_counter.most_common(10):
            kommune_navn = adapter.get_kommune_navn({"kommune": kommune_nummer})
            print(f"  {kommune_nummer} ({kommune_navn}): {count:,} documents")
        print()
        
    except Exception as e:
        print(f"Error processing dataset: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

