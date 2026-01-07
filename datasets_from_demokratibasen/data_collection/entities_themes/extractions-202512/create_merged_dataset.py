#!/usr/bin/env python3
"""
Create a merged dataset from 1584 and 28081 CSV files in the same format as 43221 dataset.
"""

import csv
import json
import sys
from pathlib import Path

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

# File paths
base_dir = Path(__file__).parent.parent.parent.parent
file_1584 = base_dir / "raw_training_data" / "1584-url-tekst-oppsummering-from-prod-20251125.csv"
file_28081 = base_dir / "raw_training_data" / "28081-url-tekst-oppsummering-from-prod-20251215.csv"
output_dir = base_dir / "datasets" / "text_summary_dataset_29665_examples"
output_file = output_dir / "processed_data.jsonl"


def read_csv_file(csv_file):
    """Read a CSV file and return a list of dictionaries."""
    documents = []
    print(f"Reading {csv_file.name}...", file=sys.stderr)
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):  # Start at 2 because row 1 is header
            dok_id = row.get('dok_id', '').strip()
            if not dok_id:
                print(f"Warning: Row {row_num} in {csv_file.name} has no dok_id, skipping", file=sys.stderr)
                continue
            
            # Get text and summary
            text = row.get('text', '').strip()
            oppsummering = row.get('oppsummering', '').strip()
            
            # Skip if both text and summary are empty
            if not text and not oppsummering:
                print(f"Warning: Row {row_num} in {csv_file.name} has empty text and summary, skipping", file=sys.stderr)
                continue
            
            documents.append({
                'dok_id': dok_id,
                'kommune': row.get('kommune', '').strip(),
                'url': row.get('url', '').strip(),
                'dok_type': row.get('dok_type', '').strip(),
                'dok_tittel': row.get('dok_tittel', '').strip(),
                'text': text,
                'oppsum_tittel': row.get('oppsum_tittel', '').strip(),
                'oppsummering': oppsummering,
                'personer': row.get('personer', '').strip(),
                'nokkelord': row.get('nokkelord', '').strip(),
                'nyhetsverdi': row.get('nyhetsverdi', '').strip(),
            })
    
    print(f"  Read {len(documents)} documents from {csv_file.name}", file=sys.stderr)
    return documents


def convert_to_jsonl_format(documents):
    """Convert documents to JSONL format matching 43221 dataset."""
    jsonl_entries = []
    dok_ids_seen = set()
    duplicates = 0
    
    for doc in documents:
        dok_id = doc['dok_id']
        
        # Check for duplicates (keep first occurrence)
        if dok_id in dok_ids_seen:
            duplicates += 1
            continue
        dok_ids_seen.add(dok_id)
        
        # Build input field: "Dokument: {dok_tittel}\n\n{text}"
        dok_tittel = doc['dok_tittel'] or ''
        text = doc['text'] or ''
        if dok_tittel:
            input_text = f"Dokument: {dok_tittel}\n\n{text}"
        else:
            input_text = text
        
        # Build output field (summary)
        output_text = doc['oppsummering'] or ''
        
        # Parse kommune as integer if possible
        kommune = doc['kommune']
        try:
            kommune_int = int(kommune) if kommune else None
        except (ValueError, TypeError):
            kommune_int = None
        
        # Parse nyhetsverdi as float if possible
        nyhetsverdi = doc['nyhetsverdi']
        try:
            nyhetsverdi_float = float(nyhetsverdi) if nyhetsverdi else None
        except (ValueError, TypeError):
            nyhetsverdi_float = None
        
        # Create JSONL entry matching 43221 format
        entry = {
            "input": input_text,
            "output": output_text,
            "metadata": {
                "dokument_id": dok_id,
                "doc_type": doc['dok_type'] or None,
                "kommune": kommune_int,
                "personer": doc['personer'] or None,
                "nokkelord": doc['nokkelord'] or None,
                "nyhetsverdi": nyhetsverdi_float
            }
        }
        
        # Remove None values from metadata to match format
        entry["metadata"] = {k: v for k, v in entry["metadata"].items() if v is not None}
        
        jsonl_entries.append(entry)
    
    if duplicates > 0:
        print(f"Warning: Skipped {duplicates} duplicate dok_ids", file=sys.stderr)
    
    return jsonl_entries


def main():
    print("=" * 80, file=sys.stderr)
    print("Creating merged dataset from 1584 and 28081 CSV files", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print()
    
    # Read both CSV files
    docs_1584 = read_csv_file(file_1584)
    docs_28081 = read_csv_file(file_28081)
    
    print()
    print(f"Total documents from 1584: {len(docs_1584):,}", file=sys.stderr)
    print(f"Total documents from 28081: {len(docs_28081):,}", file=sys.stderr)
    
    # Merge documents
    all_docs = docs_1584 + docs_28081
    print(f"Total documents before deduplication: {len(all_docs):,}", file=sys.stderr)
    
    # Convert to JSONL format (deduplication happens here)
    jsonl_entries = convert_to_jsonl_format(all_docs)
    
    print()
    print(f"Total unique documents: {len(jsonl_entries):,}", file=sys.stderr)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write JSONL file
    print(f"\nWriting to {output_file}...", file=sys.stderr)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✅ Successfully created dataset with {len(jsonl_entries):,} documents", file=sys.stderr)
    print(f"   Output: {output_file}", file=sys.stderr)
    
    # Print summary statistics
    print()
    print("=" * 80, file=sys.stderr)
    print("Summary Statistics", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    
    empty_inputs = sum(1 for e in jsonl_entries if not e['input'].strip())
    empty_outputs = sum(1 for e in jsonl_entries if not e['output'].strip())
    
    print(f"Total entries: {len(jsonl_entries):,}", file=sys.stderr)
    print(f"Empty inputs: {empty_inputs:,}", file=sys.stderr)
    print(f"Empty outputs: {empty_outputs:,}", file=sys.stderr)
    print(f"Entries with both input and output: {len(jsonl_entries) - empty_inputs - empty_outputs:,}", file=sys.stderr)


if __name__ == "__main__":
    main()

