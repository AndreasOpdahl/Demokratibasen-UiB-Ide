#!/usr/bin/env python3
"""
Clean text summary datasets by:
1. Replacing ambiguous unicode characters with spaces
2. Cleaning whitespace (removing repeated whitespace, normalizing line feeds)
3. Removing duplicate documents based on duplicates JSON files
"""

import sys
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Set, Tuple, List, Any
from collections import defaultdict

# LINE FEED character
LINE_FEED = '\n'


def _is_ambiguous_unicode_char(char: str) -> bool:
    """
    Check if a character is ambiguous unicode (same logic as analyse_dataset.py).
    
    Returns True if the character should be replaced with a space.
    """
    code_point = ord(char)
    
    # Skip ASCII characters (0-127)
    if code_point <= 127:
        return False
    
    # Allowed legitimate characters (same as analyse_dataset.py)
    allowed_code_points = {
        0x00A7,  # SECTION SIGN
        0x00AE,  # REGISTERED SIGN
        0x00B0,  # DEGREE SIGN
        0x00B1,  # PLUS-MINUS SIGN
        0x00B2,  # SUPERSCRIPT TWO
        0x00B7,  # MIDDLE DOT
        0x00C5, 0x00E5,  # Å, å
        0x00C6, 0x00E6,  # Æ, æ
        0x00D8, 0x00F8,  # Ø, ø
        0x00F7,  # DIVISION SIGN
        0x0111,  # LATIN SMALL LETTER D WITH STROKE (đ)
        0x2010,  # HYPHEN
        0x2013,  # EN DASH
        0x2014,  # EM DASH
        0x2018, 0x2019,  # LEFT/RIGHT SINGLE QUOTATION MARK
        0x201C, 0x201D,  # LEFT/RIGHT DOUBLE QUOTATION MARK
        0x2022,  # BULLET
        0x2032,  # PRIME
        0x20AC,  # EURO SIGN
        0x2248,  # ALMOST EQUAL TO
        0x2264,  # LESS-THAN OR EQUAL TO
        0x2265,  # GREATER-THAN OR EQUAL TO
        0x25A0,  # BLACK SQUARE
        0x25CB,  # WHITE CIRCLE
        0x25CF,  # BLACK CIRCLE
        0x03BC,  # GREEK SMALL LETTER MU
        # Latin characters with diacritics
        0x00E0, 0x00E1, 0x00E2, 0x00E3, 0x00E4,  # à, á, â, ã, ä
        0x00E8, 0x00E9, 0x00EA, 0x00EB,  # è, é, ê, ë
        0x00EC, 0x00ED, 0x00EE, 0x00EF,  # ì, í, î, ï
        0x00F2, 0x00F3, 0x00F4, 0x00F5, 0x00F6,  # ò, ó, ô, õ, ö
        0x00F9, 0x00FA, 0x00FB, 0x00FC,  # ù, ú, û, ü
        0x00C0, 0x00C1, 0x00C2, 0x00C3, 0x00C4,  # À, Á, Â, Ã, Ä
        0x00C8, 0x00C9, 0x00CA, 0x00CB,  # È, É, Ê, Ë
        0x00CC, 0x00CD, 0x00CE, 0x00CF,  # Ì, Í, Î, Ï
        0x00D2, 0x00D3, 0x00D4, 0x00D5, 0x00D6,  # Ò, Ó, Ô, Õ, Ö
        0x00D9, 0x00DA, 0x00DB, 0x00DC,  # Ù, Ú, Û, Ü
        # Combining diacritics
        0x0301,  # COMBINING ACUTE ACCENT
        0x0308,  # COMBINING DIAERESIS
        0x030A,  # COMBINING RING ABOVE
    }
    
    # Skip allowed legitimate characters
    if code_point in allowed_code_points:
        return False
    
    # Check for specific problematic character ranges
    is_ambiguous = (
        (0x0430 <= code_point <= 0x044F) or  # Cyrillic small letters (homoglyphs)
        (0xFF00 <= code_point <= 0xFFEF) or  # Full-width characters
        (0x2000 <= code_point <= 0x200B) or  # Various spaces (en quad, em quad, thin space, etc.)
        code_point == 0x202F or               # Narrow no-break space
        code_point == 0x205F or               # Medium mathematical space
        code_point == 0x3000 or               # Ideographic space
        (0xFE00 <= code_point <= 0xFE0F) or   # Variation selectors
        (0x200C <= code_point <= 0x200D) or   # Zero-width non-joiner/joiner
        (0x2060 <= code_point <= 0x206F) or   # Word joiner, invisible separator, etc.
        (0xF000 <= code_point <= 0xFFFF) or   # Private Use Area
        code_point == 0xFFFD                  # REPLACEMENT CHARACTER (indicates encoding error)
    )
    
    return is_ambiguous


def _replace_ambiguous_unicode(text: str) -> Tuple[str, int]:
    """
    Replace ambiguous unicode characters with spaces.
    
    Returns:
        (cleaned_text, num_chars_replaced)
    """
    result = []
    num_replaced = 0
    
    for char in text:
        if _is_ambiguous_unicode_char(char):
            result.append(' ')
            num_replaced += 1
        else:
            result.append(char)
    
    return ''.join(result), num_replaced


def _clean_whitespace(text: str) -> Tuple[str, int]:
    """
    Clean whitespace in text:
    1. Replace >=2 consecutive whitespace characters with single space, but keep LINE FEEDs in place
    2. Remove whitespace characters between two LINE FEEDs
    3. Replace >2 consecutive LINE FEEDs with 2 LINE FEEDs
    
    Returns:
        (cleaned_text, num_chars_removed)
    """
    original_len = len(text)
    
    # Step 1: Replace >=2 consecutive whitespace (but not LINE FEED) with single space
    # This keeps LINE FEEDs in place while collapsing other whitespace
    # Pattern: whitespace characters (space, tab, etc.) but not LINE FEED, repeated 2+ times
    text = re.sub(r'[ \t\r\f\v]{2,}', ' ', text)
    
    # Step 2: Remove whitespace between two LINE FEEDs
    # Pattern: LINE FEED, followed by whitespace (but not LINE FEED), followed by LINE FEED
    text = re.sub(r'\n[ \t\r\f\v]+\n', '\n\n', text)
    
    # Step 3: Replace >2 consecutive LINE FEEDs with 2 LINE FEEDs
    # Pattern: 3 or more consecutive LINE FEEDs
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    cleaned_len = len(text)
    num_removed = original_len - cleaned_len
    
    return text, num_removed


def _load_duplicate_documents(duplicates_file: Path) -> Set[str]:
    """
    Load document IDs that should be removed (duplicates) from a duplicates JSON file.
    
    Only considers clusters with "comparison_type": "input_and_summary_comparison".
    For each cluster, keeps the first document and marks the rest as duplicates.
    
    Returns:
        Set of dokument_id strings that should be removed
    """
    duplicates_to_remove = set()
    
    if not duplicates_file.exists():
        return duplicates_to_remove
    
    try:
        with open(duplicates_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        clusters = data.get('clusters', [])
        
        for cluster in clusters:
            # Only process clusters with "input_and_summary_comparison"
            if cluster.get('comparison_type') != 'input_and_summary_comparison':
                continue
            
            documents = cluster.get('documents', [])
            if len(documents) < 2:
                continue
            
            # Keep the first document, mark the rest as duplicates
            for doc in documents[1:]:
                doc_id = doc.get('dokument_id')
                if doc_id:
                    duplicates_to_remove.add(doc_id)
        
        print(f"  Loaded {len(duplicates_to_remove)} duplicate document IDs from {duplicates_file.name}", file=sys.stderr)
    
    except Exception as e:
        print(f"  Warning: Failed to load duplicates from {duplicates_file}: {e}", file=sys.stderr)
    
    return duplicates_to_remove


def _clean_text_field(text: str, stats: Dict[str, Any]) -> str:
    """
    Clean a text field (input or output) by replacing ambiguous unicode and cleaning whitespace.
    
    Updates stats with counts of removed characters.
    """
    original_text = text
    
    # Replace ambiguous unicode
    text, ambiguous_count = _replace_ambiguous_unicode(text)
    stats['ambiguous_unicode_chars'] += ambiguous_count
    
    # Clean whitespace
    text, whitespace_count = _clean_whitespace(text)
    stats['whitespace_chars'] += whitespace_count
    
    return text


def _clean_example(example: Dict[str, Any], duplicates_to_remove: Set[str], stats: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Clean a single example from the dataset.
    
    Returns:
        (cleaned_example, should_keep)
    """
    metadata = example.get('metadata', {})
    doc_id = metadata.get('dokument_id')
    
    # Check if this document should be removed as a duplicate
    if doc_id and doc_id in duplicates_to_remove:
        stats['duplicate_documents'] += 1
        return None, False
    
    # Clean input field
    if 'input' in example:
        example['input'] = _clean_text_field(str(example['input']), stats)
    
    # Clean output field
    if 'output' in example:
        example['output'] = _clean_text_field(str(example['output']), stats)
    
    return example, True


def _process_dataset_file(
    input_file: Path,
    output_file: Path,
    duplicates_to_remove: Set[str],
    stats: Dict[str, Any]
) -> None:
    """
    Process a single dataset file (JSONL format).
    """
    print(f"  Processing {input_file.name}...", file=sys.stderr)
    
    file_stats = {
        'ambiguous_unicode_chars': 0,
        'whitespace_chars': 0,
        'duplicate_documents': 0,
        'total_examples': 0,
        'kept_examples': 0
    }
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            try:
                example = json.loads(line)
                file_stats['total_examples'] += 1
                
                cleaned_example, should_keep = _clean_example(example, duplicates_to_remove, file_stats)
                
                if should_keep and cleaned_example is not None:
                    f_out.write(json.dumps(cleaned_example, ensure_ascii=False) + '\n')
                    file_stats['kept_examples'] += 1
            
            except json.JSONDecodeError as e:
                print(f"    Warning: Failed to parse JSON line: {e}", file=sys.stderr)
                continue
    
    # Update global stats
    stats['ambiguous_unicode_chars'] += file_stats['ambiguous_unicode_chars']
    stats['whitespace_chars'] += file_stats['whitespace_chars']
    stats['duplicate_documents'] += file_stats['duplicate_documents']
    stats['total_examples'] += file_stats['total_examples']
    stats['kept_examples'] += file_stats['kept_examples']
    
    print(f"    Kept {file_stats['kept_examples']}/{file_stats['total_examples']} examples", file=sys.stderr)
    print(f"    Removed {file_stats['ambiguous_unicode_chars']} ambiguous unicode chars, {file_stats['whitespace_chars']} whitespace chars", file=sys.stderr)
    print(f"    Removed {file_stats['duplicate_documents']} duplicate documents", file=sys.stderr)


def main():
    datasets_dir = Path(__file__).parent / 'prepared_datasets'
    cleaned_dir = Path(__file__).parent / 'cleaned_datasets'
    
    if not datasets_dir.exists():
        print(f"Error: Datasets directory not found: {datasets_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Only process text_summary_dataset_ALL_examples folder
    dataset_folder = datasets_dir / 'text_summary_dataset_ALL_examples'
    
    if not dataset_folder.exists():
        print(f"Error: Dataset folder not found: {dataset_folder}", file=sys.stderr)
        sys.exit(1)
    
    if not dataset_folder.is_dir():
        print(f"Error: {dataset_folder} is not a directory", file=sys.stderr)
        sys.exit(1)
    
    # Create cleaned_datasets directory
    cleaned_dir.mkdir(exist_ok=True)
    
    print(f"Processing {dataset_folder.name}...", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    
    # Global statistics
    global_stats = {
        'ambiguous_unicode_chars': 0,
        'whitespace_chars': 0,
        'duplicate_documents': 0,
        'total_examples': 0,
        'kept_examples': 0
    }
    
    # Create corresponding folder in cleaned_datasets
    cleaned_folder = cleaned_dir / dataset_folder.name
    cleaned_folder.mkdir(exist_ok=True)
    
    # Find duplicates file
    duplicates_files = list(dataset_folder.glob('*_duplicates_threshold_10.json'))
    duplicates_to_remove = set()
    
    if duplicates_files:
        # Use the first matching file (should only be one per folder)
        duplicates_to_remove = _load_duplicate_documents(duplicates_files[0])
    else:
        print(f"  No duplicates file found (looking for *_duplicates_threshold_10.json)", file=sys.stderr)
    
    # Find all text_summary_examples_*.jsonl files (excluding embeddings files)
    jsonl_files = sorted([f for f in dataset_folder.iterdir() 
                          if f.is_file() and f.name.startswith('text_summary_examples_') 
                          and f.name.endswith('.jsonl')
                          and not f.name.endswith('_embeddings.jsonl')])
    
    if not jsonl_files:
        print(f"  No text_summary_examples_*.jsonl files found", file=sys.stderr)
        sys.exit(1)
    
    # Process each JSONL file
    for jsonl_file in jsonl_files:
        output_file = cleaned_folder / jsonl_file.name
        _process_dataset_file(jsonl_file, output_file, duplicates_to_remove, global_stats)
    
    # Print final statistics
    print("\n" + "=" * 80, file=sys.stderr)
    print("CLEANING SUMMARY", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"Total examples processed: {global_stats['total_examples']:,}", file=sys.stderr)
    print(f"Examples kept: {global_stats['kept_examples']:,}", file=sys.stderr)
    print(f"Examples removed (duplicates): {global_stats['duplicate_documents']:,}", file=sys.stderr)
    print(f"Total ambiguous unicode characters replaced: {global_stats['ambiguous_unicode_chars']:,}", file=sys.stderr)
    print(f"Total whitespace characters removed: {global_stats['whitespace_chars']:,}", file=sys.stderr)
    
    if global_stats['total_examples'] > 0:
        avg_ambiguous = global_stats['ambiguous_unicode_chars'] / global_stats['total_examples']
        avg_whitespace = global_stats['whitespace_chars'] / global_stats['total_examples']
        print(f"\nAverage ambiguous unicode characters removed per example: {avg_ambiguous:.2f}", file=sys.stderr)
        print(f"Average whitespace characters removed per example: {avg_whitespace:.2f}", file=sys.stderr)
    
    if global_stats['duplicate_documents'] > 0:
        print(f"\nTotal duplicate documents removed: {global_stats['duplicate_documents']:,}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"\nCleaned datasets saved to: {cleaned_dir}", file=sys.stderr)


if __name__ == '__main__':
    main()

