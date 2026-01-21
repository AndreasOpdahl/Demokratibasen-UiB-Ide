#!/usr/bin/env python3
"""
Convert Bergen dataset from individual JSON files to text_summary_dataset format.

Reads JSON files from the FROM folder and converts them to the TO folder format,
filtering documents that don't meet quality criteria.
"""

import json
import re
import unicodedata
import random
from pathlib import Path
from typing import Dict, Any, Set, Tuple

# Paths
FROM_FOLDER = Path("/home/sinoa/Local/Tools/VSCode/Demokratibasen-UiB-Ide/datasets_from_demokratibasen/data_collection/entities_themes/extractions-202512/extracted-data/dataset-Bergen-2017-2023-all-input-tokens-max-1000-output-tokens-gpt-inferencing-202512/gpt-4o-mini")
TO_FOLDER = Path("/home/sinoa/Local/Tools/VSCode/Demokratibasen-UiB-Ide/datasets_from_demokratibasen/prepared_datasets/text_summary_dataset_bergen_2017_2023")
OUTPUT_FILE = TO_FOLDER / "text_summary_examples_bergen_2017_2023.jsonl"


def _count_alphanumeric(text: str) -> int:
    """Count the number of alphanumeric characters in a string."""
    return len(re.findall(r'[a-zA-Z0-9]', text))


def _check_ambiguous_unicode(text: str) -> Set[Tuple[str, str]]:
    """
    Check for ambiguous unicode characters in text.
    
    Only flags truly problematic characters, not legitimate Norwegian characters
    or common punctuation.
    
    Returns:
        Set of tuples (char, name) for ambiguous unicode characters found.
    """
    ambiguous_chars = set()
    
    # Legitimate Norwegian characters and common punctuation to allow
    # Norwegian: Å (U+00C5), å (U+00E5), Æ (U+00C6), æ (U+00E6), Ø (U+00D8), ø (U+00F8)
    # Common punctuation: en dash – (U+2013), em dash — (U+2014), section § (U+00A7)
    # bullet • (U+2022), quotes " " (U+201C, U+201D), ' (U+2019)
    allowed_code_points = {
        0x00C5, 0x00E5,  # Å, å
        0x00C6, 0x00E6,  # Æ, æ
        0x00D8, 0x00F8,  # Ø, ø
        0x2013,  # EN DASH
        0x2014,  # EM DASH
        0x00A7,  # SECTION SIGN
        0x2022,  # BULLET
        0x201C, 0x201D,  # LEFT/RIGHT DOUBLE QUOTATION MARK
        0x2019,  # RIGHT SINGLE QUOTATION MARK
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
    }
    
    for char in text:
        code_point = ord(char)
        
        # Skip ASCII characters (0-127)
        if code_point <= 127:
            continue
        
        # Skip allowed legitimate characters
        if code_point in allowed_code_points:
            continue
        
        # Check for specific problematic character ranges only
        # (not East Asian Width 'A' which includes legitimate characters)
        is_ambiguous = (
            (0x0430 <= code_point <= 0x044F) or  # Cyrillic small letters (homoglyphs)
            (0xFF00 <= code_point <= 0xFFEF) or  # Full-width characters
            (0x2000 <= code_point <= 0x200B) or  # Various spaces (en quad, em quad, thin space, etc.)
            code_point == 0x202F or               # Narrow no-break space
            code_point == 0x205F or               # Medium mathematical space
            code_point == 0x3000 or               # Ideographic space
            (0xFE00 <= code_point <= 0xFE0F) or   # Variation selectors
            (0x200C <= code_point <= 0x200D) or   # Zero-width non-joiner/joiner
            (0x2060 <= code_point <= 0x206F)      # Word joiner, invisible separator, etc.
        )
        
        if is_ambiguous:
            try:
                name = unicodedata.name(char, 'UNNAMED')
                ambiguous_chars.add((char, name))
            except ValueError:
                ambiguous_chars.add((char, f'U+{code_point:04X}'))
    
    return ambiguous_chars


def should_include_document(from_doc: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check if a document should be included in the output.
    
    Returns:
        (should_include, reason) tuple
    """
    text = str(from_doc.get("text", ""))
    oppsummering = str(from_doc.get("oppsummering", ""))
    
    # Check alphanumeric character count
    text_alnum = _count_alphanumeric(text)
    oppsum_alnum = _count_alphanumeric(oppsummering)
    
    if text_alnum < 10:
        return False, f"text has only {text_alnum} alphanumeric characters (< 10)"
    if oppsum_alnum < 10:
        return False, f"oppsummering has only {oppsum_alnum} alphanumeric characters (< 10)"
    
    # Check for ambiguous unicode
    text_ambiguous = _check_ambiguous_unicode(text)
    oppsum_ambiguous = _check_ambiguous_unicode(oppsummering)
    
    if text_ambiguous:
        return False, f"text contains ambiguous unicode characters: {list(text_ambiguous)[:3]}"
    if oppsum_ambiguous:
        return False, f"oppsummering contains ambiguous unicode characters: {list(oppsum_ambiguous)[:3]}"
    
    return True, ""


def convert_document(from_doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a document from FROM format to TO format.
    
    FROM format fields:
    - dok_id -> metadata.dokument_id
    - dok_type -> metadata.doc_type
    - kommune -> metadata.kommune
    - personer -> metadata.personer
    - nokkelord -> metadata.nokkelord
    - nyhetsverdi -> metadata.nyhetsverdi
    - text -> input
    - oppsummering -> output
    """
    to_doc = {
        "input": str(from_doc.get("text", "")),
        "output": str(from_doc.get("oppsummering", "")),
        "metadata": {
            "dokument_id": from_doc.get("dok_id", ""),
            "doc_type": from_doc.get("dok_type", ""),
            "kommune": from_doc.get("kommune"),
            "personer": from_doc.get("personer", ""),
            "nokkelord": from_doc.get("nokkelord", ""),
            "nyhetsverdi": from_doc.get("nyhetsverdi"),
        }
    }
    return to_doc


def main():
    """Main conversion function."""
    # Create output directory
    TO_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files from FROM folder
    json_files = sorted(FROM_FOLDER.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in FROM folder")
    
    # Process all files
    converted_docs = []
    skipped_count = 0
    skipped_reasons = {}
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                from_doc = json.load(f)
            
            should_include, reason = should_include_document(from_doc)
            
            if should_include:
                to_doc = convert_document(from_doc)
                converted_docs.append(to_doc)
            else:
                skipped_count += 1
                if reason not in skipped_reasons:
                    skipped_reasons[reason] = 0
                skipped_reasons[reason] += 1
                
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            skipped_count += 1
            if "Error" not in skipped_reasons:
                skipped_reasons["Error"] = 0
            skipped_reasons["Error"] += 1
    
    print(f"\nConversion complete:")
    print(f"  Converted: {len(converted_docs)} documents")
    print(f"  Skipped: {skipped_count} documents")
    if skipped_reasons:
        print(f"\nSkipped reasons:")
        for reason, count in sorted(skipped_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    
    # Shuffle for random partition
    random.seed(42)  # For reproducibility
    random.shuffle(converted_docs)
    
    # Partition: 90% train, 5% val, 5% test
    total = len(converted_docs)
    train_size = int(total * 0.90)
    val_size = int(total * 0.05)
    # test_size = total - train_size - val_size  # Remaining goes to test
    # test_size = total - train_size - val_size  # Remaining goes to test
    
    train_docs = converted_docs[:train_size]
    val_docs = converted_docs[train_size:train_size + val_size]
    test_docs = converted_docs[train_size + val_size:]
    
    # Save all documents to main file
    print(f"\nSaving all documents to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for doc in converted_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    # Save train split
    train_file = TO_FOLDER / "text_summary_examples_bergen_2017_2023_train.jsonl"
    print(f"Saving train split ({len(train_docs)} documents) to: {train_file}")
    with open(train_file, "w", encoding="utf-8") as f:
        for doc in train_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    # Save val split
    val_file = TO_FOLDER / "text_summary_examples_bergen_2017_2023_val.jsonl"
    print(f"Saving val split ({len(val_docs)} documents) to: {val_file}")
    with open(val_file, "w", encoding="utf-8") as f:
        for doc in val_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    # Save test split
    test_file = TO_FOLDER / "text_summary_examples_bergen_2017_2023_test.jsonl"
    print(f"Saving test split ({len(test_docs)} documents) to: {test_file}")
    with open(test_file, "w", encoding="utf-8") as f:
        for doc in test_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    # Report final counts
    print(f"\nFinal document counts:")
    print(f"  Total (all): {total} documents")
    if total > 0:
        print(f"  Train: {len(train_docs)} documents ({100*len(train_docs)/total:.1f}%)")
        print(f"  Val: {len(val_docs)} documents ({100*len(val_docs)/total:.1f}%)")
        print(f"  Test: {len(test_docs)} documents ({100*len(test_docs)/total:.1f}%)")
    else:
        print(f"  Train: {len(train_docs)} documents")
        print(f"  Val: {len(val_docs)} documents")
        print(f"  Test: {len(test_docs)} documents")


if __name__ == "__main__":
    main()
