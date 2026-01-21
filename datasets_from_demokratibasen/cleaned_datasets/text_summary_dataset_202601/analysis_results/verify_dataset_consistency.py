#!/usr/bin/env python3
"""Verify dataset consistency: document IDs match across files and splits."""
import json
import sys
from pathlib import Path

def extract_doc_ids(file_path, is_embeddings_file=False, suppress_warnings=False):
    """Extract all dokument_id values from a JSONL file."""
    doc_ids = set()
    missing_count = 0
    json_errors = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                    # Embeddings file has dokument_id at top level, not in metadata
                    if is_embeddings_file:
                        doc_id = example.get('dokument_id')
                    else:
                        metadata = example.get('metadata', {})
                        doc_id = metadata.get('dokument_id')
                    
                    if doc_id:
                        doc_ids.add(doc_id)
                    else:
                        missing_count += 1
                except json.JSONDecodeError:
                    json_errors += 1
                    if not suppress_warnings and json_errors <= 5:
                        print(f'Warning: JSON decode error on line {line_num} in {file_path.name}', file=sys.stderr)
    except Exception as e:
        print(f'Error reading {file_path}: {e}', file=sys.stderr)
        sys.exit(1)
    
    if missing_count > 0 and not suppress_warnings:
        print(f'  Note: {missing_count} lines in {file_path.name} have no dokument_id', file=sys.stderr)
    if json_errors > 5 and not suppress_warnings:
        print(f'  Note: {json_errors} JSON decode errors in {file_path.name} (showing first 5)', file=sys.stderr)
    
    return doc_ids

def main():
    base_dir = Path('.')
    
    # File paths
    main_file = base_dir / 'text_summary_examples_ALL.jsonl'
    embeddings_file = base_dir / 'text_summary_examples_ALL_embeddings.jsonl'
    train_file = base_dir / 'text_summary_examples_ALL_train.jsonl'
    val_file = base_dir / 'text_summary_examples_ALL_val.jsonl'
    test_file = base_dir / 'text_summary_examples_ALL_test.jsonl'
    
    print('=' * 80)
    print('DATASET CONSISTENCY CHECK')
    print('=' * 80)
    print()
    
    all_checks_passed = True
    
    # Check 1: Main file vs embeddings file
    print('1. Checking main file vs embeddings file...')
    if not main_file.exists():
        print(f'  ERROR: {main_file.name} not found', file=sys.stderr)
        all_checks_passed = False
    elif not embeddings_file.exists():
        print(f'  ERROR: {embeddings_file.name} not found', file=sys.stderr)
        all_checks_passed = False
    else:
        main_ids = extract_doc_ids(main_file, is_embeddings_file=False, suppress_warnings=True)
        embeddings_ids = extract_doc_ids(embeddings_file, is_embeddings_file=True, suppress_warnings=True)
        
        only_in_main = main_ids - embeddings_ids
        only_in_embeddings = embeddings_ids - main_ids
        
        if only_in_main or only_in_embeddings:
            print(f'  FAILED: Document IDs do not match!')
            if only_in_main:
                print(f'    {len(only_in_main)} IDs in main file but not in embeddings: {list(only_in_main)[:5]}...')
            if only_in_embeddings:
                print(f'    {len(only_in_embeddings)} IDs in embeddings but not in main file: {list(only_in_embeddings)[:5]}...')
            all_checks_passed = False
        else:
            print(f'  PASSED: {len(main_ids):,} document IDs match between main file and embeddings file')
    print()
    
    # Check 2: Main file vs union of train/val/test (two-way equivalence)
    print('2. Checking main file vs union of train/val/test files...')
    if not all(f.exists() for f in [train_file, val_file, test_file]):
        print(f'  ERROR: One or more split files not found', file=sys.stderr)
        all_checks_passed = False
    else:
        main_ids = extract_doc_ids(main_file, suppress_warnings=True)
        train_ids = extract_doc_ids(train_file, suppress_warnings=True)
        val_ids = extract_doc_ids(val_file, suppress_warnings=True)
        test_ids = extract_doc_ids(test_file, suppress_warnings=True)
        
        split_union = train_ids | val_ids | test_ids
        
        only_in_main = main_ids - split_union
        only_in_splits = split_union - main_ids
        
        if only_in_main or only_in_splits:
            print(f'  FAILED: Document IDs do not match!')
            if only_in_main:
                print(f'    {len(only_in_main)} IDs in main file but not in any split: {list(only_in_main)[:5]}...')
            if only_in_splits:
                print(f'    {len(only_in_splits)} IDs in splits but not in main file: {list(only_in_splits)[:5]}...')
            all_checks_passed = False
        else:
            print(f'  PASSED: {len(main_ids):,} document IDs match between main file and union of splits')
    print()
    
    # Check 3: No overlapping IDs between train/val/test
    print('3. Checking for overlapping document IDs between train/val/test...')
    if not all(f.exists() for f in [train_file, val_file, test_file]):
        print(f'  ERROR: One or more split files not found', file=sys.stderr)
        all_checks_passed = False
    else:
        train_ids = extract_doc_ids(train_file, suppress_warnings=True)
        val_ids = extract_doc_ids(val_file, suppress_warnings=True)
        test_ids = extract_doc_ids(test_file, suppress_warnings=True)
        
        train_val_overlap = train_ids & val_ids
        train_test_overlap = train_ids & test_ids
        val_test_overlap = val_ids & test_ids
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            print(f'  FAILED: Found overlapping document IDs!')
            if train_val_overlap:
                print(f'    {len(train_val_overlap)} IDs in both train and val: {list(train_val_overlap)[:5]}...')
            if train_test_overlap:
                print(f'    {len(train_test_overlap)} IDs in both train and test: {list(train_test_overlap)[:5]}...')
            if val_test_overlap:
                print(f'    {len(val_test_overlap)} IDs in both val and test: {list(val_test_overlap)[:5]}...')
            all_checks_passed = False
        else:
            print(f'  PASSED: No overlapping document IDs between train/val/test')
            print(f'    Train: {len(train_ids):,} documents')
            print(f'    Val: {len(val_ids):,} documents')
            print(f'    Test: {len(test_ids):,} documents')
    print()
    
    # Check 4: Split ratios approximately 90-5-5%
    print('4. Checking train/val/test split ratios...')
    if not all(f.exists() for f in [train_file, val_file, test_file]):
        print(f'  ERROR: One or more split files not found', file=sys.stderr)
        all_checks_passed = False
    else:
        train_ids = extract_doc_ids(train_file, suppress_warnings=True)
        val_ids = extract_doc_ids(val_file, suppress_warnings=True)
        test_ids = extract_doc_ids(test_file, suppress_warnings=True)
        
        total = len(train_ids) + len(val_ids) + len(test_ids)
        if total == 0:
            print(f'  ERROR: Total documents is 0', file=sys.stderr)
            all_checks_passed = False
        else:
            train_pct = (len(train_ids) / total) * 100
            val_pct = (len(val_ids) / total) * 100
            test_pct = (len(test_ids) / total) * 100
            
            print(f'  Split ratios:')
            print(f'    Train: {len(train_ids):,} documents ({train_pct:.2f}%)')
            print(f'    Val: {len(val_ids):,} documents ({val_pct:.2f}%)')
            print(f'    Test: {len(test_ids):,} documents ({test_pct:.2f}%)')
            print(f'    Total: {total:,} documents')
            
            # Check if ratios are approximately 90-5-5% (allow 2% tolerance)
            target_train = 90.0
            target_val = 5.0
            target_test = 5.0
            tolerance = 2.0
            
            train_ok = abs(train_pct - target_train) <= tolerance
            val_ok = abs(val_pct - target_val) <= tolerance
            test_ok = abs(test_pct - target_test) <= tolerance
            
            if train_ok and val_ok and test_ok:
                print(f'  PASSED: Split ratios are approximately 90-5-5% (within {tolerance}% tolerance)')
            else:
                print(f'  FAILED: Split ratios are not approximately 90-5-5%')
                if not train_ok:
                    print(f'    Train: {train_pct:.2f}% (expected ~{target_train}% ± {tolerance}%)')
                if not val_ok:
                    print(f'    Val: {val_pct:.2f}% (expected ~{target_val}% ± {tolerance}%)')
                if not test_ok:
                    print(f'    Test: {test_pct:.2f}% (expected ~{target_test}% ± {tolerance}%)')
                all_checks_passed = False
    print()
    
    # Summary
    print('=' * 80)
    if all_checks_passed:
        print('ALL CHECKS PASSED ✓')
        sys.exit(0)
    else:
        print('SOME CHECKS FAILED ✗')
        sys.exit(1)

if __name__ == '__main__':
    main()
