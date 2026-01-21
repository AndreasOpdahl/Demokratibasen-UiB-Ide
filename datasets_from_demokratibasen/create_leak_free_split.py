#!/usr/bin/env python3
"""
Create a leak-free train-val-test split by:
1. Collecting document IDs from old train/val files
2. Allocating old train IDs to new train
3. Allocating old val IDs to new val
4. Filling the rest to achieve 90%-5%-5% split
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, Set

def extract_doc_ids(file_path: Path) -> Set[str]:
    """Extract dokument_id values from a JSONL file."""
    if not file_path.exists():
        return set()
    doc_ids = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                metadata = example.get('metadata', {})
                doc_id = metadata.get('dokument_id')
                if doc_id:
                    doc_ids.add(doc_id)
            except json.JSONDecodeError:
                continue
    return doc_ids

def load_jsonl(file_path: Path) -> Dict[str, dict]:
    """Load JSONL file into a dictionary keyed by dokument_id."""
    if not file_path.exists():
        return {}
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                metadata = example.get('metadata', {})
                doc_id = metadata.get('dokument_id')
                if doc_id:
                    data[doc_id] = example
            except json.JSONDecodeError:
                continue
    return data

def save_jsonl(data: Dict[str, dict], file_path: Path):
    """Save dictionary to JSONL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for doc_id in sorted(data.keys()):
            f.write(json.dumps(data[doc_id], ensure_ascii=False) + '\n')

def main():
    # File paths
    base_dir = Path("datasets_from_demokratibasen/prepared_datasets")
    
    # Old train files
    old_train_file_1 = base_dir / "OLD_text_summary_dataset_12811_examples_gpt_35_turbo/text_summary_examples_202505_and_06_train.jsonl"
    old_train_file_2 = base_dir / "text_summary_dataset_202505_to_10/text_summary_examples_202505_to_10_train.jsonl"
    
    # Old val files
    old_val_file_1 = base_dir / "OLD_text_summary_dataset_12811_examples_gpt_35_turbo/text_summary_examples_202505_and_06_val.jsonl"
    old_val_file_2 = base_dir / "text_summary_dataset_202505_to_10/text_summary_examples_202505_to_10_val.jsonl"
    
    # Source file
    source_file = base_dir / "text_summary_dataset_ALL_examples/text_summary_examples_ALL.jsonl"
    
    # Output files
    output_dir = base_dir / "text_summary_dataset_ALL_examples"
    new_train_file = output_dir / "text_summary_examples_ALL_train.jsonl"
    new_val_file = output_dir / "text_summary_examples_ALL_val.jsonl"
    new_test_file = output_dir / "text_summary_examples_ALL_test.jsonl"
    
    print("=" * 70)
    print("STEP 1: Collect OLD_TRAIN and OLD_VAL document IDs")
    print("=" * 70)
    
    # Collect OLD_TRAIN IDs
    old_train_ids_1 = extract_doc_ids(old_train_file_1)
    old_train_ids_2 = extract_doc_ids(old_train_file_2)
    OLD_TRAIN = old_train_ids_1 | old_train_ids_2
    
    print(f"Old train file 1: {len(old_train_ids_1)} document IDs")
    print(f"Old train file 2: {len(old_train_ids_2)} document IDs")
    print(f"OLD_TRAIN (union): {len(OLD_TRAIN)} document IDs")
    
    # Collect OLD_VAL IDs
    old_val_ids_1 = extract_doc_ids(old_val_file_1)
    old_val_ids_2 = extract_doc_ids(old_val_file_2)
    OLD_VAL = old_val_ids_1 | old_val_ids_2
    
    print(f"\nOld val file 1: {len(old_val_ids_1)} document IDs")
    print(f"Old val file 2: {len(old_val_ids_2)} document IDs")
    print(f"OLD_VAL (union): {len(OLD_VAL)} document IDs")
    
    # Check for overlaps between OLD_TRAIN and OLD_VAL
    train_val_overlap = OLD_TRAIN & OLD_VAL
    if train_val_overlap:
        print(f"\nWARNING: {len(train_val_overlap)} document IDs appear in both OLD_TRAIN and OLD_VAL!")
        print(f"These will be allocated to train (OLD_TRAIN takes precedence)")
        # Remove from OLD_VAL
        OLD_VAL = OLD_VAL - OLD_TRAIN
        print(f"OLD_VAL after removing overlap: {len(OLD_VAL)} document IDs")
    
    print("\n" + "=" * 70)
    print("STEP 2: Load source dataset")
    print("=" * 70)
    
    source_data = load_jsonl(source_file)
    print(f"Source file: {len(source_data)} document IDs")
    
    # Check coverage
    source_ids_in_old_train = set(source_data.keys()) & OLD_TRAIN
    source_ids_in_old_val = set(source_data.keys()) & OLD_VAL
    
    print(f"IDs from OLD_TRAIN in source: {len(source_ids_in_old_train)}")
    print(f"IDs from OLD_VAL in source: {len(source_ids_in_old_val)}")
    
    print("\n" + "=" * 70)
    print("STEP 3: Allocate documents to new split")
    print("=" * 70)
    
    # Initialize new split
    new_train_data = {}
    new_val_data = {}
    new_test_data = {}
    
    # Rule 1: All OLD_TRAIN IDs go to new train
    for doc_id in OLD_TRAIN:
        if doc_id in source_data:
            new_train_data[doc_id] = source_data[doc_id]
    
    print(f"Allocated {len(new_train_data)} documents from OLD_TRAIN to new train")
    
    # Rule 2: All OLD_VAL IDs go to new val (and NOT to train or test)
    for doc_id in OLD_VAL:
        if doc_id in source_data:
            if doc_id not in new_train_data:  # Should not happen, but check anyway
                new_val_data[doc_id] = source_data[doc_id]
    
    print(f"Allocated {len(new_val_data)} documents from OLD_VAL to new val")
    
    # Remaining documents (not in OLD_TRAIN or OLD_VAL)
    remaining_ids = set(source_data.keys()) - OLD_TRAIN - OLD_VAL
    print(f"Remaining documents to allocate: {len(remaining_ids)}")
    
    # Calculate target sizes for 90%-5%-5% split
    total_docs = len(source_data)
    target_train_pct = 90.0
    target_val_pct = 5.0
    target_test_pct = 5.0
    
    target_train_count = int(total_docs * target_train_pct / 100)
    target_val_count = int(total_docs * target_val_pct / 100)
    target_test_count = total_docs - target_train_count - target_val_count
    
    print(f"\nTarget split (90%-5%-5%):")
    print(f"  Train: {target_train_count} ({target_train_pct}%)")
    print(f"  Val: {target_val_count} ({target_val_pct}%)")
    print(f"  Test: {target_test_count} ({target_test_pct}%)")
    print(f"  Total: {total_docs}")
    
    # Calculate how many more documents we need in each set
    train_needed = max(0, target_train_count - len(new_train_data))
    val_needed = max(0, target_val_count - len(new_val_data))
    test_needed = target_test_count - len(new_test_data)
    
    print(f"\nAdditional documents needed:")
    print(f"  Train: {train_needed}")
    print(f"  Val: {val_needed}")
    print(f"  Test: {test_needed}")
    print(f"  Total needed: {train_needed + val_needed + test_needed}")
    print(f"  Available: {len(remaining_ids)}")
    
    if train_needed + val_needed + test_needed > len(remaining_ids):
        print(f"\nWARNING: Not enough remaining documents to fill targets!")
        print(f"Will allocate all remaining documents to train")
        train_needed = len(remaining_ids)
        val_needed = 0
        test_needed = 0
    
    # Randomly allocate remaining documents
    remaining_list = sorted(list(remaining_ids))
    random.seed(42)  # For reproducibility
    random.shuffle(remaining_list)
    
    # Allocate to val first (to meet val target)
    for i in range(min(val_needed, len(remaining_list))):
        doc_id = remaining_list[i]
        new_val_data[doc_id] = source_data[doc_id]
    
    # Allocate to test
    test_start = val_needed
    test_end = test_start + test_needed
    for i in range(test_start, min(test_end, len(remaining_list))):
        doc_id = remaining_list[i]
        new_test_data[doc_id] = source_data[doc_id]
    
    # Allocate remaining to train
    train_start = test_end
    for i in range(train_start, len(remaining_list)):
        doc_id = remaining_list[i]
        new_train_data[doc_id] = source_data[doc_id]
    
    print(f"\nAllocated remaining documents:")
    print(f"  Train: {len(new_train_data)} total (added {len(new_train_data) - len(OLD_TRAIN & set(source_data.keys()))})")
    print(f"  Val: {len(new_val_data)} total (added {len(new_val_data) - len(OLD_VAL & set(source_data.keys()))})")
    print(f"  Test: {len(new_test_data)} total")
    
    # Final split
    total_final = len(new_train_data) + len(new_val_data) + len(new_test_data)
    train_pct_final = 100 * len(new_train_data) / total_final if total_final > 0 else 0
    val_pct_final = 100 * len(new_val_data) / total_final if total_final > 0 else 0
    test_pct_final = 100 * len(new_test_data) / total_final if total_final > 0 else 0
    
    print(f"\nFinal split:")
    print(f"  Train: {len(new_train_data)} ({train_pct_final:.2f}%)")
    print(f"  Val: {len(new_val_data)} ({val_pct_final:.2f}%)")
    print(f"  Test: {len(new_test_data)} ({test_pct_final:.2f}%)")
    print(f"  Total: {total_final}")
    
    print("\n" + "=" * 70)
    print("STEP 4: Save files and verify")
    print("=" * 70)
    
    # Save files
    save_jsonl(new_train_data, new_train_file)
    save_jsonl(new_val_data, new_val_file)
    save_jsonl(new_test_data, new_test_file)
    
    print(f"Saved files:")
    print(f"  Train: {new_train_file}")
    print(f"  Val: {new_val_file}")
    print(f"  Test: {new_test_file}")
    
    # Verify no leakage
    print("\n" + "=" * 70)
    print("STEP 5: Verify no leakage")
    print("=" * 70)
    
    new_train_ids = set(new_train_data.keys())
    new_val_ids = set(new_val_data.keys())
    new_test_ids = set(new_test_data.keys())
    
    # Check 1: OLD_TRAIN should be in new train (not in val or test)
    old_train_in_new_train = OLD_TRAIN & new_train_ids
    old_train_in_new_val = OLD_TRAIN & new_val_ids
    old_train_in_new_test = OLD_TRAIN & new_test_ids
    
    print(f"\nOLD_TRAIN leakage check:")
    print(f"  OLD_TRAIN IDs in new train: {len(old_train_in_new_train)}")
    print(f"  OLD_TRAIN IDs in new val: {len(old_train_in_new_val)} (should be 0)")
    print(f"  OLD_TRAIN IDs in new test: {len(old_train_in_new_test)} (should be 0)")
    
    if old_train_in_new_val or old_train_in_new_test:
        print("ERROR: Leakage detected! OLD_TRAIN IDs found in new val or test!")
        sys.exit(1)
    
    # Check 2: OLD_VAL should be in new val (not in train or test)
    old_val_in_new_train = OLD_VAL & new_train_ids
    old_val_in_new_val = OLD_VAL & new_val_ids
    old_val_in_new_test = OLD_VAL & new_test_ids
    
    print(f"\nOLD_VAL leakage check:")
    print(f"  OLD_VAL IDs in new train: {len(old_val_in_new_train)} (should be 0)")
    print(f"  OLD_VAL IDs in new val: {len(old_val_in_new_val)}")
    print(f"  OLD_VAL IDs in new test: {len(old_val_in_new_test)} (should be 0)")
    
    if old_val_in_new_train or old_val_in_new_test:
        print("ERROR: Leakage detected! OLD_VAL IDs found in new train or test!")
        sys.exit(1)
    
    # Check 3: No overlaps between new split files
    val_test_overlap = new_val_ids & new_test_ids
    val_train_overlap = new_val_ids & new_train_ids
    test_train_overlap = new_test_ids & new_train_ids
    
    print(f"\nOverlap check between new split files:")
    print(f"  Val-Test overlap: {len(val_test_overlap)} (should be 0)")
    print(f"  Val-Train overlap: {len(val_train_overlap)} (should be 0)")
    print(f"  Test-Train overlap: {len(test_train_overlap)} (should be 0)")
    
    if val_test_overlap or val_train_overlap or test_train_overlap:
        print("ERROR: Overlaps detected between new split files!")
        sys.exit(1)
    
    # Check 4: All source documents are in the split
    all_new_split_ids = new_train_ids | new_val_ids | new_test_ids
    source_ids_missing = set(source_data.keys()) - all_new_split_ids
    
    print(f"\nCoverage check:")
    print(f"  Source file: {len(source_data)} document IDs")
    print(f"  New split files: {len(all_new_split_ids)} document IDs")
    print(f"  Missing from split: {len(source_ids_missing)} (should be 0)")
    
    if source_ids_missing:
        print("ERROR: Some source documents are missing from the split!")
        sys.exit(1)
    
    print("\n✓ All checks passed!")
    print("\n" + "=" * 70)
    print("LEAKAGE ELIMINATION ANALYSIS")
    print("=" * 70)
    print("\nDoes this procedure eliminate leakage relative to the two older datasets?")
    print("\nYES - This procedure eliminates leakage because:")
    print("1. All documents that were used for training in the old datasets are")
    print("   allocated to the new training set, so they cannot appear in")
    print("   validation or test sets (no train->val/test leakage).")
    print("2. All documents that were used for validation in the old datasets are")
    print("   allocated to the new validation set, so they cannot appear in")
    print("   the test set (no val->test leakage).")
    print("3. The new test set only contains documents that were never used for")
    print("   training or validation in the old datasets, ensuring a clean test set.")
    print("=" * 70)

if __name__ == "__main__":
    main()
