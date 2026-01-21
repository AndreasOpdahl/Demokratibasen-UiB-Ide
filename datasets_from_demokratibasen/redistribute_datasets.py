#!/usr/bin/env python3
"""
Script to redistribute datasets by:
1. Checking overlaps between val/test files and original train files
2. Moving overlapping documents to train
3. Redistributing to achieve ~5%-5%-90% split
4. Moving random documents from train to test to reach 5% (excluding original train files)
5. Verifying no overlaps remain
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, Set, List

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
    base_dir = Path("cleaned_datasets/text_summary_dataset_ALL_examples")
    
    val_file_all = base_dir / "text_summary_examples_ALL_val.jsonl"
    test_file_all = base_dir / "text_summary_examples_ALL_test.jsonl"
    train_file_all = base_dir / "text_summary_examples_ALL_train.jsonl"
    
    val_file_2025 = base_dir / "text_summary_examples_202505_to_10_val.jsonl"
    test_file_2025 = base_dir / "text_summary_examples_202505_to_10_test.jsonl"
    
    # Original train files for overlap checking
    train_file_1 = Path("prepared_datasets/text_summary_dataset_202505_to_10/text_summary_examples_202505_to_10_train.jsonl")
    train_file_2 = Path("prepared_datasets/text_summary_dataset_202505_to_12/text_summary_examples_202505_to_12_train.jsonl")
    
    # Source file for final check
    source_file = base_dir / "text_summary_examples_202505_to_10.jsonl"
    
    print("=" * 70)
    print("STEP 1: Check overlaps with original train files")
    print("=" * 70)
    
    # Extract IDs from original train files
    train_ids_1 = extract_doc_ids(train_file_1)
    train_ids_2 = extract_doc_ids(train_file_2)
    all_train_ids = train_ids_1 | train_ids_2
    
    print(f"Train file 1: {len(train_ids_1)} document IDs")
    print(f"Train file 2: {len(train_ids_2)} document IDs")
    print(f"Combined train IDs: {len(all_train_ids)} document IDs")
    
    # Determine which val/test files to use
    if val_file_all.exists() and test_file_all.exists():
        val_file = val_file_all
        test_file = test_file_all
        print(f"\nUsing existing ALL files")
    else:
        val_file = val_file_all
        test_file = test_file_all
        # Copy from 2025 files if ALL files don't exist
        if val_file_2025.exists() and test_file_2025.exists():
            print(f"\nCreating ALL files from 202505_to_10 files...")
            val_data_source = load_jsonl(val_file_2025)
            test_data_source = load_jsonl(test_file_2025)
            save_jsonl(val_data_source, val_file)
            save_jsonl(test_data_source, test_file)
            print(f"Created {val_file.name} with {len(val_data_source)} documents")
            print(f"Created {test_file.name} with {len(test_data_source)} documents")
        else:
            print(f"ERROR: Source files not found!")
            sys.exit(1)
    
    # Load current data
    val_data = load_jsonl(val_file)
    test_data = load_jsonl(test_file)
    train_data = load_jsonl(train_file_all) if train_file_all.exists() else {}
    
    val_ids = set(val_data.keys())
    test_ids = set(test_data.keys())
    
    print(f"\nVal file: {len(val_ids)} document IDs")
    print(f"Test file: {len(test_ids)} document IDs")
    print(f"Train file: {len(train_data)} document IDs")
    
    # Check overlaps with original train files
    val_overlap = val_ids & all_train_ids
    test_overlap = test_ids & all_train_ids
    
    print(f"\nOverlap between val and train files: {len(val_overlap)} document IDs")
    print(f"Overlap between test and train files: {len(test_overlap)} document IDs")
    
    # Move overlapping documents
    moved_from_val = 0
    moved_from_test = 0
    
    for doc_id in list(val_overlap):
        if doc_id in val_data:
            train_data[doc_id] = val_data.pop(doc_id)
            moved_from_val += 1
    
    for doc_id in list(test_overlap):
        if doc_id in test_data:
            train_data[doc_id] = test_data.pop(doc_id)
            moved_from_test += 1
    
    if moved_from_val > 0 or moved_from_test > 0:
        save_jsonl(val_data, val_file)
        save_jsonl(test_data, test_file)
        save_jsonl(train_data, train_file_all)
        print(f"\nMoved {moved_from_val} documents from val to train")
        print(f"Moved {moved_from_test} documents from test to train")
    
    print("\n" + "=" * 70)
    print("STEP 2: Verify no overlaps with original train files")
    print("=" * 70)
    
    val_ids_new = set(val_data.keys())
    test_ids_new = set(test_data.keys())
    
    val_overlap_new = val_ids_new & all_train_ids
    test_overlap_new = test_ids_new & all_train_ids
    
    print(f"Overlap between val and train files: {len(val_overlap_new)} document IDs")
    print(f"Overlap between test and train files: {len(test_overlap_new)} document IDs")
    
    if val_overlap_new or test_overlap_new:
        print("ERROR: Overlaps still exist!")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("STEP 3: Verify no overlaps between val/test/train")
    print("=" * 70)
    
    train_ids_new = set(train_data.keys())
    
    val_test_overlap = val_ids_new & test_ids_new
    val_train_overlap = val_ids_new & train_ids_new
    test_train_overlap = test_ids_new & train_ids_new
    
    print(f"Overlap between val and test: {len(val_test_overlap)} document IDs")
    print(f"Overlap between val and train: {len(val_train_overlap)} document IDs")
    print(f"Overlap between test and train: {len(test_train_overlap)} document IDs")
    
    # Move any remaining overlaps to train
    # When val and test overlap, move from val to train and remove from test
    moved_val_test = 0
    moved_val_train = 0
    moved_test_train = 0
    
    for doc_id in list(val_test_overlap):
        if doc_id in val_data:
            # Move from val to train, and remove from test (they're duplicates)
            train_data[doc_id] = val_data.pop(doc_id)
            if doc_id in test_data:
                test_data.pop(doc_id)
            moved_val_test += 1
    
    for doc_id in list(val_train_overlap):
        if doc_id in val_data:
            train_data[doc_id] = val_data.pop(doc_id)
            moved_val_train += 1
    
    for doc_id in list(test_train_overlap):
        if doc_id in test_data:
            train_data[doc_id] = test_data.pop(doc_id)
            moved_test_train += 1
    
    if moved_val_test > 0 or moved_val_train > 0 or moved_test_train > 0:
        save_jsonl(val_data, val_file)
        save_jsonl(test_data, test_file)
        save_jsonl(train_data, train_file_all)
        print(f"\nMoved {moved_val_test} documents from val (val-test overlap, also removed from test)")
        print(f"Moved {moved_val_train} documents from val (val-train overlap)")
        print(f"Moved {moved_test_train} documents from test (test-train overlap)")
        
        # Re-check
        val_ids_new = set(val_data.keys())
        test_ids_new = set(test_data.keys())
        train_ids_new = set(train_data.keys())
        
        val_test_overlap = val_ids_new & test_ids_new
        val_train_overlap = val_ids_new & train_ids_new
        test_train_overlap = test_ids_new & train_ids_new
        
        print(f"\nAfter moving overlaps:")
        print(f"Overlap between val and test: {len(val_test_overlap)} document IDs")
        print(f"Overlap between val and train: {len(val_train_overlap)} document IDs")
        print(f"Overlap between test and train: {len(test_train_overlap)} document IDs")
        
        if val_test_overlap or val_train_overlap or test_train_overlap:
            print("ERROR: Overlaps still exist after moving!")
            sys.exit(1)
    
    print("\n" + "=" * 70)
    print("STEP 4: Redistribute to ~5%-5%-90% split")
    print("=" * 70)
    
    total = len(val_data) + len(test_data) + len(train_data)
    print(f"Current split:")
    val_pct = 100 * len(val_data) / total if total > 0 else 0
    test_pct = 100 * len(test_data) / total if total > 0 else 0
    train_pct = 100 * len(train_data) / total if total > 0 else 0
    print(f"  Val: {len(val_data)} ({val_pct:.2f}%)")
    print(f"  Test: {len(test_data)} ({test_pct:.2f}%)")
    print(f"  Train: {len(train_data)} ({train_pct:.2f}%)")
    print(f"  Total: {total}")
    
    target_val_pct = 5.0
    target_test_pct = 5.0
    target_train_pct = 90.0
    
    target_val = int(total * target_val_pct / 100)
    target_test = int(total * target_test_pct / 100)
    target_train = total - target_val - target_test
    
    print(f"\nTarget split:")
    print(f"  Val: {target_val} ({target_val_pct}%)")
    print(f"  Test: {target_test} ({target_test_pct}%)")
    print(f"  Train: {target_train} ({target_train_pct}%)")
    
    # Calculate how many to move
    val_to_move = max(0, len(val_data) - target_val)
    test_to_move = max(0, len(test_data) - target_test)
    
    print(f"\nNeed to move:")
    print(f"  {val_to_move} documents from val to train")
    print(f"  {test_to_move} documents from test to train")
    
    # Move documents from val (ensure no overlap with train)
    val_list = sorted(list(val_data.keys()))
    moved_from_val = 0
    for i in range(min(val_to_move, len(val_list))):
        doc_id = val_list[i]
        if doc_id not in train_data:  # Avoid duplicates
            train_data[doc_id] = val_data.pop(doc_id)
            moved_from_val += 1
    
    # Move documents from test (ensure no overlap with train)
    test_list = sorted(list(test_data.keys()))
    moved_from_test = 0
    for i in range(min(test_to_move, len(test_list))):
        doc_id = test_list[i]
        if doc_id not in train_data:  # Avoid duplicates
            train_data[doc_id] = test_data.pop(doc_id)
            moved_from_test += 1
    
    if moved_from_val > 0 or moved_from_test > 0:
        save_jsonl(val_data, val_file)
        save_jsonl(test_data, test_file)
        save_jsonl(train_data, train_file_all)
        print(f"\nMoved {moved_from_val} documents from val to train")
        print(f"Moved {moved_from_test} documents from test to train")
    
    # Final split after redistribution
    total_after_redist = len(val_data) + len(test_data) + len(train_data)
    val_pct_after = 100 * len(val_data) / total_after_redist if total_after_redist > 0 else 0
    test_pct_after = 100 * len(test_data) / total_after_redist if total_after_redist > 0 else 0
    train_pct_after = 100 * len(train_data) / total_after_redist if total_after_redist > 0 else 0
    
    print(f"\nSplit after redistribution:")
    print(f"  Val: {len(val_data)} ({val_pct_after:.2f}%)")
    print(f"  Test: {len(test_data)} ({test_pct_after:.2f}%)")
    print(f"  Train: {len(train_data)} ({train_pct_after:.2f}%)")
    print(f"  Total: {total_after_redist}")
    
    # Move random documents from train to test until test reaches 5%
    target_test_pct = 5.0
    target_test_count = int(total_after_redist * target_test_pct / 100)
    test_needed = max(0, target_test_count - len(test_data))
    
    if test_needed > 0:
        print(f"\n" + "=" * 70)
        print("STEP 4b: Move random documents from train to test to reach 5%")
        print("=" * 70)
        print(f"Test currently has {len(test_data)} documents ({test_pct_after:.2f}%)")
        print(f"Target: {target_test_count} documents (5.00%)")
        print(f"Need to move {test_needed} documents from train to test")
        print(f"\nExcluding documents already in original train files...")
        
        # Get IDs from original train files
        train_ids_original = all_train_ids
        
        # Filter train documents to exclude those in original train files
        train_candidates = [
            doc_id for doc_id in train_data.keys()
            if doc_id not in train_ids_original
        ]
        
        print(f"Train has {len(train_data)} documents")
        print(f"Documents in original train files: {len(train_ids_original)}")
        print(f"Available candidates (not in original train files): {len(train_candidates)}")
        
        if len(train_candidates) < test_needed:
            print(f"WARNING: Only {len(train_candidates)} candidates available, but {test_needed} needed")
            test_needed = len(train_candidates)
        
        # Randomly sample documents to move
        random.seed(42)  # For reproducibility
        docs_to_move = random.sample(train_candidates, test_needed)
        
        # Move documents from train to test
        moved_from_train_to_test = 0
        for doc_id in docs_to_move:
            if doc_id in train_data:
                test_data[doc_id] = train_data.pop(doc_id)
                moved_from_train_to_test += 1
        
        if moved_from_train_to_test > 0:
            save_jsonl(val_data, val_file)
            save_jsonl(test_data, test_file)
            save_jsonl(train_data, train_file_all)
            print(f"\nMoved {moved_from_train_to_test} random documents from train to test")
    
    # Final split
    total_final = len(val_data) + len(test_data) + len(train_data)
    val_pct_final = 100 * len(val_data) / total_final if total_final > 0 else 0
    test_pct_final = 100 * len(test_data) / total_final if total_final > 0 else 0
    train_pct_final = 100 * len(train_data) / total_final if total_final > 0 else 0
    
    print(f"\nFinal split:")
    print(f"  Val: {len(val_data)} ({val_pct_final:.2f}%)")
    print(f"  Test: {len(test_data)} ({test_pct_final:.2f}%)")
    print(f"  Train: {len(train_data)} ({train_pct_final:.2f}%)")
    print(f"  Total: {total_final}")
    
    print("\n" + "=" * 70)
    print("STEP 5: Add missing documents from source file to train")
    print("=" * 70)
    
    # Load source file and add missing documents to train
    source_data = load_jsonl(source_file)
    val_ids_final = set(val_data.keys())
    test_ids_final = set(test_data.keys())
    train_ids_final = set(train_data.keys())
    
    all_split_ids = val_ids_final | test_ids_final | train_ids_final
    source_ids_missing = set(source_data.keys()) - all_split_ids
    
    print(f"Source file ({source_file.name}): {len(source_data)} document IDs")
    print(f"Current split files: {len(all_split_ids)} document IDs")
    print(f"IDs in source but not in split: {len(source_ids_missing)} document IDs")
    
    if source_ids_missing:
        print(f"\nAdding {len(source_ids_missing)} missing documents to train...")
        added_count = 0
        for doc_id in source_ids_missing:
            if doc_id in source_data and doc_id not in train_data:
                train_data[doc_id] = source_data[doc_id]
                added_count += 1
        
        if added_count > 0:
            save_jsonl(train_data, train_file_all)
            print(f"Added {added_count} documents to train")
    
    # Update counts after adding missing documents
    val_ids_final = set(val_data.keys())
    test_ids_final = set(test_data.keys())
    train_ids_final = set(train_data.keys())
    all_split_ids = val_ids_final | test_ids_final | train_ids_final
    
    total_after_add = len(val_data) + len(test_data) + len(train_data)
    val_pct_after_add = 100 * len(val_data) / total_after_add if total_after_add > 0 else 0
    test_pct_after_add = 100 * len(test_data) / total_after_add if total_after_add > 0 else 0
    train_pct_after_add = 100 * len(train_data) / total_after_add if total_after_add > 0 else 0
    
    print(f"\nFinal split after adding missing documents:")
    print(f"  Val: {len(val_data)} ({val_pct_after_add:.2f}%)")
    print(f"  Test: {len(test_data)} ({test_pct_after_add:.2f}%)")
    print(f"  Train: {len(train_data)} ({train_pct_after_add:.2f}%)")
    print(f"  Total: {total_after_add}")
    
    print("\n" + "=" * 70)
    print("STEP 6: Final verification")
    print("=" * 70)
    
    # Check 1: All IDs from source file are in val/test/train
    source_ids = set(source_data.keys())
    source_ids_missing_final = source_ids - all_split_ids
    
    print(f"\nCheck 1: All IDs from source file are in val/test/train")
    print(f"Source file ({source_file.name}): {len(source_ids)} document IDs")
    print(f"Val file: {len(val_ids_final)} document IDs")
    print(f"Test file: {len(test_ids_final)} document IDs")
    print(f"Train file: {len(train_ids_final)} document IDs")
    print(f"Total in split files: {len(all_split_ids)} document IDs")
    print(f"IDs in source file that are also in split: {len(source_ids & all_split_ids)} document IDs")
    print(f"IDs in source file but not in split: {len(source_ids_missing_final)} document IDs")
    
    if source_ids_missing_final:
        print(f"ERROR: {len(source_ids_missing_final)} IDs from source file are missing from split files!")
        print(f"First 10 missing IDs:")
        for doc_id in sorted(list(source_ids_missing_final))[:10]:
            print(f"  {doc_id}")
        sys.exit(1)
    else:
        print("✓ All IDs from source file are in split files")
    
    # Check 2: No overlaps between val/test/train
    print(f"\nCheck 2: No overlapping IDs between val/test/train")
    val_test_overlap_final = val_ids_final & test_ids_final
    val_train_overlap_final = val_ids_final & train_ids_final
    test_train_overlap_final = test_ids_final & train_ids_final
    
    print(f"Overlap between val and test: {len(val_test_overlap_final)} document IDs")
    print(f"Overlap between val and train: {len(val_train_overlap_final)} document IDs")
    print(f"Overlap between test and train: {len(test_train_overlap_final)} document IDs")
    
    if val_test_overlap_final or val_train_overlap_final or test_train_overlap_final:
        print("ERROR: Overlaps exist between split files!")
        if val_test_overlap_final:
            print(f"  First 10 overlapping IDs between val and test:")
            for doc_id in sorted(list(val_test_overlap_final))[:10]:
                print(f"    {doc_id}")
        if val_train_overlap_final:
            print(f"  First 10 overlapping IDs between val and train:")
            for doc_id in sorted(list(val_train_overlap_final))[:10]:
                print(f"    {doc_id}")
        if test_train_overlap_final:
            print(f"  First 10 overlapping IDs between test and train:")
            for doc_id in sorted(list(test_train_overlap_final))[:10]:
                print(f"    {doc_id}")
        sys.exit(1)
    else:
        print("✓ No overlapping IDs between split files")
    
    print("\n✓ All checks passed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
