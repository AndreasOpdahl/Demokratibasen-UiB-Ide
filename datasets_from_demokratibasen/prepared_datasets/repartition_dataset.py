#!/usr/bin/env python3
"""
Repartition a JSONL dataset file into train/val/test splits with specified ratios.
"""

import json
import random
from pathlib import Path

INPUT_FILE = Path("/home/sinoa/Local/Tools/VSCode/Demokratibasen-UiB-Ide/datasets_from_demokratibasen/prepared_datasets/text_summary_dataset_13077_examples/text_summary_examples_202505.jsonl")
OUTPUT_DIR = INPUT_FILE.parent

# Split ratios: 80% train, 10% val, 10% test
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


def main():
    """Repartition the dataset file."""
    # Read all documents
    print(f"Reading documents from: {INPUT_FILE}")
    documents = []
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    doc = json.loads(line)
                    documents.append(doc)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    
    total = len(documents)
    print(f"Loaded {total} documents")
    
    # Shuffle for random partition
    random.seed(42)  # For reproducibility
    random.shuffle(documents)
    
    # Calculate split sizes
    train_size = int(total * TRAIN_RATIO)
    val_size = int(total * VAL_RATIO)
    # test_size = total - train_size - val_size  # Remaining goes to test
    
    train_docs = documents[:train_size]
    val_docs = documents[train_size:train_size + val_size]
    test_docs = documents[train_size + val_size:]
    
    # Save train split
    train_file = OUTPUT_DIR / "text_summary_examples_202505_train.jsonl"
    print(f"\nSaving train split ({len(train_docs)} documents, {100*len(train_docs)/total:.1f}%) to: {train_file}")
    with open(train_file, "w", encoding="utf-8") as f:
        for doc in train_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    # Save val split
    val_file = OUTPUT_DIR / "text_summary_examples_202505_val.jsonl"
    print(f"Saving val split ({len(val_docs)} documents, {100*len(val_docs)/total:.1f}%) to: {val_file}")
    with open(val_file, "w", encoding="utf-8") as f:
        for doc in val_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    # Save test split
    test_file = OUTPUT_DIR / "text_summary_examples_202505_test.jsonl"
    print(f"Saving test split ({len(test_docs)} documents, {100*len(test_docs)/total:.1f}%) to: {test_file}")
    with open(test_file, "w", encoding="utf-8") as f:
        for doc in test_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    # Report final counts
    print(f"\nFinal document counts:")
    print(f"  Total: {total} documents")
    print(f"  Train: {len(train_docs)} documents ({100*len(train_docs)/total:.1f}%)")
    print(f"  Val: {len(val_docs)} documents ({100*len(val_docs)/total:.1f}%)")
    print(f"  Test: {len(test_docs)} documents ({100*len(test_docs)/total:.1f}%)")


if __name__ == "__main__":
    main()
