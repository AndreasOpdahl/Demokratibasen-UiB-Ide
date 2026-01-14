#!/usr/bin/env python3
"""
Merge dataset files from text_summary_dataset_121466_examples, text_summary_dataset_13077_examples, and text_summary_dataset_47215_examples.

Merges corresponding files (_train.jsonl, _val.jsonl, _test.jsonl, and main .jsonl files).
For duplicate IDs:
- If one summary is <50 chars, choose the one with longer summary
- Otherwise, choose from 47215 folder (last folder)
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict


def load_jsonl(file_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load a JSONL file and return a dictionary mapping dokument_id to example."""
    examples = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                    metadata = example.get('metadata', {})
                    doc_id = metadata.get('dokument_id')
                    if doc_id:
                        examples[doc_id] = example
                except json.JSONDecodeError as e:
                    print(f"  Warning: Failed to parse JSON at line {line_num} in {file_path.name}: {e}", file=sys.stderr)
                    continue
    except Exception as e:
        print(f"  Error: Failed to load {file_path}: {e}", file=sys.stderr)
    
    return examples


def find_corresponding_files(
    folder1: Path,
    folder2: Path,
    folder3: Path
) -> List[Tuple[Path, Path, Path, str]]:
    """
    Find corresponding files between three folders based on suffix.
    
    Matches files with same suffix (_train, _val, _test, or main).
    
    Returns list of tuples: (file1, file2, file3, output_name)
    """
    matches = []
    
    # Find all text_summary_examples_*.jsonl files in all folders
    files1_list = sorted([f for f in folder1.glob("text_summary_examples_*.jsonl")])
    files2_list = sorted([f for f in folder2.glob("text_summary_examples_*.jsonl")])
    files3_list = sorted([f for f in folder3.glob("text_summary_examples_*.jsonl")])
    
    # Group files by suffix
    files1_by_suffix = {'main': [], '_train': [], '_val': [], '_test': []}
    files2_by_suffix = {'main': [], '_train': [], '_val': [], '_test': []}
    files3_by_suffix = {'main': [], '_train': [], '_val': [], '_test': []}
    
    for f in files1_list:
        stem = f.stem
        if stem.endswith('_train'):
            files1_by_suffix['_train'].append(f)
        elif stem.endswith('_val'):
            files1_by_suffix['_val'].append(f)
        elif stem.endswith('_test'):
            files1_by_suffix['_test'].append(f)
        else:
            files1_by_suffix['main'].append(f)
    
    for f in files2_list:
        stem = f.stem
        if stem.endswith('_train'):
            files2_by_suffix['_train'].append(f)
        elif stem.endswith('_val'):
            files2_by_suffix['_val'].append(f)
        elif stem.endswith('_test'):
            files2_by_suffix['_test'].append(f)
        else:
            files2_by_suffix['main'].append(f)
    
    for f in files3_list:
        stem = f.stem
        if stem.endswith('_train'):
            files3_by_suffix['_train'].append(f)
        elif stem.endswith('_val'):
            files3_by_suffix['_val'].append(f)
        elif stem.endswith('_test'):
            files3_by_suffix['_test'].append(f)
        else:
            files3_by_suffix['main'].append(f)
    
    # Match files by suffix
    for suffix in ['main', '_train', '_val', '_test']:
        files1 = files1_by_suffix[suffix]
        files2 = files2_by_suffix[suffix]
        files3 = files3_by_suffix[suffix]
        
        if not files1 and not files2 and not files3:
            continue
        
        if not files1:
            print(f"  Warning: No {suffix if suffix != 'main' else 'main'} files in folder1 (121466)", file=sys.stderr)
            continue
        
        if not files2:
            print(f"  Warning: No {suffix if suffix != 'main' else 'main'} files in folder2 (13077)", file=sys.stderr)
            continue
        
        if not files3:
            print(f"  Warning: No {suffix if suffix != 'main' else 'main'} files in folder3 (47215)", file=sys.stderr)
            continue
        
        # Use the first file from each list (should only be one of each type)
        file1 = files1[0]
        file2 = files2[0]
        file3 = files3[0]
        
        # Generate output name based on folder3's file name (as base)
        output_name = file3.stem + '.jsonl'
        matches.append((file1, file2, file3, output_name))
    
    return matches


def merge_three_examples(
    examples1: Dict[str, Dict[str, Any]],
    examples2: Dict[str, Dict[str, Any]],
    examples3: Dict[str, Dict[str, Any]],
    source1_name: str,
    source2_name: str,
    source3_name: str
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Merge three dictionaries of examples.
    
    Returns:
        (merged_examples, statistics)
    """
    merged = {}
    stats = {
        'total_from_121466': len(examples1),
        'total_from_13077': len(examples2),
        'total_from_47215': len(examples3),
        'unique_in_121466': 0,
        'unique_in_13077': 0,
        'unique_in_47215': 0,
        'overlapping_ids': 0,
        'different_texts_pairs': 0,  # Count of pairs with different texts
        'different_summaries_pairs': 0,  # Count of pairs with different summaries
        'conflicts_resolved_by_length': 0,
        'conflicts_resolved_by_source': 0,
        'final_count': 0
    }
    
    # Find all unique IDs across all three sources
    all_ids = set(examples1.keys()) | set(examples2.keys()) | set(examples3.keys())
    
    for doc_id in all_ids:
        ex1 = examples1.get(doc_id)
        ex2 = examples2.get(doc_id)
        ex3 = examples3.get(doc_id)
        
        # Count how many sources have this ID
        sources_with_id = [ex for ex in [ex1, ex2, ex3] if ex is not None]
        num_sources = len(sources_with_id)
        
        if num_sources == 0:
            continue
        elif num_sources == 1:
            # Only in one source
            if ex1:
                merged[doc_id] = ex1
                stats['unique_in_121466'] += 1
            elif ex2:
                merged[doc_id] = ex2
                stats['unique_in_13077'] += 1
            elif ex3:
                merged[doc_id] = ex3
                stats['unique_in_47215'] += 1
        else:
            # Multiple sources have this ID - need to resolve conflict
            stats['overlapping_ids'] += 1
            
            # Collect all examples that have this ID
            all_examples = []
            if ex1:
                all_examples.append(('121466', ex1))
            if ex2:
                all_examples.append(('13077', ex2))
            if ex3:
                all_examples.append(('47215', ex3))
            
            # Count pairs with different texts/summaries
            texts = {}
            summaries = {}
            for source, ex in all_examples:
                texts[source] = str(ex.get('input', ''))
                summaries[source] = str(ex.get('output', ''))
            
            # Check all pairs for differences
            sources_list = [s for s, _ in all_examples]
            for i, source1 in enumerate(sources_list):
                for source2 in sources_list[i+1:]:
                    if texts[source1] != texts[source2]:
                        stats['different_texts_pairs'] += 1
                    if summaries[source1] != summaries[source2]:
                        stats['different_summaries_pairs'] += 1
            
            # Resolve conflict
            chosen_source = None
            chosen_example = None
            
            # Rule 1: If any summary is <50 chars, choose the one with longest summary
            summary_lengths = [(s, len(summaries[s])) for s, _ in all_examples]
            lengths_only = [l for _, l in summary_lengths]
            min_length = min(lengths_only)
            
            if min_length < 50:
                # At least one summary is short - choose the longest
                chosen_source, max_len = max(summary_lengths, key=lambda x: x[1])
                stats['conflicts_resolved_by_length'] += 1
                for source, ex in all_examples:
                    if source == chosen_source:
                        chosen_example = ex
                        break
            else:
                # All summaries are >=50 chars - choose from 47215 (last folder)
                chosen_source = '47215'
                stats['conflicts_resolved_by_source'] += 1
                # Prefer 47215, but if it doesn't exist, fall back to others
                if ex3:
                    chosen_example = ex3
                elif ex2:
                    chosen_example = ex2
                    chosen_source = '13077'
                else:
                    chosen_example = ex1
                    chosen_source = '121466'
            
            merged[doc_id] = chosen_example
    
    stats['final_count'] = len(merged)
    
    return merged, stats


def save_jsonl(examples: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    """Save examples to a JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc_id, example in sorted(examples.items()):
            f.write(json.dumps(example, ensure_ascii=False) + '\n')


def main():
    # Script is now in the datasets folder, so parent is datasets directory
    datasets_dir = Path(__file__).parent
    merged_dir = datasets_dir / 'text_summary_dataset_ALL_examples'
    
    folder_121466 = datasets_dir / 'text_summary_dataset_121466_examples'
    folder_13077 = datasets_dir / 'text_summary_dataset_13077_examples'
    folder_47215 = datasets_dir / 'text_summary_dataset_47215_examples'
    
    if not folder_121466.exists():
        print(f"Error: Folder not found: {folder_121466}", file=sys.stderr)
        sys.exit(1)
    
    if not folder_13077.exists():
        print(f"Error: Folder not found: {folder_13077}", file=sys.stderr)
        sys.exit(1)
    
    if not folder_47215.exists():
        print(f"Error: Folder not found: {folder_47215}", file=sys.stderr)
        sys.exit(1)
    
    # Create merged directory
    merged_dir.mkdir(exist_ok=True)
    
    print("Finding corresponding files...", file=sys.stderr)
    
    # Find all corresponding file triplets
    all_file_triplets = find_corresponding_files(folder_121466, folder_13077, folder_47215)
    
    if not all_file_triplets:
        print("Error: No corresponding files found", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(all_file_triplets)} file triplets to merge\n", file=sys.stderr)
    
    # Global statistics
    global_stats = {
        'total_files_merged': 0,
        'total_from_121466': 0,
        'total_from_13077': 0,
        'total_from_47215': 0,
        'total_unique_in_121466': 0,
        'total_unique_in_13077': 0,
        'total_unique_in_47215': 0,
        'total_overlapping_ids': 0,
        'total_different_texts_pairs': 0,
        'total_different_summaries_pairs': 0,
        'total_conflicts_resolved_by_length': 0,
        'total_conflicts_resolved_by_source': 0,
        'total_final_count': 0
    }
    
    # Process each file triplet
    for file1, file2, file3, output_name in all_file_triplets:
        print(f"Merging {file1.name} + {file2.name} + {file3.name} -> {output_name}", file=sys.stderr)
        
        # Load examples from all three files
        examples1 = load_jsonl(file1)
        examples2 = load_jsonl(file2)
        examples3 = load_jsonl(file3)
        
        print(f"  Loaded {len(examples1)} from 121466, {len(examples2)} from 13077, {len(examples3)} from 47215", file=sys.stderr)
        
        # Merge examples
        merged, stats = merge_three_examples(examples1, examples2, examples3, file1.name, file2.name, file3.name)
        
        print(f"  Merged result: {stats['final_count']} unique examples", file=sys.stderr)
        print(f"    Overlapping IDs: {stats['overlapping_ids']}", file=sys.stderr)
        if stats['overlapping_ids'] > 0:
            print(f"      - Pairs with different texts: {stats['different_texts_pairs']}", file=sys.stderr)
            print(f"      - Pairs with different summaries: {stats['different_summaries_pairs']}", file=sys.stderr)
            print(f"      - Resolved by length: {stats['conflicts_resolved_by_length']}", file=sys.stderr)
            print(f"      - Resolved by source (47215): {stats['conflicts_resolved_by_source']}", file=sys.stderr)
        print(file=sys.stderr)
        
        # Update global stats
        global_stats['total_files_merged'] += 1
        global_stats['total_from_121466'] += stats['total_from_121466']
        global_stats['total_from_13077'] += stats['total_from_13077']
        global_stats['total_from_47215'] += stats['total_from_47215']
        global_stats['total_unique_in_121466'] += stats['unique_in_121466']
        global_stats['total_unique_in_13077'] += stats['unique_in_13077']
        global_stats['total_unique_in_47215'] += stats['unique_in_47215']
        global_stats['total_overlapping_ids'] += stats['overlapping_ids']
        global_stats['total_different_texts_pairs'] += stats['different_texts_pairs']
        global_stats['total_different_summaries_pairs'] += stats['different_summaries_pairs']
        global_stats['total_conflicts_resolved_by_length'] += stats['conflicts_resolved_by_length']
        global_stats['total_conflicts_resolved_by_source'] += stats['conflicts_resolved_by_source']
        global_stats['total_final_count'] += stats['final_count']
        
        # Save merged file
        output_path = merged_dir / output_name
        save_jsonl(merged, output_path)
        print(f"  Saved to {output_path.name}\n", file=sys.stderr)
    
    # Print summary statistics
    print("=" * 80, file=sys.stderr)
    print("MERGE SUMMARY", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"Total files merged: {global_stats['total_files_merged']}", file=sys.stderr)
    print(f"Total examples from 121466 folder: {global_stats['total_from_121466']:,}", file=sys.stderr)
    print(f"Total examples from 13077 folder: {global_stats['total_from_13077']:,}", file=sys.stderr)
    print(f"Total examples from 47215 folder: {global_stats['total_from_47215']:,}", file=sys.stderr)
    print(f"Unique examples only in 121466: {global_stats['total_unique_in_121466']:,}", file=sys.stderr)
    print(f"Unique examples only in 13077: {global_stats['total_unique_in_13077']:,}", file=sys.stderr)
    print(f"Unique examples only in 47215: {global_stats['total_unique_in_47215']:,}", file=sys.stderr)
    print(f"Overlapping IDs (conflicts): {global_stats['total_overlapping_ids']:,}", file=sys.stderr)
    if global_stats['total_overlapping_ids'] > 0:
        print(f"  - Pairs with different texts: {global_stats['total_different_texts_pairs']:,}", file=sys.stderr)
        print(f"  - Pairs with different summaries: {global_stats['total_different_summaries_pairs']:,}", file=sys.stderr)
        print(f"  - Conflicts resolved by length (shorter summary <50 chars): {global_stats['total_conflicts_resolved_by_length']:,}", file=sys.stderr)
        print(f"  - Conflicts resolved by source (chose 47215): {global_stats['total_conflicts_resolved_by_source']:,}", file=sys.stderr)
    print(f"Final total unique examples: {global_stats['total_final_count']:,}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"\nMerged files saved to: {merged_dir}", file=sys.stderr)


if __name__ == '__main__':
    main()
