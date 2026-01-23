#!/usr/bin/env python3
"""
Add Language Identification (LID) metadata to text summary datasets.

This script:
- Uses NbAiLab/nb-nordic-lid to classify input and output texts as 'nob' (Bokmål) or 'nno' (Nynorsk)
- Adds input_lid, input_lid_conf, output_lid, output_lid_conf to metadata fields
- Updates JSONL files in place
- Generates statistics about LID distribution

Requires:
- fasttext
- huggingface-hub
- numpy<2.0 (fastText is incompatible with NumPy 2.0+)
"""

from __future__ import annotations

import sys
import json
import argparse
import warnings
from collections import Counter
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Create a dummy tqdm that just returns the iterable
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Try to import fasttext and huggingface_hub for LID
try:
    import fasttext
    from huggingface_hub import hf_hub_download
    LID_AVAILABLE = True
except ImportError:
    LID_AVAILABLE = False

# Import kommune name translation
from kommuner.kommune import kommunenavn


def _iter_examples(path: Path):
    """Yield examples from a JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _get_lid_model():
    """Get or load the NbAiLab/nb-nordic-lid model."""
    if not LID_AVAILABLE:
        print("Error: fasttext and huggingface_hub libraries are not available.", file=sys.stderr)
        print("Install them with: pip install fasttext huggingface-hub", file=sys.stderr)
        sys.exit(1)
    
    # Check NumPy version compatibility with fastText
    try:
        import numpy as np
        numpy_version = np.__version__
        major_version = int(numpy_version.split('.')[0])
        if major_version >= 2:
            print("WARNING: NumPy 2.0+ is incompatible with fastText.", file=sys.stderr)
            print("FastText will fail with 'Unable to avoid copy' errors.", file=sys.stderr)
            print("Please downgrade NumPy: pip install 'numpy<2.0'", file=sys.stderr)
            print("Or upgrade fasttext if a NumPy 2.0 compatible version becomes available.", file=sys.stderr)
    except Exception:
        pass  # If we can't check NumPy version, continue anyway
    
    try:
        model_name = "nb-nordic-lid.ftz"
        model_path = hf_hub_download("NbAiLab/nb-nordic-lid", model_name)
        model = fasttext.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error: Failed to load LID model: {e}", file=sys.stderr)
        sys.exit(1)


def _classify_lid(text: str, model) -> Optional[Tuple[str, float]]:
    """
    Classify text using LID model and return the max-confidence label and confidence score.
    
    Args:
        text: Text to classify. Must not be None (FATAL error if None).
        model: FastText LID model.
    
    Returns:
        Tuple of (lid, confidence) where lid is ISO 639-3 code (e.g., 'nob', 'nno').
        Returns None if classification fails or text is not Norwegian.
    
    Raises:
        ValueError: If text is None (FATAL error - should never happen).
        RuntimeError: If NumPy 2.0 compatibility issue occurs.
    """
    # FATAL ERROR: None text should never be passed to this function
    if text is None:
        raise ValueError("FATAL ERROR: _classify_lid received None text. This indicates a programming error.")
    
    if not text or not text.strip():
        return None
    
    try:
        # FastText predict processes one line at a time, so replace newlines with spaces
        # Also limit length to avoid issues with very long texts
        text_processed = text.replace('\n', ' ').replace('\r', ' ').strip()
        
        # Skip if text is too short or empty after processing
        if not text_processed or len(text_processed) < 3:
            return None
        
        # Limit text length to avoid memory issues (FastText can handle long texts, but this is safer)
        # Most LID models work well with first 1000 characters
        if len(text_processed) > 1000:
            text_processed = text_processed[:1000]
        
        # Get top-1 prediction (max confidence label)
        # Suppress warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            try:
                labels, scores = model.predict(text_processed, k=1)
            except ValueError as ve:
                # Handle NumPy 2.0 compatibility issue with fastText
                error_msg = str(ve)
                if "Unable to avoid copy" in error_msg or "np.array" in error_msg:
                    raise RuntimeError(
                        "FastText is incompatible with NumPy 2.0. "
                        "Please downgrade NumPy to version 1.x: pip install 'numpy<2.0'"
                    ) from ve
                raise  # Re-raise other ValueErrors
        
        if not labels or len(labels) == 0:
            return None
        
        # Get the top label and confidence score
        label = labels[0]
        confidence = float(scores[0]) if len(scores) > 0 else 0.0
        
        # Extract language code from label (e.g., '__label__nno' -> 'nno')
        if label.startswith('__label__'):
            lang_code = label[9:]  # Remove '__label__' prefix
        else:
            lang_code = label
        
        # Return ISO 639-3 code for Norwegian variants (nob = Bokmål, nno = Nynorsk)
        if lang_code == 'nno':
            return ('nno', confidence)
        elif lang_code == 'nob':
            return ('nob', confidence)
        else:
            # Not Norwegian, return the language code anyway (as per user's change)
            return (lang_code, confidence)
    except RuntimeError as e:
        # NumPy 2.0 compatibility error - this is a fatal error that should be reported
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        raise  # Re-raise as this is a configuration issue that needs to be fixed
    except Exception as e:
        # Only print warning if it's not the expected newline warning or NumPy compatibility issue
        error_msg = str(e)
        if "predict processes one line at a time" not in error_msg and "Unable to avoid copy" not in error_msg:
            print(f"Warning: Failed to classify text with LID: {e}", file=sys.stderr)
        return None


def _add_lid_to_jsonl_file(input_path: Path, output_path: Optional[Path] = None, model=None):
    """
    Add input_lid and output_lid metadata fields to a JSONL file.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file (if None, overwrites input)
        model: LID model (if None, will load it)
    
    Returns:
        Statistics dictionary with LID counts
    """
    if model is None:
        model = _get_lid_model()
    
    if output_path is None:
        output_path = input_path
    
    # Statistics
    input_lid_counter = Counter()
    output_lid_counter = Counter()
    lid_match_counter = Counter()  # True/False for whether input_lid == output_lid
    kommune_lid_stats: Dict[int, Dict[str, int]] = {}  # kommune -> {input_lid: count, output_lid: count, matches: count}
    
    # Process file
    examples = []
    for example in tqdm(_iter_examples(input_path), desc="Classifying LID", unit="examples"):
        # Get input and output texts, ensuring they are not None
        input_raw = example.get("input")
        output_raw = example.get("output")
        
        # FATAL ERROR: None values should not exist in the dataset
        if input_raw is None:
            raise ValueError(
                f"FATAL ERROR: Found None 'input' field in example. "
                f"This indicates corrupted data. Dokument ID: {example.get('metadata', {}).get('dokument_id', 'unknown')}"
            )
        if output_raw is None:
            raise ValueError(
                f"FATAL ERROR: Found None 'output' field in example. "
                f"This indicates corrupted data. Dokument ID: {example.get('metadata', {}).get('dokument_id', 'unknown')}"
            )
        
        # Convert to string (should never be None at this point)
        input_text = str(input_raw)
        output_text = str(output_raw)
        
        # Classify - returns (lid, confidence) or None
        input_result = _classify_lid(input_text, model)
        output_result = _classify_lid(output_text, model)
        
        # Add to metadata
        if "metadata" not in example:
            example["metadata"] = {}
        
        input_lid = None
        input_lid_conf = None
        if input_result:
            input_lid, input_lid_conf = input_result
            example["metadata"]["input_lid"] = input_lid
            example["metadata"]["input_lid_conf"] = input_lid_conf
            input_lid_counter[input_lid] += 1
        
        output_lid = None
        output_lid_conf = None
        if output_result:
            output_lid, output_lid_conf = output_result
            example["metadata"]["output_lid"] = output_lid
            example["metadata"]["output_lid_conf"] = output_lid_conf
            output_lid_counter[output_lid] += 1
        
        # Track matches
        if input_lid and output_lid:
            lid_match_counter[input_lid == output_lid] += 1
        
        # Track per kommune
        kommune = example.get("metadata", {}).get("kommune")
        if kommune is not None:
            if kommune not in kommune_lid_stats:
                kommune_lid_stats[kommune] = {
                    "input_nynorsk": 0,
                    "input_bokmål": 0,
                    "output_nynorsk": 0,
                    "output_bokmål": 0,
                    "matches": 0,
                    "total": 0
                }
            
            stats = kommune_lid_stats[kommune]
            stats["total"] += 1
            
            if input_lid == "nno":
                stats["input_nynorsk"] += 1
            elif input_lid == "nob":
                stats["input_bokmål"] += 1
            
            if output_lid == "nno":
                stats["output_nynorsk"] += 1
            elif output_lid == "nob":
                stats["output_bokmål"] += 1
            
            if input_lid and output_lid and input_lid == output_lid:
                stats["matches"] += 1
        
        examples.append(example)
    
    # Write updated file
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    # Build statistics
    stats = {
        "input_lid_distribution": dict(input_lid_counter),
        "output_lid_distribution": dict(output_lid_counter),
        "lid_matches": {
            "identical": lid_match_counter.get(True, 0),
            "different": lid_match_counter.get(False, 0),
            "total_with_both": sum(lid_match_counter.values())
        },
        "kommune_statistics": {}
    }
    
    # Add kommune names to kommune statistics
    for kommune_nummer, kommune_stats in kommune_lid_stats.items():
        navn = kommunenavn(kommune_nummer)
        stats["kommune_statistics"][f"{navn} ({kommune_nummer})"] = kommune_stats
    
    return stats


def _save_lid_statistics(lid_stats: Dict[str, Any], output_path: Path):
    """Save LID statistics to a JSON file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(lid_stats, f, indent=2, ensure_ascii=False, default=str)
        print(f"LID statistics saved to: {output_path}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Failed to save LID statistics: {e}", file=sys.stderr)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add Language Identification (LID) metadata to text summary datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process main file
  python add_lids.py --dataset 155452_text_summary_examples --dataset-folder ../cleaned_datasets/text_summary_dataset_202601
  
  # Process all splits (train/val/test)
  python add_lids.py --dataset 155452_text_summary_examples --process-splits --dataset-folder ../cleaned_datasets/text_summary_dataset_202601
  
  # Custom output file for statistics
  python add_lids.py --dataset 155452_text_summary_examples --lid-output lid_stats.json
        """
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        help="Dataset suffix (the part after 'text_summary_dataset_', e.g., '155452_text_summary_examples')",
    )
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--train",
        action="store_const",
        const="train",
        dest="split",
        help="Load the training split file",
    )
    split_group.add_argument(
        "--val",
        action="store_const",
        const="val",
        dest="split",
        help="Load the validation split file",
    )
    split_group.add_argument(
        "--test",
        action="store_const",
        const="test",
        dest="split",
        help="Load the test split file",
    )
    parser.add_argument(
        "--dataset-folder",
        type=str,
        default=None,
        help="Folder containing the dataset subfolders (e.g., '../cleaned_datasets'). Defaults to script directory if not provided.",
    )
    parser.add_argument(
        "--lid-output",
        type=str,
        default=None,
        help="Output file path for LID statistics JSON (default: <dataset>_lid_statistics.json)",
    )
    parser.add_argument(
        "--process-splits",
        action="store_true",
        help="Also process train/val/test split files",
    )
    
    args = parser.parse_args(argv)
    
    # Determine datasets root
    if args.dataset_folder:
        datasets_root = Path(args.dataset_folder).resolve()
    else:
        datasets_root = Path(__file__).parent
    
    # Construct folder name
    folder_name = f"text_summary_dataset_{args.dataset}"
    folder_path = datasets_root / folder_name
    
    if not folder_path.exists():
        print(f"Error: Dataset folder not found: {folder_path}", file=sys.stderr)
        return 1
    
    if not LID_AVAILABLE:
        print("Error: fasttext and huggingface_hub are required for LID analysis.", file=sys.stderr)
        print("Install them with: pip install fasttext huggingface-hub", file=sys.stderr)
        return 1
    
    try:
        # Load LID model once
        print("Loading LID model...", file=sys.stderr)
        lid_model = _get_lid_model()
        print("LID model loaded successfully.", file=sys.stderr)
        
        # Collect all files to process
        files_to_process = []
        
        # Find the main file (or split file if specified)
        if args.split:
            pattern = f"*_{args.split}.jsonl"
            matching_files = list(folder_path.glob(pattern))
            if not matching_files:
                print(f"Error: No {args.split} file found in {folder_path}", file=sys.stderr)
                return 1
            files_to_process.append(matching_files[0])
        else:
            # Find the plain file
            all_jsonl = list(folder_path.glob("*.jsonl"))
            plain_files = [
                f for f in all_jsonl
                if not any(f.name.endswith(f"_{s}.jsonl") for s in ["train", "val", "test"])
                and not f.name.endswith("_embeddings.jsonl")
                and not f.name.endswith("_lid_statistics.json")
            ]
            if not plain_files:
                print(f"Error: No plain file found in {folder_path}", file=sys.stderr)
                return 1
            files_to_process.append(plain_files[0])
            
            # Add split files if requested
            if args.process_splits:
                for split in ["train", "val", "test"]:
                    pattern = f"*_{split}.jsonl"
                    matching_files = list(folder_path.glob(pattern))
                    if matching_files:
                        files_to_process.append(matching_files[0])
        
        # Process each file
        all_stats = {}
        combined_stats = {
            "input_lid_distribution": Counter(),
            "output_lid_distribution": Counter(),
            "lid_matches": {"identical": 0, "different": 0, "total_with_both": 0},
            "kommune_statistics": {}
        }
        
        for file_path in files_to_process:
            print(f"\nProcessing {file_path.name}...", file=sys.stderr)
            file_stats = _add_lid_to_jsonl_file(file_path, model=lid_model)
            
            # Store stats for this file
            file_key = file_path.stem
            all_stats[file_key] = file_stats
            
            # Combine stats
            combined_stats["input_lid_distribution"].update(file_stats["input_lid_distribution"])
            combined_stats["output_lid_distribution"].update(file_stats["output_lid_distribution"])
            combined_stats["lid_matches"]["identical"] += file_stats["lid_matches"]["identical"]
            combined_stats["lid_matches"]["different"] += file_stats["lid_matches"]["different"]
            combined_stats["lid_matches"]["total_with_both"] += file_stats["lid_matches"]["total_with_both"]
            
            # Merge kommune statistics
            for kommune_key, kommune_data in file_stats["kommune_statistics"].items():
                if kommune_key not in combined_stats["kommune_statistics"]:
                    combined_stats["kommune_statistics"][kommune_key] = {
                        "input_nynorsk": 0,
                        "input_bokmål": 0,
                        "output_nynorsk": 0,
                        "output_bokmål": 0,
                        "matches": 0,
                        "total": 0
                    }
                for key in ["input_nynorsk", "input_bokmål", "output_nynorsk", "output_bokmål", "matches", "total"]:
                    combined_stats["kommune_statistics"][kommune_key][key] += kommune_data[key]
        
        # Convert Counter to dict for JSON serialization
        combined_stats["input_lid_distribution"] = dict(combined_stats["input_lid_distribution"])
        combined_stats["output_lid_distribution"] = dict(combined_stats["output_lid_distribution"])
        
        # Build final statistics structure
        final_stats = {
            "dataset": folder_name,
            "files_processed": [f.name for f in files_to_process],
            "per_file_statistics": all_stats,
            "combined_statistics": combined_stats
        }
        
        # Print statistics
        print("\n" + "=" * 80, file=sys.stderr)
        print("LID ANALYSIS RESULTS", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(f"\nInput LID Distribution (combined):", file=sys.stderr)
        for lid, count in sorted(combined_stats["input_lid_distribution"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {lid}: {count:,}", file=sys.stderr)
        
        print(f"\nOutput LID Distribution (combined):", file=sys.stderr)
        for lid, count in sorted(combined_stats["output_lid_distribution"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {lid}: {count:,}", file=sys.stderr)
        
        matches = combined_stats["lid_matches"]
        if matches["total_with_both"] > 0:
            match_pct = (matches["identical"] / matches["total_with_both"]) * 100
            diff_pct = (matches["different"] / matches["total_with_both"]) * 100
            print(f"\nLID Match Statistics (combined):", file=sys.stderr)
            print(f"  Identical (input_lid == output_lid): {matches['identical']:,} ({match_pct:.1f}%)", file=sys.stderr)
            print(f"  Different (input_lid != output_lid): {matches['different']:,} ({diff_pct:.1f}%)", file=sys.stderr)
            print(f"  Total with both LIDs: {matches['total_with_both']:,}", file=sys.stderr)
        
        print(f"\nKommune Statistics (top 10 by total):", file=sys.stderr)
        sorted_kommuner = sorted(
            combined_stats["kommune_statistics"].items(),
            key=lambda x: x[1]["total"],
            reverse=True
        )[:10]
        for kommune_name, kommune_data in sorted_kommuner:
            print(f"\n  {kommune_name}:", file=sys.stderr)
            print(f"    Total: {kommune_data['total']:,}", file=sys.stderr)
            print(f"    Input - Nynorsk: {kommune_data['input_nynorsk']:,}, Bokmål: {kommune_data['input_bokmål']:,}", file=sys.stderr)
            print(f"    Output - Nynorsk: {kommune_data['output_nynorsk']:,}, Bokmål: {kommune_data['output_bokmål']:,}", file=sys.stderr)
            if kommune_data['total'] > 0:
                match_pct = (kommune_data['matches'] / kommune_data['total']) * 100
                print(f"    Matches: {kommune_data['matches']:,} ({match_pct:.1f}%)", file=sys.stderr)
        
        print("\n" + "=" * 80 + "\n", file=sys.stderr)
        
        # Save statistics
        if args.lid_output:
            output_path = Path(args.lid_output)
        else:
            # Build default output filename
            if args.split:
                output_stem = files_to_process[0].stem
            else:
                output_stem = files_to_process[0].stem
            output_path = folder_path / f"{output_stem}_lid_statistics.json"
        
        _save_lid_statistics(final_stats, output_path)
        
        return 0
        
    except Exception as e:
        print(f"Error in LID analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
