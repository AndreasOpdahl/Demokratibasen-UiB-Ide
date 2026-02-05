"""
Dataset loading utilities for JSONL files.

This module provides shared functions for loading JSONL datasets with:
- Git LFS pointer detection
- File size validation
- JSON parsing with error handling
- Consistent error messages
"""

import json
import os
from typing import List, Dict, Optional


def load_jsonl_dataset(
    file_path: str,
    dataset_type: str = "dataset",
    raise_on_error: bool = False
) -> Optional[List[Dict]]:
    """
    Load JSONL dataset with Git LFS pointer detection and error handling.
    
    Args:
        file_path: Path to JSONL file
        dataset_type: Type name for error messages (e.g., "training", "validation")
        raise_on_error: If True, raise exceptions instead of returning None
    
    Returns:
        List of parsed JSON objects, or None if error occurred and raise_on_error=False
    
    Raises:
        FileNotFoundError: If file doesn't exist (only if raise_on_error=True)
        ValueError: If file is Git LFS pointer or invalid JSON (only if raise_on_error=True)
    """
    # Check if file exists
    if not os.path.exists(file_path):
        error_msg = f"ERROR: {dataset_type.capitalize()} dataset file does not exist: {file_path}"
        if raise_on_error:
            raise FileNotFoundError(error_msg)
        print(error_msg)
        return None
    
    # Check file size (Git LFS pointers are typically < 200 bytes)
    file_size = os.path.getsize(file_path)
    if file_size < 200:
        print(f"WARNING: {dataset_type.capitalize()} dataset file is very small ({file_size} bytes).")
        print(f"         This might be a Git LFS pointer file. Please ensure the actual file is downloaded.")
    
    # Read and parse JSONL file
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            # Check if it's a Git LFS pointer
            if first_line.strip().startswith('version https://git-lfs.github.com/spec/v1'):
                error_msg = (
                    f"ERROR: {dataset_type.capitalize()} dataset file appears to be a Git LFS pointer, not actual data.\n"
                    f"       Please download the actual file using: git lfs pull\n"
                    f"       Or ensure the file at {file_path} contains actual JSONL data."
                )
                if raise_on_error:
                    raise ValueError(error_msg)
                print(error_msg)
                return None
            
            # Reset file pointer and read all lines
            f.seek(0)
            line_num = 0
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as json_err:
                    error_msg = (
                        f"ERROR: Invalid JSON on line {line_num} of {dataset_type} dataset:\n"
                        f"       {str(json_err)}\n"
                        f"       Line content (first 200 chars): {line[:200]}"
                    )
                    if raise_on_error:
                        raise ValueError(error_msg)
                    print(error_msg)
                    return None
        
        if len(data) == 0:
            error_msg = f"ERROR: {dataset_type.capitalize()} dataset file is empty or contains no valid JSON lines: {file_path}"
            if raise_on_error:
                raise ValueError(error_msg)
            print(error_msg)
            return None
        
        print(f"Successfully loaded {len(data)} {dataset_type} examples")
        return data
        
    except Exception as e:
        error_msg = f"Error reading {dataset_type} dataset: {e}\nFile path: {file_path}"
        if raise_on_error:
            raise
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None
