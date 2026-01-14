#!/usr/bin/env python3
"""
Copy files from FROM folder to TO folder, but only if they don't already exist in TO.
Never overwrite existing files.
"""

import shutil
from pathlib import Path

FROM_FOLDER = Path("/home/sinoa/Local/Tools/VSCode/Demokratibasen-UiB-Ide/datasets_from_demokratibasen/data_collection/entities_themes/extractions-202512/extracted-data/dataset-202505-max-2048-input-tokens-max-1000-output-tokens-gpt-inferencing-202512/gpt-4o-mini")
TO_FOLDER = Path("/home/sinoa/Local/Tools/VSCode/Demokratibasen-UiB-Ide/datasets_from_demokratibasen/data_collection/entities_themes/extractions-202512/extracted-data/dataset-202505-all-input-tokens-max-1000-output-tokens-gpt-inferencing-202512/gpt-4o-mini")


def main():
    """Copy files from FROM to TO, skipping existing files."""
    # Ensure TO folder exists
    TO_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Get all files in TO folder (to check for existing files)
    existing_files = set()
    for file_path in TO_FOLDER.glob("*.json"):
        existing_files.add(file_path.name)
    
    print(f"Found {len(existing_files)} existing files in TO folder")
    
    # Get all files in FROM folder
    from_files = list(FROM_FOLDER.glob("*.json"))
    print(f"Found {len(from_files)} files in FROM folder")
    
    # Copy files that don't exist in TO
    copied_count = 0
    skipped_count = 0
    
    for from_file in from_files:
        to_file = TO_FOLDER / from_file.name
        
        if from_file.name in existing_files:
            skipped_count += 1
            if skipped_count <= 10 or skipped_count % 1000 == 0:
                print(f"Skipping {from_file.name} (already exists)")
        else:
            try:
                shutil.copy2(from_file, to_file)
                copied_count += 1
                if copied_count <= 10 or copied_count % 1000 == 0:
                    print(f"Copied {from_file.name} ({copied_count} files copied so far)")
            except Exception as e:
                print(f"Error copying {from_file.name}: {e}")
    
    print(f"\nCopy operation complete:")
    print(f"  Copied: {copied_count} files")
    print(f"  Skipped (already exist): {skipped_count} files")
    print(f"  Total files in TO folder now: {len(existing_files) + copied_count}")


if __name__ == "__main__":
    main()
