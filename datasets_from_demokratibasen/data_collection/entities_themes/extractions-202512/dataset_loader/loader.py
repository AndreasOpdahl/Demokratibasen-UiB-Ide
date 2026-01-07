"""
Dataset loader for JSONL files and directories of JSON files.

Loads a dataset by name and provides an iterator interface.
Supports both:
- JSONL files (one JSON object per line)
- Directories containing individual JSON files (*.json)
"""

import json
import sys
from pathlib import Path
from typing import Type, Iterator
from .dataset_registry import get_dataset_path, get_dataset_adapter


class DatasetLoader:
    """
    Load and iterate through documents from a registered dataset.
    
    Supports both JSONL files and directories of JSON files.
    
    Filters out documents with:
    - Empty or whitespace-only text
    - Text shorter than 10 characters
    - No alphanumeric content
    
    Sanitizes text content to ensure valid JSON encoding.
    
    Args:
        dataset_name: Name of the dataset (e.g., "dataset-202505" or "dataset-202510")
        root: Optional root path for resolving dataset file paths.
             If not provided, uses the default root from dataset_registry.
    """
    
    def __init__(self, dataset_name: str, root: Path = None):
        from .dataset_registry import set_root_path, get_root_path
        
        # Set root path if provided
        if root is not None:
            set_root_path(root)
        
        # Get dataset file/directory path and adapter
        self.dataset_name = dataset_name
        self.path = get_dataset_path(dataset_name, root)
        self.adapter_class = get_dataset_adapter(dataset_name)
        
        if not self.path.exists():
            sys.exit(f"Fant ikke {self.path}")
        
        # Determine if path is a file or directory
        self.is_directory = self.path.is_dir()
        self.is_file = self.path.is_file()
        
        if not (self.is_directory or self.is_file):
            sys.exit(f"Path exists but is neither file nor directory: {self.path}")
        
        # Backward compatibility: file_path property (returns path for files, None for directories)
        self.file_path = self.path if self.is_file else None

    def _sanitise_json_string(self, raw: str) -> str:
        """
        Returnerer 'raw' der
        - alle kontrolltegn (< 0x20) inne i strenger er escaped
        - et "nakent" dobbelt-anførselstegn inne i en streng skrives om til \"
            (gjelder også sekvensen "").
        """
        out, in_str, esc = [], False, False
        it = iter(enumerate(raw))
        for i, ch in it:
            if esc:                       # forrige tegn var '\'
                out.append(ch)
                esc = False
                continue

            if ch == '\\':                # start av escape
                out.append(ch)
                esc = True
                continue

            if ch == '"':                 # anførselstegn
                if in_str:
                    # Sjekk om dette egentlig er et "" (ulovlig) som betyr "
                    nxt = raw[i + 1] if i + 1 < len(raw) else ''
                    if nxt == '"':        # fant ""
                        out.extend(['\\', '"'])      # legg inn \"
                        next(it)                     # hopp over andre "
                        continue
                in_str = not in_str                  # toggl streng-modus
                out.append(ch)
                continue

            if in_str and ord(ch) < 0x20:           # kontrolltegn i streng
                if ch == '\t':
                    out.extend(['\\', 't'])
                elif ch == '\n':
                    out.extend(['\\', 'n'])
                elif ch == '\r':
                    out.extend(['\\', 'r'])
                else:
                    out.extend(['\\', 'u', *f'{ord(ch):04x}'])
            else:
                out.append(ch)

        return ''.join(out)

    def _iter_json_files(self) -> Iterator[Path]:
        """Iterator over JSON files in the directory."""
        if self.is_directory:
            # Sort for deterministic ordering
            yield from sorted(self.path.glob("*.json"))
        else:
            # For single file, don't yield anything (will be handled as JSONL)
            pass
    
    def _iter_jsonl_lines(self) -> Iterator[str]:
        """Iterator over lines in a JSONL file."""
        if self.is_file:
            with open(self.path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        yield line
    
    def _process_document(self, doc: dict, adapter) -> tuple:
        """Process a single document through the adapter and filters."""
        # Check if adapter has document type filtering
        if hasattr(adapter, 'should_include_document'):
            if not adapter.should_include_document(doc):
                return None
        
        # Use adapter to normalize field names
        normalized = adapter.normalize(doc)
        
        doc_id = normalized["dok_id"]
        kommune_nummer = normalized["kommune_nummer"]
        kommune_navn = normalized["kommune_navn"]
        text = normalized["tekst"]
        
        # Filter out documents with empty or whitespace-only text
        if not text.strip():
            print(f"Warning: Skipping document {doc_id}: empty or whitespace-only text", file=sys.stderr)
            return None
        
        # Filter out documents with insufficient text content
        if len(text) < 10:
            print(f"Warning: Skipping document {doc_id}: text too short ({len(text)} characters, minimum 10)", file=sys.stderr)
            return None
        
        # Filter out documents with no alphanumeric content
        if not any(c.isalnum() for c in text):
            print(f"Warning: Skipping document {doc_id}: text contains no alphanumeric characters", file=sys.stderr)
            return None
        
        # Sanitise the text
        text = self._sanitise_json_string(text)
        
        return (doc_id, kommune_nummer, kommune_navn, text)
    
    def __call__(self):
        """
        Iterate through documents from either a JSONL file or a directory of JSON files.
        
        Yields:
            Tuple of (doc_id, kommune_nummer, kommune_navn, text) for each valid document
        """
        adapter = self.adapter_class()
        
        if self.is_directory:
            # Process directory of JSON files
            for json_file in self._iter_json_files():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        doc = json.load(f)
                    
                    result = self._process_document(doc, adapter)
                    if result is not None:
                        yield result
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON file {json_file.name}: {e}", file=sys.stderr)
                    continue
                except Exception as e:
                    print(f"Warning: Error processing {json_file.name}: {e}", file=sys.stderr)
                    continue
        
        elif self.is_file:
            # Process JSONL file (one JSON object per line)
            for line in self._iter_jsonl_lines():
                try:
                    doc = json.loads(line)
                    
                    result = self._process_document(doc, adapter)
                    if result is not None:
                        yield result
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}", file=sys.stderr)
                    continue

