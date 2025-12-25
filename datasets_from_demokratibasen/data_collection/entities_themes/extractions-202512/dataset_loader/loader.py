"""
Dataset loader for JSONL files.

Loads a dataset by name and provides an iterator interface.
"""

import json
import sys
from pathlib import Path
from typing import Type
from .dataset_registry import get_dataset_path, get_dataset_adapter


class DatasetLoader:
    """
    Load and iterate through documents from a registered dataset.
    
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
        
        # Get dataset file path and adapter
        self.dataset_name = dataset_name
        self.file_path = get_dataset_path(dataset_name, root)
        self.adapter_class = get_dataset_adapter(dataset_name)
        
        if not self.file_path.exists():
            sys.exit(f"Fant ikke {self.file_path}")

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

    def __call__(self):
        """
        Iterate through documents, reading and processing one line at a time.
        
        Yields:
            Tuple of (doc_id, kommune_nummer, kommune_navn, text) for each valid document
        """
        adapter = self.adapter_class()
        
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    doc = json.loads(line)
                    
                    # Check if adapter has document type filtering
                    if hasattr(adapter, 'should_include_document'):
                        if not adapter.should_include_document(doc):
                            continue
                    
                    # Use adapter to normalize field names
                    normalized = adapter.normalize(doc)
                    
                    doc_id = normalized["dok_id"]
                    kommune_nummer = normalized["kommune_nummer"]
                    kommune_navn = normalized["kommune_navn"]
                    text = normalized["tekst"]
                    
                    # Filter out documents with empty or whitespace-only text
                    if not text.strip():
                        print(f"Warning: Skipping document {doc_id}: empty or whitespace-only text", file=sys.stderr)
                        continue
                    
                    # Filter out documents with insufficient text content
                    if len(text) < 10:
                        print(f"Warning: Skipping document {doc_id}: text too short ({len(text)} characters, minimum 10)", file=sys.stderr)
                        continue
                    
                    # Filter out documents with no alphanumeric content
                    if not any(c.isalnum() for c in text):
                        print(f"Warning: Skipping document {doc_id}: text contains no alphanumeric characters", file=sys.stderr)
                        continue
                    
                    # Sanitise the text
                    text = self._sanitise_json_string(text)
                    
                    # Yield the processed document
                    yield (doc_id, kommune_nummer, kommune_navn, text)
                    
                except json.JSONDecodeError as e:
                    # Skip invalid JSON lines but log the error
                    print(f"Warning: Skipping invalid JSON line: {e}", file=sys.stderr)
                    continue

