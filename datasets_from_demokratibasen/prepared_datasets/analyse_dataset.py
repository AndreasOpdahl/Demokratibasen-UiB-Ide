#!/usr/bin/env python3
"""
Text summary dataset analysis script.

Analyzes text_summary_datasets_*_examples datasets:
- Counts total examples
- Distribution of doc_type
- Distribution of kommune
- Summary length statistics
- Input length statistics
- Sample examples
"""

from __future__ import annotations

import sys
import json
import argparse
import re
import warnings
import unicodedata
from dataclasses import dataclass, asdict
from collections import Counter
from typing import Optional, Dict, Any, List, Set, Tuple
from pathlib import Path

# Suppress pynvml deprecation warning from torch
warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Create a dummy tqdm that just returns the iterable
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Try to import UMAP for 2D projections
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Try to import matplotlib for histograms
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Try to import tiktoken for token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    # Use cl100k_base encoding (used by GPT-4, GPT-3.5-turbo, etc.)
    _tokenizer = tiktoken.get_encoding("cl100k_base")
except ImportError:
    TIKTOKEN_AVAILABLE = False
    _tokenizer = None

# Try to import transformers for embeddings
try:
    # Suppress pynvml deprecation warning before importing torch
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")
        import torch
        from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Try to import FAISS for fast similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

import numpy as np

# Import kommune name translation
from kommuner.kommune import kommunenavn


@dataclass
class AnalysisResult:
    dataset_name: str
    path: str
    total_examples: int
    doc_type_counter: Dict[str, int]
    kommune_counter: Dict[int, int]
    summary_lengths: List[int]
    input_lengths: List[int]
    summary_token_lengths: List[int]
    input_token_lengths: List[int]
    cosine_distances: List[float]
    input_embeddings: List[np.ndarray]
    output_embeddings: List[np.ndarray]
    doc_types: List[str]
    samples: List[Dict[str, Any]]


def _count_alphanumeric(text: str) -> int:
    """Count the number of alphanumeric characters in a string."""
    return len(re.findall(r'[a-zA-Z0-9]', text))


def _should_filter_document(example: Dict[str, Any]) -> bool:
    """
    Check if a document should be filtered out (has < 10 alphanumeric characters in input or output).
    
    Returns:
        True if document should be filtered out, False otherwise
    """
    input_text = str(example.get("input", ""))
    output_text = str(example.get("output", ""))
    
    input_alnum = _count_alphanumeric(input_text)
    output_alnum = _count_alphanumeric(output_text)
    
    return input_alnum < 10 or output_alnum < 10


def _check_ambiguous_unicode(text: str) -> Set[Tuple[str, str]]:
    """
    Check for ambiguous unicode characters in text.
    
    Only flags truly problematic characters, not legitimate Norwegian characters
    or common punctuation. Ambiguous characters are those that can cause issues
    with text processing, matching, and display.
    
    Returns:
        Set of tuples (char, name) for ambiguous unicode characters found.
    """
    ambiguous_chars = set()
    
    # Legitimate Norwegian characters and common punctuation to allow
    # Norwegian: Å (U+00C5), å (U+00E5), Æ (U+00C6), æ (U+00E6), Ø (U+00D8), ø (U+00F8)
    # Common punctuation: en dash – (U+2013), em dash — (U+2014), section § (U+00A7)
    # bullet • (U+2022), quotes " " (U+201C, U+201D), ' (U+2018, U+2019)
    # Mathematical symbols: ±, ², °, ÷, ≈, ≤, ≥, μ, ′
    # Currency: €
    # Other common: ®, ·, đ
    allowed_code_points = {
        0x00A7,  # SECTION SIGN
        0x00AE,  # REGISTERED SIGN
        0x00B0,  # DEGREE SIGN
        0x00B1,  # PLUS-MINUS SIGN
        0x00B2,  # SUPERSCRIPT TWO
        0x00B7,  # MIDDLE DOT
        0x00C5, 0x00E5,  # Å, å
        0x00C6, 0x00E6,  # Æ, æ
        0x00D8, 0x00F8,  # Ø, ø
        0x00F7,  # DIVISION SIGN
        0x0111,  # LATIN SMALL LETTER D WITH STROKE (đ)
        0x2010,  # HYPHEN
        0x2013,  # EN DASH
        0x2014,  # EM DASH
        0x2018, 0x2019,  # LEFT/RIGHT SINGLE QUOTATION MARK
        0x201C, 0x201D,  # LEFT/RIGHT DOUBLE QUOTATION MARK
        0x2022,  # BULLET
        0x2032,  # PRIME
        0x20AC,  # EURO SIGN
        0x2248,  # ALMOST EQUAL TO
        0x2264,  # LESS-THAN OR EQUAL TO
        0x2265,  # GREATER-THAN OR EQUAL TO
        0x25A0,  # BLACK SQUARE
        0x25CB,  # WHITE CIRCLE
        0x25CF,  # BLACK CIRCLE
        0x03BC,  # GREEK SMALL LETTER MU
        # Latin characters with diacritics (common in Norwegian and other languages)
        0x00E0, 0x00E1, 0x00E2, 0x00E3, 0x00E4,  # à, á, â, ã, ä
        0x00E8, 0x00E9, 0x00EA, 0x00EB,  # è, é, ê, ë
        0x00EC, 0x00ED, 0x00EE, 0x00EF,  # ì, í, î, ï
        0x00F2, 0x00F3, 0x00F4, 0x00F5, 0x00F6,  # ò, ó, ô, õ, ö
        0x00F9, 0x00FA, 0x00FB, 0x00FC,  # ù, ú, û, ü
        0x00C0, 0x00C1, 0x00C2, 0x00C3, 0x00C4,  # À, Á, Â, Ã, Ä
        0x00C8, 0x00C9, 0x00CA, 0x00CB,  # È, É, Ê, Ë
        0x00CC, 0x00CD, 0x00CE, 0x00CF,  # Ì, Í, Î, Ï
        0x00D2, 0x00D3, 0x00D4, 0x00D5, 0x00D6,  # Ò, Ó, Ô, Õ, Ö
        0x00D9, 0x00DA, 0x00DB, 0x00DC,  # Ù, Ú, Û, Ü
        # Combining diacritics (used in proper Unicode normalization)
        0x0301,  # COMBINING ACUTE ACCENT
        0x0308,  # COMBINING DIAERESIS
        0x030A,  # COMBINING RING ABOVE
    }
    
    for char in text:
        code_point = ord(char)
        
        # Skip ASCII characters (0-127)
        if code_point <= 127:
            continue
        
        # Skip allowed legitimate characters
        if code_point in allowed_code_points:
            continue
        
        # Check for specific problematic character ranges only
        # (not East Asian Width 'A' which includes legitimate characters)
        is_ambiguous = (
            (0x0430 <= code_point <= 0x044F) or  # Cyrillic small letters (homoglyphs)
            (0xFF00 <= code_point <= 0xFFEF) or  # Full-width characters
            (0x2000 <= code_point <= 0x200B) or  # Various spaces (en quad, em quad, thin space, etc.)
            code_point == 0x202F or               # Narrow no-break space
            code_point == 0x205F or               # Medium mathematical space
            code_point == 0x3000 or               # Ideographic space
            (0xFE00 <= code_point <= 0xFE0F) or   # Variation selectors
            (0x200C <= code_point <= 0x200D) or   # Zero-width non-joiner/joiner
            (0x2060 <= code_point <= 0x206F) or   # Word joiner, invisible separator, etc.
            (0xF000 <= code_point <= 0xFFFF) or   # Private Use Area (except we allow some common ones above)
            code_point == 0xFFFD                  # REPLACEMENT CHARACTER (indicates encoding error)
        )
        
        if is_ambiguous:
            try:
                name = unicodedata.name(char, 'UNNAMED')
                ambiguous_chars.add((char, name))
            except ValueError:
                ambiguous_chars.add((char, f'U+{code_point:04X}'))
    
    return ambiguous_chars


def _count_tokens(text: str) -> int:
    """Count the number of tokens in a string using tiktoken."""
    if not TIKTOKEN_AVAILABLE or _tokenizer is None:
        return 0
    try:
        return len(_tokenizer.encode(text))
    except Exception:
        return 0


def _get_embedding_model():
    """Get or load the NbAiLab/nb-bert-large model and tokenizer. Requires CUDA."""
    if not TRANSFORMERS_AVAILABLE:
        print("Error: transformers library is not available.", file=sys.stderr)
        sys.exit(1)
    
    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. Embeddings require CUDA device.", file=sys.stderr)
        sys.exit(1)
    
    try:
        model_name = "NbAiLab/nb-bert-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        # Move model to CUDA device
        device = torch.device("cuda")
        model = model.to(device)
        return model, tokenizer, device
    except Exception as e:
        print(f"Error: Failed to load embedding model: {e}", file=sys.stderr)
        sys.exit(1)


def _create_embedding(text: str, model, tokenizer, device) -> Optional[np.ndarray]:
    """Create embedding for a text using mean-pooling, masking padding, and L2-normalization on CUDA."""
    if model is None or tokenizer is None:
        return None
    
    try:
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # Move inputs to CUDA device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Get attention mask
        attention_mask = inputs["attention_mask"]  # [batch_size, seq_len]
        
        # Mean-pool with masking
        # Expand mask to match embedding dimensions
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # Sum embeddings, masking out padding
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        # Sum of mask (number of non-padding tokens)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        # Mean pooling
        mean_embeddings = sum_embeddings / sum_mask
        
        # Convert to numpy and L2-normalize
        embedding = mean_embeddings[0].cpu().numpy()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    except Exception as e:
        print(f"Warning: Failed to create embedding: {e}", file=sys.stderr)
        return None


def _cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine distance (1 - cosine similarity) between two embeddings."""
    if emb1 is None or emb2 is None:
        return float('nan')
    
    # Cosine similarity = dot product of normalized vectors
    cosine_sim = np.dot(emb1, emb2)
    # Cosine distance = 1 - cosine similarity
    cosine_dist = 1.0 - cosine_sim
    return float(cosine_dist)


def _get_embeddings_path(data_path: Path) -> Path:
    """Get the path for embeddings file corresponding to a data file."""
    stem = data_path.stem
    return data_path.parent / f"{stem}_embeddings.jsonl"


def _load_embeddings(embeddings_path: Path, data_file_path: Optional[Path] = None) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load embeddings from a JSONL file. Returns dict mapping dokument_id to embeddings.
    
    If data_file_path is provided, filters out embeddings for documents that don't pass
    the alphanumeric character threshold (removes embeddings for filtered documents).
    """
    embeddings_dict = {}
    if not embeddings_path.exists():
        return embeddings_dict
    
    try:
        with open(embeddings_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    dokument_id = data.get("dokument_id")
                    if dokument_id:
                        embeddings_dict[dokument_id] = {
                            "input_embedding": np.array(data.get("input_embedding", [])),
                            "output_embedding": np.array(data.get("output_embedding", []))
                        }
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception as e:
        print(f"Warning: Failed to load embeddings from {embeddings_path}: {e}", file=sys.stderr)
    
    # Filter out embeddings for documents that don't pass the threshold
    if data_file_path and data_file_path.exists():
        # Build a set of valid document IDs
        valid_doc_ids = set()
        for example in _iter_examples(data_file_path):
            if not _should_filter_document(example):
                metadata = example.get("metadata", {})
                doc_id = metadata.get("dokument_id")
                if doc_id:
                    valid_doc_ids.add(doc_id)
        
        # Remove embeddings for filtered documents
        filtered_count = 0
        doc_ids_to_remove = []
        for doc_id in embeddings_dict:
            if doc_id not in valid_doc_ids:
                doc_ids_to_remove.append(doc_id)
                filtered_count += 1
        
        for doc_id in doc_ids_to_remove:
            del embeddings_dict[doc_id]
        
        if filtered_count > 0:
            print(f"Removed {filtered_count} embeddings for filtered-out documents", file=sys.stderr)
            # Clean the embeddings file
            _clean_embeddings_file(embeddings_path, valid_doc_ids)
    
    return embeddings_dict


def _clean_embeddings_file(embeddings_path: Path, valid_doc_ids: Set[str]):
    """
    Clean an embeddings file by removing entries for documents not in valid_doc_ids.
    Rewrites the file with only valid embeddings.
    """
    try:
        cleaned_data = []
        with open(embeddings_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    dokument_id = data.get("dokument_id")
                    if dokument_id and dokument_id in valid_doc_ids:
                        cleaned_data.append(data)
                except (json.JSONDecodeError, KeyError):
                    continue
        
        # Rewrite the file with only valid embeddings
        with open(embeddings_path, "w", encoding="utf-8") as f:
            for item in cleaned_data:
                f.write(json.dumps(item) + "\n")
        
        print(f"Cleaned embeddings file: removed entries for filtered-out documents", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Failed to clean embeddings file: {e}", file=sys.stderr)


def _save_embeddings(embeddings_path: Path, embeddings_data: List[Dict[str, Any]]):
    """Save embeddings to a JSONL file."""
    try:
        with open(embeddings_path, "w", encoding="utf-8") as f:
            for item in embeddings_data:
                # Convert numpy arrays to lists for JSON serialization
                item_copy = item.copy()
                if "input_embedding" in item_copy and isinstance(item_copy["input_embedding"], np.ndarray):
                    item_copy["input_embedding"] = item_copy["input_embedding"].tolist()
                if "output_embedding" in item_copy and isinstance(item_copy["output_embedding"], np.ndarray):
                    item_copy["output_embedding"] = item_copy["output_embedding"].tolist()
                f.write(json.dumps(item_copy) + "\n")
    except Exception as e:
        print(f"Warning: Failed to save embeddings to {embeddings_path}: {e}", file=sys.stderr)


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


def analyze_dataset(
    dataset_suffix: str,
    split: Optional[str] = None,
    datasets_root: Optional[Path] = None,
    num_examples: Optional[int] = None
) -> AnalysisResult:
    """
    Analyze a text_summary dataset and return structured results.
    
    Args:
        dataset_suffix: The suffix after "text_summary_dataset_" (e.g., "12811_examples")
        split: Optional split name ("train", "val", or "test")
        datasets_root: Root directory containing the datasets (defaults to script directory)
        num_examples: Optional number of sample examples to collect (None to skip)
    """
    if datasets_root is None:
        datasets_root = Path(__file__).parent
    
    # Construct folder name
    folder_name = f"text_summary_dataset_{dataset_suffix}"
    folder_path = datasets_root / folder_name
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {folder_path}")
    
    # Find the appropriate file
    if split:
        # Look for files with the split suffix
        pattern = f"*_{split}.jsonl"
        matching_files = list(folder_path.glob(pattern))
        if not matching_files:
            raise FileNotFoundError(
                f"No {split} file found in {folder_path}. "
                f"Expected pattern: *_{split}.jsonl"
            )
        file_path = matching_files[0]
    else:
        # Look for the plain file (without _train/_test/_val, and not embeddings)
        # Find all JSONL files and filter out those with split suffixes and embeddings files
        all_jsonl = list(folder_path.glob("*.jsonl"))
        plain_files = [
            f for f in all_jsonl
            if not any(f.name.endswith(f"_{s}.jsonl") for s in ["train", "val", "test"])
            and not f.name.endswith("_embeddings.jsonl")
        ]
        if not plain_files:
            raise FileNotFoundError(
                f"No plain file found in {folder_path}. "
                f"Expected a file without _train/_test/_val suffix and not _embeddings.jsonl."
            )
        file_path = plain_files[0]
    
    # Check for ambiguous unicode characters in input data file
    print("Checking for ambiguous unicode characters in input data...", file=sys.stderr)
    ambiguous_chars_found: Set[Tuple[str, str]] = set()
    sample_positions: List[Tuple[int, str, str]] = []  # (line_num, field, char_info)
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                    # Check input field
                    input_text = str(example.get("input", ""))
                    if input_text:
                        input_ambiguous = _check_ambiguous_unicode(input_text)
                        if input_ambiguous:
                            ambiguous_chars_found.update(input_ambiguous)
                            # Store first few occurrences with line numbers
                            if len(sample_positions) < 10:
                                for char, name in list(input_ambiguous)[:3]:
                                    sample_positions.append((line_num, "input", f"{char} ({name}, U+{ord(char):04X})"))
                    
                    # Check output field
                    output_text = str(example.get("output", ""))
                    if output_text:
                        output_ambiguous = _check_ambiguous_unicode(output_text)
                        if output_ambiguous:
                            ambiguous_chars_found.update(output_ambiguous)
                            # Store first few occurrences with line numbers
                            if len(sample_positions) < 10:
                                for char, name in list(output_ambiguous)[:3]:
                                    sample_positions.append((line_num, "output", f"{char} ({name}, U+{ord(char):04X})"))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Warning: Could not check for ambiguous unicode: {e}", file=sys.stderr)
    
    # Warn if ambiguous unicode found
    if ambiguous_chars_found:
        print(f"\nWARNING: Found {len(ambiguous_chars_found)} ambiguous unicode character(s) in {file_path.name}", file=sys.stderr)
        print("Ambiguous unicode characters may look like ASCII but are different code points.", file=sys.stderr)
        print("This can cause issues with text processing, matching, and display.", file=sys.stderr)
        print("\nAmbiguous characters found:", file=sys.stderr)
        for char, name in sorted(ambiguous_chars_found, key=lambda x: ord(x[0])):
            try:
                ascii_repr = repr(char)
            except:
                ascii_repr = f"U+{ord(char):04X}"
            print(f"  {char} ({name}, U+{ord(char):04X}, repr: {ascii_repr})", file=sys.stderr)
        
        if sample_positions:
            print("\nSample occurrences:", file=sys.stderr)
            for line_num, field, char_info in sample_positions[:5]:
                print(f"  Line {line_num}, field '{field}': {char_info}", file=sys.stderr)
            if len(sample_positions) > 5:
                print(f"  ... and {len(sample_positions) - 5} more occurrences", file=sys.stderr)
        print(file=sys.stderr)
    
    # Handle embeddings
    embeddings_path = _get_embeddings_path(file_path)
    embeddings_dict = {}
    
    # Check if embeddings file exists and is up-to-date
    # If data file is newer than embeddings file, regenerate embeddings
    if embeddings_path.exists():
        data_mtime = file_path.stat().st_mtime
        embeddings_mtime = embeddings_path.stat().st_mtime
        if data_mtime > embeddings_mtime:
            print(f"Data file {file_path.name} is newer than embeddings file {embeddings_path.name}. Regenerating embeddings...", file=sys.stderr)
            embeddings_path.unlink()  # Delete old embeddings file
    
    # For split files, check if plain embeddings exist
    if split:
        # Look for plain embeddings file (find the corresponding plain data file first)
        all_jsonl = list(folder_path.glob("*.jsonl"))
        plain_data_files = [
            f for f in all_jsonl
            if not any(f.name.endswith(f"_{s}.jsonl") for s in ["train", "val", "test"])
            and not f.name.endswith("embeddings.jsonl")
        ]
        if plain_data_files:
            plain_data_file = plain_data_files[0]
            plain_embeddings_path = _get_embeddings_path(plain_data_file)
            if plain_embeddings_path.exists():
                # Check if plain data file is newer than plain embeddings file
                plain_data_mtime = plain_data_file.stat().st_mtime
                plain_embeddings_mtime = plain_embeddings_path.stat().st_mtime
                if plain_data_mtime > plain_embeddings_mtime:
                    print(f"Plain data file {plain_data_file.name} is newer than plain embeddings file {plain_embeddings_path.name}. Regenerating embeddings...", file=sys.stderr)
                    plain_embeddings_path.unlink()  # Delete old embeddings file
            
            # Load embeddings if they still exist (after potential deletion)
            if plain_embeddings_path.exists():
                # Load all plain embeddings (filtering out invalid documents)
                plain_embeddings = _load_embeddings(plain_embeddings_path, data_file_path=plain_data_file)
                # Get dokument_ids from current split file (only valid ones)
                split_doc_ids = set()
                for example in _iter_examples(file_path):
                    if not _should_filter_document(example):
                        metadata = example.get("metadata", {})
                        doc_id = metadata.get("dokument_id")
                        if doc_id:
                            split_doc_ids.add(doc_id)
                # Extract embeddings for documents in this split
                for doc_id in split_doc_ids:
                    if doc_id in plain_embeddings:
                        embeddings_dict[doc_id] = plain_embeddings[doc_id]
                print(f"Loaded {len(embeddings_dict)} embeddings from plain embeddings file for {split} split", file=sys.stderr)
    
    # If embeddings don't exist, generate them (only for documents that pass filtering)
    if not embeddings_dict and not embeddings_path.exists():
        if not TRANSFORMERS_AVAILABLE:
            print("Warning: transformers not available, cannot generate embeddings", file=sys.stderr)
        else:
            print(f"Generating embeddings for {file_path.name} (only for documents passing filter)...", file=sys.stderr)
            model, tokenizer, device = _get_embedding_model()
            
            # Filter examples first, then generate embeddings only for valid ones
            examples_list = list(_iter_examples(file_path))
            valid_examples = [ex for ex in examples_list if not _should_filter_document(ex)]
            num_docs = len(valid_examples)
            
            print(f"Generating embeddings for {num_docs} valid documents (filtered out {len(examples_list) - num_docs} documents)", file=sys.stderr)
            
            embeddings_data = []
            for example in tqdm(valid_examples, total=num_docs, desc="Generating embeddings", unit="docs"):
                metadata = example.get("metadata", {})
                dokument_id = metadata.get("dokument_id", "unknown")
                input_text = str(example.get("input", ""))
                output_text = str(example.get("output", ""))
                
                input_emb = _create_embedding(input_text, model, tokenizer, device)
                output_emb = _create_embedding(output_text, model, tokenizer, device)
                
                if input_emb is not None and output_emb is not None:
                    embeddings_dict[dokument_id] = {
                        "input_embedding": input_emb,
                        "output_embedding": output_emb
                    }
                    embeddings_data.append({
                        "dokument_id": dokument_id,
                        "input_embedding": input_emb.tolist(),
                        "output_embedding": output_emb.tolist()
                    })
            
            if embeddings_data:
                _save_embeddings(embeddings_path, embeddings_data)
                print(f"Saved {len(embeddings_data)} embeddings to {embeddings_path.name}", file=sys.stderr)
    
    # If embeddings file exists but not loaded, load it (and filter out invalid documents)
    if not embeddings_dict and embeddings_path.exists():
        embeddings_dict = _load_embeddings(embeddings_path, data_file_path=file_path)
        print(f"Loaded {len(embeddings_dict)} embeddings from {embeddings_path.name}", file=sys.stderr)
    
    # Analyze the dataset
    total_examples = 0
    doc_type_counter: Counter[str] = Counter()
    kommune_counter: Counter[int] = Counter()
    summary_lengths: List[int] = []
    input_lengths: List[int] = []
    summary_token_lengths: List[int] = []
    input_token_lengths: List[int] = []
    cosine_distances: List[float] = []
    samples: List[Dict[str, Any]] = []
    
    # Collect embeddings and doc_types for 2D projection
    input_embeddings_list: List[np.ndarray] = []
    output_embeddings_list: List[np.ndarray] = []
    doc_types_list: List[str] = []
    
    # Count documents with insufficient alphanumeric characters
    input_low_alnum_count = 0
    output_low_alnum_count = 0
    input_low_alnum_doc_types: Counter[str] = Counter()
    output_low_alnum_doc_types: Counter[str] = Counter()
    
    for example in _iter_examples(file_path):
        total_examples += 1
        
        # Extract metadata
        metadata = example.get("metadata", {})
        doc_type = metadata.get("doc_type")
        kommune = metadata.get("kommune")
        
        # Extract text lengths (characters and tokens)
        # Convert to string in case they're not (e.g., float, None)
        input_text = str(example.get("input", ""))
        output_text = str(example.get("output", ""))
        
        # Check for insufficient alphanumeric characters
        input_alnum = _count_alphanumeric(input_text)
        output_alnum = _count_alphanumeric(output_text)
        
        # Track short texts/summaries and exclude from analysis
        has_short_input = input_alnum < 10
        has_short_output = output_alnum < 10
        
        if has_short_input:
            input_low_alnum_count += 1
            # Track doc_type for short input texts
            if doc_type:
                input_low_alnum_doc_types[doc_type] += 1
            else:
                input_low_alnum_doc_types["unknown"] += 1
        
        if has_short_output:
            output_low_alnum_count += 1
            # Track doc_type for short summaries
            if doc_type:
                output_low_alnum_doc_types[doc_type] += 1
            else:
                output_low_alnum_doc_types["unknown"] += 1
        
        # Exclude documents with short texts or summaries from further analysis
        if has_short_input or has_short_output:
            continue
        
        # Only process documents with sufficient alphanumeric characters
        if doc_type:
            doc_type_counter[doc_type] += 1
        
        if kommune is not None:
            kommune_counter[kommune] += 1
        
        input_lengths.append(len(input_text))
        summary_lengths.append(len(output_text))
        input_token_lengths.append(_count_tokens(input_text))
        summary_token_lengths.append(_count_tokens(output_text))
        
        # Calculate cosine distance if embeddings available
        dokument_id = metadata.get("dokument_id")
        if dokument_id and dokument_id in embeddings_dict:
            input_emb = embeddings_dict[dokument_id]["input_embedding"]
            output_emb = embeddings_dict[dokument_id]["output_embedding"]
            cosine_dist = _cosine_distance(input_emb, output_emb)
            if not np.isnan(cosine_dist):
                cosine_distances.append(cosine_dist)
            
            # Collect embeddings for 2D projection
            input_embeddings_list.append(input_emb)
            output_embeddings_list.append(output_emb)
            doc_types_list.append(doc_type if doc_type else "unknown")
        
        # Collect samples if requested
        if num_examples is not None and len(samples) < num_examples:
            samples.append({
                "dokument_id": metadata.get("dokument_id"),
                "doc_type": doc_type,
                "kommune": kommune,
                "input_length": len(input_text),
                "output_length": len(output_text),
                "nyhetsverdi": metadata.get("nyhetsverdi"),
            })
    
    # Print dataset analysis header before warnings
    print("=" * 80, file=sys.stderr)
    print(f"Dataset Analysis: {folder_name}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(file=sys.stderr)
    
    # Print summary warnings for documents with insufficient alphanumeric characters
    if input_low_alnum_count > 0:
        print(
            f"WARNING:\n{input_low_alnum_count:,} document(s) have input text with "
            f"less than 10 alphanumeric characters",
            file=sys.stderr
        )
        # Print document type distribution for short input texts
        if input_low_alnum_doc_types:
            total_short = sum(input_low_alnum_doc_types.values())
            for doc_type, count in sorted(
                input_low_alnum_doc_types.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (count / total_short) * 100
                print(
                    f"  {doc_type}: {count} short texts ({percentage:.1f}%)",
                    file=sys.stderr
                )
        print()
    
    if output_low_alnum_count > 0:
        print(
            f"WARNING:\n{output_low_alnum_count:,} document(s) have summary text with "
            f"less than 10 alphanumeric characters",
            file=sys.stderr
        )
        # Print document type distribution for short summaries
        if output_low_alnum_doc_types:
            total_short = sum(output_low_alnum_doc_types.values())
            for doc_type, count in sorted(
                output_low_alnum_doc_types.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (count / total_short) * 100
                print(
                    f"  {doc_type}: {count} short summaries ({percentage:.1f}%)",
                    file=sys.stderr
                )
        print()
    
    return AnalysisResult(
        dataset_name=folder_name,
        path=str(file_path),
        total_examples=total_examples,
        doc_type_counter=dict(doc_type_counter),
        kommune_counter=dict(kommune_counter),
        summary_lengths=summary_lengths,
        input_lengths=input_lengths,
        summary_token_lengths=summary_token_lengths,
        input_token_lengths=input_token_lengths,
        cosine_distances=cosine_distances,
        input_embeddings=input_embeddings_list,
        output_embeddings=output_embeddings_list,
        doc_types=doc_types_list,
        samples=samples,
    )


def _plot_histograms(result: AnalysisResult):
    """Plot side-by-side histograms for input and summary text lengths (in tokens) and cosine distances."""
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib is not available. Histograms will not be displayed.", file=sys.stderr)
        return
    
    if not result.input_token_lengths or not result.summary_token_lengths:
        if not TIKTOKEN_AVAILABLE:
            print("Warning: tiktoken is not available. Histograms will not be displayed.", file=sys.stderr)
        return
    
    # Calculate max token length for x-axis (separate for each histogram)
    max_input_tokens = max(result.input_token_lengths)
    max_summary_tokens = max(result.summary_token_lengths)
    
    # Cap input tokens at 10000 for histogram (values >= 10000 go to the last bin)
    input_tokens_capped = [min(tokens, 10000) for tokens in result.input_token_lengths]
    input_max_for_hist = 10000
    
    # Determine if we have cosine distances
    has_cosine_distances = result.cosine_distances and len(result.cosine_distances) > 0
    
    # Create figure with three subplots side by side
    if has_cosine_distances:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot input text token length histogram (capped at 10000)
    ax1.hist(input_tokens_capped, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Length (tokens)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Input Text Length Distribution')
    ax1.set_xlim(0, input_max_for_hist)
    ax1.grid(True, alpha=0.3)
    
    # Plot summary text token length histogram
    ax2.hist(result.summary_token_lengths, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax2.set_xlabel('Length (tokens)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Summary Text Length Distribution')
    ax2.set_xlim(0, max_summary_tokens)
    ax2.grid(True, alpha=0.3)
    
    # Plot cosine distance histogram if available
    if has_cosine_distances:
        max_cosine_dist = max(result.cosine_distances)
        ax3.hist(result.cosine_distances, bins=50, edgecolor='black', alpha=0.7, color='green')
        ax3.set_xlabel('Cosine Distance')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Text-Summary Embedding Distance')
        ax3.set_xlim(0, max_cosine_dist)
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure to file (Agg backend doesn't support plt.show())
    output_file = Path(result.path).parent / f"length_distributions_{Path(result.path).stem}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Histograms saved to: {output_file}", file=sys.stderr)
    print()


def _project_embeddings_2d(embeddings: List[np.ndarray], use_pca: bool = True) -> Optional[np.ndarray]:
    """Project embeddings to 2D using UMAP with cosine metric. Optionally use PCA pre-step."""
    if not UMAP_AVAILABLE:
        return None
    
    if not embeddings or len(embeddings) == 0:
        return None
    
    try:
        # Stack embeddings into a matrix
        embeddings_matrix = np.vstack(embeddings)
        
        # Optional PCA pre-step for BERT-like embeddings
        if use_pca and len(embeddings_matrix[0]) > 200:
            try:
                from sklearn.decomposition import PCA
                # Reduce to 100 dims (good balance for BERT-large)
                pca = PCA(n_components=min(100, len(embeddings_matrix[0]) - 1))
                embeddings_matrix = pca.fit_transform(embeddings_matrix)
            except ImportError:
                # sklearn not available, skip PCA
                pass
        
        # UMAP with cosine metric
        # Suppress UMAP warning about n_jobs when random_state is set
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*n_jobs.*random_state.*")
            reducer = umap.UMAP(
                n_components=2,
                metric='cosine',
                n_neighbors=min(30, len(embeddings) - 1),  # Adjust based on data size
                min_dist=0.1,
                random_state=42
            )
            projection = reducer.fit_transform(embeddings_matrix)
        return projection
    except Exception as e:
        print(f"Warning: Failed to project embeddings to 2D: {e}", file=sys.stderr)
        return None


def _plot_embedding_projections(result: AnalysisResult):
    """Plot 2D projections of input and output embeddings, colored by document type."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    if not UMAP_AVAILABLE:
        print("Warning: UMAP is not available. 2D projections will not be displayed.", file=sys.stderr)
        return
    
    if not result.input_embeddings or not result.output_embeddings:
        return
    
    # Project embeddings to 2D
    input_projection = _project_embeddings_2d(result.input_embeddings)
    output_projection = _project_embeddings_2d(result.output_embeddings)
    
    if input_projection is None or output_projection is None:
        return
    
    # Get unique document types and assign colors
    unique_doc_types = sorted(set(result.doc_types))
    # Use a colormap for different document types (suppress deprecation warning)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
        try:
            cmap = matplotlib.colormaps.get_cmap('tab10')
        except AttributeError:
            # Fallback for older matplotlib versions
            cmap = plt.cm.get_cmap('tab10')
    colors = {doc_type: cmap(i % 10) for i, doc_type in enumerate(unique_doc_types)}
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot input embeddings
    for doc_type in unique_doc_types:
        mask = np.array([dt == doc_type for dt in result.doc_types])
        if np.any(mask):
            ax1.scatter(
                input_projection[mask, 0],
                input_projection[mask, 1],
                c=[colors[doc_type]],
                label=doc_type,
                alpha=0.6,
                s=10
            )
    ax1.set_xlabel('UMAP Dimension 1')
    ax1.set_ylabel('UMAP Dimension 2')
    ax1.set_title('Input Text Embeddings (2D Projection)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot output embeddings
    for doc_type in unique_doc_types:
        mask = np.array([dt == doc_type for dt in result.doc_types])
        if np.any(mask):
            ax2.scatter(
                output_projection[mask, 0],
                output_projection[mask, 1],
                c=[colors[doc_type]],
                label=doc_type,
                alpha=0.6,
                s=10
            )
    ax2.set_xlabel('UMAP Dimension 1')
    ax2.set_ylabel('UMAP Dimension 2')
    ax2.set_title('Summary Text Embeddings (2D Projection)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = Path(result.path).parent / f"embedding_projections_{Path(result.path).stem}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Embedding projections saved to: {output_file}", file=sys.stderr)
    print()


def _print_result(result: AnalysisResult):
    # Header already printed before warnings, skip it here
    print(f"Dataset path: {result.path}")
    print()
    
    print(f"Total examples: {result.total_examples:,}")
    print()
    
    # Calculate statistics
    if result.input_lengths:
        avg_input = sum(result.input_lengths) / len(result.input_lengths)
        min_input = min(result.input_lengths)
        max_input = max(result.input_lengths)
        
        # Calculate token statistics if available
        if result.input_token_lengths and TIKTOKEN_AVAILABLE:
            avg_input_tokens = sum(result.input_token_lengths) / len(result.input_token_lengths)
            min_input_tokens = min(result.input_token_lengths)
            max_input_tokens = max(result.input_token_lengths)
            token_str = f", ~{avg_input_tokens:.1f} tiktokens"
            min_token_str = f", ~{min_input_tokens:.0f} tiktokens"
            max_token_str = f", ~{max_input_tokens:.0f} tiktokens"
        else:
            token_str = ""
            min_token_str = ""
            max_token_str = ""
        
        print("INPUT TEXT STATISTICS:")
        print(f"  Average length: {avg_input:.1f} characters{token_str}")
        print(f"  Min length: {min_input:,} characters{min_token_str}")
        print(f"  Max length: {max_input:,} characters{max_token_str}")
        
        # Calculate percentage of inputs > 2048 tiktokens
        if result.input_token_lengths and TIKTOKEN_AVAILABLE:
            count_above_2048 = sum(1 for tokens in result.input_token_lengths if tokens > 2048)
            percentage_above_2048 = (count_above_2048 / len(result.input_token_lengths)) * 100
            print(f"  Percentage length > 2048 tiktokens: {percentage_above_2048:.1f}%")
        
        print()
    
    if result.summary_lengths:
        avg_output = sum(result.summary_lengths) / len(result.summary_lengths)
        min_output = min(result.summary_lengths)
        max_output = max(result.summary_lengths)
        
        # Calculate token statistics if available
        if result.summary_token_lengths and TIKTOKEN_AVAILABLE:
            avg_output_tokens = sum(result.summary_token_lengths) / len(result.summary_token_lengths)
            min_output_tokens = min(result.summary_token_lengths)
            max_output_tokens = max(result.summary_token_lengths)
            token_str = f", ~{avg_output_tokens:.1f} tiktokens"
            min_token_str = f", ~{min_output_tokens:.0f} tiktokens"
            max_token_str = f", ~{max_output_tokens:.0f} tiktokens"
        else:
            token_str = ""
            min_token_str = ""
            max_token_str = ""
        
        print("SUMMARY TEXT STATISTICS:")
        print(f"  Average length: {avg_output:.1f} characters{token_str}")
        print(f"  Min length: {min_output:,} characters{min_token_str}")
        print(f"  Max length: {max_output:,} characters{max_token_str}")
        print()
    
    # Cosine distance statistics
    if result.cosine_distances and len(result.cosine_distances) > 0:
        avg_distance = sum(result.cosine_distances) / len(result.cosine_distances)
        min_distance = min(result.cosine_distances)
        max_distance = max(result.cosine_distances)
        print("TEXT-SUMMARY EMBEDDING DISTANCE STATISTICS:")
        print(f"  Average distance: {avg_distance:.4f}")
        print(f"  Min distance: {min_distance:.4f}")
        print(f"  Max distance: {max_distance:.4f}")
        print()
    
    # Plot histograms
    _plot_histograms(result)
    
    # Plot 2D embedding projections
    _plot_embedding_projections(result)
    
    print("=" * 80)
    print("DOCUMENT TYPE DISTRIBUTION")
    print("=" * 80)
    if result.doc_type_counter:
        total = sum(result.doc_type_counter.values()) or 1
        for doc_type, count in sorted(
            result.doc_type_counter.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {doc_type}: {count:,} examples ({(count/total)*100:.1f}%)")
    else:
        print("  No document types found.")
    print()
    
    print("=" * 80)
    print("KOMMUNE DISTRIBUTION (top 10)")
    print("=" * 80)
    for kommune_nummer, count in sorted(
        result.kommune_counter.items(), key=lambda x: x[1], reverse=True
    )[:10]:
        navn = kommunenavn(kommune_nummer)
        print(f"  {navn} ({kommune_nummer}): {count:,} examples")
    print()
    
    if result.samples:
        print("=" * 80)
        print("SAMPLE EXAMPLES")
        print("=" * 80)
        for i, sample in enumerate(result.samples, 1):
            print(f"Example {i}:")
            print(f"  Document ID: {sample.get('dokument_id')}")
            print(f"  Document Type: {sample.get('doc_type', 'N/A')}")
            kommune_nummer = sample.get('kommune')
            if kommune_nummer is not None:
                navn = kommunenavn(kommune_nummer)
                print(f"  Kommune: {navn} ({kommune_nummer})")
            else:
                print(f"  Kommune: N/A")
            print(f"  Input length: {sample.get('input_length', 0):,} characters")
            print(f"  Output length: {sample.get('output_length', 0):,} characters")
            if sample.get('nyhetsverdi') is not None:
                print(f"  Nyhetsverdi: {sample.get('nyhetsverdi')}")
            print()


def _save_analysis_results(result: AnalysisResult):
    """Save analysis results to a JSON file."""
    # Get the data file path and construct output filename
    data_file_path = Path(result.path)
    output_file = data_file_path.parent / f"{data_file_path.stem}_analysis_results.json"
    
    # Calculate statistics
    stats = {
        "dataset_name": result.dataset_name,
        "data_file": result.path,
        "total_examples": result.total_examples,
        "doc_type_distribution": dict(result.doc_type_counter),
    }
    
    # Add kommune names to kommune distribution
    kommune_distribution_with_names = {}
    for kommune_nummer, count in result.kommune_counter.items():
        navn = kommunenavn(kommune_nummer)
        kommune_distribution_with_names[f"{navn} ({kommune_nummer})"] = count
    stats["kommune_distribution"] = kommune_distribution_with_names
    
    # Input text statistics
    if result.input_lengths:
        stats["input_text"] = {
            "average_length_chars": sum(result.input_lengths) / len(result.input_lengths),
            "min_length_chars": min(result.input_lengths),
            "max_length_chars": max(result.input_lengths),
        }
        if result.input_token_lengths and TIKTOKEN_AVAILABLE:
            stats["input_text"]["average_length_tokens"] = sum(result.input_token_lengths) / len(result.input_token_lengths)
            stats["input_text"]["min_length_tokens"] = min(result.input_token_lengths)
            stats["input_text"]["max_length_tokens"] = max(result.input_token_lengths)
            count_above_2048 = sum(1 for tokens in result.input_token_lengths if tokens > 2048)
            stats["input_text"]["percentage_above_2048_tokens"] = (count_above_2048 / len(result.input_token_lengths)) * 100
    
    # Summary text statistics
    if result.summary_lengths:
        stats["summary_text"] = {
            "average_length_chars": sum(result.summary_lengths) / len(result.summary_lengths),
            "min_length_chars": min(result.summary_lengths),
            "max_length_chars": max(result.summary_lengths),
        }
        if result.summary_token_lengths and TIKTOKEN_AVAILABLE:
            stats["summary_text"]["average_length_tokens"] = sum(result.summary_token_lengths) / len(result.summary_token_lengths)
            stats["summary_text"]["min_length_tokens"] = min(result.summary_token_lengths)
            stats["summary_text"]["max_length_tokens"] = max(result.summary_token_lengths)
    
    # Cosine distance statistics
    if result.cosine_distances and len(result.cosine_distances) > 0:
        stats["embedding_distance"] = {
            "average_distance": sum(result.cosine_distances) / len(result.cosine_distances),
            "min_distance": min(result.cosine_distances),
            "max_distance": max(result.cosine_distances),
        }
    
    # Save to JSON
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        print(f"Analysis results saved to: {output_file}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Failed to save analysis results: {e}", file=sys.stderr)


def _union_find_connected_components(edges: List[tuple[int, int]], num_nodes: int) -> List[List[int]]:
    """
    Find connected components using Union-Find (Disjoint Set Union) algorithm.
    
    Args:
        edges: List of (i, j) pairs representing edges in the graph
        num_nodes: Total number of nodes
    
    Returns:
        List of lists, where each inner list contains node indices in a connected component
    """
    # Initialize parent array for union-find
    parent = list(range(num_nodes))
    rank = [0] * num_nodes
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]
    
    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            # Union by rank
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            elif rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_y] = root_x
                rank[root_x] += 1
    
    # Union all edges
    for i, j in edges:
        union(i, j)
    
    # Collect components
    components_dict = {}
    for i in range(num_nodes):
        root = find(i)
        if root not in components_dict:
            components_dict[root] = []
        components_dict[root].append(i)
    
    return list(components_dict.values())


def _star_cluster(
    cluster: List[int],
    similarity_matrix: np.ndarray,
    threshold: float
) -> List[List[int]]:
    """
    Apply star clustering to a connected component to avoid chain over-clustering.
    
    For each cluster, picks a representative (highest degree node) and only includes
    items that are within threshold to the representative.
    
    Args:
        cluster: List of node indices in the component
        similarity_matrix: Precomputed similarity matrix (cosine similarity for L2-normalized vectors)
        threshold: Similarity threshold
    
    Returns:
        List of star clusters (sub-clusters within the input cluster)
    """
    if len(cluster) <= 1:
        return [cluster]
    
    # Build subgraph of similarities within this cluster
    cluster_similarities = {}
    for i in cluster:
        for j in cluster:
            if i < j:
                sim = similarity_matrix[i, j]
                cluster_similarities[(i, j)] = sim
    
    # Count degrees (number of neighbors above threshold) within cluster
    degrees = {node: 0 for node in cluster}
    for (i, j), sim in cluster_similarities.items():
        if sim >= threshold:
            degrees[i] += 1
            degrees[j] += 1
    
    # Pick representative (highest degree, tie-break by smallest index)
    representative = max(cluster, key=lambda x: (degrees[x], -x))
    
    # Build star: representative + all nodes directly connected to it above threshold
    star_cluster = [representative]
    for node in cluster:
        if node != representative:
            # Since matrix is symmetric, we can access either direction
            sim = similarity_matrix[representative, node]
            if sim >= threshold:
                star_cluster.append(node)
    
    # Handle remaining nodes (if any) recursively
    remaining = [node for node in cluster if node not in star_cluster]
    if remaining:
        # Recursively cluster remaining nodes
        remaining_clusters = _star_cluster(remaining, similarity_matrix, threshold)
        return [star_cluster] + remaining_clusters
    else:
        return [star_cluster]


def _find_duplicates_single_type(
    embeddings_dict: Dict[str, Dict[str, np.ndarray]],
    similarity_threshold: float,
    k_neighbors: int,
    use_star_clustering: bool,
    embedding_type: str,
    comparison_name: str,
    low_memory: bool = False,
) -> Dict[str, Any]:
    """
    Find duplicates for a single comparison type (input comparison or summary comparison).
    
    Args:
        embeddings_dict: Dictionary mapping dokument_id to embeddings
        similarity_threshold: Minimum cosine similarity to consider as duplicate
        k_neighbors: Number of nearest neighbors to search
        use_star_clustering: If True, apply star clustering
        embedding_type: "input" or "output"
        comparison_name: Human-readable name for this comparison type
    
    Returns:
        Dictionary with clusters, statistics, etc.
    """
    if not FAISS_AVAILABLE:
        raise RuntimeError("FAISS is not available. Please install it: pip install faiss-cpu (or use conda: conda install -c pytorch faiss-gpu for GPU support)")
    
    # Extract document IDs and embeddings
    doc_ids = list(embeddings_dict.keys())
    embeddings = []
    valid_doc_ids = []
    
    # Standard comparison: same embedding type vs same embedding type
    for doc_id in doc_ids:
        emb_key = f"{embedding_type}_embedding"
        if emb_key in embeddings_dict[doc_id]:
            emb = embeddings_dict[doc_id][emb_key]
            if emb is not None and len(emb) > 0:
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                    embeddings.append(emb)
                    valid_doc_ids.append(doc_id)
    
    if len(embeddings) < 2:
        raise ValueError(f"Need at least 2 documents with {embedding_type} embeddings, found {len(embeddings)}")
    
    print(f"Found {len(embeddings)} documents for {comparison_name} comparison", file=sys.stderr)
    
    # Stack embeddings into a matrix (N x D)
    embeddings_matrix = np.vstack(embeddings).astype(np.float32)
    num_docs, dim = embeddings_matrix.shape
    
    print(f"Building FAISS index for {num_docs} documents (dim={dim})...", file=sys.stderr)
    
    # Build FAISS index (IndexFlatIP for inner product = cosine similarity for L2-normalized vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_matrix)
    
    # Search for k_neighbors (we'll search k_neighbors+1 to exclude self)
    k_search = min(k_neighbors + 1, num_docs)
    
    print(f"Searching for {k_neighbors} nearest neighbors per document (similarity >= {similarity_threshold:.4f})...", file=sys.stderr)
    
    # Find neighbors
    similarities, indices = index.search(embeddings_matrix, k_search)
    
    # Build edges: (i, j) pairs where similarity >= threshold (excluding self)
    # Also build similarity matrix for star clustering (if enabled and not in low-memory mode)
    edges = []
    similarity_pairs = {}
    
    similarity_matrix = None
    use_star_clustering_effective = use_star_clustering and not low_memory
    if use_star_clustering_effective:
        try:
            similarity_matrix = np.zeros((num_docs, num_docs), dtype=np.float32)
        except MemoryError:
            # Fall back to low-memory mode without a full similarity matrix
            print(
                "Warning: Failed to allocate full similarity matrix for star clustering "
                f"({num_docs}x{num_docs}). Falling back to low-memory duplicate detection "
                "without star clustering.",
                file=sys.stderr,
            )
            similarity_matrix = None
            use_star_clustering_effective = False
    
    for i in tqdm(range(num_docs), desc=f"Finding {comparison_name} pairs", unit="docs"):
        for idx_in_results, j in enumerate(indices[i]):
            if j == i:  # Skip self
                continue
            sim = float(similarities[i, idx_in_results])
            # Store in similarity matrix (symmetric) if available
            if similarity_matrix is not None:
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
            
            if sim >= similarity_threshold:
                # Store edge with smaller index first for consistency
                edge = (min(i, j), max(i, j))
                if edge not in similarity_pairs:
                    edges.append(edge)
                    similarity_pairs[edge] = sim
    
    print(f"Found {len(edges)} {comparison_name} pairs above similarity threshold {similarity_threshold:.4f}", file=sys.stderr)
    
    # Build connected components
    print("Building connected components...", file=sys.stderr)
    components = _union_find_connected_components(edges, num_docs)
    
    # Filter to only components with 2+ documents (duplicates)
    duplicate_components = [comp for comp in components if len(comp) >= 2]
    
    print(f"Found {len(duplicate_components)} {comparison_name} clusters ({len(components) - len(duplicate_components)} singletons)", file=sys.stderr)
    
    # Optionally apply star clustering (only if we successfully built a similarity matrix)
    if use_star_clustering_effective and similarity_matrix is not None and duplicate_components:
        print(f"Applying star clustering to {comparison_name} clusters...", file=sys.stderr)
        star_clusters = []
        for comp in tqdm(duplicate_components, desc="Star clustering", unit="components"):
            sub_clusters = _star_cluster(comp, similarity_matrix, similarity_threshold)
            star_clusters.extend(sub_clusters)
        duplicate_components = star_clusters
        print(f"After star clustering: {len(duplicate_components)} {comparison_name} clusters", file=sys.stderr)
    
    # Convert component indices to dokument_ids
    clusters = [[valid_doc_ids[idx] for idx in comp] for comp in duplicate_components]
    cluster_sizes = [len(cluster) for cluster in clusters]
    
    # Calculate statistics
    total_duplicate_docs = sum(cluster_sizes)
    num_singletons = num_docs - total_duplicate_docs
    num_unique_docs = num_singletons + len(clusters)
    
    statistics = {
        "total_documents": num_docs,
        "num_clusters": len(clusters),
        "total_duplicate_documents": total_duplicate_docs,
        "num_singletons": num_singletons,
        "num_unique_documents": num_unique_docs,
        "largest_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "similarity_threshold": similarity_threshold,
        "k_neighbors": k_neighbors,
        "use_star_clustering": use_star_clustering,
        "comparison_type": comparison_name
    }
    
    # Build doc_id to index mapping
    doc_id_to_index = {doc_id: i for i, doc_id in enumerate(valid_doc_ids)}
    
    return {
        "clusters": clusters,
        "cluster_sizes": cluster_sizes,
        "similarity_pairs": similarity_pairs,
        "statistics": statistics,
        "doc_id_to_index": doc_id_to_index
    }


def find_duplicates(
    embeddings_path: Path,
    similarity_threshold: float = 0.99,
    k_neighbors: int = 50,
    use_star_clustering: bool = False,
    text_similarity: bool = True,
    summary_similarity: bool = True,
    low_memory: bool = False,
) -> Dict[str, Any]:
    """
    Find duplicate and near-duplicate documents using FAISS k-NN search and clustering.
    
    Compares:
    - Input comparison: All input texts against each other
    - Summary comparison: All output summaries against each other
    
    Args:
        embeddings_path: Path to the embeddings JSONL file
        similarity_threshold: Minimum cosine similarity to consider as duplicate (0.99 = very similar)
        k_neighbors: Number of nearest neighbors to search for each document
        use_star_clustering: If True, apply star clustering to avoid chain over-clustering
        text_similarity: If True, compare all input texts against each other
        summary_similarity: If True, compare all output summaries against each other
    
    Returns:
        Dictionary containing:
        - text_clusters: List of clusters from input comparison
        - summary_clusters: List of clusters from summary comparison
        - statistics: Combined statistics about all duplicate detection types
    """
    if not FAISS_AVAILABLE:
        raise RuntimeError("FAISS is not available. Please install it: pip install faiss-cpu (or use conda: conda install -c pytorch faiss-gpu for GPU support)")
    
    print(f"Loading embeddings from {embeddings_path}...", file=sys.stderr)
    
    # Find corresponding data file to filter embeddings
    data_file = embeddings_path.parent / embeddings_path.name.replace("_embeddings.jsonl", ".jsonl")
    if not data_file.exists():
        # Try to find any matching data file
        all_jsonl = list(embeddings_path.parent.glob("*.jsonl"))
        plain_files = [
            f for f in all_jsonl
            if not any(f.name.endswith(f"_{s}.jsonl") for s in ["train", "val", "test"])
            and not f.name.endswith("embeddings.jsonl")
        ]
        if plain_files:
            data_file = plain_files[0]
    
    embeddings_dict = _load_embeddings(embeddings_path, data_file_path=data_file if data_file.exists() else None)
    
    if not embeddings_dict:
        raise ValueError(f"No embeddings found in {embeddings_path}")
    
    results = {}
    all_statistics = []
    
    # Input comparison
    if text_similarity:
        print("\n" + "=" * 80, file=sys.stderr)
        print("INPUT COMPARISON", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        try:
            text_result = _find_duplicates_single_type(
                embeddings_dict, similarity_threshold, k_neighbors, use_star_clustering,
                "input", "input_comparison", low_memory=low_memory
            )
            results["text_clusters"] = text_result["clusters"]
            results["text_statistics"] = text_result["statistics"]
            results["text_similarity_pairs"] = text_result["similarity_pairs"]
            results["text_doc_id_to_index"] = text_result["doc_id_to_index"]
            all_statistics.append(("input_comparison", text_result["statistics"]))
        except Exception as e:
            print(f"Error in input comparison: {e}", file=sys.stderr)
            results["text_clusters"] = []
            results["text_statistics"] = {}
    
    # Summary comparison
    if summary_similarity:
        print("\n" + "=" * 80, file=sys.stderr)
        print("SUMMARY COMPARISON", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        try:
            summary_result = _find_duplicates_single_type(
                embeddings_dict, similarity_threshold, k_neighbors, use_star_clustering,
                "output", "summary_comparison", low_memory=low_memory
            )
            results["summary_clusters"] = summary_result["clusters"]
            results["summary_statistics"] = summary_result["statistics"]
            results["summary_similarity_pairs"] = summary_result["similarity_pairs"]
            results["summary_doc_id_to_index"] = summary_result["doc_id_to_index"]
            all_statistics.append(("summary_comparison", summary_result["statistics"]))
        except Exception as e:
            print(f"Error in summary comparison: {e}", file=sys.stderr)
            results["summary_clusters"] = []
            results["summary_statistics"] = {}
    
    # Input and Summary comparison (documents that are duplicates in both input AND summary)
    if text_similarity and summary_similarity and "text_clusters" in results and "summary_clusters" in results:
        print("\n" + "=" * 80, file=sys.stderr)
        print("INPUT AND SUMMARY COMPARISON", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        try:
            # Build sets of document pairs from input clusters
            input_pairs = set()
            for cluster in results["text_clusters"]:
                for i, doc_id1 in enumerate(cluster):
                    for j, doc_id2 in enumerate(cluster):
                        if i < j:
                            # Store as sorted tuple for consistency
                            pair = tuple(sorted([doc_id1, doc_id2]))
                            input_pairs.add(pair)
            
            # Build sets of document pairs from summary clusters
            summary_pairs = set()
            for cluster in results["summary_clusters"]:
                for i, doc_id1 in enumerate(cluster):
                    for j, doc_id2 in enumerate(cluster):
                        if i < j:
                            # Store as sorted tuple for consistency
                            pair = tuple(sorted([doc_id1, doc_id2]))
                            summary_pairs.add(pair)
            
            # Find pairs that are in BOTH input and summary clusters
            intersection_pairs = input_pairs & summary_pairs
            
            print(f"Found {len(intersection_pairs)} document pairs that are duplicates in both input and summary", file=sys.stderr)
            
            if intersection_pairs:
                # Build clusters from intersection pairs using union-find
                # First, collect all unique document IDs
                all_doc_ids = set()
                for pair in intersection_pairs:
                    all_doc_ids.update(pair)
                
                # Create mapping from doc_id to index
                doc_id_list = list(all_doc_ids)
                doc_id_to_index = {doc_id: i for i, doc_id in enumerate(doc_id_list)}
                
                # Build edges from intersection pairs
                edges = []
                similarity_pairs = {}
                text_doc_id_to_index = results.get("text_doc_id_to_index", {})
                summary_doc_id_to_index = results.get("summary_doc_id_to_index", {})
                text_similarity_pairs = results.get("text_similarity_pairs", {})
                summary_similarity_pairs = results.get("summary_similarity_pairs", {})
                
                for pair in intersection_pairs:
                    idx1 = doc_id_to_index[pair[0]]
                    idx2 = doc_id_to_index[pair[1]]
                    edges.append((idx1, idx2))
                    
                    # Look up similarities from input and summary comparisons
                    input_sim = None
                    summary_sim = None
                    
                    # Try to get input similarity
                    text_idx1 = text_doc_id_to_index.get(pair[0])
                    text_idx2 = text_doc_id_to_index.get(pair[1])
                    if text_idx1 is not None and text_idx2 is not None:
                        edge_key1 = (min(text_idx1, text_idx2), max(text_idx1, text_idx2))
                        edge_key2 = (text_idx1, text_idx2)
                        input_sim = text_similarity_pairs.get(edge_key1) or text_similarity_pairs.get(edge_key2)
                    
                    # Try to get summary similarity
                    summary_idx1 = summary_doc_id_to_index.get(pair[0])
                    summary_idx2 = summary_doc_id_to_index.get(pair[1])
                    if summary_idx1 is not None and summary_idx2 is not None:
                        edge_key1 = (min(summary_idx1, summary_idx2), max(summary_idx1, summary_idx2))
                        edge_key2 = (summary_idx1, summary_idx2)
                        summary_sim = summary_similarity_pairs.get(edge_key1) or summary_similarity_pairs.get(edge_key2)
                    
                    # Use minimum similarity (more conservative)
                    if input_sim is not None and summary_sim is not None:
                        sim = min(input_sim, summary_sim)
                    elif input_sim is not None:
                        sim = input_sim
                    elif summary_sim is not None:
                        sim = summary_sim
                    else:
                        sim = similarity_threshold  # Default if not found
                    similarity_pairs[(idx1, idx2)] = sim
                    similarity_pairs[(idx2, idx1)] = sim
                
                # Build connected components
                print("Building connected components...", file=sys.stderr)
                components = _union_find_connected_components(edges, len(doc_id_list))
                duplicate_components = [comp for comp in components if len(comp) >= 2]
                
                print(f"Found {len(duplicate_components)} input_and_summary_comparison clusters ({len(components) - len(duplicate_components)} singletons)", file=sys.stderr)
                
                # Convert component indices to dokument_ids
                clusters = [[doc_id_list[idx] for idx in comp] for comp in duplicate_components]
                cluster_sizes = [len(cluster) for cluster in clusters]
                
                # Calculate statistics
                total_duplicate_docs = sum(cluster_sizes)
                num_singletons = len(doc_id_list) - total_duplicate_docs
                num_unique_docs = num_singletons + len(clusters)
                
                statistics = {
                    "total_documents": len(doc_id_list),
                    "num_clusters": len(clusters),
                    "total_duplicate_documents": total_duplicate_docs,
                    "num_singletons": num_singletons,
                    "num_unique_documents": num_unique_docs,
                    "largest_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
                    "similarity_threshold": similarity_threshold,
                    "k_neighbors": k_neighbors,
                    "use_star_clustering": use_star_clustering,
                    "comparison_type": "input_and_summary_comparison"
                }
                
                results["input_and_summary_clusters"] = clusters
                results["input_and_summary_statistics"] = statistics
                results["input_and_summary_similarity_pairs"] = similarity_pairs
                results["input_and_summary_doc_id_to_index"] = doc_id_to_index
                all_statistics.append(("input_and_summary_comparison", statistics))
            else:
                print("No document pairs found that are duplicates in both input and summary", file=sys.stderr)
                results["input_and_summary_clusters"] = []
                results["input_and_summary_statistics"] = {}
                results["input_and_summary_similarity_pairs"] = {}
                results["input_and_summary_doc_id_to_index"] = {}
        except Exception as e:
            print(f"Error in input and summary comparison: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            results["input_and_summary_clusters"] = []
            results["input_and_summary_statistics"] = {}
    
    # Combined statistics
    combined_stats = {
        "similarity_threshold": similarity_threshold,
        "k_neighbors": k_neighbors,
        "use_star_clustering": use_star_clustering,
        "comparisons_performed": [name for name, _ in all_statistics]
    }
    
    for name, stats in all_statistics:
        for key, value in stats.items():
            if key not in ["similarity_threshold", "k_neighbors", "use_star_clustering", "comparison_type"]:
                combined_stats[f"{name}_{key}"] = value
    
    results["statistics"] = combined_stats
    
    return results


def _order_cluster_by_similarity(
    cluster: List[str],
    similarity_pairs: Dict,
    doc_id_to_index: Dict[str, int]
) -> List[str]:
    """
    Order documents in a cluster by similarity.
    The first two documents form the most similar pair.
    Approximately the first 10 documents are ordered by similarity to already-ordered documents.
    The rest are appended in any order.
    """
    if len(cluster) <= 1:
        return cluster
    
    # Build similarity matrix for documents in this cluster
    cluster_similarities = {}
    for i, doc_id1 in enumerate(cluster):
        idx1 = doc_id_to_index.get(doc_id1)
        if idx1 is None:
            continue
        for j, doc_id2 in enumerate(cluster):
            if i >= j:
                continue
            idx2 = doc_id_to_index.get(doc_id2)
            if idx2 is None:
                continue
            
            # Try different edge key formats
            edge_key1 = (min(idx1, idx2), max(idx1, idx2))
            edge_key2 = (idx1, idx2)
            sim = similarity_pairs.get(edge_key1) or similarity_pairs.get(edge_key2)
            if sim is not None:
                cluster_similarities[(doc_id1, doc_id2)] = float(sim)
                cluster_similarities[(doc_id2, doc_id1)] = float(sim)
    
    if not cluster_similarities:
        # No similarities found, return original order
        return cluster
    
    # Find the pair with highest similarity (first two documents)
    best_pair = None
    best_sim = -1.0
    for (doc_id1, doc_id2), sim in cluster_similarities.items():
        if sim > best_sim:
            best_sim = sim
            best_pair = (doc_id1, doc_id2)
    
    if best_pair is None:
        return cluster
    
    # Start with the best pair
    ordered = [best_pair[0], best_pair[1]]
    remaining = [doc_id for doc_id in cluster if doc_id not in ordered]
    
    # Greedily add remaining documents: add the one most similar to any already-ordered document
    # Only order approximately the first 10 documents carefully
    target_ordered_size = min(10, len(cluster))
    while len(ordered) < target_ordered_size and remaining:
        best_next = None
        best_next_sim = -1.0
        
        for doc_id in remaining:
            # Find max similarity to any already-ordered document
            max_sim = -1.0
            for ordered_doc in ordered:
                sim = cluster_similarities.get((doc_id, ordered_doc), -1.0)
                if sim > max_sim:
                    max_sim = sim
            
            if max_sim > best_next_sim:
                best_next_sim = max_sim
                best_next = doc_id
        
        if best_next is not None:
            ordered.append(best_next)
            remaining.remove(best_next)
        else:
            # No similarity found, break and append rest
            break
    
    # Append remaining documents in any order
    ordered.extend(remaining)
    
    return ordered


def _save_clusters(
    clusters: List[List[str]],
    similarity_pairs: Dict,
    doc_id_to_index: Dict[str, int],
    data_dict: Dict[str, Dict],
    comparison_type: str
) -> List[Dict]:
    """Helper function to save clusters for a single comparison type."""
    saved_clusters = []
    for cluster in clusters:
        # Order cluster by similarity
        ordered_cluster = _order_cluster_by_similarity(cluster, similarity_pairs, doc_id_to_index)
        
        # Calculate min and max similarity within the cluster
        cluster_similarities = []
        for i, doc_id1 in enumerate(ordered_cluster):
            idx1 = doc_id_to_index.get(doc_id1)
            if idx1 is None:
                continue
            for j, doc_id2 in enumerate(ordered_cluster):
                if i >= j:
                    continue
                idx2 = doc_id_to_index.get(doc_id2)
                if idx2 is None:
                    continue
                
                # Try different edge key formats
                edge_key1 = (min(idx1, idx2), max(idx1, idx2))
                edge_key2 = (idx1, idx2)
                sim = similarity_pairs.get(edge_key1) or similarity_pairs.get(edge_key2)
                if sim is not None:
                    cluster_similarities.append(float(sim))
        
        min_similarity = min(cluster_similarities) if cluster_similarities else None
        max_similarity = max(cluster_similarities) if cluster_similarities else None
        
        cluster_data = {
            "cluster_id": len(saved_clusters),
            "size": len(ordered_cluster),
            "comparison_type": comparison_type,
            "min_similarity": min_similarity,
            "max_similarity": max_similarity,
            "documents": []
        }
        
        # Build document info in similarity order
        for doc_id in ordered_cluster:
            doc_info = data_dict.get(doc_id, {"dokument_id": doc_id})
            cluster_data["documents"].append(doc_info)
        
        saved_clusters.append(cluster_data)
    return saved_clusters


def save_duplicate_results(
    results: Dict[str, Any],
    output_path: Path,
    embeddings_path: Path
):
    """
    Save duplicate detection results to a JSON file.
    
    Args:
        results: Results from find_duplicates() (may contain text_clusters, summary_clusters)
        output_path: Path to save the JSON output
        embeddings_path: Path to embeddings file (used to find corresponding data file)
    """
    # Collect all document IDs that are in clusters (only load metadata for these)
    doc_ids_needed = set()
    if "text_clusters" in results and results["text_clusters"]:
        for cluster in results["text_clusters"]:
            doc_ids_needed.update(cluster)
    if "summary_clusters" in results and results["summary_clusters"]:
        for cluster in results["summary_clusters"]:
            doc_ids_needed.update(cluster)
    if "input_and_summary_clusters" in results and results["input_and_summary_clusters"]:
        for cluster in results["input_and_summary_clusters"]:
            doc_ids_needed.update(cluster)
    
    # Load original data to include document metadata (only for documents in clusters)
    data_dict = {}
    data_file = embeddings_path.parent / embeddings_path.name.replace("_embeddings.jsonl", ".jsonl")
    if data_file.exists() and doc_ids_needed:
        print(f"Loading document metadata for {len(doc_ids_needed)} documents in clusters from {data_file.name}...", file=sys.stderr)
        loaded_count = 0
        for example in _iter_examples(data_file):
            metadata = example.get("metadata", {})
            doc_id = metadata.get("dokument_id")
            if doc_id and doc_id in doc_ids_needed:
                data_dict[doc_id] = {
                    "dokument_id": doc_id,
                    "doc_type": metadata.get("doc_type"),
                    "kommune": metadata.get("kommune"),
                    "input_preview": str(example.get("input", ""))[:200] if example.get("input") else None,
                    "output_preview": str(example.get("output", ""))[:200] if example.get("output") else None
                }
                loaded_count += 1
                # Early exit if we've loaded all needed documents
                if loaded_count >= len(doc_ids_needed):
                    break
        print(f"Loaded metadata for {len(data_dict)} documents", file=sys.stderr)
    
    # Build output structure
    output_data = {
        "statistics": results["statistics"],
        "clusters": []
    }
    
    # Save input clusters (sorted by size, largest first)
    if "text_clusters" in results and results["text_clusters"]:
        # Sort clusters by size (descending)
        sorted_text_clusters = sorted(results["text_clusters"], key=len, reverse=True)
        text_clusters = _save_clusters(
            sorted_text_clusters,
            results.get("text_similarity_pairs", {}),
            results.get("text_doc_id_to_index", {}),
            data_dict,
            "input_comparison"
        )
        output_data["clusters"].extend(text_clusters)
    
    # Save summary clusters (sorted by size, largest first)
    if "summary_clusters" in results and results["summary_clusters"]:
        # Sort clusters by size (descending)
        sorted_summary_clusters = sorted(results["summary_clusters"], key=len, reverse=True)
        summary_clusters = _save_clusters(
            sorted_summary_clusters,
            results.get("summary_similarity_pairs", {}),
            results.get("summary_doc_id_to_index", {}),
            data_dict,
            "summary_comparison"
        )
        output_data["clusters"].extend(summary_clusters)
    
    # Save input and summary clusters (sorted by size, largest first)
    if "input_and_summary_clusters" in results and results["input_and_summary_clusters"]:
        # Sort clusters by size (descending)
        sorted_input_and_summary_clusters = sorted(results["input_and_summary_clusters"], key=len, reverse=True)
        input_and_summary_clusters = _save_clusters(
            sorted_input_and_summary_clusters,
            results.get("input_and_summary_similarity_pairs", {}),
            results.get("input_and_summary_doc_id_to_index", {}),
            data_dict,
            "input_and_summary_comparison"
        )
        output_data["clusters"].extend(input_and_summary_clusters)
    
    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"Duplicate detection results saved to {output_path}", file=sys.stderr)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a text_summary dataset or find duplicate documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard analysis
  python analyse_dataset.py --dataset 12811_examples
  python analyse_dataset.py --dataset 12811_examples --train
  python analyse_dataset.py --dataset 12811_examples --examples 5
  
  # Find duplicates (both comparison types by default)
  python analyse_dataset.py --dataset 12811_examples --find-duplicates
  python analyse_dataset.py --dataset 12811_examples --find-duplicates --similarity-threshold 0.95 --star-clustering
  
  # Find duplicates with specific comparison types
  python analyse_dataset.py --dataset 12811_examples --find-duplicates --text-similarity --no-summary-similarity
  python analyse_dataset.py --dataset 12811_examples --find-duplicates --summary-similarity
        """
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        help="Dataset suffix (the part after 'text_summary_dataset_', e.g., '12811_examples')",
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
        "--examples",
        type=int,
        default=None,
        metavar="N",
        help="Number of sample examples to display (default: don't show examples)",
    )
    
    # Duplicate detection arguments
    parser.add_argument(
        "--find-duplicates",
        action="store_true",
        help="Find duplicate and near-duplicate documents using embeddings",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.99,
        help="Minimum cosine similarity to consider as duplicate (default: 0.99, range: 0.0-1.0)",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=50,
        help="Number of nearest neighbors to search for each document (default: 50)",
    )
    parser.add_argument(
        "--star-clustering",
        action="store_true",
        help="Apply star clustering to avoid chain over-clustering",
    )
    parser.add_argument(
        "--low-memory-duplicates",
        action="store_true",
        help="Use low-memory duplicate detection (no full similarity matrix, no star clustering). "
             "Useful for very large datasets where a dense similarity matrix would not fit in memory.",
    )
    parser.add_argument(
        "--text-similarity",
        action="store_true",
        default=True,
        help="Compare all input texts against each other (default: enabled)",
    )
    parser.add_argument(
        "--no-text-similarity",
        dest="text_similarity",
        action="store_false",
        help="Disable input comparison",
    )
    parser.add_argument(
        "--summary-similarity",
        action="store_true",
        default=True,
        help="Compare all output summaries against each other (default: enabled)",
    )
    parser.add_argument(
        "--no-summary-similarity",
        dest="summary_similarity",
        action="store_false",
        help="Disable summary comparison",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path for duplicate detection results (default: <dataset>_duplicates.json)",
    )
    
    args = parser.parse_args(argv)
    
    # Determine datasets root
    if args.dataset_folder:
        datasets_root = Path(args.dataset_folder).resolve()
    else:
        datasets_root = Path(__file__).parent
    
    # Handle duplicate detection mode
    if args.find_duplicates:
        if not FAISS_AVAILABLE:
            print("Error: FAISS is required for duplicate detection. Install it with: pip install faiss-cpu (or use conda: conda install -c pytorch faiss-gpu for GPU support)", file=sys.stderr)
            return 1
        
        # Construct folder name
        folder_name = f"text_summary_dataset_{args.dataset}"
        folder_path = datasets_root / folder_name
        
        if not folder_path.exists():
            print(f"Error: Dataset folder not found: {folder_path}", file=sys.stderr)
            return 1
        
        # Find the appropriate data file
        if args.split:
            pattern = f"*_{args.split}.jsonl"
            matching_files = list(folder_path.glob(pattern))
            if not matching_files:
                print(f"Error: No {args.split} file found in {folder_path}", file=sys.stderr)
                return 1
            file_path = matching_files[0]
        else:
            all_jsonl = list(folder_path.glob("*.jsonl"))
            plain_files = [
                f for f in all_jsonl
                if not any(f.name.endswith(f"_{s}.jsonl") for s in ["train", "val", "test"])
                and not f.name.endswith("embeddings.jsonl")
            ]
            if not plain_files:
                print(f"Error: No plain file found in {folder_path}", file=sys.stderr)
                return 1
            file_path = plain_files[0]
        
        # Get embeddings path
        embeddings_path = _get_embeddings_path(file_path)
        
        # Check if embeddings file exists and is up-to-date
        if embeddings_path.exists():
            data_mtime = file_path.stat().st_mtime
            embeddings_mtime = embeddings_path.stat().st_mtime
            if data_mtime > embeddings_mtime:
                print(f"Error: Data file {file_path.name} is newer than embeddings file {embeddings_path.name}.", file=sys.stderr)
                print("Please run the standard analysis first to regenerate embeddings.", file=sys.stderr)
                return 1
        
        if not embeddings_path.exists():
            print(f"Error: Embeddings file not found: {embeddings_path}", file=sys.stderr)
            print("Please run the standard analysis first to generate embeddings.", file=sys.stderr)
            return 1
        
        try:
            # Find duplicates
            results = find_duplicates(
                embeddings_path=embeddings_path,
                similarity_threshold=args.similarity_threshold,
                k_neighbors=args.k_neighbors,
                use_star_clustering=args.star_clustering,
                text_similarity=args.text_similarity,
                summary_similarity=args.summary_similarity,
                low_memory=args.low_memory_duplicates,
            )
            
            # Print statistics
            stats = results["statistics"]
            print("\n" + "=" * 80, file=sys.stderr)
            print("DUPLICATE DETECTION RESULTS", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            print(f"Similarity threshold: {stats['similarity_threshold']:.4f}", file=sys.stderr)
            print(f"Using star clustering: {stats['use_star_clustering']}", file=sys.stderr)
            print(f"Comparisons performed: {', '.join(stats.get('comparisons_performed', []))}", file=sys.stderr)
            print(file=sys.stderr)
            
            # Print statistics for each comparison type
            if "input_comparison_total_documents" in stats:
                print("INPUT COMPARISON:", file=sys.stderr)
                print(f"  Total documents: {stats['input_comparison_total_documents']:,}", file=sys.stderr)
                print(f"  Number of clusters: {stats['input_comparison_num_clusters']:,}", file=sys.stderr)
                print(f"  Total duplicate documents: {stats['input_comparison_total_duplicate_documents']:,}", file=sys.stderr)
                print(f"  Largest cluster size: {stats['input_comparison_largest_cluster_size']:,}", file=sys.stderr)
                print(file=sys.stderr)
            
            if "summary_comparison_total_documents" in stats:
                print("SUMMARY COMPARISON:", file=sys.stderr)
                print(f"  Total documents: {stats['summary_comparison_total_documents']:,}", file=sys.stderr)
                print(f"  Number of clusters: {stats['summary_comparison_num_clusters']:,}", file=sys.stderr)
                print(f"  Total duplicate documents: {stats['summary_comparison_total_duplicate_documents']:,}", file=sys.stderr)
                print(f"  Largest cluster size: {stats['summary_comparison_largest_cluster_size']:,}", file=sys.stderr)
                print(file=sys.stderr)
            
            if "input_and_summary_comparison_total_documents" in stats:
                print("INPUT AND SUMMARY COMPARISON:", file=sys.stderr)
                print(f"  Total documents: {stats['input_and_summary_comparison_total_documents']:,}", file=sys.stderr)
                print(f"  Number of clusters: {stats['input_and_summary_comparison_num_clusters']:,}", file=sys.stderr)
                print(f"  Total duplicate documents: {stats['input_and_summary_comparison_total_duplicate_documents']:,}", file=sys.stderr)
                print(f"  Largest cluster size: {stats['input_and_summary_comparison_largest_cluster_size']:,}", file=sys.stderr)
                print(file=sys.stderr)
            
            print("=" * 80 + "\n", file=sys.stderr)
            
            # Save results
            if args.output:
                output_path = Path(args.output)
            else:
                # Build default output filename with threshold suffix, e.g. _threshold_099.json for 0.99
                threshold_str = str(args.similarity_threshold).replace('.', '')
                output_stem = embeddings_path.stem.replace('_embeddings', '')
                output_path = embeddings_path.parent / f"{output_stem}_duplicates_threshold_{threshold_str}.json"
            
            save_duplicate_results(
                results, 
                output_path, 
                embeddings_path
            )
            
            # Print sample clusters from each type
            all_clusters = []
            if "text_clusters" in results and results["text_clusters"]:
                all_clusters.extend([("input_comparison", c) for c in results["text_clusters"][:3]])
            if "summary_clusters" in results and results["summary_clusters"]:
                all_clusters.extend([("summary_comparison", c) for c in results["summary_clusters"][:3]])
            if "input_and_summary_clusters" in results and results["input_and_summary_clusters"]:
                all_clusters.extend([("input_and_summary_comparison", c) for c in results["input_and_summary_clusters"][:3]])
            
            if all_clusters:
                print("\nSample duplicate clusters (first 3 of each type):", file=sys.stderr)
                for comp_type, cluster in all_clusters[:9]:
                    print(f"\n{comp_type.upper()} Cluster (size: {len(cluster)}):", file=sys.stderr)
                    for doc_id in cluster[:10]:  # Show first 10 documents
                        print(f"  - {doc_id}", file=sys.stderr)
                    if len(cluster) > 10:
                        print(f"  ... and {len(cluster) - 10} more", file=sys.stderr)
            
            return 0
            
        except Exception as e:
            print(f"Error finding duplicates: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1
    
    # Standard analysis mode
    try:
        result = analyze_dataset(
            args.dataset,
            split=args.split,
            datasets_root=datasets_root,
            num_examples=args.examples
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error analyzing dataset: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    _print_result(result)
    
    # Save analysis results to JSON file
    _save_analysis_results(result)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

