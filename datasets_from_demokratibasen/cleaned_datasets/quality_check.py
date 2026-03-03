#!/usr/bin/env python3
"""
Quality check script for text_summary_dataset_202601.
Analyzes language distribution, token counts, embedding distances, and word statistics.
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

# Try to import tiktoken
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    _tokenizer = tiktoken.get_encoding("cl100k_base")
except ImportError:
    TIKTOKEN_AVAILABLE = False
    _tokenizer = None
    print("Warning: tiktoken not available. Token counts will not be calculated.", file=sys.stderr)


def count_tokens(text: str) -> Optional[int]:
    """Count tokens in text using tiktoken."""
    if not TIKTOKEN_AVAILABLE or _tokenizer is None:
        return None
    try:
        return len(_tokenizer.encode(text))
    except Exception:
        return None


def is_sami_language(metadata: Dict) -> bool:
    """Check if document is in Sami language based on metadata."""
    input_lid = metadata.get("input_lid", "").lower()
    # Sami language codes: sme (Northern Sami), smj (Lule Sami), sma (Southern Sami), smn (Inari Sami), etc.
    sami_codes = {"sme", "smj", "sma", "smn", "sms", "sju"}
    return input_lid in sami_codes


def extract_norwegian_words(text: str) -> List[str]:
    """
    Extract Norwegian words:
    - 1-2 character words (from the short-word list)
    - any consecutive string of 3+ Norwegian letters
    A word is defined as a sequence of Norwegian letters with word boundaries.
    """
    short_words = {
        "i", "å", "av", "at", "da", "de", "du", "ei", "en", "er", "et", "ga", "gå",
        "ja", "jo", "la", "le", "lo", "ly", "me", "mi", "no", "nå", "om", "og", "på", "se", "si", "så", "ta",
        "ti", "to", "ut", "vi", "øv", "øy", "år"
    }
    # Find all Norwegian-letter words (1+ length), then filter
    tokens = re.findall(r'\b[a-zA-ZæøåÆØÅ]{1,}\b', text)
    words: List[str] = []
    for tok in tokens:
        tok_lower = tok.lower()
        if len(tok_lower) >= 3 or tok_lower in short_words:
            words.append(tok)
    return words


def calculate_embedding_distance(input_emb: List[float], output_emb: List[float]) -> float:
    """Calculate cosine distance between two embeddings."""
    if not input_emb or not output_emb:
        return float('nan')
    
    vec1 = np.array(input_emb)
    vec2 = np.array(output_emb)
    
    # Cosine similarity = dot product (since vectors are L2-normalized)
    cosine_sim = np.dot(vec1, vec2)
    # Cosine distance = 1 - cosine similarity
    cosine_dist = 1.0 - cosine_sim
    return float(cosine_dist)


def main():
    base_dir = Path(__file__).parent / "text_summary_dataset_202601"
    # After pruning, use the consolidated 149978_* files
    data_file = base_dir / "149978_text_summary_examples.jsonl"
    embeddings_file = base_dir / "149978_text_summary_examples_embeddings.jsonl"
    analysis_dir = base_dir / "analysis_results"
    analysis_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("QUALITY CHECK ANALYSIS")
    print("=" * 80)
    print()
    
    # Load embeddings
    print("Loading embeddings...")
    embeddings_dict = {}
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                doc_id = obj.get('dokument_id')
                if doc_id:
                    embeddings_dict[doc_id] = {
                        'input_embedding': obj.get('input_embedding'),
                        'output_embedding': obj.get('output_embedding'),
                        'embedding_distance': obj.get('embedding_distance')
                    }
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(embeddings_dict):,} embeddings")
    print()
    
    # Analyze dataset
    print("Analyzing dataset...")
    
    sami_docs = []
    sami_distances = []
    
    tokens_100_or_less = []
    tokens_100_distances = []
    
    tokens_50_or_less = []
    tokens_50_distances = []
    
    all_norwegian_words = []
    words_chars_ratios = []
    
    total_docs = 0
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                obj = json.loads(line)
                metadata = obj.get('metadata', {})
                doc_id = metadata.get('dokument_id')
                input_text = obj.get('input', '')
                
                if not doc_id or not input_text:
                    continue
                
                total_docs += 1
                
                # Check if Sami language
                if is_sami_language(metadata):
                    sami_docs.append(doc_id)
                    if doc_id in embeddings_dict:
                        dist = embeddings_dict[doc_id].get('embedding_distance')
                        if dist is not None:
                            sami_distances.append(dist)
                
                # Count tokens
                token_count = count_tokens(input_text)
                
                # Check token thresholds
                if token_count is not None:
                    if token_count <= 100:
                        tokens_100_or_less.append(doc_id)
                        if doc_id in embeddings_dict:
                            dist = embeddings_dict[doc_id].get('embedding_distance')
                            if dist is not None:
                                tokens_100_distances.append(dist)
                    
                    if token_count <= 50:
                        tokens_50_or_less.append(doc_id)
                        if doc_id in embeddings_dict:
                            dist = embeddings_dict[doc_id].get('embedding_distance')
                            if dist is not None:
                                tokens_50_distances.append(dist)
                
                # Extract Norwegian words
                words = extract_norwegian_words(input_text)
                all_norwegian_words.extend(words)
                
                # Calculate words/characters ratio
                char_count = len(input_text)
                if char_count > 0:
                    ratio = len(words) / char_count if words else 0.0
                    words_chars_ratios.append(ratio)
                
            except json.JSONDecodeError:
                continue
    
    print(f"Total documents analyzed: {total_docs:,}")
    print()
    
    # Print results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    # Sami language analysis
    print("SAMI LANGUAGE ANALYSIS:")
    print(f"  Number of input texts in Sami language: {len(sami_docs):,}")
    if sami_distances:
        print(f"  Embedding distance statistics for Sami documents:")
        print(f"    Min:  {min(sami_distances):.6f}")
        print(f"    Max:  {max(sami_distances):.6f}")
        print(f"    Mean: {np.mean(sami_distances):.6f}")
    else:
        print(f"  No embedding distances available for Sami documents")
    print()
    
    # 100 tokens or less
    print("100 TIKTOKENS OR LESS:")
    print(f"  Number of input texts with 100 tiktokens or less: {len(tokens_100_or_less):,}")
    if tokens_100_distances:
        print(f"  Embedding distance statistics:")
        print(f"    Min:  {min(tokens_100_distances):.6f}")
        print(f"    Max:  {max(tokens_100_distances):.6f}")
        print(f"    Mean: {np.mean(tokens_100_distances):.6f}")
    else:
        print(f"  No embedding distances available")
    print()
    
    # 50 tokens or less
    print("50 TIKTOKENS OR LESS:")
    print(f"  Number of input texts with 50 tiktokens or less: {len(tokens_50_or_less):,}")
    if tokens_50_distances:
        print(f"  Embedding distance statistics:")
        print(f"    Min:  {min(tokens_50_distances):.6f}")
        print(f"    Max:  {max(tokens_50_distances):.6f}")
        print(f"    Mean: {np.mean(tokens_50_distances):.6f}")
    else:
        print(f"  No embedding distances available")
    print()
    
    # Norwegian words analysis
    print("NORWEGIAN WORDS (1-2 characters):")
    word_counter = Counter(all_norwegian_words)
    unique_words = sorted(word_counter.keys())
    print(f"  Total occurrences: {len(all_norwegian_words):,}")
    print(f"  Unique words: {len(unique_words):,}")
    
    # Save word list to file
    words_file = analysis_dir / "norwegian_1_2_char_words.txt"
    with open(words_file, 'w', encoding='utf-8') as f:
        for word in unique_words:
            f.write(f"{word}\n")
    print(f"  Word list saved to: {words_file.name}")
    
    # Words/characters ratio
    if words_chars_ratios:
        print(f"  Words/characters ratio statistics:")
        print(f"    Min:  {min(words_chars_ratios):.6f}")
        print(f"    Max:  {max(words_chars_ratios):.6f}")
        print(f"    Mean: {np.mean(words_chars_ratios):.6f}")
    print()
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
