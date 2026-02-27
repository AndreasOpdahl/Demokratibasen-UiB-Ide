#!/usr/bin/env python3
"""
Calculate cosine distances between input and output embeddings,
add embedding_distance field, and output statistics and top 20 largest distances.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def cosine_distance(embedding1, embedding2):
    """
    Calculate cosine distance between two L2-normalized embeddings.
    Since embeddings are L2-normalized, cosine distance = 1 - dot_product
    """
    # Convert to numpy arrays for efficient computation
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Cosine similarity = dot product (since vectors are normalized)
    cosine_sim = np.dot(vec1, vec2)
    
    # Cosine distance = 1 - cosine similarity
    distance = 1.0 - cosine_sim
    
    return float(distance)

def load_texts_from_main_file(main_file_path):
    """Load input and output texts from main file, keyed by dokument_id."""
    texts = {}
    with open(main_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                metadata = obj.get('metadata', {})
                doc_id = metadata.get('dokument_id')
                if doc_id:
                    texts[doc_id] = {
                        'input': obj.get('input', ''),
                        'output': obj.get('output', '')
                    }
            except json.JSONDecodeError:
                continue
    return texts

def main():
    base_dir = Path("/media/disk_7300G/sinoa/Local/Tools/GitProjects/Demokratibasen-UiB-Ide/datasets_from_demokratibasen/cleaned_datasets/text_summary_dataset_202601")
    
    embeddings_file = base_dir / "155452_text_summary_examples_embeddings.jsonl"
    main_file = base_dir / "155452_text_summary_examples.jsonl"
    output_file = base_dir / "large_distance_intput_output_pairs.json"
    
    print("=" * 80)
    print("CALCULATING EMBEDDING DISTANCES")
    print("=" * 80)
    print()
    
    # Load input/output texts from main file
    print("Loading texts from main file...")
    texts_dict = load_texts_from_main_file(main_file)
    print(f"Loaded {len(texts_dict):,} text records")
    print()
    
    # Process embeddings file
    print(f"Processing {embeddings_file.name}...")
    
    distances = []
    records_with_distances = []
    records_to_update = []
    
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                obj = json.loads(line)
                doc_id = obj.get('dokument_id')
                input_emb = obj.get('input_embedding')
                output_emb = obj.get('output_embedding')
                
                if input_emb is None or output_emb is None:
                    print(f"Warning: Missing embeddings on line {line_num}")
                    continue
                
                # Calculate cosine distance
                distance = cosine_distance(input_emb, output_emb)
                
                # Add distance field
                obj['embedding_distance'] = distance
                records_to_update.append(json.dumps(obj, ensure_ascii=False) + '\n')
                
                distances.append(distance)
                records_with_distances.append({
                    'doc_id': doc_id,
                    'distance': distance
                })
                
            except json.JSONDecodeError as e:
                print(f"Warning: JSON decode error on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
    
    # Write updated records back to embeddings file
    print(f"Writing updated records with distances back to {embeddings_file.name}...")
    with open(embeddings_file, 'w', encoding='utf-8') as f:
        f.writelines(records_to_update)
    
    # Calculate statistics
    if distances:
        distances_array = np.array(distances)
        mean_dist = float(np.mean(distances_array))
        max_dist = float(np.max(distances_array))
        min_dist = float(np.min(distances_array))
        
        print()
        print("=" * 80)
        print("DISTANCE STATISTICS")
        print("=" * 80)
        print(f"Mean distance: {mean_dist:.6f}")
        print(f"Max distance:  {max_dist:.6f}")
        print(f"Min distance:  {min_dist:.6f}")
        print(f"Total records: {len(distances):,}")
        print()
    else:
        print("ERROR: No distances calculated!")
        return
    
    # Find top 20 largest distances
    print("Finding top 20 largest distances...")
    records_with_distances.sort(key=lambda x: x['distance'], reverse=True)
    top_20 = records_with_distances[:20]
    
    # Get input/output texts for top 20
    top_20_triples = []
    missing_count = 0
    
    for record in top_20:
        doc_id = record['doc_id']
        distance = record['distance']
        
        if doc_id in texts_dict:
            text_data = texts_dict[doc_id]
            top_20_triples.append({
                'distance': distance,
                'input': text_data['input'],
                'output': text_data['output']
            })
        else:
            missing_count += 1
            print(f"Warning: Could not find text for document ID: {doc_id}")
    
    if missing_count > 0:
        print(f"Warning: {missing_count} documents from top 20 could not be found in main file")
    
    # Write top 20 to output file
    print(f"Writing top 20 largest distances to {output_file.name}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(top_20_triples, f, indent=2, ensure_ascii=False)
    
    print()
    print("=" * 80)
    print("COMPLETED")
    print("=" * 80)
    print(f"Top 20 distances written to: {output_file.name}")
    print(f"Embeddings file updated with distance field")

if __name__ == "__main__":
    main()
