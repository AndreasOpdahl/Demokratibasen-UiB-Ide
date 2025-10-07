# analyze_data.py
import pandas as pd
from pathlib import Path
import json

def analyze_data_alignment(features_dir: str, labels_csv: str):
    """
    Analyzes the alignment between feature files and labels.
    """
    # Read labels
    labels_df = pd.read_csv(labels_csv)
    label_ids = set(labels_df['dokument_id'].tolist())
    print(f"Total labels in CSV: {len(label_ids)}")
    
    # Get feature files
    features_path = Path(features_dir)
    json_files = list(features_path.glob("*.json"))
    print(f"Total JSON feature files: {len(json_files)}")
    
    # Extract IDs from JSON files
    feature_ids = set()
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'dokument_id' in data:
                    feature_ids.add(data['dokument_id'])
        except:
            continue
    
    print(f"Unique dokument_id in features: {len(feature_ids)}")
    
    # Find matches and mismatches
    matches = label_ids.intersection(feature_ids)
    missing_in_labels = feature_ids - label_ids
    missing_in_features = label_ids - feature_ids
    
    print(f"\nMatching dokument_id: {len(matches)}")
    print(f"Features without labels: {len(missing_in_labels)}")
    print(f"Labels without features: {len(missing_in_features)}")
    
    # Save details to files
    with open("missing_in_labels.txt", "w") as f:
        for doc_id in sorted(list(missing_in_labels)):
            f.write(f"{doc_id}\n")
    
    with open("missing_in_features.txt", "w") as f:
        for doc_id in sorted(list(missing_in_features)):
            f.write(f"{doc_id}\n")
    
    print("\nDetails saved to missing_in_labels.txt and missing_in_features.txt")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze data alignment')
    parser.add_argument('--features_dir', type=str, required=True,
                       help='Directory containing JSON feature files')
    parser.add_argument('--labels_csv', type=str, required=True,
                       help='CSV file containing labels')
    
    args = parser.parse_args()
    
    analyze_data_alignment(
        features_dir=args.features_dir,
        labels_csv=args.labels_csv
    )
