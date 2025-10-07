# data_processor.py
import json
import random
import pandas as pd
from pathlib import Path
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_norwegian_documents(features_dir: str, labels_csv: str, output_file: str):
    """
    Processes Norwegian documents with separate features (JSON) and labels (CSV).
    Provides detailed reporting on mismatches.
    """
    # Read the CSV labels
    logger.info(f"Reading labels from {labels_csv}")
    labels_df = pd.read_csv(labels_csv)
    
    # Check if dokument_id column exists
    if 'dokument_id' not in labels_df.columns:
        logger.error("CSV file does not contain 'dokument_id' column")
        return
    
    labels_df.set_index('dokument_id', inplace=True)
    logger.info(f"Found {len(labels_df)} labels in CSV")
    
    features_path = Path(features_dir)
    output_data = []
    missing_labels = []
    processed_count = 0
    
    # Get all JSON files
    json_files = list(features_path.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in {features_dir}")
    
    # Process each JSON file in the features directory
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                feature_data = json.load(f)
            
            dokument_id = feature_data.get("dokument_id")
            
            if not dokument_id:
                logger.warning(f"JSON file {json_file.name} missing dokument_id")
                continue
            
            # Find matching label in CSV
            if dokument_id in labels_df.index:
                label_row = labels_df.loc[dokument_id]
                
                # Create training example
                processed_entry = {
                    "input": f"Dokument: {feature_data.get('tittel', '')}\n\n{feature_data.get('tekst', '')}",
                    "output": label_row.get("oppsummering", ""),
                    "metadata": {
                        "dokument_id": dokument_id,
                        "doc_type": feature_data.get("doc_type", ""),
                        "kommune": feature_data.get("kommune", ""),
                        "personer": label_row.get("personer", ""),
                        "nokkelord": label_row.get("nokkelord", ""),
                        "nyhetsverdi": label_row.get("nyhetsverdi", "")
                    }
                }
                
                output_data.append(processed_entry)
                processed_count += 1
            else:
                missing_labels.append(dokument_id)
                
        except Exception as e:
            logger.error(f"Error processing {json_file}: {str(e)}")
    
    # Save as JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in output_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Generate report
    logger.info(f"Processed {processed_count} documents successfully")
    logger.info(f"Found {len(missing_labels)} documents without matching labels")
    
    
    if missing_labels:
        # Save missing IDs to a file for reference
        with open("missing_labels_report.txt", "w") as report_file:
            report_file.write("Documents without matching labels:\n")
            for doc_id in missing_labels[:100]:  # Limit to first 100
                report_file.write(f"{doc_id}\n")
            if len(missing_labels) > 100:
                report_file.write(f"... and {len(missing_labels) - 100} more\n")
        
        logger.info(f"Missing labels report saved to missing_labels_report.txt")



def split_data(input_file, train_file, val_file, test_file, test_size=0.1, val_size=0.1):

    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    random.shuffle(data)

    test_split = int(len(data) * (1 - test_size))
    val_split = int(test_split * (1 - val_size))

    train_data = data[:val_split]
    val_data = data[val_split:test_split]
    test_data = data[test_split:]

    # Save training data
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Save validation data
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Save test data
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Split complete: {len(train_data)} training samples, {len(val_data)} validation samples, {len(test_data)} test samples")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process Norwegian documents')
    parser.add_argument('--features_dir', type=str, required=True,
                       help='Directory containing JSON feature files')
    parser.add_argument('--labels_csv', type=str, required=True,
                       help='CSV file containing labels')
    parser.add_argument('--output_file', type=str, default='/app/data/output/processed_data.jsonl',
                       help='Output JSONL file path')

    parser.add_argument('--split', action='store_true',
                       help='Split data into training, validation, and test sets')
    parser.add_argument('--test_size', type=float, default=0.1,
                       help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Proportion of training data to use for validation')

    args = parser.parse_args()

    process_norwegian_documents(
        features_dir=args.features_dir,
        labels_csv=args.labels_csv,
        output_file=args.output_file
    )

    if args.split:
        split_data(
            args.output_file,
            args.output_file.replace('.jsonl', '_train.jsonl'),
            args.output_file.replace('.jsonl', '_val.jsonl'),
            args.output_file.replace('.jsonl', '_test.jsonl'),
            test_size=args.test_size,
            val_size=args.val_size
        )
