from datetime import date
import json
from pathlib import Path

import pandas as pd


# Get today's date
TODAY = date.today().strftime("%Y%m%d")
LOCAL_FOLDER = Path(f'./{TODAY}-Demokratibasen-prod')

# Define the texts directory (OpenAI batch files)
TEXT_DIR = LOCAL_FOLDER / f"batch-files-{TODAY}"

# Define the summary file (from Demokratibasen-prod)
SUMMARY_CSV_FILE = LOCAL_FOLDER / f"./url_oppsummering_from_prod_{TODAY}.csv"

# What to do
EXTRACT_TEXTS_FROM_BATCH_INPUT_FILES = True
EXTRACT_SUMMARIES_FROM_BATCH_OUTPUT_FILES = False
EXTRACT_SUMMARIES_FROM_SUMMARY_CSV_FILE = True
JOIN_SUMMARIES_AND_TEXTS = True \
    and EXTRACT_TEXTS_FROM_BATCH_INPUT_FILES \
    and (EXTRACT_SUMMARIES_FROM_BATCH_OUTPUT_FILES \
            or EXTRACT_SUMMARIES_FROM_SUMMARY_CSV_FILE)


if EXTRACT_TEXTS_FROM_BATCH_INPUT_FILES:
    # Load all JSON files from the texts directory
    json_dir = TEXT_DIR / "input_files"
    json_files = list(json_dir.glob("file-*.json"))

    print(f"\n\nFound {len(json_files)} JSON files in the texts directory")

    # Load all JSON files into a list
    # Handle both single JSON objects and JSONL format (multiple objects per file)
    texts_data = []
    errors = []

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Check if file contains multiple JSON objects (JSONL format)
                if '\n{' in content:
                    # JSONL format - multiple JSON objects, one per line
                    for line_num, line in enumerate(content.split('\n'), 1):
                        if line.strip():
                            try:
                                texts_data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                errors.append(f"{json_file.name} (line {line_num}): {str(e)}")
                else:
                    # Single JSON object
                    texts_data.append(json.loads(content))
        except Exception as e:
            errors.append(f"{json_file.name}: {str(e)}")

    if len(errors) > 0:
        print(f"\n⚠️  Encountered {len(errors)} errors while loading files:")
        print(f"JSON parsing returned:")
        print(f"    {errors[0][:200]}")
        print(f"    ... and {len(errors) - 1} more error(s)")


    # Process texts_data to extract specific fields
    processed_texts = []
    seen_ids = {}  # Track IDs we've already processed: {custom_id: (processed_record, index_in_list)}
    duplicate_warnings = []
    
    def model_priority(model_name):
        """Return priority value for model (higher is better)"""
        if model_name is None:
            return 0
        model_lower = model_name.lower()
        if 'gpt-4' in model_lower:
            return 2
        elif 'gpt-3.5' in model_lower:
            return 1
        else:
            return 0
    
    for item in texts_data:
        # Extract user message content
        text = None
        if 'body' in item and 'messages' in item['body']:
            for message in item['body']['messages']:
                if message.get('role') == 'user':
                    text = message.get('content')
                    break
        
        # Create processed record
        processed_record = {
            'custom_id': item.get('custom_id'),
            'model': item.get('body', {}).get('model'),
            'max_tokens': item.get('body', {}).get('max_tokens'),
            'text': text
        }
        
        custom_id = processed_record['custom_id']
        
        # Check if we've seen this ID before
        if custom_id in seen_ids:
            # Compare with existing record
            existing_record, existing_idx = seen_ids[custom_id]
            
            if existing_record != processed_record:
                # Check if the only difference is the model
                differences = []
                for key in processed_record:
                    if existing_record.get(key) != processed_record.get(key):
                        differences.append(key)
                
                # If only model differs, choose the better model
                if differences == ['model']:
                    existing_model = existing_record.get('model')
                    new_model = processed_record.get('model')
                    
                    if model_priority(new_model) > model_priority(existing_model):
                        # Replace with better model
                        processed_texts[existing_idx] = processed_record
                        seen_ids[custom_id] = (processed_record, existing_idx)
                        # No warning needed - this is expected behavior
                    # else: keep existing (better or equal model)
                else:
                    # Other differences exist - warn about discrepancy
                    diff_strs = []
                    for key in differences:
                        diff_strs.append(f"{key}: {existing_record.get(key)!r} vs {processed_record.get(key)!r}")
                    duplicate_warnings.append(
                        f"ID '{custom_id}' has different values: {', '.join(diff_strs)}"
                    )
            # Skip adding duplicate (either already have better model, or warned about differences)
            continue
        else:
            # First time seeing this ID
            idx = len(processed_texts)
            seen_ids[custom_id] = (processed_record, idx)
        processed_texts.append(processed_record)
    
    # Report duplicate warnings
    if len(duplicate_warnings) > 1:
        print(f"\n⚠️  Found {len(duplicate_warnings)} duplicate ID(s) with different values in input files:")
        print(f"Duplicate check returned:")
        print(f"    {duplicate_warnings[0][:200]}")
        print(f"    ... and {len(duplicate_warnings) - 1} more warning(s)")



    # Convert to DataFrame
    df_texts = pd.DataFrame(processed_texts)

    # Convert max_tokens to integer
    df_texts['max_tokens'] = df_texts['max_tokens'].astype('Int64')

    print(f"\nLoaded texts: {len(df_texts)} rows (duplicates removed during processing)")
    print(f"Columns: {df_texts.columns.tolist()}")
    
    # Verify no duplicates made it through (sanity check)
    duplicates = df_texts['custom_id'].duplicated()
    if duplicates.any():
        num_duplicates = duplicates.sum()
        print(f"\n⚠️  ERROR: Found {num_duplicates} duplicate custom_id(s) - this should not happen!")
        duplicate_ids = df_texts[duplicates]['custom_id'].tolist()
        print(f"  Duplicate IDs: {duplicate_ids[:10]}")  # Show first 10
        if len(duplicate_ids) > 10:
            print(f"  ... and {len(duplicate_ids) - 10} more")
    else:
        print(f"✓ All custom_ids are unique")


if EXTRACT_SUMMARIES_FROM_BATCH_OUTPUT_FILES:
    # Load all JSON files from the output_files directory
    json_dir = TEXT_DIR / "output_files"
    json_files = list(json_dir.glob("file-*.json"))

    print(f"\n\nFound {len(json_files)} JSON files in the output directory")

    # Load all JSON files into a list
    # Handle both single JSON objects and JSONL format (multiple objects per file)
    summaries_data = []
    errors = []

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Check if file contains multiple JSON objects (JSONL format)
                if '\n{' in content:
                    # JSONL format - multiple JSON objects, one per line
                    for line_num, line in enumerate(content.split('\n'), 1):
                        if line.strip():
                            try:
                                summaries_data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                errors.append(f"{json_file.name} (line {line_num}): {str(e)}")
                else:
                    # Single JSON object
                    summaries_data.append(json.loads(content))
        except Exception as e:
            errors.append(f"{json_file.name}: {str(e)}")

    if errors:
        print(f"\n⚠️  Encountered {len(errors)} errors while loading files:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    # Process summaries_data to extract specific fields
    processed_summaries = []
    seen_ids = {}  # Track IDs we've already processed: {dok_id: processed_record}
    duplicate_warnings = []
    
    for item in summaries_data:
        # Extract assistant message content from response
        summary_content = None
        if 'response' in item and 'body' in item['response']:
            body = item['response']['body']
            if 'choices' in body and len(body['choices']) > 0:
                message = body['choices'][0].get('message', {})
                if message.get('role') == 'assistant':
                    summary_content = message.get('content')
        
        # Parse the summary content as JSON to extract fields
        if summary_content:
            try:
                summary_json = json.loads(summary_content)
                
                # Create processed record with the expected columns
                # Map the output field names to the CSV column names
                processed_record = {
                    'dok_id': item.get('custom_id'),
                    'kommune': summary_json.get('kommune'),  # May not be in output
                    'url': summary_json.get('url'),  # May not be in output
                    'dok_type': summary_json.get('dok_type'),  # May not be in output
                    'dok_tittel': summary_json.get('dok_tittel'),  # May not be in output
                    'oppsum_tittel': summary_json.get('summary_title'),  # Field name mapping
                    'oppsummering': summary_json.get('summary_body'),  # Field name mapping
                    'personer': summary_json.get('persons_mentioned'),  # Field name mapping
                    'nokkelord': summary_json.get('keywords'),  # Field name mapping
                    'nyhetsverdi': summary_json.get('news_score')  # Field name mapping
                }
                
                dok_id = processed_record['dok_id']
                
                # Check if we've seen this ID before
                if dok_id in seen_ids:
                    # Compare with existing record
                    existing = seen_ids[dok_id]
                    if existing != processed_record:
                        # Values are different - warn about discrepancy
                        differences = []
                        for key in processed_record:
                            if existing.get(key) != processed_record.get(key):
                                differences.append(f"{key}: {existing.get(key)!r} vs {processed_record.get(key)!r}")
                        duplicate_warnings.append(
                            f"ID '{dok_id}' has different values: {', '.join(differences)}"
                        )
                    # Skip this duplicate (keep first occurrence)
                    continue
                else:
                    # First time seeing this ID
                    seen_ids[dok_id] = processed_record
                    processed_summaries.append(processed_record)
                    
            except json.JSONDecodeError as e:
                errors.append(f"Failed to parse summary content for {item.get('custom_id')}: {str(e)}")
            except Exception as e:
                errors.append(f"Error processing {item.get('custom_id')}: {str(e)}")
    
    # Report duplicate warnings
    if duplicate_warnings:
        print(f"\n⚠️  Found {len(duplicate_warnings)} duplicate ID(s) with different values in output files:")
        for warning in duplicate_warnings[:10]:
            print(f"  - {warning}")
        if len(duplicate_warnings) > 10:
            print(f"  ... and {len(duplicate_warnings) - 10} more")

    # Convert to DataFrame
    df_summaries = pd.DataFrame(processed_summaries)

    print(f"\nLoaded summaries: {len(df_summaries)} rows (duplicates removed during processing)")
    print(f"Columns: {df_summaries.columns.tolist()}")
    
    # Verify no duplicates made it through (sanity check)
    duplicates = df_summaries['dok_id'].duplicated()
    if duplicates.any():
        num_duplicates = duplicates.sum()
        print(f"\n⚠️  ERROR: Found {num_duplicates} duplicate dok_id(s) - this should not happen!")
        duplicate_ids = df_summaries[duplicates]['dok_id'].tolist()
        print(f"  Duplicate IDs: {duplicate_ids[:10]}")  # Show first 10
        if len(duplicate_ids) > 10:
            print(f"  ... and {len(duplicate_ids) - 10} more")
    else:
        print(f"✓ All dok_ids are unique")


if EXTRACT_SUMMARIES_FROM_SUMMARY_CSV_FILE:

    # Load the CSV file with summaries
    SUMMARY_csv_file = SUMMARY_CSV_FILE
    df_summaries = pd.read_csv(SUMMARY_csv_file)

    print(f"\nLoaded summaries CSV: {len(df_summaries)} rows")
    print(f"Columns: {df_summaries.columns.tolist()}")
    
    # Check for duplicate dok_ids
    duplicates = df_summaries['dok_id'].duplicated()
    if duplicates.any():
        num_duplicates = duplicates.sum()
        print(f"\n⚠️  WARNING: Found {num_duplicates} duplicate dok_id(s) in CSV file!")
        duplicate_ids = df_summaries[duplicates]['dok_id'].tolist()
        print(f"  Duplicate IDs: {duplicate_ids[:10]}")  # Show first 10
        if len(duplicate_ids) > 10:
            print(f"  ... and {len(duplicate_ids) - 10} more")
    else:
        print(f"✓ All dok_ids are unique")


if JOIN_SUMMARIES_AND_TEXTS:
    # Inner join df_texts and df_summaries
    # Join on df_texts.custom_id = df_summaries.dok_id
    df_joined = df_texts.merge(  # type: ignore
        df_summaries,  # type: ignore
        left_on='custom_id',
        right_on='dok_id',
        how='inner'
    )

    # Drop the custom_id column since we now have dok_id
    df_joined = df_joined.drop(columns=['custom_id'])

    # Reorder
    df_joined = df_joined[[
        'dok_id', 
        'kommune', 
        'url', 
        'dok_type',
        'dok_tittel', 
        'text', 
        'model', 
        'max_tokens', 
        'oppsum_tittel', 
        'oppsummering', 
        'personer',
        'nokkelord', 
        'nyhetsverdi'
    ]]

    print(f"\n\nJoined DataFrame:")
    print(f"Rows: {len(df_joined)}")
    print(f"Columns: {df_joined.columns.tolist()}")


    # File name
    output_fn = str(LOCAL_FOLDER / f"{len(df_joined)}-url-tekst-oppsummering-{TODAY}")

    # Save to pickle and csv
    df_joined.to_pickle(output_fn + ".pkl")
    df_joined.to_csv(output_fn + ".csv", index=False)
