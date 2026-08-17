import pandas as pd
import json
from pathlib import Path

# Define the sources directory
sources_dir = Path(__file__).parent

# Load the CSV file with summaries
csv_file = sources_dir / "url-oppsummering-from-prod20251215.csv"
df_summaries = pd.read_csv(csv_file)

print(f"Loaded summaries CSV: {len(df_summaries)} rows")
print(f"Columns: {df_summaries.columns.tolist()}")
print(f"\nFirst few rows:")
print(df_summaries.head())

# Load all JSON files from the texts directory
json_dir = sources_dir / "input_files"
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

if errors:
    print(f"\n⚠️  Encountered {len(errors)} errors while loading files:")
    for error in errors[:10]:  # Show first 10 errors
        print(f"  - {error}")
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more errors")


# Process texts_data to extract specific fields
processed_texts = []
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
    processed_texts.append(processed_record)

# Convert to DataFrame
df_texts = pd.DataFrame(processed_texts)

# Convert max_tokens to integer
df_texts['max_tokens'] = df_texts['max_tokens'].astype('Int64')

print(f"\nLoaded texts: {len(df_texts)} rows")
print(f"Columns: {df_texts.columns.tolist()}")
print(f"\nFirst few rows:")
print(df_texts.head())

# Inner join df_texts and df_summaries
# Join on df_texts.custom_id = df_summaries.dok_id
df_joined = df_texts.merge(
    df_summaries,
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

# Save to pickle and csv
#df_joined.to_pickle("44118-url-tekst-oppsummering-20250930.pkl")
#df_joined.to_csv("44118-url-tekst-oppsummering-20250930.csv", index=False)
