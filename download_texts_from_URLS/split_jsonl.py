import json
import os

# Input JSONL file path
input_file = 'dokumenter.jsonl'
# Output directory for individual JSON files
output_dir = 'dokument_jsons'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read and split JSONL file
with open(input_file, 'r', encoding='utf-8') as infile:
    for line in infile:
        if not line.strip():
            continue  # skip empty lines
        obj = json.loads(line)
        obj_id = f'{obj.get('kommune')}_{obj.get('dokument_id')}'
        if obj_id is None:
            print("Warning: Object without 'id' field found. Skipping.")
            continue
        output_path = os.path.join(output_dir, f"{obj_id}.json")
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(obj, outfile, indent=2)

print(f"Finished splitting JSONL file into individual JSON files in '{output_dir}'")
