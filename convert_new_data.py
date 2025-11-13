import os
import json
import pandas as pd
from tqdm import tqdm

# -----------------------------
# 1) Load data
# -----------------------------
new_data = pd.read_csv("training_data_new.csv")

print("Number of rows: ", new_data.shape[0])
print("Number of columns: ", new_data.shape[1])
print("Columns: ", list(new_data.columns))

# Select + rename
df_selected = (
    new_data[['dok_id', 'kommune', 'url', 'dok_type', 'dok_tittel', 'text']]
    .rename(columns={
        'dok_id': 'dokument_id',
        'dok_type': 'doc_type',
        'dok_tittel': 'tittel',
        'text': 'tekst'
    })
)

print("New Columns: ", list(df_selected.columns))
print("Number of rows: ", df_selected.shape[0])
print("Number of columns: ", df_selected.shape[1])

# -----------------------------
# 2) Ensure output directories
# -----------------------------
directories = ["new_features", "duplicate_features"]
for directory in directories:
    print(f"Checking if '{directory}' is present..")
    os.makedirs(directory, exist_ok=True)
    print(f"Directory '{directory}' is ready.")

# -----------------------------
# 3) Helpers
# -----------------------------
def sanitize_for_filename(value: str) -> str:
    """
    Make a safe filename token from a string-like value.
    - Convert NaN/None/empty to 'unknown'
    - Strip whitespace
    - Replace problematic characters
    """
    if pd.isna(value):
        return "unknown"
    s = str(value).strip()
    if not s:
        return "unknown"
    # Replace spaces and slashes and a few other filesystem-problematic chars
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        s = s.replace(ch, '_')
    s = s.replace(' ', '_')
    return s

def row_to_json_safe(row: pd.Series) -> str:
    """
    Convert a pandas Series to JSON with:
    - NaN -> None (valid JSON null)
    - Indentation
    """
    # Replace NaN with None so the JSON is valid
    clean = row.where(pd.notna(row), None).to_dict()
    return json.dumps(clean, indent=4, ensure_ascii=False, allow_nan=False)

# -----------------------------
# 4) Process rows
# -----------------------------
processed = set()   # use a set for faster membership checks
duplicates = []

# tqdm over iterrows with total for a consistent progress bar
for idx, (_, row) in enumerate(tqdm(df_selected.iterrows(), total=len(df_selected), desc="Processing")):
    doc_id = row['dokument_id']
    kommune_safe = sanitize_for_filename(row['kommune'])

    if doc_id not in processed:
        processed.add(doc_id)

        filename = f"new_features/{kommune_safe}_{doc_id}.json"
        row_json = row_to_json_safe(row)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(row_json)
    else:
        # Duplicate
        duplicates.append(doc_id)
        # Use the actual loop index value (idx) in the filename (not row['index'])
        filename = f"duplicate_features/{kommune_safe}_{doc_id}_{idx}.json"
        row_json = row_to_json_safe(row)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(row_json)

# -----------------------------
# 5) Write duplicates list ONCE
# -----------------------------
if duplicates:
    # unique list while preserving order
    seen = set()
    unique_duplicates = [x for x in duplicates if not (x in seen or seen.add(x))]

    with open("duplicates.txt", 'w', encoding='utf-8') as f:
        for dup in unique_duplicates:
            f.write(f"{dup}\n")

print(f"Done. Wrote {len(processed)} unique records and {len(duplicates)} duplicates.")
