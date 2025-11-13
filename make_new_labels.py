import pandas as pd

# -----------------------------
# 1) Load data
# -----------------------------
INPUT_CSV = "training_data_new.csv"
new_data = pd.read_csv(INPUT_CSV)
print(f"Original Data Shape: {new_data.shape}")

# -----------------------------
# 2) Select & rename columns
# -----------------------------
df_selected = (
    new_data[['dok_id', 'dok_tittel', 'oppsummering', 'personer', 'nokkelord', 'nyhetsverdi']]
    .rename(columns={
        'dok_id': 'dokument_id',
        'dok_tittel': 'tittel'
    })
)

# Add batch_id column
df_selected['batch_id'] = ''

# Reorder columns
desired_order = ['dokument_id', 'batch_id', 'tittel', 'oppsummering', 'personer', 'nokkelord', 'nyhetsverdi']
df_selected = df_selected[desired_order]

print(f"Selected Data Shape: {df_selected.shape}")

# -----------------------------
# 3) Split into uniques & duplicates
# -----------------------------
is_dup = df_selected.duplicated(subset=['dokument_id'], keep='first')

df_uniques = df_selected[~is_dup].copy()
df_duplicates = df_selected[is_dup].copy()

# Print shapes
print(f"Unique Rows Shape: {df_uniques.shape}")
print(f"Duplicate Rows Shape: {df_duplicates.shape}")

# -----------------------------
# 4) Save to CSVs in same directory
# -----------------------------
df_uniques.to_csv("new_labels.csv", index=False, encoding='utf-8')
df_duplicates.to_csv("duplicate_labels.csv", index=False, encoding='utf-8')

print(f"Saved {len(df_uniques):,} unique rows to: new_labels.csv")
print(f"Saved {len(df_duplicates):,} duplicate rows to: duplicate_labels.csv")
