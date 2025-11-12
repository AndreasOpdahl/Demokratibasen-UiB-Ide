import os
import sys
import traceback
import pandas as pd

# -----------------------------
# Config
# -----------------------------
OLD_CSV = "labels.csv"
NEW_CSV = "new_labels.csv"
MERGED_CSV = "labels_latest.csv"
AUDIT_KEY_DUPS_CSV = "labels_key_duplicates_audit.csv"

CANONICAL_ORDER = [
    "dokument_id", "batch_id", "tittel", "oppsummering", "personer", "nokkelord", "nyhetsverdi"
]
PRIMARY_KEY = ["dokument_id"]  # keep first by this key

def print_header(title: str):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def load_csv_as_str(path: str) -> pd.DataFrame:
    """Load CSV as strings, trim whitespace, keep empty tokens as empty strings."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False,
                     na_values=["", "NA", "NaN", "null", "None"])
    # Avoid deprecated applymap; use apply on columns that are object/string
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col].dtype):
            df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def normalize_empty(df: pd.DataFrame) -> pd.DataFrame:
    """Convert whitespace-only strings to empty string for all object columns."""
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col].dtype):
            df[col] = df[col].map(lambda x: "" if isinstance(x, str) and x.strip() == "" else x)
    return df

def ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Ensure all columns exist; add missing as empty strings."""
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df

def reorder_columns_union(df: pd.DataFrame, canonical_first: list[str]) -> pd.DataFrame:
    """Place canonical columns first; keep any extras after in stable order."""
    extra = [c for c in df.columns if c not in canonical_first]
    return df[canonical_first + extra]

def main():
    try:
        print_header("Context")
        print(f"Working directory: {os.getcwd()}")
        print(f"Old CSV exists: {os.path.abspath(OLD_CSV)} -> {os.path.exists(OLD_CSV)}")
        print(f"New CSV exists: {os.path.abspath(NEW_CSV)} -> {os.path.exists(NEW_CSV)}")

        # -----------------------------
        # 1) Load
        # -----------------------------
        print_header("Load")
        old = load_csv_as_str(OLD_CSV)
        new = load_csv_as_str(NEW_CSV)
        print(f"Old shape: {old.shape}")
        print(f"New shape: {new.shape}")

        # -----------------------------
        # 2) Schema alignment
        # -----------------------------
        union_cols = sorted(set(old.columns).union(new.columns))
        canonical_present = [c for c in CANONICAL_ORDER if c in union_cols]

        old = ensure_columns(old, union_cols)
        new = ensure_columns(new, union_cols)

        old = reorder_columns_union(old, canonical_present)
        new = reorder_columns_union(new, canonical_present)

        old = normalize_empty(old)
        new = normalize_empty(new)

        # -----------------------------
        # 3) Concatenate (old first, then new)
        # -----------------------------
        print_header("Concatenate")
        merged = pd.concat([old, new], ignore_index=True)
        print(f"Merged pre-dedup shape: {merged.shape}")

        # -----------------------------
        # 4) Remove exact full-row duplicates
        # -----------------------------
        print_header("Full-row duplicates")
        pre = merged.shape[0]
        merged = merged.drop_duplicates(keep="first")
        print(f"Removed {pre - merged.shape[0]} exact duplicates; shape now {merged.shape}")

        # -----------------------------
        # 5) Key duplicate handling: KEEP FIRST (old has precedence)
        # -----------------------------
        print_header("Key duplicates by dokument_id (keep first)")
        if not all(k in merged.columns for k in PRIMARY_KEY):
            missing = [k for k in PRIMARY_KEY if k not in merged.columns]
            print(f"WARNING: Missing key columns: {missing}")
        else:
            key_dups_mask = merged.duplicated(subset=PRIMARY_KEY, keep="first")
            key_dups = merged.loc[key_dups_mask].copy()
            print(f"Found {len(key_dups)} key-duplicate rows to drop (keeping first occurrence).")

            # Write audit (even if empty -> header-only)
            audit_abs = os.path.abspath(AUDIT_KEY_DUPS_CSV)
            key_dups.to_csv(audit_abs, index=False, encoding="utf-8")
            print(f"Audit written to: {audit_abs}")

            # Drop key-duplicates (keep first)
            merged = merged[~key_dups_mask].copy()

        # -----------------------------
        # 6) Null/empty sanity
        # -----------------------------
        print_header("Null/Empty ratios")
        for col in merged.columns:
            # Treat empty string as empty; merged is string-typed
            ratio = (merged[col].astype(str) == "").mean()
            print(f"{col}: {ratio:.2%} empty")

        if "dokument_id" in merged.columns:
            missing_ids = (merged["dokument_id"] == "").sum()
            if missing_ids > 0:
                missing_abs = os.path.abspath("labels_missing_dokument_id.csv")
                merged.loc[merged["dokument_id"] == ""].to_csv(missing_abs, index=False, encoding="utf-8")
                print(f"WARNING: {missing_ids} rows missing dokument_id (saved to {missing_abs}).")

        # -----------------------------
        # 7) Final ordering and save
        # -----------------------------
        print_header("Finalize & Save")
        merged = reorder_columns_union(merged, canonical_present)

        out_abs = os.path.abspath(MERGED_CSV)
        merged.to_csv(out_abs, index=False, encoding="utf-8")
        print(f"Saved merged labels to: {out_abs}")

        # Verify file exists and show final shape
        print(f"File exists after save? {os.path.exists(out_abs)}")
        print(f"Final shape: {merged.shape}")

    except Exception as e:
        print("\n[ERROR] An exception occurred. Full traceback:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
