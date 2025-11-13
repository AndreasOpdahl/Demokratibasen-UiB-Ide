import os
import json
import pandas as pd

# ------------- Config -------------
LABELS_PATH = "labels_final.csv"      # or "labels_final.csv" if you want to check the synced set
FEATURES_DIR = "features_synced"       # or "features_synced" if you synced
REQUIRED_KEYS = ["dokument_id", "kommune", "url", "doc_type", "tittel", "tekst"]

# If set True, the script will raise an AssertionError when sets don't match
STRICT_ASSERT = False                   # set to True if you want a hard fail

# ------------- Helpers -------------
def iter_json_files(root):
    for base, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith(".json"):
                yield os.path.join(base, name)

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)

# ------------- Main -------------
def main():
    # 1) Load labels
    labels = pd.read_csv(LABELS_PATH, dtype=str, keep_default_na=False)
    labels["dokument_id"] = labels["dokument_id"].astype(str).str.strip()
    label_ids = set(labels["dokument_id"])
    print(f"Labels file: {os.path.abspath(LABELS_PATH)}")
    print(f"Label rows:  {len(labels)}")
    print(f"Unique label dokument_id: {len(label_ids)}")

    # 2) Scan features
    feature_ids = set()
    invalid_features = 0
    missing_key_counts = {k: 0 for k in REQUIRED_KEYS}

    for path in iter_json_files(FEATURES_DIR):
        data, err = load_json(path)
        if err or not isinstance(data, dict):
            invalid_features += 1
            continue
        # Check required keys presence (sanity)
        for k in REQUIRED_KEYS:
            if k not in data:
                missing_key_counts[k] += 1
        did = str(data.get("dokument_id", "")).strip()
        if did:
            feature_ids.add(did)

    print(f"\nFeatures dir: {os.path.abspath(FEATURES_DIR)}")
    print(f"Unique feature dokument_id: {len(feature_ids)}")
    if invalid_features:
        print(f"Warning: {invalid_features} feature files failed to parse as JSON.")
    if any(missing_key_counts.values()):
        print("Missing required keys in some features (sanity check):")
        for k, c in missing_key_counts.items():
            if c:
                print(f"  - {k}: {c} files missing this key")

    # 3) Set comparison
    labels_only = sorted(label_ids - feature_ids)
    features_only = sorted(feature_ids - label_ids)
    intersect = sorted(label_ids & feature_ids)

    print("\n=== Set Comparison ===")
    print(f"Labels only (no feature): {len(labels_only)}")
    print(f"Features only (no label): {len(features_only)}")
    print(f"Intersection (both):      {len(intersect)}")

    # Show a few examples if mismatches exist
    if labels_only[:5]:
        print(f"\nSample labels-only IDs (first 5): {labels_only[:5]}")
    if features_only[:5]:
        print(f"Sample features-only IDs (first 5): {features_only[:5]}")

    # 4) Optional strict assert
    if STRICT_ASSERT:
        assert len(labels_only) == 0, "There are labels without features."
        assert len(features_only) == 0, "There are features without labels."
        print("\nSTRICT ASSERT PASSED: feature and label IDs match exactly.")

if __name__ == "__main__":
    main()
