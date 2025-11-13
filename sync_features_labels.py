import os
import json
import shutil
import pandas as pd

# -----------------------------
# Config
# -----------------------------
LABELS_IN = "labels_latest.csv"
FEATURES_IN_DIR = "features_latest"

LABELS_OUT = "labels_final.csv"
FEATURES_SYNCED_DIR = "features_synced"  # new folder with features that have labels

AUDIT_DIR = "audits"
AUDIT_LABELS_MISSING_FEATURE = os.path.join(AUDIT_DIR, "audit_labels_missing_feature.csv")
AUDIT_FEATURES_MISSING_LABEL = os.path.join(AUDIT_DIR, "audit_features_missing_label.csv")

REQUIRED_KEYS = ["dokument_id", "kommune", "url", "doc_type", "tittel", "tekst"]

# -----------------------------
# Helpers
# -----------------------------
def print_header(title: str):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def iter_json_files(root_dir: str):
    for base, _, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith(".json"):
                yield os.path.join(base, name)

def load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj
    except Exception as e:
        return {"__error__": str(e)}

def has_required_keys(obj: dict, keys: list[str]):
    missing = [k for k in keys if k not in obj]
    return (len(missing) == 0), missing

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -----------------------------
# Main
# -----------------------------
def main():
    print_header("Context")
    print(f"Working directory: {os.getcwd()}")
    print(f"Labels CSV:        {os.path.abspath(LABELS_IN)} (exists={os.path.exists(LABELS_IN)})")
    print(f"Features dir:      {os.path.abspath(FEATURES_IN_DIR)} (exists={os.path.isdir(FEATURES_IN_DIR)})")

    ensure_dir(AUDIT_DIR)
    ensure_dir(FEATURES_SYNCED_DIR)

    # 1) Load labels
    labels = pd.read_csv(LABELS_IN, dtype=str, keep_default_na=False)
    labels = labels.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    print(f"Loaded labels_latest.csv with shape: {labels.shape}")

    # Sanity: no duplicate dokument_id in labels (should be 0 after your previous keep-first merge)
    dups_mask = labels.duplicated(subset=["dokument_id"], keep="first")
    dups_count = dups_mask.sum()
    print(f"Label duplicate check by dokument_id: {dups_count} duplicates")
    if dups_count > 0:
        # keep first, drop the rest (defensive; should be zero)
        labels = labels[~dups_mask].copy()
        print(f"Dropped {dups_count} duplicate label rows; new shape: {labels.shape}")

    label_ids = set(labels["dokument_id"].astype(str))

    # 2) Index features by dokument_id (with minimal sanity: required keys)
    print_header("Index features")
    feature_index = {}  # dokument_id -> file path
    invalid_feature_rows = []

    count_scanned = 0
    for p in iter_json_files(FEATURES_IN_DIR):
        count_scanned += 1
        data = load_json(p)
        if "__error__" in data:
            invalid_feature_rows.append({"path": os.path.abspath(p), "reason": data["__error__"], "dokument_id": ""})
            continue

        ok, missing = has_required_keys(data, REQUIRED_KEYS)
        if not ok:
            invalid_feature_rows.append({
                "path": os.path.abspath(p),
                "reason": f"Missing keys: {','.join(missing)}",
                "dokument_id": data.get("dokument_id", "")
            })
            continue

        did = str(data.get("dokument_id", "")).strip()
        if not did:
            invalid_feature_rows.append({"path": os.path.abspath(p), "reason": "Empty dokument_id", "dokument_id": ""})
            continue

        # If multiple files share the same dokument_id in features_latest, keep the first we encounter
        # (features_latest should already be deduped, but be defensive)
        if did not in feature_index:
            feature_index[did] = p

    print(f"Scanned features: {count_scanned}")
    print(f"Valid indexed features: {len(feature_index)}")
    if invalid_feature_rows:
        pd.DataFrame(invalid_feature_rows).to_csv(
            os.path.join(AUDIT_DIR, "audit_features_invalid.csv"), index=False, encoding="utf-8"
        )
        print(f"→ Wrote invalid features audit: {os.path.join(AUDIT_DIR, 'audit_features_invalid.csv')}")

    # 3) Compare sets
    label_only = sorted(label_ids.difference(feature_index.keys()))
    feature_only = sorted(set(feature_index.keys()).difference(label_ids))
    intersect_ids = sorted(label_ids.intersection(feature_index.keys()))

    print_header("Set comparison")
    print(f"Labels only (no feature): {len(label_only)}")
    print(f"Features only (no label): {len(feature_only)}")
    print(f"Intersection (both):      {len(intersect_ids)}")

    # 4) Audits
    if label_only:
        labels[labels["dokument_id"].isin(label_only)].to_csv(
            AUDIT_LABELS_MISSING_FEATURE, index=False, encoding="utf-8"
        )
        print(f"→ Wrote labels missing features: {os.path.abspath(AUDIT_LABELS_MISSING_FEATURE)}")

    if feature_only:
        pd.DataFrame({"dokument_id": feature_only,
                      "path": [os.path.abspath(feature_index[d]) for d in feature_only]}).to_csv(
            AUDIT_FEATURES_MISSING_LABEL, index=False, encoding="utf-8"
        )
        print(f"→ Wrote features missing labels: {os.path.abspath(AUDIT_FEATURES_MISSING_LABEL)}")

    # 5) Write cleaned labels (labels_final = only intersection)
    print_header("Write labels_final.csv")
    labels_final = labels[labels["dokument_id"].isin(set(intersect_ids))].copy()
    labels_final.to_csv(LABELS_OUT, index=False, encoding="utf-8")
    print(f"Saved labels_final.csv: {os.path.abspath(LABELS_OUT)} (shape={labels_final.shape})")

    # 6) (Recommended) Build features_synced with only intersecting features
    print_header("Build features_synced (copy only matched features)")
    copied = 0
    for did in intersect_ids:
        src = feature_index[did]
        # Use existing filename; alternatively, enforce {kommune}_{dokument_id}.json by re-reading JSON.
        dst = os.path.join(FEATURES_SYNCED_DIR, os.path.basename(src))
        shutil.copy2(src, dst)
        copied += 1
    print(f"Copied {copied} features to: {os.path.abspath(FEATURES_SYNCED_DIR)}")

    # 7) Summary
    print_header("Summary")
    print(f"Labels input:           {len(label_ids)}")
    print(f"Features input:         {len(feature_index)}")
    print(f"Labels only (no feat):  {len(label_only)}")
    print(f"Features only (no lab): {len(feature_only)}")
    print(f"Synced (intersection):  {len(intersect_ids)}")
    print(f"labels_final.csv rows:  {labels_final.shape[0]}")
    print(f"features_synced files:  {copied}")

if __name__ == "__main__":
    main()
