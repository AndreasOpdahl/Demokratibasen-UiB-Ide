import os
import json
import csv

# -----------------------------
# Config
# -----------------------------
OLD_DIR = "features"
NEW_DIR = "new_features"
OUT_DIR = "features_latest"

REQUIRED_KEYS = ["dokument_id", "kommune", "url", "doc_type", "tittel", "tekst"]

DUPS_AUDIT = "features_key_duplicates_audit.csv"
INVALID_AUDIT = "features_invalid_records.csv"

# -----------------------------
# Helpers
# -----------------------------
def print_header(title: str):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def sanitize_for_filename(value) -> str:
    """Make a safe filename token from any value."""
    s = "unknown" if value is None else str(value).strip()
    if not s:
        s = "unknown"
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        s = s.replace(ch, '_')
    s = s.replace(' ', '_')
    return s

def iter_json_files(root_dir: str):
    """Yield all .json file paths under root_dir (recursively), in sorted order for stability."""
    for base, _, files in os.walk(root_dir):
        for name in sorted(files):
            if name.lower().endswith(".json"):
                yield os.path.join(base, name)

def load_json(path: str):
    """Load a JSON object from file; return (data, error_message)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None, "Top-level JSON is not an object"
        return data, None
    except Exception as e:
        return None, f"JSON parse error: {e}"

def has_required_keys(obj: dict, keys: list[str]):
    """Return (bool, missing_keys_list)."""
    missing = [k for k in keys if k not in obj]
    return (len(missing) == 0), missing

def write_json(path: str, obj: dict):
    """Write JSON with stable formatting, UTF-8."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# -----------------------------
# Main
# -----------------------------
def main():
    print_header("Context")
    print(f"Working directory: {os.getcwd()}")
    print(f"Old features dir:  {os.path.abspath(OLD_DIR)} (exists={os.path.exists(OLD_DIR)})")
    print(f"New features dir:  {os.path.abspath(NEW_DIR)} (exists={os.path.exists(NEW_DIR)})")

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output dir ready:  {os.path.abspath(OUT_DIR)}")

    # Stats
    stats = {
        "scanned_old": 0,
        "scanned_new": 0,
        "valid_old": 0,
        "valid_new": 0,
        "written": 0,
        "duplicates": 0,
        "invalid": 0
    }

    # Tracking
    seen_ids = set()
    duplicates_rows = []  # rows for CSV audit
    invalid_rows = []     # rows for CSV audit

    # Phase order matters: OLD first (kept), NEW second (dedup against OLD)
    for source_dir, label in [(OLD_DIR, "old"), (NEW_DIR, "new")]:
        if not os.path.isdir(source_dir):
            print(f"NOTE: Source directory missing: {source_dir} (skipping)")
            continue

        for path in iter_json_files(source_dir):
            if label == "old":
                stats["scanned_old"] += 1
            else:
                stats["scanned_new"] += 1

            data, err = load_json(path)
            if err:
                stats["invalid"] += 1
                invalid_rows.append({
                    "source": label,
                    "path": os.path.abspath(path),
                    "reason": err,
                    "dokument_id": ""
                })
                continue

            ok, missing = has_required_keys(data, REQUIRED_KEYS)
            if not ok:
                stats["invalid"] += 1
                invalid_rows.append({
                    "source": label,
                    "path": os.path.abspath(path),
                    "reason": f"Missing keys: {','.join(missing)}",
                    "dokument_id": data.get("dokument_id", "")
                })
                continue

            # Count valid
            if label == "old":
                stats["valid_old"] += 1
            else:
                stats["valid_new"] += 1

            dokument_id = str(data.get("dokument_id", "")).strip()
            if not dokument_id:
                stats["invalid"] += 1
                invalid_rows.append({
                    "source": label,
                    "path": os.path.abspath(path),
                    "reason": "Empty dokument_id",
                    "dokument_id": ""
                })
                continue

            if dokument_id in seen_ids:
                stats["duplicates"] += 1
                duplicates_rows.append({
                    "dokument_id": dokument_id,
                    "source": label,
                    "path": os.path.abspath(path)
                })
                continue

            # First time we see this dokument_id → write it
            seen_ids.add(dokument_id)
            kommune_safe = sanitize_for_filename(data.get("kommune"))
            out_name = f"{kommune_safe}_{dokument_id}.json"
            out_path = os.path.join(OUT_DIR, out_name)

            # Write JSON as-is (no cleanup other than formatting)
            write_json(out_path, data)
            stats["written"] += 1

    # Write audits
    if duplicates_rows:
        with open(DUPS_AUDIT, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dokument_id", "source", "path"])
            writer.writeheader()
            writer.writerows(duplicates_rows)

    if invalid_rows:
        with open(INVALID_AUDIT, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source", "path", "reason", "dokument_id"])
            writer.writeheader()
            writer.writerows(invalid_rows)

    # Summary
    print_header("Summary")
    print(f"Scanned (old):            {stats['scanned_old']}")
    print(f"Scanned (new):            {stats['scanned_new']}")
    print(f"Valid (old):              {stats['valid_old']}")
    print(f"Valid (new):              {stats['valid_new']}")
    print(f"Duplicates skipped:       {stats['duplicates']}")
    print(f"Invalid skipped:          {stats['invalid']}")
    print(f"Wrote to features_latest: {stats['written']}")
    if duplicates_rows:
        print(f"→ Key-duplicates audit:   {os.path.abspath(DUPS_AUDIT)}")
    if invalid_rows:
        print(f"→ Invalid records audit:  {os.path.abspath(INVALID_AUDIT)}")

if __name__ == "__main__":
    main()
