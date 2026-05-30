import re
from pathlib import Path

# ----------------------------------------
# CONFIG
# ----------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = SCRIPT_DIR / "to be delited"
LOCAL_BASE = SCRIPT_DIR / "Data" / "eval" / "New folder"

# Optional overrides when auto-matching summary stem -> local folder name fails
LOCAL_FOLDER_OVERRIDES = {
    "nb-gpt-j-6b": "nb-gpt-j-6",
}

# Example `ls -l` line:
# -rw-r--r-- 1 sinoa ansatt 13598974 april 27 08:00 path/to/file.jsonl
LS_LINE = re.compile(
    r"^-rw[-\w]+\s+\d+\s+\S+\s+\S+\s+(\d+)\s+\S+\s+\d+\s+(?:\d+:\d+|\d{4})\s+(.+)$"
)


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def resolve_local_folder(summary_stem: str, local_folders: list[str]) -> str | None:
    if summary_stem in LOCAL_FOLDER_OVERRIDES:
        target = LOCAL_FOLDER_OVERRIDES[summary_stem]
        return target if target in local_folders else None

    norm_stem = normalize_name(summary_stem)
    matches = []
    for folder in local_folders:
        norm_folder = normalize_name(folder)
        if norm_stem == norm_folder or norm_stem.startswith(norm_folder) or norm_folder.startswith(norm_stem):
            matches.append(folder)

    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    # Prefer the longest folder name (most specific) on ambiguous matches
    return max(matches, key=len)


def parse_reference_listing(path: Path) -> dict[str, int]:
    """Return {basename: size_bytes} from an `ls -l` listing file."""
    files: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = LS_LINE.match(line)
            if match:
                size = int(match.group(1))
                basename = Path(match.group(2)).name
                files[basename] = size
    return files


def compare_listing(reference: dict[str, int], local_root: Path) -> dict:
    """Compare only files that exist locally against the server listing."""
    local_files = {
        p.name: p.stat().st_size
        for p in local_root.iterdir()
        if p.is_file()
    }

    matched = []
    size_mismatches = []
    not_in_listing = []

    for basename in sorted(local_files):
        actual_size = local_files[basename]
        if basename not in reference:
            not_in_listing.append(basename)
            continue
        expected_size = reference[basename]
        if actual_size == expected_size:
            matched.append(basename)
        else:
            size_mismatches.append((basename, expected_size, actual_size))

    return {
        "checked": len(local_files),
        "matched": matched,
        "size_mismatches": size_mismatches,
        "not_in_listing": not_in_listing,
    }


def print_model_report(summary_name: str, local_folder: str, result: dict) -> None:
    matched = result["matched"]
    size_mismatches = result["size_mismatches"]
    not_in_listing = result["not_in_listing"]
    checked = result["checked"]

    print("=" * 70)
    print(f"Summary:  {summary_name}")
    print(f"Local:    {LOCAL_BASE / local_folder}")
    print("-" * 70)
    print(f"  Local files checked:   {checked}")
    print(f"  Same size (match):     {len(matched)}")
    print(f"  Different size:        {len(size_mismatches)}")
    print(f"  Not in server listing: {len(not_in_listing)}")

    if checked and len(matched) == checked:
        print("  => All local files match the server listing (identical size).")
    elif size_mismatches:
        print("  => Some local files differ in size from the server listing (re-download?).")
    elif not_in_listing and not size_mismatches:
        print("  => Local files match where listed; some files have no server entry.")

    if size_mismatches:
        print("\n  DIFFERENT SIZE (not the same file):")
        for basename, expected, actual in size_mismatches:
            print(f"    {basename}")
            print(f"      server: {expected} bytes")
            print(f"      local:  {actual} bytes")

    if not_in_listing:
        print("\n  NOT IN SERVER LISTING:")
        for name in not_in_listing:
            print(f"    {name}")


def main() -> None:
    if not REFERENCE_DIR.is_dir():
        raise SystemExit(f"Reference folder not found: {REFERENCE_DIR}")
    if not LOCAL_BASE.is_dir():
        raise SystemExit(f"Local base folder not found: {LOCAL_BASE}")

    summary_files = sorted(REFERENCE_DIR.glob("*.txt"))
    if not summary_files:
        raise SystemExit(f"No *.txt listings in {REFERENCE_DIR}")

    local_folders = sorted(p.name for p in LOCAL_BASE.iterdir() if p.is_dir())

    totals = {"checked": 0, "matched": 0, "mismatch": 0, "not_in_listing": 0}
    unmapped = []

    for summary_path in summary_files:
        stem = summary_path.stem
        local_folder = resolve_local_folder(stem, local_folders)

        if local_folder is None:
            unmapped.append(stem)
            print("=" * 70)
            print(f"Summary:  {summary_path.name}")
            print(f"  ERROR: No matching folder under {LOCAL_BASE}")
            print(f"  Available: {', '.join(local_folders) or '(none)'}")
            continue

        reference = parse_reference_listing(summary_path)
        if not reference:
            print("=" * 70)
            print(f"Summary:  {summary_path.name}")
            print(f"  WARNING: No files parsed from listing (check format).")
            continue

        result = compare_listing(reference, LOCAL_BASE / local_folder)
        print_model_report(summary_path.name, local_folder, result)

        totals["checked"] += result["checked"]
        totals["matched"] += len(result["matched"])
        totals["mismatch"] += len(result["size_mismatches"])
        totals["not_in_listing"] += len(result["not_in_listing"])

    print("\n" + "=" * 70)
    print("OVERALL")
    print(f"  Models compared:       {len(summary_files) - len(unmapped)}")
    print(f"  Unmapped summaries:    {len(unmapped)}")
    print(f"  Local files checked:   {totals['checked']}")
    print(f"  Same size (match):     {totals['matched']}")
    print(f"  Different size:        {totals['mismatch']}")
    print(f"  Not in server listing: {totals['not_in_listing']}")
    print("=" * 70)

    if unmapped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
