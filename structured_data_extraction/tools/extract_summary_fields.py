import argparse
from pathlib import Path
from typing import Dict, List, Set

import ijson


FIELD_TO_OUTPUT = {
    "hva_saken_gjelder": "issues",
    "tema": "topics",
    "viktige_personer": "people",
    "viktige_steder": "places",
    "viktige_organisasjoner": "organisations",
    "viktige_hendelser": "events",
    "viktige_tidspunkter": "times",
}


def collect_from_properties(properties: Dict[str, List[List[str]]], field: str, dest: Set[str]) -> None:
    values = properties.get(field, [])
    for model_values in values:
        for item in model_values:
            if item:
                dest.add(item.strip())


def collect_from_counts(counts: List[List]) -> Set[str]:
    extracted = set()
    for entry in counts:
        if not entry:
            continue
        extracted.add(entry[0])
    return extracted


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract summary fields from a combined summary JSON.")
    parser.add_argument(
        "--summary-file",
        type=Path,
        required=True,
        help="Path to the combined summary JSON (e.g. summaries/dataset-...json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_data/extractions"),
        help="Output directory for extracted text files",
    )
    args = parser.parse_args()

    if not args.summary_file.exists():
        parser.error(f"{args.summary_file} does not exist")

    extracted: Dict[str, Set[str]] = {out: set() for out in FIELD_TO_OUTPUT.values()}

    with args.summary_file.open("rb") as f:
        for analysis in ijson.items(f, "document_analyses.item"):
            properties = analysis.get("properties", {})
            counts = analysis.get("counts", {})
            for field, out_name in FIELD_TO_OUTPUT.items():
                if field == "hva_saken_gjelder":
                    collect_from_properties(properties, field, extracted[out_name])
                else:
                    extracted[out_name].update(collect_from_counts(counts.get(field, [])))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for field, data in extracted.items():
        out_path = args.output_dir / f"{field}.txt"
        with out_path.open("w", encoding="utf-8") as fout:
            for entry in sorted(data):
                fout.write(entry + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
