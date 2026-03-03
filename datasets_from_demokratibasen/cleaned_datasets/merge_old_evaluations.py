#!/usr/bin/env python3
"""
Merge OLD_pred_ref_summary_evaluations and OLD_summarisation_evaluations into one folder.

Each merged file has top-level fields (in order):
  - dokument_id
  - reference_metrics (from OLD_pred_ref_summary_evaluations)
  - hygiene_metrics (from OLD_summarisation_evaluations: "hygiene" or "hygiene_metrics")
  - faithfulness_metrics (from OLD_summarisation_evaluations: "faithfulness" or "faithfulness_metrics")

Only merges files that exist in BOTH source folders.
Output: summarisation_evaluations (never overwrites existing files)
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent / "text_summary_dataset_202601" / "baseline_metrics"
PRED_REF_DIR = BASE_DIR / "OLD_pred_ref_summary_evaluations"
SUMMARISATION_DIR = BASE_DIR / "OLD_summarisation_evaluations"
OUTPUT_DIR = BASE_DIR / "summarisation_evaluations"


def main():
    pred_ref_files = {f.name: f for f in PRED_REF_DIR.glob("*.json")}
    summarisation_files = {f.name: f for f in SUMMARISATION_DIR.glob("*.json")}

    common = set(pred_ref_files) & set(summarisation_files)
    print(f"Files in OLD_pred_ref_summary_evaluations: {len(pred_ref_files)}")
    print(f"Files in OLD_summarisation_evaluations: {len(summarisation_files)}")
    print(f"Common files to merge: {len(common)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    merged_count = 0
    skipped_count = 0
    for filename in sorted(common):
        out_path = OUTPUT_DIR / filename
        if out_path.exists():
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if "reference_metrics" in existing:
                    skipped_count += 1
                    continue
            except (json.JSONDecodeError, OSError):
                pass

        with open(pred_ref_files[filename], "r", encoding="utf-8") as f:
            pred_ref = json.load(f)
        with open(summarisation_files[filename], "r", encoding="utf-8") as f:
            summarisation = json.load(f)

        hygiene = summarisation.get("hygiene") or summarisation.get("hygiene_metrics") or {}
        faithfulness = summarisation.get("faithfulness") or summarisation.get("faithfulness_metrics") or {}
        merged = {
            "dokument_id": pred_ref.get("dokument_id") or summarisation.get("dokument_id"),
            "reference_metrics": pred_ref.get("reference_metrics", {}),
            "hygiene_metrics": hygiene,
            "faithfulness_metrics": faithfulness,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False, default=str)

        merged_count += 1
        if merged_count % 2000 == 0:
            print(f"  Merged {merged_count}...")

    print(f"Done. Merged {merged_count} files into {OUTPUT_DIR}")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} already-merged files (not overwritten)")


if __name__ == "__main__":
    main()
