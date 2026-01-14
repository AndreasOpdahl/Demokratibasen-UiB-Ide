#!/usr/bin/env python3
"""
Unified dataset analysis script.

Analyzes any registered dataset (e.g., 'dataset-202505', 'dataset-202510'):
- Counts raw vs filtered documents
- Counts documents with/without dok_type (raw and filtered)
- Distribution of dok_type (filtered and raw)
- Kommune distribution (filtered)
- Collects sample docs by type
"""

from __future__ import annotations

import sys
import json
import argparse
from dataclasses import dataclass, asdict
from collections import Counter
from typing import Optional, Dict, Any
from pathlib import Path

# Ensure local package import when executed directly
script_dir = Path(__file__).parent  # dataset_loader/
parent_dir = script_dir.parent  # parent of dataset_loader/
sys.path.insert(0, str(parent_dir))

from dataset_loader import DatasetLoader, get_dataset_adapter, kommunenavn  # type: ignore


@dataclass
class AnalysisResult:
    dataset_name: str
    path: str
    total_documents_raw: int
    total_documents_filtered: int
    documents_with_type_filtered: int
    documents_without_type_filtered: int
    documents_with_type_raw: int
    documents_without_type_raw: int
    type_counter_filtered: Dict[str, int]
    type_counter_raw: Dict[str, int]
    kommune_counter: Dict[int, int]
    samples_by_type: Dict[str, Dict[str, Any]]
    samples: list


def _iter_raw_documents(path: Path):
    """Yield raw documents from a JSONL file or a directory of JSON files."""
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    elif path.is_dir():
        for jf in sorted(path.glob("*.json")):
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                yield doc
            except json.JSONDecodeError:
                continue


def analyze_dataset(dataset_name: str, root: Optional[Path] = None) -> AnalysisResult:
    """
    Analyze a registered dataset and return structured results.
    """
    loader = DatasetLoader(dataset_name, root=root)
    adapter_cls = get_dataset_adapter(dataset_name)
    adapter = adapter_cls()

    # First pass: filtered iteration through loader
    total_documents_filtered = 0
    documents_with_type_filtered = 0
    documents_without_type_filtered = 0
    type_counter_filtered: Counter[str] = Counter()
    kommune_counter: Counter[int] = Counter()
    samples = []
    samples_by_type: Dict[str, Dict[str, Any]] = {}
    filtered_doc_ids = set()

    for doc_id, kommune_nummer, kommune_navn, text in loader():
        total_documents_filtered += 1
        filtered_doc_ids.add(doc_id)
        if kommune_nummer is not None:
            kommune_counter[kommune_nummer] += 1
        if len(samples) < 5:
            samples.append(
                {
                    "doc_id": doc_id,
                    "kommune_nummer": kommune_nummer,
                    "kommune_navn": kommune_navn,
                    "text_length": len(text),
                }
            )

    # Second pass: raw docs via adapter.normalize
    total_documents_raw = 0
    documents_with_type_raw = 0
    documents_without_type_raw = 0
    type_counter_raw: Counter[str] = Counter()

    raw_path = loader.path
    for raw_doc in _iter_raw_documents(raw_path):
        total_documents_raw += 1
        try:
            normalized = adapter.normalize(raw_doc)
        except Exception:
            continue

        doc_id = normalized.get("dok_id")
        doc_type = normalized.get("dok_type")

        if doc_type:
            documents_with_type_raw += 1
            type_counter_raw[doc_type] += 1
        else:
            documents_without_type_raw += 1

        if doc_id in filtered_doc_ids:
            if doc_type:
                documents_with_type_filtered += 1
                type_counter_filtered[doc_type] += 1
                if (
                    doc_type not in samples_by_type
                    and len(samples_by_type) < 10
                ):
                    samples_by_type[doc_type] = {
                        "doc_id": doc_id,
                        "dok_type": doc_type,
                        "dok_tittel": normalized.get("dok_tittel"),
                        "kommune_navn": normalized.get("kommune_navn"),
                    }
            else:
                documents_without_type_filtered += 1

    return AnalysisResult(
        dataset_name=dataset_name,
        path=str(raw_path),
        total_documents_raw=total_documents_raw,
        total_documents_filtered=total_documents_filtered,
        documents_with_type_filtered=documents_with_type_filtered,
        documents_without_type_filtered=documents_without_type_filtered,
        documents_with_type_raw=documents_with_type_raw,
        documents_without_type_raw=documents_without_type_raw,
        type_counter_filtered=dict(type_counter_filtered),
        type_counter_raw=dict(type_counter_raw),
        kommune_counter=dict(kommune_counter),
        samples_by_type=samples_by_type,
        samples=samples,
    )


def _print_result(result: AnalysisResult):
    print("=" * 80)
    print(f"Dataset Analysis: {result.dataset_name}")
    print("=" * 80)
    print()
    print(f"Dataset path: {result.path}")
    print()

    if result.total_documents_raw - result.total_documents_filtered > 0:
        print(f"Total documents in raw dataset: {result.total_documents_raw:,}")
        print(f"Documents filtered out: {result.total_documents_raw - result.total_documents_filtered:,}")
        print()
        print("RAW DOCUMENTS (all in file):")
        print(f"  Documents with dok_type: {result.documents_with_type_raw:,}")
        print(f"  Documents without dok_type: {result.documents_without_type_raw:,}")
        print()

    print(f"Total documents (after filtering: >=10 characters and alphanumeric): {result.total_documents_filtered:,}")
        
    print()

    print("FILTERED DOCUMENTS (used in processing):")
    print(f"  Documents with dok_type: {result.documents_with_type_filtered:,}")
    print(f"  Documents without dok_type: {result.documents_without_type_filtered:,}")
    print()

    print("=" * 80)
    print("DOCUMENT TYPES DISTRIBUTION (FILTERED)")
    print("=" * 80)
    if result.type_counter_filtered:
        total = sum(result.type_counter_filtered.values()) or 1
        for t, c in sorted(
            result.type_counter_filtered.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {t}: {c:,} documents ({(c/total)*100:.1f}%)")
    else:
        print("  No document types found.")
    print()

    print("=" * 80)
    print("KOMMUNE DISTRIBUTION (top 10, filtered)")
    print("=" * 80)
    for kommune_nummer, count in sorted(
        result.kommune_counter.items(), key=lambda x: x[1], reverse=True
    )[:10]:
        navn = kommunenavn(kommune_nummer) if kommune_nummer is not None else "ukjent kommune"
        print(f"  {kommune_nummer} ({navn}): {count:,} documents")
    print()

    if result.samples_by_type:
        print("=" * 80)
        print("SAMPLE DOCUMENTS BY TYPE")
        print("=" * 80)
        for doc_type, sample in result.samples_by_type.items():
            print(f"Type: {doc_type}")
            print(f"  Document ID: {sample.get('doc_id')}")
            title = sample.get("dok_tittel")
            if title:
                if len(title) > 80:
                    title = title[:80] + "..."
                print(f"  Title: {title}")
            print(f"  Kommune: {sample.get('kommune_navn')}")
            print()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze a registered dataset.")
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        help="Dataset name (e.g., dataset-202505, dataset-202510)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Optional repository root to resolve dataset path.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root) if args.root else None
    try:
        result = analyze_dataset(args.dataset, root=root)
    except SystemExit:
        # bubbled up from DatasetLoader on missing path
        raise
    except Exception as e:
        print(f"Error analyzing dataset: {e}", file=sys.stderr)
        return 1

    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

