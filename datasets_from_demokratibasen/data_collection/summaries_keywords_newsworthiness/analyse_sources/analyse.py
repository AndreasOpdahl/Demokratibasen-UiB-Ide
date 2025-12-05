"""
Collections of source file paths grouped by role.
"""

import csv
import json
from pathlib import Path
import sys
from collections import Counter

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(1024 * 1024 * 1024)


BASE_DIR = Path(__file__).resolve().parent.parent

description_files = [
    "sources/111188-demokratibasen-prod-urls-20250921.csv",
    "sources/111721-demokratibasen-test-urls-20250920.csv",
    "sources/12243-demokratibasen-uib-ide-urls-20250920.csv",
    "sources/36812-demokratibasen-urls-20250528.csv",
]
description_schema = ['dokument_id', 'doc_type', 'kommune', 'tittel', 'url']
description_map = {
    'dokument_id': 'dok_id',
    'doc_type': 'dok_type',
    'kommune': 'kommune',
    'tittel': 'dok_tittel',
    'url': 'url',
}

json_texts_files = [
    "sources/103908-dokumenter-texts-20250921.jsonl",
    "sources/36812-demokratibasen-texts-20250528.jsonl",
]
json_texts_schema = ['dokument_id', 'doc_type', 'kommune', 'tittel', 'url', 'tekst']
json_texts_map = {
    'dokument_id': 'dok_id',
    'doc_type': 'dok_type',
    'kommune': 'kommune',
    'tittel': 'dok_tittel',
    'url': 'url',
    'tekst': 'tekst',
}

csv_texts_files = [
    "sources/12243-demokratibasen-uib-ide-texts-20250920.csv",
]
csv_texts_schema = ['dokument_id', 'doc_type', 'kommune', 'tittel', 'url', 'doc_tekst']
csv_texts_map = {
    'dokument_id': 'dok_id',
    'doc_type': 'dok_type',
    'kommune': 'kommune',
    'tittel': 'dok_tittel',
    'url': 'url',
    'doc_tekst': 'tekst',
}

inferences_files = [
    "sources/17569-demokratibasen-inferences-20250624.csv",
    "sources/29281-demokratibasen-prod-inferences-20250921.csv",
    "sources/29602-demokratibasen-test-inferences-20250920.csv",
    "sources/6100-demokratibasen-uib-ide-inferences-20250920.csv",
]
inferences_schema = ['dokument_id', 'batch_id', 'tittel', 'oppsummering', 'personer', 'nokkelord', 'nyhetsverdi']
inferences_map = {
    'dokument_id': 'dok_id',
    'batch_id': None,
    'tittel': 'oppsum_tittel',
    'oppsummering': 'oppsummering',
    'personer': 'personer',
    'nokkelord': 'nokkelord',
    'nyhetsverdi': 'nyhetsverdi',
}

description_and_inference_files = [
    "sources/44118-url-oppsummering-20250930.csv",
    "extract_data_from_prod/50301_url_oppsummering_from_prod_20251026.csv",
]
description_and_inference_schema = ['dok_id', 'kommune', 'dok_type', 'dok_tittel', 'url', 'oppsum_tittel', 'oppsummering', 'personer', 'nokkelord', 'nyhetsverdi']
# misses: 'tekst', 'modell', 'maks_tokens'

joined_example_files = [
    "training_data/17720-examples-from-prod-20250930.csv",
    "training_data/27725-url-tekst-oppsummering-20251026.csv",
]
joined_example_schema = ['dok_id', 'kommune', 'url', 'dok_type', 'dok_tittel', 'text', 'model', 'max_tokens', 'oppsum_tittel', 'oppsummering', 'personer', 'nokkelord', 'nyhetsverdi']
joined_example_map = {
    'dok_id': 'dok_id',
    'kommune': 'kommune',
    'url': 'url',
    'dok_type': 'dok_type',
    'dok_tittel': 'dok_tittel',
    'text': 'tekst',
    'model': 'modell',
    'max_tokens': 'maks_tokens',
    'oppsum_tittel': 'oppsum_tittel',
    'oppsummering': 'oppsummering',
    'personer': 'personer',
    'nokkelord': 'nokkelord',
    'nyhetsverdi': 'nyhetsverdi',
}



def canonical_data(files, key_map):
    """Load files and apply a key-mapping to produce canonical dicts."""
    canonical = {}
    for path in files:
        full_path = Path(path)
        if not full_path.is_absolute():
            full_path = BASE_DIR / full_path
        rows = []
        if full_path.suffix.lower() == ".csv":
            with full_path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    mapped = {}
                    for source_key, value in row.items():
                        canonical_key = key_map.get(source_key)
                        if canonical_key is None:
                            continue
                        mapped[canonical_key] = value
                    rows.append(mapped)
        elif full_path.suffix.lower() == ".jsonl":
            with full_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    mapped = {}
                    for source_key, value in obj.items():
                        canonical_key = key_map.get(source_key)
                        if canonical_key is None:
                            continue
                        mapped[canonical_key] = value
                    rows.append(mapped)
        else:
            raise ValueError(f"Unsupported file type: {path}")

        canonical[path] = rows
    return canonical

canonical_descriptions = canonical_data(description_files, description_map)
canonical_json_texts = canonical_data(json_texts_files, json_texts_map)
canonical_csv_texts = canonical_data(csv_texts_files, csv_texts_map)
canonical_inferences = canonical_data(inferences_files, inferences_map)
canonical_joined_examples = canonical_data(joined_example_files, joined_example_map)

canonical_data = {
    "Descriptions": canonical_descriptions,
    "JSON Texts": canonical_json_texts,
    "CSV Texts": canonical_csv_texts,
    "Inferences": canonical_inferences,
    "Joined Examples": canonical_joined_examples,
}
