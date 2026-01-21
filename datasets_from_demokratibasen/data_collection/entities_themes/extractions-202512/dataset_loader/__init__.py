"""
Dataset loader package for loading and processing JSONL files.

This package provides:
- DatasetLoader: Main class for loading datasets by name
- DatasetAdapter202505: Adapter for 2025-05 format (field name translation)
- DatasetAdapter202510: Adapter for 2025-10 format (processed_data.jsonl)
- dataset_registry: Registry mapping dataset names to paths and adapters
- kommunenavn: Function for translating kommune IDs to names
- KOMMUNENAVN: Dictionary mapping kommune IDs to names
"""

from .loader import DatasetLoader
from .dataset_adapter_202505 import DatasetAdapter202505
from .dataset_adapter_202510 import DatasetAdapter202510
from .dataset_registry import (
    DATASET_REGISTRY,
    get_dataset_path,
    get_dataset_adapter,
    list_datasets,
    set_root_path,
    get_root_path,
    register_dataset,
)
from .kommune import kommunenavn, KOMMUNENAVN

__all__ = [
    "DatasetLoader",
    "DatasetAdapter202505",
    "DatasetAdapter202510",
    "DATASET_REGISTRY",
    "get_dataset_path",
    "get_dataset_adapter",
    "list_datasets",
    "set_root_path",
    "get_root_path",
    "register_dataset",
    "kommunenavn",
    "KOMMUNENAVN",
]

