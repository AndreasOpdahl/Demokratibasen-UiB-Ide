"""
Dataset registry mapping dataset names to file paths and adapters.
"""

from pathlib import Path
from typing import Dict, Tuple, Type, Optional
from .dataset_adapter_202505 import DatasetAdapter202505
from .dataset_adapter_202510 import DatasetAdapter202510
from .dataset_adapter_202601 import DatasetAdapter202601
from .dataset_adapter_bergen_2017_2023 import DatasetAdapterBergen2017_2023


# Default root path (can be overridden)
_DEFAULT_ROOT = None


def set_root_path(root: Path):
    """Set the root path for resolving dataset file paths."""
    global _DEFAULT_ROOT
    _DEFAULT_ROOT = root


def get_root_path() -> Optional[Path]:
    """Get the current root path."""
    return _DEFAULT_ROOT


# Dataset registry: maps dataset name to (file_path_relative_to_root, adapter_class)
DATASET_REGISTRY: Dict[str, Tuple[str, Type]] = {
    "dataset-202505": (
        "case_documents_summary/data_raw/dokumenter.jsonl",
        DatasetAdapter202505
    ),
    "dataset-202510": (
        "datasets_from_demokratibasen/prepared_datasets/text_summary_dataset_43221_examples/text_summary_examples_202505_to_10.jsonl",
        DatasetAdapter202510
    ),
    "dataset-Bergen-2017-2023": (
        "../Kommunebasen-Bergen/ekstraher_tekster_og_typer/dok_tekster",
        DatasetAdapterBergen2017_2023
    ),
    # 2026-01 cleaned text summary datasets (train/val/test splits)
    "dataset-202601-train": (
        "datasets_from_demokratibasen/cleaned_datasets/text_summary_dataset_202601/149978_text_summary_examples_train.jsonl",
        DatasetAdapter202601,
    ),
    "dataset-202601-val": (
        "datasets_from_demokratibasen/cleaned_datasets/text_summary_dataset_202601/149978_text_summary_examples_val.jsonl",
        DatasetAdapter202601,
    ),
    "dataset-202601-test": (
        "datasets_from_demokratibasen/cleaned_datasets/text_summary_dataset_202601/149978_text_summary_examples_test.jsonl",
        DatasetAdapter202601,
    ),
}


def _find_repo_root(start_path: Path) -> Path:
    """
    Find the repository root by looking for common markers.
    
    Looks for:
    - .git directory
    - datasets_from_demokratibasen directory (repo-specific marker)
    
    Args:
        start_path: Path to start searching from (typically __file__)
        
    Returns:
        Repository root path
        
    Raises:
        ValueError: If repository root cannot be found
    """
    current = start_path.resolve()
    
    # Try to find repo root by looking for markers
    for _ in range(10):  # Limit search depth
        # Check for .git directory (common repo marker)
        if (current / ".git").exists():
            return current
        # Check for repo-specific marker
        if (current / "datasets_from_demokratibasen").exists():
            return current
        # Check for case_documents_summary (another marker)
        if (current / "case_documents_summary").exists():
            return current
        
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent
    
    raise ValueError(
        f"Could not find repository root starting from {start_path}. "
        "Please provide root parameter or call set_root_path()."
    )


def get_dataset_path(dataset_name: str, root: Optional[Path] = None) -> Path:
    """
    Get the full path to a dataset file or directory.
    
    Args:
        dataset_name: Name of the dataset (e.g., "dataset-202505")
        root: Root path for resolving relative paths. If None, tries to find repo root automatically.
        
    Returns:
        Full path to the dataset file (JSONL) or directory (for datasets with individual JSON files)
        
    Raises:
        ValueError: If dataset name is not in registry or root cannot be determined
    """
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available datasets: {available}"
        )
    
    root_path = root or _DEFAULT_ROOT
    if root_path is None:
        # Try to find repo root automatically
        # Start from the dataset_registry.py file location
        registry_file = Path(__file__).resolve()
        root_path = _find_repo_root(registry_file)
    
    relative_path, _ = DATASET_REGISTRY[dataset_name]
    return root_path / relative_path


def get_dataset_adapter(dataset_name: str) -> Type:
    """
    Get the adapter class for a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., "dataset-202505")
        
    Returns:
        Adapter class for the dataset
        
    Raises:
        ValueError: If dataset name is not in registry
    """
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available datasets: {available}"
        )
    
    _, adapter_class = DATASET_REGISTRY[dataset_name]
    return adapter_class


def list_datasets() -> list[str]:
    """List all available dataset names."""
    return list(DATASET_REGISTRY.keys())


def register_dataset(name: str, file_path: str, adapter_class: Type):
    """
    Register a new dataset (useful for testing).
    
    Args:
        name: Dataset name
        file_path: Relative path from root
        adapter_class: Adapter class to use
    """
    DATASET_REGISTRY[name] = (file_path, adapter_class)

