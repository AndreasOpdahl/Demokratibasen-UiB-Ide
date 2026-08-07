"""Defaults for building human annotation datasets from G-Eval exports."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Data moved out of the repo (2026-06) into the shared OneDrive folder. Override with
# CHECKPOINT_SELECTION_DATA_DIR if your OneDrive root or the dataset snapshot name differs.
DATA_ROOT: Path = Path(
    os.environ.get("CHECKPOINT_SELECTION_DATA_DIR")
    or (
        Path(os.environ.get("ONEDRIVE", str(Path.home() / "OneDrive")))
        / "Shared"
        / "Demokratibasen-UiB-Ide"
        / "EvaluationDatasets"
        / "CheckpointSelection"
        / "Data_202606"
    )
)

DEFAULT_CHECKPOINT_DIR = REPO_ROOT / ".deepeval" / "geval_judgment_checkpoints" / "winners"
DEFAULT_CONTEXT_EXPORT_DIR = REPO_ROOT / ".deepeval" / "geval_exports" / "winners"
DEFAULT_GEVAL_EXPORT_DIR = DEFAULT_CONTEXT_EXPORT_DIR

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

DIMENSIONS: tuple[str, ...] = (
    "relevance",
    "consistency",
    "newsworthiness",
    "hygiene",
)

# Short id prefix for item_id (e.g. rel-01) and UI colors for each dimension.
DIMENSION_ID_PREFIX: dict[str, str] = {
    "relevance": "rel",
    "consistency": "con",
    "newsworthiness": "new",
    "hygiene": "hyg",
}

DIMENSION_COLORS: dict[str, str] = {
    "relevance": "#1976d2",
    "consistency": "#388e3c",
    "newsworthiness": "#f57c00",
    "hygiene": "#7b1fa2",
}

DEFAULT_JUDGES: tuple[str, ...] = (
    "gpt-5-mini",
    "google/gemini-2.5-flash-preview-05-20",
    "anthropic/claude-3-5-haiku-20241022",
    "mistral-medium-latest",
)

REFERENCE_MODEL_ID = "GPT4o-mini"

# Stratified mix for judge-validation samples (independent per dimension).
DEFAULT_SELECTION_RATIOS: dict[str, float] = {
    "low_agreement": 0.40,
    "tie_majority": 0.10,
    "high_agreement": 0.15,
    "reference_challenged": 0.20,
    "representative": 0.15,
}

SELECTION_BUCKET_ORDER: tuple[str, ...] = (
    "low_agreement",
    "tie_majority",
    "high_agreement",
    "reference_challenged",
    "representative",
)

DEFAULT_PER_DIMENSION = 25
DEFAULT_SEED = 42
