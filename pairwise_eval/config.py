"""Defaults for dimensions, judges, and RNG seeds."""

from __future__ import annotations

import os
from pathlib import Path

# Repository root (parent of the `pairwise_eval` package)
REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv_file(path: Path) -> None:
    """Set ``os.environ`` from ``KEY=VALUE`` lines in ``.env`` (does not override existing vars)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv_file(REPO_ROOT / ".env")

EVAL_DIMENSIONS: tuple[str, ...] = ("faithfulness", "correctness", "completeness")

# Each string is both the G-Eval judge key and the API ``model`` id passed to the backend.
# Local judges use ``LOCAL_LLM_CHAT_URL`` (LM Studio–style). Ids in ``OPENAI_JUDGE_IDS`` use
# OpenAI Chat Completions (``OPENAI_CHAT_COMPLETIONS_URL`` + ``OPENAI_API_KEY``).
JUDGES: tuple[str, ...] = (
    "google/gemma-3-4b",
    #"gpt-3.5-turbo",
    # "anthropic/claude-3-5-haiku-20241022",  # example
    # "google/gemini-2.5-flash-preview-05-20",
    # "mistral-medium-latest",
)

# Subset of ``JUDGES`` that call OpenAI-compatible ``/v1/chat/completions`` (not the local server).
OPENAI_JUDGE_IDS: frozenset[str] = frozenset({"gpt-3.5-turbo"})
OPENAI_API_KEY: str | None = os.environ.get("OPENAI_API_KEY")
OPENAI_CHAT_COMPLETIONS_URL: str = os.environ.get(
    "OPENAI_CHAT_COMPLETIONS_URL", "https://api.openai.com/v1/chat/completions"
)

HUMAN_JUDGES: tuple[str, ...] = ()
LLM_JUDGES: tuple[str, ...] = tuple(j for j in JUDGES if j not in HUMAN_JUDGES)

# Synthetic ``model_id`` for the gold summary from eval JSONL ``reference`` (one row per doc in long_df).
REFERENCE_SUMMARY_MODEL_ID = "GPT4o-mini"

# Local chat API (same default as ``geval_local_judge``).
LOCAL_LLM_CHAT_URL = "http://localhost:1234/api/v1/chat"
LOCAL_LLM_TIMEOUT_S = 300.0

# First N documents (by first-seen ``doc_id`` order). None = use the full loaded corpus.
MAX_DOCUMENTS: int | None = 601  #600
# G-Eval: append-only JSONL per (judge × dimension); resume skips keys already on disk. None = off.
GEVAL_CHECKPOINT_DIR: Path | None = REPO_ROOT / ".deepeval" / "geval_judgment_checkpoints"

DEFAULT_PAIR_SEED = 42
DEFAULT_GEVAL_BASE_SEED = 42
MOCK_TIE_PROB = 0.1
