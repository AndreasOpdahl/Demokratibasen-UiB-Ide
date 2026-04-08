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


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return default
    return float(str(raw).strip())


# Each name must have a template ``Data/prompts/geval/{name}.txt`` (or set overrides in ``geval_local_judge._GEVAL_PROMPT_FILE_OVERRIDES``).
EVAL_DIMENSIONS: tuple[str, ...] = (
    "faithfulness",
    "correctness",
    "completeness",
    "newsworthiness",
)

# Each string is both the G-Eval judge key and (for cloud APIs) the model id passed to the backend.
# Local judges use ``LOCAL_LLM_CHAT_URL`` (LM Studio–style). Ids in ``OPENAI_JUDGE_IDS`` use
# OpenAI Chat Completions (``OPENAI_CHAT_COMPLETIONS_URL`` + ``OPENAI_API_KEY``). Ids in
# ``GEMINI_JUDGE_IDS`` use the Gemini REST API (``GEMINI_API_BASE`` + ``GOOGLE_API_KEY`` or
# ``GEMINI_API_KEY``); the judge id is ``google/<gemini-model-id>`` and the API model id is the
# part after ``google/``.
JUDGES: tuple[str, ...] = (
    "google/gemma-3-4b",
    "gpt-3.5-turbo",
    "google/gemini-2.5-flash-preview-05-20",
    # "anthropic/claude-3-5-haiku-20241022",  # example
    # "mistral-medium-latest",
)

# Subset of ``JUDGES`` that call OpenAI-compatible ``/v1/chat/completions`` (not the local server).
OPENAI_JUDGE_IDS: frozenset[str] = frozenset({"gpt-3.5-turbo"})
OPENAI_API_KEY: str | None = os.environ.get("OPENAI_API_KEY")
OPENAI_CHAT_COMPLETIONS_URL: str = os.environ.get(
    "OPENAI_CHAT_COMPLETIONS_URL", "https://api.openai.com/v1/chat/completions"
)

# Gemini generateContent (Google AI); judge keys look like ``google/gemini-…``.
GEMINI_JUDGE_IDS: frozenset[str] = frozenset({"google/gemini-2.5-flash-preview-05-20"})
GOOGLE_API_KEY: str | None = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
GEMINI_API_BASE: str = os.environ.get(
    "GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta"
)
# Map judge id (checkpoint / JUDGES string) → REST ``models/{id}`` name. Preview API ids are
# retired or renamed often; keep a stable judge key here and point it at a current model id.
GEMINI_JUDGE_TO_API_MODEL: dict[str, str] = {
    "google/gemini-2.5-flash-preview-05-20": "gemini-2.5-flash",
}
# Space Gemini calls so average rate stays under this (rolling minute, enforced as gap between
# consecutive calls). Free tier is often ~20 RPM; default 18 leaves headroom. Set env
# ``GEMINI_MAX_REQUESTS_PER_MINUTE`` (float); ``0`` disables throttling.
GEMINI_MAX_REQUESTS_PER_MINUTE: float = _env_float("GEMINI_MAX_REQUESTS_PER_MINUTE", 18.0)

HUMAN_JUDGES: tuple[str, ...] = ()
LLM_JUDGES: tuple[str, ...] = tuple(j for j in JUDGES if j not in HUMAN_JUDGES)

# Synthetic ``model_id`` for the gold summary from eval JSONL ``reference`` (one row per doc in long_df).
REFERENCE_SUMMARY_MODEL_ID = "GPT4o-mini"

# Local chat API (same default as ``geval_local_judge``).
LOCAL_LLM_CHAT_URL = "http://localhost:1234/api/v1/chat"
LOCAL_LLM_TIMEOUT_S = 300.0

# First N documents (by first-seen ``doc_id`` order). None = use the full loaded corpus.
MAX_DOCUMENTS: int | None = 51 #603

# Random model pairs sampled per document (capped by available combinations); see :func:`pairwise_eval.pairs.build_pairs_table`.
N_PAIRS_PER_DOCUMENT: int = 8

# --- Eval JSONL directory ---
# ``None`` → auto-detect ``REPO_ROOT / "Data" / "eval"`` (with cwd fallbacks in
# :func:`pairwise_eval.data.resolve_eval_data_dir`). If set: absolute paths are used as-is;
# relative paths are resolved under ``REPO_ROOT``.
EVAL_DATA_DIR: Path | None = None

# Winners / alternate corpus (uncomment). Keeps ``Data/eval`` as the default when this is ``None``.
# EVAL_DATA_DIR = Path("Data/eval/winners")

# --- Export root ---
# Artifacts go under ``<cwd>/.deepeval/<GEVAL_EXPORT_DIRNAME>/`` (see
# :func:`pairwise_eval.io_export.resolve_geval_export_dir`). Use a different name so a second
# run does not overwrite ``geval_exports``.
GEVAL_EXPORT_DIRNAME: str = "geval_exports"
# GEVAL_EXPORT_DIRNAME = "geval_winners_exports"

# G-Eval: append-only JSONL per (judge × dimension); resume skips keys already on disk. ``None`` = off.
# For a second dataset, point this elsewhere so judgment checkpoints do not mix with the main run.
GEVAL_CHECKPOINT_DIR: Path | None = REPO_ROOT / ".deepeval" / "geval_judgment_checkpoints"
# GEVAL_CHECKPOINT_DIR = REPO_ROOT / ".deepeval" / "geval_judgment_checkpoints_winners"

DEFAULT_PAIR_SEED = 42
DEFAULT_GEVAL_BASE_SEED = 42
MOCK_TIE_PROB = 0.1
