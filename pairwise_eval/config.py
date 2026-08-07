"""Defaults for dimensions, judges, and RNG seeds."""

from __future__ import annotations

import os
from pathlib import Path

# Repository root (parent of the `pairwise_eval` package)
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


# Each name must have a template ``DATA_ROOT/prompts/geval/{name}.txt`` (or set overrides in ``geval_local_judge._GEVAL_PROMPT_FILE_OVERRIDES``).
EVAL_DIMENSIONS: tuple[str, ...] = (
    "relevance",
    "consistency",
    "newsworthiness",
    "hygiene"
)

# Each string is both the G-Eval judge key and (for cloud APIs) the model id passed to the backend.
# The CLI pipeline's :func:`pairwise_eval.llm_judge.make_local_llm_evaluate_fn` routes **only**
# ids in ``OPENAI_JUDGE_IDS``, ``MISTRAL_JUDGE_IDS``, ``GEMINI_JUDGE_IDS``, or ``ANTHROPIC_JUDGE_IDS``;
# any other id raises ``ValueError`` (no silent LM Studio fallback). For ad‑hoc local judging,
# call ``geval_local_judge`` / ``LOCAL_LLM_CHAT_URL`` from your own script.
# OpenAI: ``OPENAI_CHAT_COMPLETIONS_URL`` + ``OPENAI_API_KEY``. Gemini: ``GEMINI_API_BASE`` +
# ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY``; judge id ``google/<…>`` with optional
# ``GEMINI_JUDGE_TO_API_MODEL`` remap. Anthropic: ``ANTHROPIC_MESSAGES_URL`` + ``ANTHROPIC_API_KEY``;
# judge id ``anthropic/<…>`` with optional ``ANTHROPIC_JUDGE_TO_API_MODEL``. Mistral: same JSON
# shape as OpenAI via ``MISTRAL_CHAT_COMPLETIONS_URL`` + ``MISTRAL_API_KEY``.
JUDGES: tuple[str, ...] = (
    #"google/gemma-3-4b",
    "gpt-5-mini",
    "google/gemini-2.5-flash-preview-05-20",
    "anthropic/claude-3-5-haiku-20241022",
    "mistral-medium-latest",
)

# Subset of ``JUDGES`` that call OpenAI's Chat Completions (not Mistral).
OPENAI_JUDGE_IDS: frozenset[str] = frozenset({"gpt-5-mini"})
OPENAI_API_KEY: str | None = os.environ.get("OPENAI_API_KEY")
OPENAI_CHAT_COMPLETIONS_URL: str = os.environ.get(
    "OPENAI_CHAT_COMPLETIONS_URL", "https://api.openai.com/v1/chat/completions"
)

MISTRAL_JUDGE_IDS: frozenset[str] = frozenset({"mistral-medium-latest"})
MISTRAL_API_KEY: str | None = os.environ.get("MISTRAL_API_KEY")
MISTRAL_CHAT_COMPLETIONS_URL: str = os.environ.get(
    "MISTRAL_CHAT_COMPLETIONS_URL", "https://api.mistral.ai/v1/chat/completions"
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
# consecutive calls: ``60 / GEMINI_MAX_REQUESTS_PER_MINUTE`` seconds). Lower RPM = longer spacing
# and fewer 429s; free tier is often ~20 RPM — default 10 (~6 s between calls) is conservative.
# Override with env ``GEMINI_MAX_REQUESTS_PER_MINUTE`` (float); ``0`` disables throttling.
GEMINI_MAX_REQUESTS_PER_MINUTE: float = _env_float("GEMINI_MAX_REQUESTS_PER_MINUTE", 10.0)

# Anthropic Messages API; judge keys look like ``anthropic/<logical-name>``.
ANTHROPIC_JUDGE_IDS: frozenset[str] = frozenset({"anthropic/claude-3-5-haiku-20241022"})
ANTHROPIC_API_KEY: str | None = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MESSAGES_URL: str = os.environ.get(
    "ANTHROPIC_MESSAGES_URL", "https://api.anthropic.com/v1/messages"
)
ANTHROPIC_VERSION: str = os.environ.get("ANTHROPIC_VERSION", "2023-06-01")
# Legacy judge id → Messages API ``model`` (Anthropic retires dated ids; keep a stable checkpoint key).
# See https://platform.claude.com/docs/en/about-claude/models/overview — use ``claude-haiku-4-5`` for the alias.
ANTHROPIC_JUDGE_TO_API_MODEL: dict[str, str] = {
    "anthropic/claude-3-5-haiku-20241022": "claude-haiku-4-5-20251001",
}

HUMAN_JUDGES: tuple[str, ...] = ()
LLM_JUDGES: tuple[str, ...] = tuple(j for j in JUDGES if j not in HUMAN_JUDGES)

# Synthetic ``model_id`` for the gold summary from eval JSONL ``reference`` (one row per doc in long_df).
REFERENCE_SUMMARY_MODEL_ID = "GPT4o-mini"

# Local chat API (same default as ``geval_local_judge``).
LOCAL_LLM_CHAT_URL = "http://localhost:1234/api/v1/chat"
LOCAL_LLM_TIMEOUT_S = 300.0

# First N documents (by first-seen ``doc_id`` order). None = use the full loaded corpus.
# For per-model checkpoint selection, keep None so Bradley–Terry / win rates use every example
# in that model folder’s JSONL files.
MAX_DOCUMENTS: int | None = 2500

# Random model pairs sampled per document (capped by available combinations); see :func:`pairwise_eval.pairs.build_pairs_table`.
N_PAIRS_PER_DOCUMENT: int = 8

# If ``True``, ``build_pairs_table`` uses a greedy *balanced* sampler that keeps a global
# counter of how often each model and each unordered pair has already been picked, and at each
# step prefers pairs whose (model, pair) counts are lowest (random tie-break). This produces
# near-uniform per-model and per-pair coverage instead of the vanilla i.i.d. uniform draw, at
# the cost of being deterministic given the seed/history. Set to ``False`` to keep the original
# uniform-random behavior (each doc samples ``N_PAIRS_PER_DOCUMENT`` pairs independently). The
# flag is forwarded by the CLI to :func:`pairwise_eval.pairs.build_pairs_table`; library callers
# pass it explicitly via the ``balanced=`` kwarg.
BALANCED_PAIR_SAMPLING: bool = True

# When set to a prior run's ``json/pairs_table.json``, the pipeline **reuses** those rows (same
# ``left`` / ``right`` / summary text) for each ``doc_id`` still present, then samples only
# **additional** unordered pairs until ``N_PAIRS_PER_DOCUMENT`` is reached — so raising
# ``N_PAIRS_PER_DOCUMENT`` does not throw away expensive judgments already in checkpoints.
# ``None`` = do not force a path; the CLI still auto-loads ``<export_dir>/json/pairs_table.json``
# when that file exists (per export leaf), which is correct for multi-model eval layouts.
# If you set this to one model's ``pairs_table.json`` while running **several** model subfolders,
# every subfolder would reuse that same file — usually wrong; leave ``None`` for auto per leaf.
# Relative paths resolve under ``REPO_ROOT``.
EXTEND_PAIRS_TABLE_JSON: Path | None = None
# EXTEND_PAIRS_TABLE_JSON = REPO_ROOT / ".deepeval" / "geval_exports" / "gemma-2b" / "json" / "pairs_table.json"

# --- Eval JSONL directory ---
# ``None`` → auto-detect ``DATA_ROOT / "eval"`` (with cwd fallbacks in
# :func:`pairwise_eval.data.resolve_eval_data_dir`). If set: absolute paths are used as-is;
# relative paths are resolved under ``DATA_ROOT``.
EVAL_DATA_DIR: Path | None = None

# Winners / alternate corpus (uncomment). Keeps ``DATA_ROOT/eval`` as the default when this is ``None``.
# EVAL_DATA_DIR = Path("winners")  # flat dir; each *.jsonl stem is <model>__checkpoint-...

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

# Multi-model CLI (``DATA_ROOT/eval/<model>/``): skip a leaf when its checkpoint folder already shows
# at least as many distinct ``doc_id`` values (union across ``*.jsonl``) as this run's subset
# after ``MAX_DOCUMENTS``. Avoids re-running and overwriting exports when ``MAX_DOCUMENTS`` was
# lowered while checkpoints still reflect more documents. **Heuristic only** — it does not
# check that every judge×dimension finished. Set ``False`` to always run every model.
GEVAL_SKIP_MODEL_IF_CHECKPOINT_DOC_COUNT_GTE_RUN: bool = False

DEFAULT_PAIR_SEED = 42
DEFAULT_GEVAL_BASE_SEED = 42
MOCK_TIE_PROB = 0.1



