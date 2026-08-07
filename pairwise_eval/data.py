"""Toy and real (JSONL) summarization data in long format."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from pairwise_eval.config import (
    DATA_ROOT,
    EVAL_DATA_DIR,
    GEVAL_EXPORT_DIRNAME,
    REFERENCE_SUMMARY_MODEL_ID,
    REPO_ROOT,
)


def append_gold_summary_as_model_rows(checkpoint_long_df: pd.DataFrame) -> pd.DataFrame:
    """Append gold-reference summaries as a normal eval ``model_id`` (see ``REFERENCE_SUMMARY_MODEL_ID``).

    Input: long table with only checkpoint rows (must include ``doc_id``, ``source_text``,
    ``reference_summary``). Output: same rows plus one extra row per doc with ``summary_text`` = gold.
    """
    need = {"doc_id", "source_text", "reference_summary"}
    missing = need - set(checkpoint_long_df.columns)
    if missing:
        raise ValueError(f"append_gold_summary_as_model_rows: missing columns {missing}")
    gold = (
        checkpoint_long_df.groupby("doc_id", sort=False)
        .first()[["source_text", "reference_summary"]]
        .reset_index()
        .assign(
            model_id=REFERENCE_SUMMARY_MODEL_ID,
            summary_text=lambda t: t["reference_summary"].astype(str),
        )
    )
    gold = gold[["doc_id", "source_text", "model_id", "summary_text", "reference_summary"]]
    return pd.concat([checkpoint_long_df, gold], ignore_index=True)


def long_df_head_documents(long_df: pd.DataFrame, max_documents: int | None) -> pd.DataFrame:
    """Subset ``long_df`` to the first N documents (by first-seen ``doc_id`` order).

    Input: full long_df; ``max_documents`` = N or ``None`` (keep all). Output: filtered copy.
    """
    if max_documents is None:
        return long_df.copy()
    if max_documents < 1:
        raise ValueError("max_documents must be >= 1 when not None")
    ids = long_df["doc_id"].unique()
    keep = ids[:max_documents]
    return long_df.loc[long_df["doc_id"].isin(keep)].copy()


def build_toy_long_df() -> pd.DataFrame:
    """Build a tiny long_df for tests: 5 docs × 5 fake checkpoints + gold row per doc.

    Input: none. Output: DataFrame with columns ``doc_id``, ``source_text``, ``model_id``,
    ``summary_text``, ``reference_summary``.
    """
    documents = [
        "The city council approved a plan to add 20 electric buses and reduce fares for students.",
        "A new study shows that regular exercise reduces the risk of heart disease by 30%.",
        "The company announced a new AI-powered product aimed at improving customer support.",
        "Heavy rainfall caused flooding in several مناطق, displacing hundreds of residents.",
        "Scientists discovered a new species of marine الحياة in the Pacific Ocean.",
    ]
    reference_summaries = [
        "The council approved electric buses and student fare reductions.",
        "Exercise significantly lowers heart disease risk.",
        "A company launched an AI tool for customer support.",
        "Flooding displaced residents after heavy rainfall.",
        "A new marine species was discovered in the Pacific.",
    ]
    models = ["ft_model_A", "ft_model_B", "ft_model_C", "ft_model_D", "ft_model_E"]
    rows: list[dict] = []
    for doc_id, (doc, ref) in enumerate(zip(documents, reference_summaries), start=1):
        for model in models:
            rows.append(
                {
                    "doc_id": f"doc_{doc_id}",
                    "source_text": doc,
                    "model_id": model,
                    "summary_text": f"{model}: {doc[:80]}...",
                    "reference_summary": ref,
                }
            )
    checkpoints = pd.DataFrame(rows)
    return append_gold_summary_as_model_rows(checkpoints)


def resolve_eval_dir_for_judgment_leaf(
    leaf_dir: Path,
    *,
    fallback_single_eval_subdir: bool = False,
) -> Path:
    """Map a per-model judgment directory to the eval JSONL folder used for that run.

    Resolution order:

    1. ``DATA_ROOT/eval/<leaf_name>/`` if it exists and contains at least one ``*.jsonl``.
    2. :func:`resolve_eval_data_dir` if that directory itself contains ``*.jsonl`` (flat layout).
    3. Immediate subdirectories of the eval root that contain ``*.jsonl``: use the subfolder
       whose name equals ``leaf_dir.name`` (case-sensitive, then case-insensitive).
    4. If still unmatched and ``fallback_single_eval_subdir`` is True and there is exactly
       one such subdir under the eval root, return that directory (with :func:`warnings.warn`).
       Use only when you know that single corpus matches the judgments (e.g. one machine,
       judgments moved under a different leaf name). Otherwise set ``EVAL_JSONL_LEAF`` in the
       notebook or add ``DATA_ROOT/eval/<leaf_name>/*.jsonl``.

    This avoids returning a bare ``DATA_ROOT/eval`` directory that only holds per-model
    subfolders (no JSONL at the top level), which would make :func:`load_eval_jsonl_long_df` fail.
    """
    name = leaf_dir.name
    repo_eval = DATA_ROOT / "eval"
    candidate = repo_eval / name
    if candidate.is_dir() and any(candidate.glob("*.jsonl")):
        return candidate.resolve()

    eval_root = resolve_eval_data_dir()
    if any(eval_root.glob("*.jsonl")):
        return eval_root.resolve()

    subdirs = discover_eval_model_subdirs(eval_root)
    if not subdirs:
        raise FileNotFoundError(
            f"No *.jsonl under {eval_root} or its immediate subfolders. "
            f"Expected either flat {eval_root}/*.jsonl or nested {eval_root}/{name}/*.jsonl "
            f"(and aligned checkpoint judgments for that eval)."
        )
    for p in subdirs:
        if p.name == name:
            return p.resolve()
    lowered = name.lower()
    for p in subdirs:
        if p.name.lower() == lowered:
            return p.resolve()
    choices = ", ".join(p.name for p in subdirs)
    if fallback_single_eval_subdir and len(subdirs) == 1:
        only = subdirs[0].resolve()
        warnings.warn(
            f"No eval folder named {name!r}; using the only available eval subdir {only.name!r} "
            f"under {eval_root}. Confirm this corpus matches these judgments.",
            UserWarning,
            stacklevel=2,
        )
        return only
    raise FileNotFoundError(
        f"No eval data for judgment leaf {name!r}: missing or empty {candidate}, "
        f"and no subfolder under {eval_root} matches that name. "
        f"Available eval subdirs (with *.jsonl): {choices}. "
        f"Set CHECKPOINT_LEAF_NAME to match one of these, add DATA_ROOT/eval/{name}/*.jsonl, "
        f"or call resolve_eval_dir_for_judgment_leaf(..., fallback_single_eval_subdir=True) "
        f"when exactly one eval subdir exists and is the correct corpus."
    )


def resolve_eval_dir_for_judgment_leaf_notebook(leaf_dir: Path) -> Path:
    """Like ``resolve_eval_dir_for_judgment_leaf(..., fallback_single_eval_subdir=True)``.

    Use this from notebooks so they still work if the active interpreter loads an **older**
    ``pairwise_eval`` where :func:`resolve_eval_dir_for_judgment_leaf` does not yet accept
    ``fallback_single_eval_subdir`` (stale install, wrong ``sys.path`` order, etc.).
    """
    import inspect

    sig = inspect.signature(resolve_eval_dir_for_judgment_leaf)
    if "fallback_single_eval_subdir" in sig.parameters:
        return resolve_eval_dir_for_judgment_leaf(leaf_dir, fallback_single_eval_subdir=True)

    try:
        return resolve_eval_dir_for_judgment_leaf(leaf_dir)
    except FileNotFoundError:
        name = leaf_dir.name
        eval_root = resolve_eval_data_dir()
        subdirs = discover_eval_model_subdirs(eval_root)
        if len(subdirs) == 1:
            only = subdirs[0].resolve()
            warnings.warn(
                f"No eval folder named {name!r}; using the only available eval subdir {only.name!r} "
                f"under {eval_root}. Confirm this corpus matches these judgments. "
                f"(Install the repo's ``pairwise_eval`` so ``fallback_single_eval_subdir`` is available.)",
                UserWarning,
                stacklevel=2,
            )
            return only
        raise


def resolve_export_dir_for_judgment_leaf(leaf_dir: Path) -> Path:
    """Export root for a judgment leaf: per-model under ``REPO_ROOT/.deepeval/…`` when eval subdir exists."""
    from pairwise_eval.io_export import resolve_geval_export_dir

    name = leaf_dir.name
    candidate = DATA_ROOT / "eval" / name
    if candidate.is_dir() and any(candidate.glob("*.jsonl")):
        leaf = GEVAL_EXPORT_DIRNAME.strip()
        return (REPO_ROOT / ".deepeval" / leaf / name).resolve()
    return resolve_geval_export_dir()


def discover_eval_model_subdirs(eval_root: Path) -> list[Path]:
    """Return immediate child directories of ``eval_root`` that contain at least one ``*.jsonl``.

    Used when checkpoints live under ``DATA_ROOT/eval/<model_name>/*.jsonl`` instead of flat
    ``DATA_ROOT/eval/*.jsonl``. Output: sorted paths (each is a directory to pass to
    :func:`load_eval_jsonl_long_df`).
    """
    if not eval_root.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(eval_root.iterdir()):
        if child.is_dir() and any(child.glob("*.jsonl")):
            out.append(child)
    return out


def resolve_eval_data_dir() -> Path:
    """Locate the eval JSONL directory.

    Uses :data:`pairwise_eval.config.EVAL_DATA_DIR` when set; otherwise ``DATA_ROOT / "eval"``
    or cwd fallbacks. Output: existing directory path.
    """
    if EVAL_DATA_DIR is not None:
        p = Path(EVAL_DATA_DIR)
        if not p.is_absolute():
            p = DATA_ROOT / p
        p = p.resolve()
        if not p.is_dir():
            raise FileNotFoundError(
                f"EVAL_DATA_DIR is set to {p} but that path is not an existing directory."
            )
        return p
    data_root_eval = DATA_ROOT / "eval"
    if data_root_eval.is_dir():
        return data_root_eval
    cwd = Path.cwd()
    for candidate in (cwd / "Data" / "eval", cwd.parent / "Data" / "eval"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find eval data. Expected at {DATA_ROOT / 'eval'} "
        "(set ONEDRIVE or CHECKPOINT_SELECTION_DATA_DIR if that's not your OneDrive layout), "
        "or a legacy Data/eval under cwd."
    )


def stack_eval_jsonl_checkpoints_long_df(eval_dir: Path) -> pd.DataFrame:
    """Load every ``*.jsonl`` in ``eval_dir`` into one long table (predictions only, no gold row).

    Input: directory of aligned JSONL files. Output: DataFrame one row per (doc, file stem).
    """
    files = sorted(eval_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files under {eval_dir}")
    for path in files:
        if path.stem == REFERENCE_SUMMARY_MODEL_ID:
            raise ValueError(
                f"{path.name}: file stem must not equal REFERENCE_SUMMARY_MODEL_ID "
                f"({REFERENCE_SUMMARY_MODEL_ID!r}); that id is reserved for the gold-summary "
                "eval model (JSONL ``reference`` column)."
            )

    rows: list[dict] = []
    baseline_input: pd.Series | None = None

    for path in files:
        model_id = path.stem
        part = pd.read_json(path, lines=True)
        required = {"input_text", "prompt", "reference", "prediction"}
        missing = required - set(part.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")

        part = part.copy()
        part["prediction"] = part["prediction"].fillna("").astype(str)

        if baseline_input is None:
            baseline_input = part["input_text"].astype(str).reset_index(drop=True)
        else:
            if not part["input_text"].astype(str).reset_index(drop=True).equals(baseline_input):
                raise ValueError(f"{path.name}: `input_text` rows do not match the first file.")

        for i, r in part.iterrows():
            rows.append(
                {
                    "doc_id": f"doc_{i + 1}",
                    "source_text": r["input_text"],
                    "model_id": model_id,
                    "summary_text": r["prediction"],
                    "reference_summary": r["reference"],
                }
            )

    return pd.DataFrame(rows)


def load_eval_jsonl_long_df(eval_dir: Path) -> pd.DataFrame:
    """Full eval long_df: stack JSONL checkpoints, then append gold-summary rows.

    Input: ``eval_dir``. Output: long_df ready for ``build_pairs_table``.
    """
    checkpoints = stack_eval_jsonl_checkpoints_long_df(eval_dir)
    return append_gold_summary_as_model_rows(checkpoints)
