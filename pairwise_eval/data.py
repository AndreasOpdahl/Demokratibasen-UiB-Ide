"""Toy and real (JSONL) summarization data in long format."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pairwise_eval.config import EVAL_DATA_DIR, REFERENCE_SUMMARY_MODEL_ID, REPO_ROOT


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


def resolve_eval_data_dir() -> Path:
    """Locate the eval JSONL directory.

    Uses :data:`pairwise_eval.config.EVAL_DATA_DIR` when set; otherwise ``REPO_ROOT / "Data" / "eval"``
    or cwd fallbacks. Output: existing directory path.
    """
    if EVAL_DATA_DIR is not None:
        p = Path(EVAL_DATA_DIR)
        if not p.is_absolute():
            p = REPO_ROOT / p
        p = p.resolve()
        if not p.is_dir():
            raise FileNotFoundError(
                f"EVAL_DATA_DIR is set to {p} but that path is not an existing directory."
            )
        return p
    repo_eval = REPO_ROOT / "Data" / "eval"
    if repo_eval.is_dir():
        return repo_eval
    cwd = Path.cwd()
    for candidate in (cwd / "Data" / "eval", cwd.parent / "Data" / "eval"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find Data/eval. Expected at <repo>/Data/eval or ./Data/eval."
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
