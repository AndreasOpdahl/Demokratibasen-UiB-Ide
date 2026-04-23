"""Run the pairwise G-Eval pipeline and write exports under ``.deepeval/<GEVAL_EXPORT_DIRNAME>/``.

If ``Data/eval`` (or :data:`~pairwise_eval.config.EVAL_DATA_DIR`) contains only subfolders with
``*.jsonl`` (one folder per base model / checkpoint family), the CLI runs the same pipeline once per
subfolder and writes ``<repo>/.deepeval/<GEVAL_EXPORT_DIRNAME>/<folder_name>/`` plus matching judgment
checkpoints under ``<repo>/.deepeval/geval_judgment_checkpoints/<folder_name>/`` when checkpoints are
enabled in config.
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from pairwise_eval import (
    build_geval_tables,
    build_toy_long_df,
    export_full_run,
    load_eval_jsonl_long_df,
    mock_evaluate_pair,
    resolve_eval_data_dir,
    resolve_geval_export_dir,
)
from pairwise_eval.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_JUDGE_IDS,
    EXTEND_PAIRS_TABLE_JSON,
    GEMINI_JUDGE_IDS,
    GOOGLE_API_KEY,
    GEVAL_CHECKPOINT_DIR,
    GEVAL_EXPORT_DIRNAME,
    JUDGES,
    MAX_DOCUMENTS,
    MISTRAL_API_KEY,
    MISTRAL_JUDGE_IDS,
    N_PAIRS_PER_DOCUMENT,
    OPENAI_API_KEY,
    OPENAI_JUDGE_IDS,
    REFERENCE_SUMMARY_MODEL_ID,
    REPO_ROOT,
)
from pairwise_eval.data import discover_eval_model_subdirs, long_df_head_documents
from pairwise_eval.llm_judge import make_local_llm_evaluate_fn
from pairwise_eval.pairs import build_pairs_table, load_pairs_table_json

# Quick toggles
USE_TOY_DATA = False
USE_LLM_JUDGE = True


def _safe_artifact_subdir(name: str) -> str:
    """Single path segment for nested ``.deepeval``
     outputs (no separators)."""
    return name.replace("\\", "_").replace("/", "_")


def _per_model_export_dir(model_key: str) -> Path:
    """Same layout as :func:`resolve_geval_export_dir`, but under ``REPO_ROOT`` and one subfolder per model."""
    leaf = GEVAL_EXPORT_DIRNAME.strip()
    safe = _safe_artifact_subdir(model_key)
    return REPO_ROOT / ".deepeval" / leaf / safe


def _per_model_checkpoint_dir(model_key: str) -> Path | None:
    if GEVAL_CHECKPOINT_DIR is None:
        return None
    return GEVAL_CHECKPOINT_DIR / _safe_artifact_subdir(model_key)


def _resolve_extend_pairs_table_json(path: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.is_file():
        raise FileNotFoundError(f"EXTEND_PAIRS_TABLE_JSON not found or not a file: {p}")
    return p


def _extend_pairs_json_for_export_dir(export_dir: Path) -> Path | None:
    """Explicit config path wins; else reuse ``export_dir/json/pairs_table.json`` if it exists."""
    if EXTEND_PAIRS_TABLE_JSON is not None:
        return _resolve_extend_pairs_table_json(EXTEND_PAIRS_TABLE_JSON)
    auto = export_dir / "json" / "pairs_table.json"
    if auto.is_file():
        return auto
    return None


def _print_judge_warnings() -> None:
    print("Judges:", list(JUDGES))
    if any(j in OPENAI_JUDGE_IDS for j in JUDGES) and not OPENAI_API_KEY:
        print(
            "Warning: OpenAI judge(s) in JUDGES but OPENAI_API_KEY is unset — "
            "those judgments will be [api_error] ties.",
            file=sys.stderr,
        )
    if any(j in GEMINI_JUDGE_IDS for j in JUDGES) and not GOOGLE_API_KEY:
        print(
            "Warning: Gemini judge(s) in JUDGES but GOOGLE_API_KEY (or GEMINI_API_KEY) is unset — "
            "those judgments will be [api_error] ties.",
            file=sys.stderr,
        )
    if any(j in ANTHROPIC_JUDGE_IDS for j in JUDGES) and not ANTHROPIC_API_KEY:
        print(
            "Warning: Anthropic judge(s) in JUDGES but ANTHROPIC_API_KEY is unset — "
            "those judgments will be [api_error] ties.",
            file=sys.stderr,
        )
    if any(j in MISTRAL_JUDGE_IDS for j in JUDGES) and not MISTRAL_API_KEY:
        print(
            "Warning: Mistral judge(s) in JUDGES but MISTRAL_API_KEY is unset — "
            "those judgments will be [api_error] ties.",
            file=sys.stderr,
        )


def run_pipeline_for_eval_dir(
    eval_dir: Path,
    *,
    export_dir: Path,
    checkpoint_dir: Path | None,
    extend_pairs_table_json: Path | None = None,
) -> Path:
    """One full run: same steps as before, with explicit eval / export / checkpoint paths."""
    long_df = load_eval_jsonl_long_df(eval_dir)
    print("Data:", eval_dir.resolve(), f"({len(long_df)} rows)")

    long_df = long_df_head_documents(long_df, MAX_DOCUMENTS)
    if MAX_DOCUMENTS is not None:
        print(f"Subset: first {MAX_DOCUMENTS} document(s)", f"({len(long_df)} rows)")

    models = sorted(long_df["model_id"].unique())
    prior_pairs_df = None
    if extend_pairs_table_json is not None:
        prior_pairs_df = load_pairs_table_json(extend_pairs_table_json)
        print(
            "Extending pairs from prior table:",
            extend_pairs_table_json.resolve(),
            f"({len(prior_pairs_df)} rows) → target {N_PAIRS_PER_DOCUMENT} per doc",
            flush=True,
        )
    pairs = build_pairs_table(
        long_df, n_pairs=N_PAIRS_PER_DOCUMENT, prior_pairs_df=prior_pairs_df
    )

    if checkpoint_dir is not None:
        print("Checkpoint dir:", checkpoint_dir.resolve())

    evaluate_fn = (
        make_local_llm_evaluate_fn(verbose=True)
        if USE_LLM_JUDGE
        else mock_evaluate_pair
    )
    geval = build_geval_tables(
        pairs, long_df, evaluate_fn=evaluate_fn, checkpoint_dir=checkpoint_dir
    )

    out = export_full_run(
        geval,
        pairs,
        long_df,
        models,
        ref_model=REFERENCE_SUMMARY_MODEL_ID,
        export_dir=export_dir,
    )
    print("Exports:", out.resolve())
    return out


def main() -> None:
    """CLI: load data → optional doc subset → pairs → G-Eval (mock or LLM) → export artifacts."""
    _print_judge_warnings()

    if USE_TOY_DATA:
        long_df = build_toy_long_df()
        print("Data: toy", f"({len(long_df)} rows)")
        long_df = long_df_head_documents(long_df, MAX_DOCUMENTS)
        if MAX_DOCUMENTS is not None:
            print(f"Subset: first {MAX_DOCUMENTS} document(s)", f"({len(long_df)} rows)")
        models = sorted(long_df["model_id"].unique())
        export_dir = resolve_geval_export_dir()
        ext_json = _extend_pairs_json_for_export_dir(export_dir)
        prior = load_pairs_table_json(ext_json) if ext_json is not None else None
        if ext_json is not None:
            print(
                "Extending pairs from prior table:",
                ext_json.resolve(),
                f"({len(prior)} rows) → target {N_PAIRS_PER_DOCUMENT} per doc",
                flush=True,
            )
        pairs = build_pairs_table(
            long_df, n_pairs=N_PAIRS_PER_DOCUMENT, prior_pairs_df=prior
        )
        ck = GEVAL_CHECKPOINT_DIR
        if ck is not None:
            print("Checkpoint dir:", ck.resolve())
        evaluate_fn = (
            make_local_llm_evaluate_fn(verbose=True)
            if USE_LLM_JUDGE
            else mock_evaluate_pair
        )
        geval = build_geval_tables(
            pairs, long_df, evaluate_fn=evaluate_fn, checkpoint_dir=ck
        )
        out = export_full_run(
            geval,
            pairs,
            long_df,
            models,
            ref_model=REFERENCE_SUMMARY_MODEL_ID,
            export_dir=export_dir,
        )
        print("Exports:", out.resolve())
        return

    eval_root = resolve_eval_data_dir()
    direct_jsonl = sorted(eval_root.glob("*.jsonl"))
    model_dirs = discover_eval_model_subdirs(eval_root)

    if direct_jsonl:
        export_here = resolve_geval_export_dir()
        run_pipeline_for_eval_dir(
            eval_root,
            export_dir=export_here,
            checkpoint_dir=GEVAL_CHECKPOINT_DIR,
            extend_pairs_table_json=_extend_pairs_json_for_export_dir(export_here),
        )
    elif model_dirs:
        print(
            f"Found {len(model_dirs)} model folder(s) under {eval_root.resolve()} — "
            "running the pipeline separately for each.",
            flush=True,
        )
        for md in model_dirs:
            key = md.name
            print("\n===", key, "===", flush=True)
            export_here = _per_model_export_dir(key)
            run_pipeline_for_eval_dir(
                md,
                export_dir=export_here,
                checkpoint_dir=_per_model_checkpoint_dir(key),
                extend_pairs_table_json=_extend_pairs_json_for_export_dir(export_here),
            )
    else:
        hint = (
            f"No *.jsonl under {eval_root.resolve()} and no model subfolders with JSONL. "
            f"Expected either flat `*.jsonl` here or subfolders like `{REPO_ROOT / 'Data' / 'eval' / '<model>'}/*.jsonl`."
        )
        raise FileNotFoundError(hint)


if __name__ == "__main__":
    main()
