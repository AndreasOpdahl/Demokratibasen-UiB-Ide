"""Run the pairwise G-Eval pipeline and write exports under ``.deepeval/<GEVAL_EXPORT_DIRNAME>/``."""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from pairwise_eval import (
    build_geval_tables,
    build_pairs_table,
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
    GEMINI_JUDGE_IDS,
    GOOGLE_API_KEY,
    GEVAL_CHECKPOINT_DIR,
    JUDGES,
    MAX_DOCUMENTS,
    MISTRAL_API_KEY,
    MISTRAL_JUDGE_IDS,
    N_PAIRS_PER_DOCUMENT,
    OPENAI_API_KEY,
    OPENAI_JUDGE_IDS,
    REFERENCE_SUMMARY_MODEL_ID,
)
from pairwise_eval.data import long_df_head_documents
from pairwise_eval.llm_judge import make_local_llm_evaluate_fn

# Quick toggles
USE_TOY_DATA = False
USE_LLM_JUDGE = True


def main() -> None:
    """CLI: load data → optional doc subset → pairs → G-Eval (mock or LLM) → export artifacts."""
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

    if USE_TOY_DATA:
        long_df = build_toy_long_df()
        print("Data: toy", f"({len(long_df)} rows)")
    else:
        eval_dir = resolve_eval_data_dir()
        long_df = load_eval_jsonl_long_df(eval_dir)
        print("Data:", eval_dir.resolve(), f"({len(long_df)} rows)")

    long_df = long_df_head_documents(long_df, MAX_DOCUMENTS)
    if MAX_DOCUMENTS is not None:
        print(f"Subset: first {MAX_DOCUMENTS} document(s)", f"({len(long_df)} rows)")

    models = sorted(long_df["model_id"].unique())
    pairs = build_pairs_table(long_df, n_pairs=N_PAIRS_PER_DOCUMENT)

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

    export_dir = resolve_geval_export_dir()
    out = export_full_run(
        geval,
        pairs,
        long_df,
        models,
        ref_model=REFERENCE_SUMMARY_MODEL_ID,
        export_dir=export_dir,
    )
    print("Exports:", out.resolve())


if __name__ == "__main__":
    main()
