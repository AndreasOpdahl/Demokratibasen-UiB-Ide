"""
Pairwise G-Eval workflow: long_df (all eval models) → pairs → judgments → win rates → exports.

Example::

    from pairwise_eval import (
        REFERENCE_SUMMARY_MODEL_ID,
        build_geval_tables,
        build_pairs_table,
        build_toy_long_df,
        export_full_run,
        mock_evaluate_pair,
    )

    long_df = build_toy_long_df()  # checkpoints + gold model REFERENCE_SUMMARY_MODEL_ID
    models = sorted(long_df["model_id"].unique())
    pairs = build_pairs_table(long_df)  # ``N_PAIRS_PER_DOCUMENT`` from :mod:`pairwise_eval.config`
    geval = build_geval_tables(pairs, long_df)
    export_full_run(geval, pairs, long_df, models, ref_model=REFERENCE_SUMMARY_MODEL_ID)
"""

from pairwise_eval.bradley_terry import (
    bradley_terry_long_table,
    bradley_terry_theta_wide,
    fit_bradley_terry,
    markdown_bradley_terry_theta,
    win_matrix_from_geval,
)
from pairwise_eval.config import (
    EVAL_DATA_DIR,
    EVAL_DIMENSIONS,
    EXTEND_PAIRS_TABLE_JSON,
    GEVAL_CHECKPOINT_DIR,
    GEVAL_EXPORT_DIRNAME,
    HUMAN_JUDGES,
    JUDGES,
    LLM_JUDGES,
    LOCAL_LLM_CHAT_URL,
    LOCAL_LLM_TIMEOUT_S,
    MAX_DOCUMENTS,
    MOCK_TIE_PROB,
    N_PAIRS_PER_DOCUMENT,
    REFERENCE_SUMMARY_MODEL_ID,
    REPO_ROOT,
)
from pairwise_eval.geval_checkpoint import (
    append_judgment_line,
    checkpoint_file_path,
    judgment_stable_key,
    load_checkpoint_index,
)
from pairwise_eval.data import (
    append_gold_summary_as_model_rows,
    build_toy_long_df,
    discover_eval_model_subdirs,
    load_eval_jsonl_long_df,
    long_df_head_documents,
    resolve_eval_data_dir,
    stack_eval_jsonl_checkpoints_long_df,
)
from pairwise_eval.llm_judge import make_local_llm_evaluate_fn, parse_geval_ab_tie
from pairwise_eval.io_export import (
    export_bradley_terry,
    export_full_run,
    export_win_rates_by_dimension_md,
    export_win_rates_paper,
    resolve_geval_export_dir,
    save_geval_json,
    write_results_summary_md,
)
from pairwise_eval.judging import (
    EvaluateFn,
    attach_doc_context,
    build_geval_tables,
    geval_by_judge,
    is_tie_row,
    mock_evaluate_pair,
    models_in_dimension,
    rng_for_judge_dimension,
)
from pairwise_eval.pairs import build_pairs_table, load_pairs_table_json
from pairwise_eval.win_rates import (
    markdown_win_rate_tables_by_dimension,
    win_rate_matrix_by_dimension,
    win_rate_table_paper,
    wins_and_opportunities_for_group,
)

__all__ = [
    "EVAL_DATA_DIR",
    "EVAL_DIMENSIONS",
    "EXTEND_PAIRS_TABLE_JSON",
    "GEVAL_CHECKPOINT_DIR",
    "GEVAL_EXPORT_DIRNAME",
    "HUMAN_JUDGES",
    "JUDGES",
    "LLM_JUDGES",
    "LOCAL_LLM_CHAT_URL",
    "LOCAL_LLM_TIMEOUT_S",
    "MAX_DOCUMENTS",
    "MOCK_TIE_PROB",
    "N_PAIRS_PER_DOCUMENT",
    "REFERENCE_SUMMARY_MODEL_ID",
    "REPO_ROOT",
    "EvaluateFn",
    "append_judgment_line",
    "append_gold_summary_as_model_rows",
    "attach_doc_context",
    "bradley_terry_long_table",
    "bradley_terry_theta_wide",
    "build_geval_tables",
    "build_pairs_table",
    "build_toy_long_df",
    "checkpoint_file_path",
    "export_bradley_terry",
    "export_full_run",
    "export_win_rates_by_dimension_md",
    "export_win_rates_paper",
    "fit_bradley_terry",
    "geval_by_judge",
    "is_tie_row",
    "judgment_stable_key",
    "load_checkpoint_index",
    "load_pairs_table_json",
    "discover_eval_model_subdirs",
    "load_eval_jsonl_long_df",
    "long_df_head_documents",
    "make_local_llm_evaluate_fn",
    "markdown_bradley_terry_theta",
    "markdown_win_rate_tables_by_dimension",
    "mock_evaluate_pair",
    "models_in_dimension",
    "parse_geval_ab_tie",
    "resolve_eval_data_dir",
    "resolve_geval_export_dir",
    "stack_eval_jsonl_checkpoints_long_df",
    "rng_for_judge_dimension",
    "save_geval_json",
    "win_matrix_from_geval",
    "win_rate_matrix_by_dimension",
    "win_rate_table_paper",
    "wins_and_opportunities_for_group",
    "write_results_summary_md",
]
