"""Write JSON/CSV/Markdown/LaTeX artifacts under ``.deepeval/<GEVAL_EXPORT_DIRNAME>/``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from pairwise_eval.bradley_terry import bradley_terry_long_table, bradley_terry_theta_wide, markdown_bradley_terry_theta
from pairwise_eval.config import (
    EVAL_DIMENSIONS,
    GEVAL_EXPORT_DIRNAME,
    HUMAN_JUDGES,
    JUDGES,
    LLM_JUDGES,
)
from pairwise_eval.win_rates import markdown_win_rate_tables_by_dimension, win_rate_table_paper

# Subfolders under the export root (keeps one flat directory from getting cluttered)
SUBJSON = "json"
SUBTABLES = "tables"
SUBREPORTS = "reports"


def _results_summary_preamble(
    geval_tables: Dict[Tuple[str, str], pd.DataFrame],
    pairs_df: pd.DataFrame | None,
    long_df: pd.DataFrame | None,
) -> str:
    """Build judges, dimensions, scope, and datapoint counts for the top of ``results_summary.md``."""
    _judge_set = {j for j, _ in geval_tables.keys()}
    # Match column order in win-rate tables (:data:`JUDGES`), then any extra keys lexicographically.
    judges = [j for j in JUDGES if j in _judge_set] + sorted(_judge_set - set(JUDGES))
    _dim_set = {d for _, d in geval_tables.keys()}
    dimensions = [d for d in EVAL_DIMENSIONS if d in _dim_set] + sorted(_dim_set - set(EVAL_DIMENSIONS))
    judge_line = ", ".join(f"`{j}`" for j in judges) if judges else "_(none)_"
    dim_line = ", ".join(f"`{d}`" for d in dimensions) if dimensions else "_(none)_"
    lines = [
        f"**Judges:** {judge_line}",
        f"**Dimensions:** {dim_line}",
    ]

    if long_df is not None and "doc_id" in long_df.columns:
        lines.append(
            f"**Documents in subset:** {int(long_df['doc_id'].nunique())} distinct `doc_id`."
        )

    sizes = [len(df) for df in geval_tables.values()] if geval_tables else []
    total_rows = sum(sizes)
    n_j, n_d = len(judges), len(dimensions)
    if geval_tables and sizes:
        if len(set(sizes)) == 1:
            r = sizes[0]
            n_tables = len(geval_tables)
            parts = [
                f"**Datapoints:** {total_rows} pairwise judgments total "
                f"({r} rows per G-Eval table × {n_tables} table(s), one per judge × dimension)."
            ]
            if pairs_df is not None and len(pairs_df) == r and n_j and n_d:
                jw = "judge" if n_j == 1 else "judges"
                dm = "dimension" if n_d == 1 else "dimensions"
                parts.append(
                    f"Equivalent to {len(pairs_df)} pair comparisons × {n_d} {dm} × {n_j} {jw}."
                )
            lines.extend(parts)
        else:
            mn, mx = min(sizes), max(sizes)
            lines.append(
                f"**Datapoints:** {total_rows} pairwise judgments total across all tables "
                f"({mn}–{mx} rows per judge×dimension; see `json/` for per-file counts)."
            )
    elif pairs_df is not None:
        lines.append(
            f"**Pairwise comparisons:** {len(pairs_df)} rows in the pairs table "
            "(G-Eval tables empty or not passed)."
        )

    return "\n".join(lines)


def _json_stem_fragment(s: str) -> str:
    """Sanitize a string for safe use inside JSON filenames (no path separators).

    Input: raw judge/model id. Output: string with ``\\`` and ``/`` replaced by ``__``.
    """
    return s.replace("\\", "__").replace("/", "__")


def resolve_geval_export_dir() -> Path:
    """Export root: ``.deepeval/<GEVAL_EXPORT_DIRNAME>`` (or same name under cwd if cwd is ``.deepeval``).

    Input: none. Output: Path (may not exist yet). Directory name comes from
    :data:`pairwise_eval.config.GEVAL_EXPORT_DIRNAME`.
    """
    leaf = GEVAL_EXPORT_DIRNAME.strip()
    if not leaf or "/" in leaf or "\\" in leaf or leaf in (".", ".."):
        raise ValueError(
            "GEVAL_EXPORT_DIRNAME must be a single folder name (no path separators), "
            "e.g. geval_exports or geval_winners_exports."
        )
    cwd = Path.cwd()
    if cwd.name == ".deepeval":
        return cwd / leaf
    return cwd / ".deepeval" / leaf


def _ensure_export_layout(root: Path) -> tuple[Path, Path, Path]:
    """Ensure ``root/json``, ``root/tables``, ``root/reports`` exist.

    Input: export root. Output: ``(json_dir, tables_dir, reports_dir)``.
    """
    j = root / SUBJSON
    t = root / SUBTABLES
    r = root / SUBREPORTS
    j.mkdir(parents=True, exist_ok=True)
    t.mkdir(parents=True, exist_ok=True)
    r.mkdir(parents=True, exist_ok=True)
    return j, t, r


def save_geval_json(
    geval_tables: Dict[Tuple[str, str], pd.DataFrame],
    pairs_df: pd.DataFrame,
    long_df: pd.DataFrame,
    export_dir: Path | None = None,
) -> Path:
    """Write ``pairs_table``, ``summarization_long``, and each G-Eval table to ``json/*.json`` + manifest.

    Input: geval tables, pairs, long_df; optional ``export_dir``. Output: root path written under.
    """
    out = export_dir or resolve_geval_export_dir()
    json_dir, _, _ = _ensure_export_layout(out)
    manifest: dict = {
        "export_dir": str(out.resolve()),
        "layout": {
            "json": f"{SUBJSON}/",
            "tables": f"{SUBTABLES}/",
            "reports": f"{SUBREPORTS}/",
        },
        "files": [],
    }

    def write_df(stem: str, frame: pd.DataFrame) -> None:
        """Write one DataFrame to ``json/{stem}.json`` and record it in ``manifest``."""
        rel = f"{SUBJSON}/{stem}.json"
        path = out / rel
        frame.to_json(path, orient="records", indent=2, force_ascii=False)
        manifest["files"].append({"path": rel, "rows": len(frame)})

    write_df("pairs_table", pairs_df)
    write_df("summarization_long", long_df)
    for (judge_id, dimension), tbl in geval_tables.items():
        jpart = _json_stem_fragment(judge_id)
        write_df(f"geval__{jpart}__{dimension}", tbl)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def export_win_rates_paper(
    geval_tables: Dict[Tuple[str, str], pd.DataFrame],
    model_order: list[str],
    export_dir: Path | None = None,
) -> Path:
    """Export pooled human/LLM win-rate table to CSV, LaTeX, and Markdown under ``tables/`` and ``reports/``."""
    out = export_dir or resolve_geval_export_dir()
    _, tables_dir, reports_dir = _ensure_export_layout(out)
    wr = win_rate_table_paper(geval_tables, model_order=model_order)
    wr.to_csv(tables_dir / "win_rates_by_model.csv", index=False)
    wr.to_latex(
        buf=reports_dir / "win_rates_by_model.tex",
        index=False,
        float_format="%.3f".format,
        caption="Pairwise win rates by judge type (human vs pooled LLM judges).",
        label="tab:winrates",
    )
    w2 = wr.copy()
    for c in ("win_rate_human", "win_rate_llm_pooled"):
        w2[c] = w2[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    cols = list(w2.columns)
    human_note = ", ".join(f"`{j}`" for j in HUMAN_JUDGES) if HUMAN_JUDGES else "_(none)_"
    llm_judge_note = ", ".join(f"`{j}`" for j in LLM_JUDGES) if LLM_JUDGES else "_(none)_"
    lines = [
        "# Pairwise win rates by model",
        "",
        f"Human judges (all dimensions pooled): {human_note}. LLM judges (all dimensions pooled): {llm_judge_note}.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, r in w2.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    lines.append("")
    (reports_dir / "win_rates_by_model.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def export_win_rates_by_dimension_md(
    geval_tables: Dict[Tuple[str, str], pd.DataFrame],
    model_order: list[str],
    export_dir: Path | None = None,
) -> Path:
    """Write ``win_rates_by_dimension.md`` (per-dimension markdown tables)."""
    out = export_dir or resolve_geval_export_dir()
    _, _, reports_dir = _ensure_export_layout(out)
    md = markdown_win_rate_tables_by_dimension(geval_tables, model_order=model_order)
    (reports_dir / "win_rates_by_dimension.md").write_text(md, encoding="utf-8")
    return out


def export_bradley_terry(
    geval_tables: Dict[Tuple[str, str], pd.DataFrame],
    model_order: list[str],
    ref_model: str,
    export_dir: Path | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Export Bradley–Terry long + per-dimension θ CSVs and a markdown report.

    Input: geval tables, model order, BT reference model id. Output: ``(export_root, bt_long)``.
    """
    out = export_dir or resolve_geval_export_dir()
    _, tables_dir, reports_dir = _ensure_export_layout(out)
    bt_long = bradley_terry_long_table(geval_tables, model_order=model_order, ref_model=ref_model)
    bt_long.to_csv(tables_dir / "bradley_terry_long.csv", index=False)
    for dim, tab in bradley_terry_theta_wide(bt_long).items():
        tab.to_csv(tables_dir / f"bradley_terry_theta__{dim}.csv")
    (reports_dir / "bradley_terry_theta_by_dimension.md").write_text(markdown_bradley_terry_theta(bt_long), encoding="utf-8")
    return out, bt_long


def write_results_summary_md(
    geval_tables: Dict[Tuple[str, str], pd.DataFrame],
    model_order: list[str],
    ref_model: str,
    export_dir: Path | None = None,
    *,
    pairs_df: pd.DataFrame | None = None,
    long_df: pd.DataFrame | None = None,
) -> Path:
    """Write ``results_summary.md``: pooled win-rate markdown + Bradley–Terry section + layout note.

    Input: geval tables, model order, BT ref model; optional ``pairs_df`` / ``long_df`` for the header scope.
    Output: path to the markdown file.
    """
    out = export_dir or resolve_geval_export_dir()
    _, _, reports_dir = _ensure_export_layout(out)
    preamble = _results_summary_preamble(geval_tables, pairs_df, long_df)
    win_md = markdown_win_rate_tables_by_dimension(geval_tables, model_order=model_order)
    bt_long = bradley_terry_long_table(geval_tables, model_order=model_order, ref_model=ref_model)
    bt_md = markdown_bradley_terry_theta(bt_long)
    bt_lines = bt_md.split("\n")
    if bt_lines and bt_lines[0].startswith("# "):
        bt_lines = bt_lines[1:]
    bt_sub = "\n".join(bt_lines).strip()
    layout_note = "\n".join(
        [
            "---",
            "",
            "## Export layout",
            "",
            f"- `{SUBJSON}/` — pairwise rows and per-judge G-Eval tables (JSON)",
            f"- `{SUBTABLES}/` — CSV summaries (win rates, Bradley–Terry)",
            f"- `{SUBREPORTS}/` — Markdown / LaTeX for reading and papers",
            "",
        ]
    )
    doc = "\n".join(
        [
            "# G-Eval results summary",
            "",
            preamble,
            "",
            f"Bradley–Terry: `{ref_model}` labels gold summaries (JSONL `reference`). "
            "Exported θ use mean-centered β (geom. mean θ = 1); odds vs any other model match the fitted BT model.",
            "",
            "---",
            "",
            "## 1. Pairwise win rates",
            "",
            win_md.strip(),
            "",
            "---",
            "",
            "## 2. Bradley–Terry strengths (θ)",
            "",
            bt_sub,
            "",
            layout_note,
        ]
    )
    path = reports_dir / "results_summary.md"
    path.write_text(doc, encoding="utf-8")
    return path


def export_full_run(
    geval_tables: Dict[Tuple[str, str], pd.DataFrame],
    pairs_df: pd.DataFrame,
    long_df: pd.DataFrame,
    model_order: list[str],
    ref_model: str,
    export_dir: Path | None = None,
) -> Path:
    """Run the full export pipeline (JSON, win rates, Bradley–Terry, results summary).

    Input: geval tables, pairs, long_df, model list, BT ``ref_model``. Output: export root path.
    """
    out = export_dir or resolve_geval_export_dir()
    save_geval_json(geval_tables, pairs_df, long_df, export_dir=out)
    export_win_rates_paper(geval_tables, model_order, export_dir=out)
    export_win_rates_by_dimension_md(geval_tables, model_order, export_dir=out)
    export_bradley_terry(geval_tables, model_order, ref_model, export_dir=out)
    write_results_summary_md(
        geval_tables,
        model_order,
        ref_model,
        export_dir=out,
        pairs_df=pairs_df,
        long_df=long_df,
    )
    return out
