"""Win-rate tables from pairwise G-Eval results (ties → 0.5 win each side)."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Tuple

import pandas as pd

from pairwise_eval.config import EVAL_DIMENSIONS, HUMAN_JUDGES, JUDGES, LLM_JUDGES
from pairwise_eval.judging import is_tie_row, models_in_dimension


def wins_and_opportunities_for_group(
    geval_tables: Dict[Tuple[str, str], pd.DataFrame],
    judge_ids: Iterable[str],
    dimensions: Tuple[str, ...],
) -> tuple[dict[str, float], dict[str, int]]:
    """Aggregate wins and comparison counts over selected judges and dimensions.

    Input: ``geval_tables``, iterable of judge ids, dimension tuple. Output: ``(wins, opps)`` dicts
    keyed by ``model_id`` (ties count as 0.5 win each side).
    """
    wins: dict[str, float] = defaultdict(float)
    opps: dict[str, int] = defaultdict(int)
    for j in judge_ids:
        for d in dimensions:
            tbl = geval_tables[(j, d)]
            for _, row in tbl.iterrows():
                opps[row["left"]] += 1
                opps[row["right"]] += 1
                if is_tie_row(row):
                    wins[row["left"]] += 0.5
                    wins[row["right"]] += 0.5
                else:
                    wins[row["chosen"]] += 1.0
    return wins, opps


def win_rate_table_paper(
    geval_tables: Dict[Tuple[str, str], pd.DataFrame],
    model_order: Iterable[str] | None = None,
    dimensions: Tuple[str, ...] = EVAL_DIMENSIONS,
) -> pd.DataFrame:
    """Paper-style table: pooled human vs pooled LLM win rates (all dimensions merged).

    Input: ``geval_tables``, optional row order. Output: DataFrame with counts and win rates per model.
    """
    w_h, o_h = wins_and_opportunities_for_group(geval_tables, HUMAN_JUDGES, dimensions)
    w_l, o_l = wins_and_opportunities_for_group(geval_tables, LLM_JUDGES, dimensions)
    models = list(model_order) if model_order is not None else sorted(set(o_h) | set(o_l))
    rows = []
    for m in models:
        oh, ol = o_h.get(m, 0), o_l.get(m, 0)
        rows.append(
            {
                "model": m,
                "n_pairwise_human": oh,
                "win_rate_human": (w_h[m] / oh) if oh else float("nan"),
                "n_pairwise_llm": ol,
                "win_rate_llm_pooled": (w_l[m] / ol) if ol else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def win_rate_matrix_by_dimension(
    geval_tables: Dict[Tuple[str, str], pd.DataFrame],
    dimension: str,
    judges: Tuple[str, ...] = JUDGES,
    model_order: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Per-model win rates for one dimension and every judge.

    Input: ``geval_tables``, dimension name, judge list, optional model order. Output: wide table.
    """
    mo = list(model_order) if model_order is not None else models_in_dimension(geval_tables, dimension)
    col_names = [f"{j}_win_rate" for j in judges]
    rows = []
    for m in mo:
        rec: dict = {"model": m}
        for j, col in zip(judges, col_names):
            tbl = geval_tables[(j, dimension)]
            opps = ((tbl["left"] == m) | (tbl["right"] == m)).sum()
            w = 0.0
            for _, row in tbl.iterrows():
                if is_tie_row(row):
                    if row["left"] == m or row["right"] == m:
                        w += 0.5
                elif row["chosen"] == m:
                    w += 1.0
            rec[col] = (w / opps) if opps else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def markdown_win_rate_tables_by_dimension(
    geval_tables: Dict[Tuple[str, str], pd.DataFrame],
    dimensions: Tuple[str, ...] = EVAL_DIMENSIONS,
    judges: Tuple[str, ...] = JUDGES,
    model_order: Iterable[str] | None = None,
) -> str:
    """Markdown tables: win rates per dimension × judge.

    Input: ``geval_tables``, dimensions, judges, optional model order. Output: markdown string.
    """
    mo = list(model_order) if model_order is not None else models_in_dimension(geval_tables, dimensions[0])
    parts: list[str] = []
    for dim in dimensions:
        mat = win_rate_matrix_by_dimension(geval_tables, dim, judges, mo)
        parts.append(f"### {dim.capitalize()}\n")
        cols = ["model"] + [f"{j}_win_rate" for j in judges]
        parts.append("| " + " | ".join(cols) + " |")
        parts.append("| " + " | ".join("---" for _ in cols) + " |")
        for _, r in mat.iterrows():
            cells = [r["model"]] + [f"{r[c]:.3f}" if pd.notna(r[c]) else "—" for c in cols[1:]]
            parts.append("| " + " | ".join(str(x) for x in cells) + " |")
        parts.append("")
    return "\n".join(parts)
