"""
Flertallsstemming over tre dommere (openai, gemini, claude)
og fire modeller (baseline, openai, gemini, claude).

Lager to resultatfiler:
- majority_vote_per_document.csv
- majority_vote_totals.csv
"""

from __future__ import annotations
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

# ---------- KONFIGURASJON ----------
JUDGE_DIRS = ["openai_judge_results", "gemini_judge_results", "claude_judge_results"]
CRITS      = ["koherens", "konsistens", "flyt", "relevans"]
MODELS     = ["baseline", "openai", "gemini", "claude"]

PAIR_RE = re.compile(r"judge_scores_([^_]+)_vs_([^.]*)\.csv")


# ---------- HJELPEFUNKSJONER ----------
def pair_totals(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Returner DataFrame med totalscore for modell a og b per dokument."""
    return pd.DataFrame({
        "dokument_id": df["dokument_id"],
        a: df[[f"{c}_{a}" for c in CRITS]].sum(axis=1),
        b: df[[f"{c}_{b}" for c in CRITS]].sum(axis=1),
    })


def winner_series(tot_df: pd.DataFrame, a: str, b: str) -> Dict[str, str]:
    """{dok_id: vinner eller 'tie'} for ett par."""
    res = {}
    for _, r in tot_df.iterrows():
        if r[a] > r[b]:
            res[r["dokument_id"]] = a
        elif r[b] > r[a]:
            res[r["dokument_id"]] = b
        else:
            res[r["dokument_id"]] = "tie"
    return res


def majority(votes: List[str]) -> str:
    """Minst 2 av 3 stemmer -> vinner, ellers 'tie'."""
    cnt = Counter(votes)
    if cnt and cnt.most_common(1)[0][1] >= 2:
        return cnt.most_common(1)[0][0]
    return "tie"

def read_all_pair_winners() -> Dict[str, Dict[str, List[str]]]:
    """
    Les alle judge_scores*-filer og samle dommer-stemmer.
    Returnerer
      { dok_id: { "<a>_vs_<b>": [win_by_openaiJudge, win_by_geminiJudge, …] } }
    """
    votes: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    for judge_dir in JUDGE_DIRS:
        for csv_path in Path(judge_dir).glob("judge_scores*_vs_*.csv"):
            m = PAIR_RE.fullmatch(csv_path.name)
            if not m:
                continue
            a, b = m.groups()
            df = pd.read_csv(csv_path)
            winners = winner_series(pair_totals(df, a, b), a, b)
            for dok_id, win in winners.items():
                votes[dok_id][f"{a}_vs_{b}"].append(win)

    return votes


def aggregate(
    votes: Dict[str, Dict[str, List[str]]]
) -> tuple[list[dict], Counter, Counter]:
    """
    - Lager per-dokument-rader med final_vinner, wins_per_model, pair_winner_…
    - Samler totals: docs_won (final-vinner) + pair_wins
    Returnerer (doc_rows, docs_won_counter, pair_wins_counter).
    """
    doc_rows: list[dict] = []
    docs_won = Counter() # endelig vinner per dokument
    pair_wins = Counter() # antall par-seire på tvers av dokumenter

    for dok_id, pair_dict in votes.items():
        pair_majorities = {p: majority(v) for p, v in pair_dict.items()}

        wins_this_doc = Counter()
        for win in pair_majorities.values():
            if win in MODELS:
                wins_this_doc[win] += 1
                pair_wins[win] += 1

        # final vinner = flest par-seire i dette dokumentet
        if wins_this_doc:
            top_model, top_cnt = wins_this_doc.most_common(1)[0]
            if list(wins_this_doc.values()).count(top_cnt) == 1:
                final = top_model
            else:
                final = "tie"
        else:
            final = "tie"

        if final in MODELS:
            docs_won[final] += 1

        row = {
            "dokument_id": dok_id,
            "final_vinner": final,
        }
        for m in MODELS:
            row[f"{m}_pair_wins"] = wins_this_doc.get(m, 0)
        row.update({f"pair_winner_{k}": v for k, v in pair_majorities.items()})
        doc_rows.append(row)

    return doc_rows, docs_won, pair_wins


# ---------- SKRIVING ----------
def write_doc_level(rows: list[dict], path: Path) -> None:
    fieldnames = ["dokument_id", "final_vinner"] \
                 + [f"{m}_pair_wins" for m in MODELS] \
                 + sorted([k for k in rows[0] if k.startswith("pair_winner_")])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)
    print(f"Suksess: Skrev {len(rows)} rader til {path}")


def write_totals(
    docs_won: Counter, pair_wins: Counter, path: Path
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["modell", "docs_won", "pair_wins"])
        for m in MODELS:
            w.writerow([m, docs_won.get(m, 0), pair_wins.get(m, 0)])
    print(f"Suksess:  Skrev totalsammendrag til {path}")


# ---------- main ----------
def main() -> None:
    votes = read_all_pair_winners()
    if not votes:
        raise SystemExit("Fant ingen judge_scores*-filer under dommer-katalogene.")

    doc_rows, docs_won, pair_wins = aggregate(votes)

    write_doc_level(doc_rows, Path("majority_vote_per_document.csv"))
    write_totals(docs_won, pair_wins, Path("majority_vote_totals.csv"))


if __name__ == "__main__":
    main()