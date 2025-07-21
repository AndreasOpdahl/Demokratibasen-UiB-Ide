"""
Bygger judge-input-fila (JSONL) for pair-wise evaluering.
Resultatet legges i LLM_as_judge/judge_input_<A>_vs_<B>.jsonl
"""

import json, pathlib
import pandas as pd

# ---------- KONFIGURASJON ----------
LABEL_A   = "baseline" # navn i JSONL (og senere i prompt)
LABEL_B   = "openai" #  ─   ─  ─

CSV_A     = "summary_generation/gemini_summary_results.csv"
CSV_B     = "summary_generation/claude_summary_results.csv"

ROOT      = pathlib.Path(__file__).resolve().parent.parent
DOC_PATH  = ROOT / "cleaning_preprocessing" / "baseline_documents_cleaned_first300.jsonl"
A_PATH    = ROOT / CSV_A
B_PATH    = ROOT / CSV_B

out_dir   = ROOT / "LLM_as_judge"
out_dir.mkdir(exist_ok=True)
OUT_PATH  = out_dir / f"judge_input_{LABEL_A}_vs_{LABEL_B}.jsonl"

# ---------- LES DOKUMENTTEKST ----------
def read_summary_csv(path: pathlib.Path, label: str,
                     candidates = ("oppsummering", "sammendrag", "summary")) -> pd.DataFrame:
    """
    Leser en CSV-fil og returnerer en df med:
        dokument_id, <label>
    Der <label> er 'openai', 'gemini' … etc.
    Velger første kolonne som matcher *candidates*.
    """
    df = pd.read_csv(path, dtype={"dokument_id": "string"})
    for col in candidates:
        if col in df.columns:
            return df[["dokument_id", col]].rename(columns={col: label})

    raise ValueError(
        f"Ingen av kolonnene {candidates} finnes i {path.name}. "
        f"Fant: {list(df.columns)}"
    )

with DOC_PATH.open(encoding="utf-8") as fh:
    doc_text = {obj["dokument_id"]: obj.get("tekst_cleaned") or obj.get("tekst")
                for obj in map(json.loads, fh)}

# ---------- LES SAMMENDRAG ----------
df_a = read_summary_csv(A_PATH, LABEL_A)
df_b = read_summary_csv(B_PATH, LABEL_B)

df = df_a.merge(df_b, on="dokument_id", how="inner")

# ---------- SKRIV JSONL ----------
with OUT_PATH.open("w", encoding="utf-8") as out:
    for _, r in df.iterrows():
        src = doc_text.get(r.dokument_id)
        if not src:          # finnes ingen tilhørende dokumenttekst
            continue
        out.write(json.dumps({
            "dokument_id": r.dokument_id,
            "source":      src,
            LABEL_A:       r[LABEL_A],
            LABEL_B:       r[LABEL_B],
        }, ensure_ascii=False) + "\n")

print(f"Skrev {len(df):,} poster til {OUT_PATH.relative_to(ROOT)}")