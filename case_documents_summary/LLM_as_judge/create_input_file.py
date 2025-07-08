"""
Bygger judge-input-fila (JSONL) for pair-wise evaluering.
Resultatet legges i LLM_as_judge/judge_input_<A>_vs_<B>.jsonl
"""

import json, pathlib
import pandas as pd

# ────────────────────── KONFIGURER HER ──────────────────────
LABEL_A   = "baseline"                       # navn i JSONL (og senere i prompt)
LABEL_B   = "gemini"                         #  ─   ─  ─

CSV_A     = "baseline/baseline_case_summaries.csv"     # har kolonnen «oppsummering»
CSV_B     = "summary_generation/gemini_summary_results.csv"       # har kolonnen «sammendrag»
# ────────────────────────────────────────────────────────────

ROOT      = pathlib.Path(__file__).resolve().parent.parent
DOC_PATH  = ROOT / "cleaning_preprocessing" / "baseline_documents_cleaned_first300.jsonl"
A_PATH    = ROOT / CSV_A
B_PATH    = ROOT / CSV_B

out_dir   = ROOT / "LLM_as_judge"
out_dir.mkdir(exist_ok=True)
OUT_PATH  = out_dir / f"judge_input_{LABEL_A}_vs_{LABEL_B}.jsonl"

# ---------- les dokumenttekst ------------------------------------------------
with DOC_PATH.open(encoding="utf-8") as f:
    doc_text = {json.loads(l)["dokument_id"]:
                json.loads(l).get("tekst_cleaned") or json.loads(l).get("tekst")
                for l in f}

# ---------- les sammendrag ---------------------------------------------------
df_a = pd.read_csv(A_PATH, dtype={"dokument_id": "string"}) \
          [["dokument_id", "oppsummering"]] \
          .rename(columns={"oppsummering": LABEL_A})

df_b = pd.read_csv(B_PATH, dtype={"dokument_id": "string"}) \
          [["dokument_id", "sammendrag"]] \
          .rename(columns={"sammendrag": LABEL_B})

df = df_a.merge(df_b, on="dokument_id", how="inner")

# ---------- skriv JSONL ------------------------------------------------------
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

print(f"Skrev {len(df):,} poster ➜ {OUT_PATH.relative_to(ROOT)}")