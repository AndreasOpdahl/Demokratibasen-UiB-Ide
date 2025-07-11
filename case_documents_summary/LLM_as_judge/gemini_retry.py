"""
Opplevde litt problemer med gemini som dommer, feilet innimellom på noen få dokumenter.
Kjør Gemini-dommer kun på dokumenter som feilet i forrige runde
og lagre poeng + forklaring i egen CSV.

Kan deretter manuelt kopiere radene inn i den opprinnelige judge-CSV-en.
"""

import csv, json, os, pathlib, re, time
from typing import Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv

# ---------- KONFIGURASJON ----------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

LABEL_A = "openai"
LABEL_B = "gemini"
CRITS   = ["koherens", "konsistens", "flyt", "relevans"]

MODEL       = "models/gemini-1.5-pro-latest"
TEMPERATURE = 0

ROOT     = pathlib.Path(__file__).resolve().parent
IN_PATH  = ROOT / "judge_input" / f"judge_input_{LABEL_A}_vs_{LABEL_B}.jsonl"
OUT_DIR  = ROOT / "gemini_judge_results"
OUT_DIR.mkdir(exist_ok=True)
CSV_OUT  = OUT_DIR / f"judge_scores_{LABEL_A}_vs_{LABEL_B}_retry.csv"

# ---------- LEGG INN ID DU ØNSKER Å KJØRE PÅ NYTT ----------
RETRY_IDS = {
    "817df11a-e657-5b5d-b18a-4536fbbed0e7",
    # legg til flere UUIDer her ved behov
}

# ---------- PROMPTER ----------
SYSTEM_PROMPT = """
Du er dyktig og hjelpsom assistent i en avisredaksjon.
Avisen dekker kommunal politikk, men journalistene rekker ikke
å lese alle saksdokumenter. I stedet leser de automatiske sammendrag.
Du skal vurdere hvilke av to sammendrag som er best, ut fra fire kriterier.
Svar KUN med gyldig JSON.
"""

USER_TMPL = """
Her følger ett saksdokument og TO forskjellige sammendrag.

============================================================
SAKSDOKUMENT (kildetekst)
------------------------------------------------------------
{source}

============================================================
SAMMENDRAG A
------------------------------------------------------------
{summary_a}

============================================================
SAMMENDRAG B
------------------------------------------------------------
{summary_b}

============================================================
KRITERIER  – bruk skala 1 (dårligst) – 5 (best)

• **Koherens** – er sammendraget velstrukturert og logisk helhetlig?
• **Konsistens** – inneholder det kun påstander som støttes av kildeteksten?
• **Flyt** – er setningene velskrevne uten formaterings- eller grammatikkfeil?
• **Relevans** – inneholder det bare den viktigste informasjonen om saken,
  uten redundans?

============================================================
OPPGAVE
------------------------------------------------------------
1. Gi en NUMERISK poengsum (1–5) til hvert sammendrag for hvert kriterium.
2. For hvert kriterium, gi én kort begrunnelse (1–2 setninger).

SVARFORMAT (gyldig JSON – ingen annen tekst):
{{
  "koherens":   {{"A": <int>, "B": <int>, "forklaring": "<tekst>"}},
  "konsistens": {{"A": <int>, "B": <int>, "forklaring": "<tekst>"}},
  "flyt":       {{"A": <int>, "B": <int>, "forklaring": "<tekst>"}},
  "relevans":   {{"A": <int>, "B": <int>, "forklaring": "<tekst>"}}
}}
"""

# ---------- HJELPEFUNKSJONER ----------
_CODE_BLOCK = re.compile(r"^```[a-zA-Z]*\n?|\n?```$", re.S)

def _parse_json(txt: str) -> Dict[str, Any]:
    """Fjerner ev. markdown-kodeblokker og returnerer JSON-objektet."""
    txt = _CODE_BLOCK.sub("", txt.strip())
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            return json.loads(m.group(0))
        raise

def llm_score(doc: str, summary_a: str, summary_b: str) -> Dict[str, Any]:
    model = genai.GenerativeModel(MODEL)
    rsp   = model.generate_content(
        [SYSTEM_PROMPT,
         USER_TMPL.format(source=doc, summary_a=summary_a, summary_b=summary_b)],
        generation_config={"temperature": TEMPERATURE}
    )
    return _parse_json(rsp.text)

# ---------- HOVEDLØP ----------
def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"Fant ikke {IN_PATH}")

    # last inn alle poster & filtrér
    with IN_PATH.open(encoding="utf-8") as f:
        rows_all = [json.loads(line) for line in f]
    rows_retry = [r for r in rows_all if r["dokument_id"] in RETRY_IDS]

    if not rows_retry:
        print("Ingen av de oppgitte ID-ene finnes i input-fila.")
        return

    with CSV_OUT.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)

        # header
        header = ["index", "dokument_id"]
        for c in CRITS:
            header += [f"{c}_{LABEL_A}", f"{c}_{LABEL_B}", f"{c}_forklaring"]
        writer.writerow(header)

        # rader
        for idx, row in enumerate(rows_retry, start=1):
            doc_id  = row["dokument_id"]
            doc_txt = row.get("document") or row.get("source")
            try:
                scores = llm_score(doc_txt, row[LABEL_A], row[LABEL_B])
            except Exception as e:
                print(f"Failed: {doc_id} ->", e)
                continue

            out = [idx, doc_id]
            for c in CRITS:
                out += [scores[c]["A"], scores[c]["B"], scores[c]["forklaring"]]
            writer.writerow(out)

            print(f"Suksess: {doc_id}")
            time.sleep(0.5)

    print("\nFerdig – nye rader lagret i", CSV_OUT.relative_to(ROOT))

if __name__ == "__main__":
    main()