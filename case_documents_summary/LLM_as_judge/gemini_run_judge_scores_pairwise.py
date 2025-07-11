"""
Kjør Gemini 1.5 Pro som dommer mellom to modeller og lagre
poeng + begrunnelse i CSV.
Nå med innebygd retry + smartere JSON-parser.
"""

import csv, json, os, pathlib, re, time
from typing import Dict, Any

import google.generativeai as genai
from dotenv import load_dotenv

# ---------- KONFIGURASJON ----------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

LABEL_A = "baseline"
LABEL_B = "claude"
CRITS   = ["koherens", "konsistens", "flyt", "relevans"]

MODEL          = "models/gemini-1.5-pro-latest"
TEMPERATURE    = 0
MAX_ATTEMPTS   = 3           # hvor mange ganger vi prøver samme dokument
MAX_CHARS_SRC  = 4_000       # klipp kildeteksten

ROOT     = pathlib.Path(__file__).resolve().parent
IN_PATH  = ROOT / "judge_input" / f"judge_input_{LABEL_A}_vs_{LABEL_B}.jsonl"
OUT_DIR  = ROOT / "gemini_judge_results" # Endre ut i fra hvilken modell som er judge
OUT_DIR.mkdir(exist_ok=True)
CSV_OUT  = OUT_DIR / f"judge_scores_{LABEL_A}_vs_{LABEL_B}.csv"

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
  uten redundans? Relevans inkluderer mulig *nyhetsverdi*.

============================================================
OPPGAVE
------------------------------------------------------------
1. Gi en NUMERISK poengsum (1–5) til hvert sammendrag for hvert kriterium.
2. For hvert kriterium, gi ÉN kort begrunnelse (1–2 setninger).

SVARFORMAT  (kun JSON):
{{
  "koherens":   {{"A": <int>, "B": <int>, "forklaring": "<tekst>"}},
  "konsistens": {{"A": <int>, "B": <int>, "forklaring": "<tekst>"}},
  "flyt":       {{"A": <int>, "B": <int>, "forklaring": "<tekst>"}},
  "relevans":   {{"A": <int>, "B": <int>, "forklaring": "<tekst>"}}
}}
"""
# ---------- PARSER ----------
RE_CODE   = re.compile(r"^```[a-zA-Z]*\n?|\n?```$", re.S)
RE_DBLQ   = re.compile(r'("")')         # dobbelttegn uten escape
RE_CTRLS  = re.compile(r'[\x00-\x1f]')  # uescaped kontrolltegn

def _parse_json_flexible(raw: str) -> Dict[str, Any]:
    """Prøver å rydde i typiske formateringsfeil før json.loads."""
    raw = RE_CODE.sub("", raw.strip())

    # fikse linjeskift / uescaped ctrl-tegn inne i strenger
    def _escape_ctrl(m: re.Match) -> str:
        ch = m.group(0)
        return "\\n" if ch in "\r\n" else f"\\u{ord(ch):04x}"
    raw = RE_CTRLS.sub(_escape_ctrl, raw)

    # fikse ""  ➜ \"
    raw = RE_DBLQ.sub(r'\\"', raw)

    # prøv rett av
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            return json.loads(m.group(0))
        raise

# ---------- LLM-KALL ----------
def llm_score(src: str, sum_a: str, sum_b: str) -> Dict[str, Any]:
    prompt = USER_TMPL.format(
        source    = src[:MAX_CHARS_SRC],
        summary_a = sum_a,
        summary_b = sum_b
    )
    model = genai.GenerativeModel(MODEL)
    rsp   = model.generate_content(
        [SYSTEM_PROMPT, prompt],
        generation_config={
            "temperature": TEMPERATURE,
            "response_mime_type": "application/json"
        }
    )
    return _parse_json_flexible(rsp.text)

# ---------- HOVEDLØP ----------
def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"Fant ikke {IN_PATH}")

    with IN_PATH.open(encoding="utf-8") as f_in, \
         CSV_OUT.open("w", encoding="utf-8", newline="") as f_out:

        writer = csv.writer(f_out, quoting=csv.QUOTE_ALL, lineterminator="\n") # pakker alt i "" pga. problemer med kommaseparator i genererte begrunnelser
        header = ["index", "dokument_id"]
        for c in CRITS:
            header += [f"{c}_{LABEL_A}", f"{c}_{LABEL_B}", f"{c}_forklaring"]
        writer.writerow(header)

        for idx, line in enumerate(f_in, start=1):
            row  = json.loads(line)
            did  = row["dokument_id"]
            doc  = row.get("source") or row.get("document", "")

            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    scores = llm_score(doc, row[LABEL_A], row[LABEL_B])
                    break
                except Exception as e:
                    if attempt == MAX_ATTEMPTS:
                        print(f"Failed: {idx:>4} {did} – ga opp ({e})")
                        scores = None
                    else:
                        time.sleep(1.5 * attempt)   # eksponentiell back-off
            if not scores:
                continue

            out = [idx, did]
            for c in CRITS:
                out += [scores[c]["A"], scores[c]["B"], scores[c]["forklaring"]]
            writer.writerow(out)
            print(f"{idx:>4} | Suksess: {did}")
            time.sleep(0.4)   # høflig pausing

    print("\nFerdig – resultat lagret i", CSV_OUT.relative_to(ROOT))

if __name__ == "__main__":
    main()