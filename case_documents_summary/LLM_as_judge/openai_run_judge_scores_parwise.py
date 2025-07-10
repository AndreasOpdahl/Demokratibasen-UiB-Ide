import json, time, pathlib, csv, os, re
import openai
from   dotenv import load_dotenv

# ---------- KONFIGURASJON ----------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

LABEL_A = "baseline" # endre om ønskelig
LABEL_B = "openai"
CRITS   = ["koherens", "konsistens", "flyt", "relevans"]

MODEL       = "gpt-4o"
TEMPERATURE = 0

MAX_CHARS_SRC = 4_000
MAX_ATTEMPTS  = 3

ROOT      = pathlib.Path(__file__).resolve().parent
IN_PATH   = ROOT / "judge_input" / f"judge_input_{LABEL_A}_vs_{LABEL_B}.jsonl"
OUT_DIR   = ROOT / "openai_judge_results"
OUT_DIR.mkdir(exist_ok=True)
CSV_OUT   = OUT_DIR / f"judge_scores_{LABEL_A}_vs_{LABEL_B}test.csv"

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
  uten redundans?  Relevans inkluderer spesielt om sammendraget får frem
  mulig *nyhetsverdi* …

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

# ---------- JSON-HJELPER ----------
RE_CODE  = re.compile(r"^```[a-zA-Z]*\n?|\n?```$", re.S)

def _parse_json_flexible(txt: str):
    txt = RE_CODE.sub("", txt.strip())
    return json.loads(txt)  # GPT-4o er stort sett ren JSON; enkel variant holder

def llm_score(doc: str, a: str, b: str) -> dict:
    prompt = USER_TMPL.format(
        source=doc[:MAX_CHARS_SRC], summary_a=a, summary_b=b
    )
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            rsp = openai.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
            )
            return _parse_json_flexible(rsp.choices[0].message.content)
        except Exception as e:
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(1.5 * attempt)

# ---------- LLM-KALL ----------
def llm_score(doc: str, a: str, b: str) -> dict:
    rsp = openai.chat.completions.create(
        model            = MODEL,
        temperature      = TEMPERATURE,
        response_format  = {"type": "json_object"},
        messages         = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_TMPL.format(
                source=doc, summary_a=a, summary_b=b)}
        ],
    )
    return json.loads(rsp.choices[0].message.content)

# ---------- HOVEDLØP ----------
def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"Fant ikke {IN_PATH}")

    with IN_PATH.open(encoding="utf-8") as fin, \
         CSV_OUT.open("w", encoding="utf-8", newline="") as fout:

        writer = csv.writer(fout, quoting=csv.QUOTE_ALL, lineterminator="\n")

        # header
        header = ["index", "dokument_id"]
        for c in CRITS:
            header += [f"{c}_{LABEL_A}", f"{c}_{LABEL_B}", f"{c}_forklaring"]
        writer.writerow(header)

        # rader
        for idx, line in enumerate(fin, start=1):
            row     = json.loads(line)
            doc_id  = row["dokument_id"]
            doc_txt = row.get("document") or row.get("source")

            try:
                scores = llm_score(doc_txt, row[LABEL_A], row[LABEL_B])
            except Exception as e:
                print("Failed:", doc_id, e)
                continue

            out = [idx, doc_id]
            for c in CRITS:
                out += [scores[c]["A"], scores[c]["B"], scores[c]["forklaring"]]
            writer.writerow(out)

            print(f"{idx} | Suksess: {doc_id}")
            time.sleep(0.5)

    print("Ferdig. Resultat lagret i:", CSV_OUT.relative_to(ROOT))

if __name__ == "__main__":
    main()