
"""
Kjør Claude-4 Sonnet som dommer mellom to modeller og
lagre poeng + én setnings begrunnelse per kriterium i CSV.

Utformat:
"index","dokument_id",
"koherens_<A>","koherens_<B>","koherens_forklaring",
"konsistens_<A>", … osv.
"""

# --- ekstra imports på toppen av fila -------------
import signal, sys

import csv, json, os, re, time, ast
from pathlib import Path
from typing import Dict, Any

import anthropic
from dotenv import load_dotenv

# ---------- KONFIGURASJON ----------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent 
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise SystemExit("Error: ANTHROPIC_API_KEY mangler i .env")

client = anthropic.Anthropic(api_key=api_key)

RE_CODE   = re.compile(r"^```[a-zA-Z]*\n?|\n?```$", re.S)
RE_DBLQ   = re.compile(r'("")')          # ""  ->  \"
RE_CTRLS  = re.compile(r"[\x00-\x1F]")   # u-escaped kontrolltegn

LABEL_A = "baseline"
LABEL_B = "claude" 
CRITS = ["koherens", "konsistens", "flyt", "relevans"]

MODEL = "claude-sonnet-4-20250514"
TEMPERATURE = 0
MAX_ATTEMPTS = 3
MAX_CHARS_SRC = 4_000

IN_PATH = SCRIPT_DIR / "judge_input" / f"judge_input_{LABEL_A}_vs_{LABEL_B}.jsonl"
OUT_DIR = SCRIPT_DIR / "claude_judge_results"
OUT_DIR.mkdir(exist_ok=True)
CSV_OUT = OUT_DIR / f"judge_scores_{LABEL_A}_vs_{LABEL_B}.csv"

# ---------- PROMPTER ----------
SYSTEM_PROMPT = """Du er dyktig og hjelpsom assistent i en avisredaksjon.
Avisen dekker kommunal politikk, men journalistene rekker ikke å lese alle
saksdokumenter. I stedet leser de automatiske sammendrag. Du skal vurdere
hvilke av to sammendrag som er best, ut fra fire kriterier. Svar KUN med gyldig JSON.""".strip()

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

• Koherens  – er sammendraget velstrukturert og logisk helhetlig?  
• Konsistens – inneholder det kun påstander som støttes av kildeteksten?  
• Flyt       – er setningene velskrevne uten formaterings- eller grammatikkfeil?  
• Relevans   – inneholder det bare den viktigste informasjonen om saken,
               uten redundans? Relevans inkluderer mulig *nyhetsverdi*.

============================================================
OPPGAVE
------------------------------------------------------------
1. Gi en NUMERISK poengsum (1–5) til hvert sammendrag for hvert kriterium.  
2. For hvert kriterium, gi ÉN kort begrunnelse (1–2 setninger).

SVARFORMAT (eneste gyldige ut fra deg):
{{
  "koherens":   {{"A": <int>, "B": <int>, "forklaring": "<tekst>"}},
  "konsistens": {{"A": <int>, "B": <int>, "forklaring": "<tekst>"}},
  "flyt":       {{"A": <int>, "B": <int>, "forklaring": "<tekst>"}},
  "relevans":   {{"A": <int>, "B": <int>, "forklaring": "<tekst>"}}
}}
""".strip()

stop_requested = False
def _sigint_handler(sig, frame):
    global stop_requested
    stop_requested = True
    print("\n⏹  Avbryter etter pågående kall ...")

signal.signal(signal.SIGINT, _sigint_handler)

# ---------- ROBUST JSON-PARSER -----------
def parse_json_flexible(raw: str) -> Dict[str, Any]:
    """
    • Fjerner evt. markdown-kodeblokker  (``` … ```)  
    • Prøver først helt vanlig json.loads()  
    • Hvis det feiler, gjør minimal opprydding og prøver igjen  
    • Deretter et «siste-håp» som bruker ast.literal_eval()
      (tåler u/escaped new-lines, enkeltsitater, manglende komma etter siste felt osv.)
    Kaster ValueError hvis alt feiler.
    """
    # 0) Trim + fjern ```blokker
    txt = RE_CODE.sub("", raw.strip())

    # 1) Ren JSON?
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass                                 # fortsett – vi prøver mer tolerant parsing

    # 2) Minimal «quick fix» som ofte holder
    cleaned = txt

    # – Claude pleier av og til å lage blanke linjer *før* {...}. Fjern alt foran første {.
    if not cleaned.lstrip().startswith("{"):
        cleaned = cleaned[cleaned.find("{") :]

    # – Fjern ASCII-kontrolltegn (0–31) som ikke er \t\r\n
    cleaned = RE_CTRLS.sub("", cleaned)

    # – Har den brukt enkelt­anførselstegn rundt nøkler/strenger?
    #   Vi prøver *bare* hvis antall " er veldig lavt.
    if cleaned.count('"') < 4 and cleaned.count("'") > 4:
        cleaned = re.sub(r"'", r'"', cleaned)

    # – Nye forsøk
    for candidate in (cleaned, cleaned.replace("\n", "\\n")):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # 3) Siste-håp: Python-literal-eval  (tåler single quotes, trailing commas mm.)
    try:
        return ast.literal_eval(cleaned)
    except Exception:
        raise ValueError("Kunne ikke parse modell­responsen som JSON:\n" + raw[:500])

# ---------- LLM-KALL ----------
def judge_one(doc: str, sum_a: str, sum_b: str) -> Dict[str, Any]:
    prompt = USER_TMPL.format(
        source    = doc[:MAX_CHARS_SRC],
        summary_a = sum_a,
        summary_b = sum_b
    )
    rsp = client.messages.create(
        model           = MODEL,
        system          = SYSTEM_PROMPT,
        messages        = [{"role": "user", "content": prompt}],
        temperature     = TEMPERATURE,
        max_tokens      = 1024,
    )
    return parse_json_flexible(rsp.content[0].text)

# ---------- HOVEDLØP ----------
def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"Fant ikke inputfilen: {IN_PATH}")

    # 1) hent dokument-ID-er som allerede er prosessert
    done_ids: set[str] = set()
    if CSV_OUT.exists():
        with CSV_OUT.open(encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            done_ids = {r["dokument_id"] for r in rdr}

    # 2) åpne CSV-filen i append-modus
    with CSV_OUT.open("a", encoding="utf-8", newline="") as out_file:
        writer = csv.writer(out_file, quoting=csv.QUOTE_ALL, lineterminator="\n")

        # skriv header hvis filen er tom
        if CSV_OUT.stat().st_size == 0:
            header = ["index", "dokument_id"]
            for c in CRITS:
                header += [f"{c}_{LABEL_A}", f"{c}_{LABEL_B}", f"{c}_forklaring"]
            writer.writerow(header)
            out_file.flush()

        # 3) les alle dokumentene inn i minnet én gang
        with IN_PATH.open(encoding="utf-8") as fin:
            docs = [json.loads(l) for l in fin]

        total = len(docs)
        for idx, row in enumerate(docs, start=1):
            if stop_requested:
                break

            did = row["dokument_id"]
            if did in done_ids:
                continue

            doc_txt = row.get("source") or row.get("document", "")
            scores: dict[str, Any] | None = None

            # retry-sløyfe
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    scores = judge_one(doc_txt, row[LABEL_A], row[LABEL_B])
                    break
                except Exception as err:
                    print(f"DokID: {did} | forsøk {attempt}/{MAX_ATTEMPTS} | FEIL: {err}")
                    if attempt == MAX_ATTEMPTS:
                        print(f"Failed: {idx:>4}/{total} {did} - ga opp")
                    else:
                        backoff = 1.8 ** attempt
                        time.sleep(backoff)

            if not scores:
                continue

            # skriv resultatet til CSV
            row_out = [idx, did]
            for c in CRITS:
                row_out += [scores[c]["A"], scores[c]["B"], scores[c]["forklaring"]]
            writer.writerow(row_out)
            out_file.flush()

            done_ids.add(did)
            print(f"{idx:>4}/{total} | Suksess: {did}")
            time.sleep(0.3)

    print(
        f"\nFerdig - {len(done_ids)} av {total} dokumenter "
        f"lagret i {CSV_OUT.relative_to(SCRIPT_DIR)}"
    )


if __name__ == "__main__":
    main()