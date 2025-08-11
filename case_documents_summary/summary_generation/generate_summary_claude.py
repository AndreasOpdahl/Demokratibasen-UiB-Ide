"""
============================================================
CSV-oversikt med CLAUDE-sammendrag
* Steg 1 : Claude-4 Sonnet (tools)  -> strukturerte fakta
* Steg 2 : Claude-4 Sonnet          -> kort sammendrag
* Steg 3 : Skriv en CSV-linje
============================================================
"""

import csv, json, os, sys, time
from pathlib import Path

import anthropic        
from dotenv import load_dotenv

# ---------- KONFIGURASJON ----------
# Load .env from the repository root so this works regardless of CWD
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# Expect the standard ANTHROPIC_API_KEY env var
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL_EXTRACT = "claude-sonnet-4-20250514"
MODEL_SUMMARY = "claude-sonnet-4-20250514"

INFILE = ROOT / "cleaning_preprocessing" / "baseline_documents_cleaned_first300.jsonl"
OUTCSV = ROOT / "summary_generation" / "claude_summary_results.csv"

KOMMUNE_NAVN = {4601: "Bergen", 5501: "Tromsø", 5536: "Lyngen"}
# ---------- VERKTØY-SCHEME (ANTHORPIC BRUKER "input_schema") ----------

SCHEMA = {
    "name": "extract_case_info",
    "description": (
        "Identifiser nøkkelopplysninger i et saksfremlegg. "
        "Trekk KUN ut informasjon om selve saken – overse header/footer-metadata "
        "som saksnr, arkivsaksnr, saksbehandler, signaturer osv."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tittel":                 {"type": "string"},
            "hva_saken_gjelder":      {"type": "string"},
            "foreslått_vedtak":      {"type": "string"},
            "forventede_konsekvenser":{"type": "string"},
            "viktige_hendelser":      {"type": "array", "items": {"type": "string"}},
            "viktige_tidspunkter":    {"type": "array", "items": {"type": "string"}},
            "viktige_personer": {
                "type": "array", "items": {"type": "string"},
                "description": "Ekskluder navn som bare nevnes i signatur/header"
            },
            "viktige_organisasjoner": {"type": "array", "items": {"type": "string"}},
            "viktige_steder":         {"type": "array", "items": {"type": "string"}},
            "tema": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3–10 sentrale tema/områder saken berører"
            },
        },
        "required": ["tittel", "hva_saken_gjelder", "tema"],
    },
}

# ---------- HJELPEFUNKSJONER ----------
def kommune_navn(kid): return KOMMUNE_NAVN.get(int(kid or 0), "en norsk kommune")
def listify(v): return [] if v is None else (v if isinstance(v, list) else [str(v)])

# ---------- HOVEDLØP ----------
def main() -> None:
    if not INFILE.exists():
        sys.exit(f"Fant ikke {INFILE}")

    OUTCSV.parent.mkdir(parents=True, exist_ok=True)

    with INFILE.open(encoding="utf-8") as fin, \
         OUTCSV.open("w", encoding="utf-8", newline="") as fout:

        writer = csv.writer(fout)
        writer.writerow(["dokument_id", "tittel", "sammendrag",
                         "personer", "tema"])

        for ln in fin:
            doc = json.loads(ln)
            text = doc.get("tekst_cleaned") or doc.get("tekst") or ""
            if not text.strip():
                continue

            kommune = kommune_navn(doc.get("kommune"))
            doc_id  = doc["dokument_id"]

            try:
                # Steg 1 : Claude-tools → element-ekstraksjon
                sys_msg_ex = (
                    f"Du er assisterende saksredaktør i {kommune}. "
                    "Ignorer administrativ metadata og ekskluder personer som "
                    "kun nevnes i signaturer eller header-informasjon. "
                    "Returner KUN JSON som følger schemaet."
                )
                ex_resp = client.messages.create(
                    model=MODEL_EXTRACT,
                    temperature=0,
                    max_tokens=4096,
                    system=sys_msg_ex,
                    messages=[{"role": "user", "content": text}],
                    tools=[SCHEMA],
                    tool_choice={"type": "tool", "name": "extract_case_info"},
                )

                invocation = ex_resp.content[0]
                info = invocation.input  # dict med feltene fra schema

                # Steg 2 : skriv kort sammendrag
                person_liste = listify(info.get("viktige_personer"))
                tema_liste   = listify(info.get("tema"))
                allowed_names = ", ".join(person_liste) or "ingen"

                news_values = (
                    "Nyhetsverdi-eksempler: hypernærhet, lokale konsekvenser, "
                    "samfunnsengasjement, løsningsorientering, lokal ansvarlighet, "
                    "lokal betydning, økonomisk/sosial samfunnsnytte, kriselederskap, "
                    "deltakende engasjement."
                )
                prompt = (
                    f"Du er journalist i en lokalavis som dekker {kommune}. "
                    "Lag et sammendrag på maks én ingress + syv setninger.\n"
                    "- Ingen punktlister eller parenteser\n"
                    f"- Ikke nevne andre personer enn: {allowed_names}\n"
                    "- Ikke legg til fakta, roller eller konsekvenser som ikke "
                    "støttes av dokumentteksten.\n"
                    f"Nyhetsverdi kan vektlegges når relevant. {news_values}\n\n"
                    "Nøkkelopplysninger som må bevares:\n"
                    f"{json.dumps(info, ensure_ascii=False, indent=2)}"
                )
                sys_msg_sum = (
                    "Du skriver korte, nøkterne sammendrag. Du bryr deg kun om "
                    "informasjon om selve saken og overser administrativ metadata. "
                    "Skriv KUN fakta som støttes av dokumentteksten."
                )

                sum_resp = client.messages.create(
                    model=MODEL_SUMMARY,
                    temperature=0.1,
                    max_tokens=800,
                    system=sys_msg_sum,
                    messages=[{"role": "user", "content": prompt}],
                )
                summary = sum_resp.content[0].text.strip()
                summary = summary.replace("**", "")

            except Exception as e:
                print("Hoppet over", doc_id, "→", e)
                time.sleep(2)
                continue

            # skriv til CSV 
            writer.writerow([
                doc_id,
                info.get("tittel", "").replace("\n", " ").strip(),
                summary.replace("\n", " ").strip(),
                ", ".join(person_liste),
                ", ".join(tema_liste),
            ])
            print("Suksess:", doc_id)
            time.sleep(1)

    print(f"\nFerdig – Claudesammendrag lagret i "
          f"{OUTCSV.relative_to(Path.cwd())}")

if __name__ == "__main__":
    main()