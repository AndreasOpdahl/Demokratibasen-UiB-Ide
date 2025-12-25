"""
============================================================
Strukturerte data-ekstraksjon med Claude
* Steg 1: Claude-4 Sonnet (tools) -> strukturerte fakta per dokument
* Steg 2: Skriv til JSONL-fil med alle ekstraherte data
============================================================
"""

import csv, json, os, sys, time
from pathlib import Path

import anthropic        
from dotenv import load_dotenv

# ---------- KONFIGURASJON ----------
# Load .env from the repository root so this works regardless of CWD
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Expect the standard ANTHROPIC_API_KEY env var
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL_EXTRACT = "claude-sonnet-4-20250514"

# moved to datasets_from_demokratibasen/data_collection/summaries_keywords_newsworthiness/202505-and-06-Demokratibasen-demo/36812-demokratibasen-texts-20250528.jsonl
INFILE = ROOT / "case_documents_summary" / "data_raw" / "dokumenter.jsonl"
OUTJSONL = ROOT / "data_extraction" / "extracted_data_claude.jsonl"

KOMMUNE_NAVN = {4601: "Bergen", 5501: "Tromsø", 5536: "Lyngen"}

# ---------- VERKTØY-SCHEME (ANTHROPIC BRUKER "input_schema") ----------
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
            "forslag_til_vedtak":      {"type": "string"},
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

# ---------- CLAUDE-KALL ----------
def claude_extract(text: str, kommune: str) -> dict:
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
    return invocation.input  # dict med feltene fra schema

# ---------- HOVEDLØP ----------
def main() -> None:
    if not INFILE.exists():
        sys.exit(f"Fant ikke {INFILE}")

    OUTJSONL.parent.mkdir(parents=True, exist_ok=True)

    with INFILE.open(encoding="utf-8") as fin, \
         OUTJSONL.open("w", encoding="utf-8") as fout:

        for ln in fin:
            if not ln.strip():
                continue
            
            doc = json.loads(ln)
            text = doc.get("tekst") or ""
            if not text.strip():
                continue

            kommune = kommune_navn(doc.get("kommune"))
            doc_id  = doc["dokument_id"]

            try:
                extracted_data = claude_extract(text, kommune)
                
                # Legg til metadata fra originalt dokument
                output_record = {
                    "dokument_id": doc_id,
                    "kommune": doc.get("kommune"),
                    "kommune_navn": kommune,
                    "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "extraction_model": MODEL_EXTRACT,
                    "extracted_data": extracted_data
                }
                
                json.dump(output_record, fout, ensure_ascii=False)
                fout.write("\n")
                print("Suksess:", doc_id)

            except Exception as e:
                print("Hoppet over", doc_id, "→", e)
                time.sleep(2)
                continue

    print(f"\nFerdig. Ekstraherte data lagret i {OUTJSONL.relative_to(Path.cwd())}")

if __name__ == "__main__":
    main()