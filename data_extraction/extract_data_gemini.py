"""
============================================================
Strukturerte data-ekstraksjon med Gemini
* Steg 1: Gemini 2.5 Flash (tools) -> strukturerte fakta per dokument
* Steg 2: Skriv til JSONL-fil med alle ekstraherte data
============================================================
"""

import csv, json, os, sys, time
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

# ---------- KONFIGURASJON ----------
# Load .env from the repository root so this works regardless of CWD
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_EXTRACT = "models/gemini-2.5-flash"

INFILE = ROOT / "case_documents_summary" / "data_raw" / "dokumenter.jsonl"
OUTJSONL = ROOT / "data_extraction" / "extracted_data_gemini.jsonl"

KOMMUNE_NAVN = {4601: "Bergen", 5501: "Tromsø", 5536: "Lyngen"}

# ---------- VERKTØY-SCHEMA (GEMINI BRUKER "parameters") ----------
# Definerer typene med en snarvei for lesbarhet
OBJECT = genai.protos.Type.OBJECT
STRING = genai.protos.Type.STRING
ARRAY = genai.protos.Type.ARRAY

SCHEMA = {
    "name": "extract_case_info",
    "description": (
        "Identifiser nøkkelopplysninger i et saksfremlegg. "
        "Trekk kun ut informasjon om selve saken – overse header/footer-metadata "
        "som saksnr, arkivsaksnr, saksbehandler, signaturer osv."
    ),
    "parameters": {
        "type": OBJECT,
        "properties": {
            "tittel":                {"type": STRING},
            "hva_saken_gjelder":     {"type": STRING},
            "foreslått_vedtak":      {"type": STRING},
            "forventede_konsekvenser":{"type": STRING},
            "viktige_hendelser":     {"type": ARRAY, "items": {"type": STRING}},
            "viktige_tidspunkter":   {"type": ARRAY, "items": {"type": STRING}},
            "viktige_personer":      {
                "type": ARRAY,
                "items": {"type": STRING},
                "description": "Ekskluder navn som bare nevnes i signatur/header (kommunedirektør, saksbehandler o.l.)"
            },
            "viktige_organisasjoner":{"type": ARRAY, "items": {"type": STRING}},
            "viktige_steder":        {"type": ARRAY, "items": {"type": STRING}},
            "tema": {
                "type": ARRAY,
                "items": {"type": STRING},
                "description": "3–10 sentrale tema/områder saken berører"
            },
        },
        "required": ["tittel", "hva_saken_gjelder", "tema"],
    },
}

extracter = genai.GenerativeModel(MODEL_EXTRACT, tools=[SCHEMA])

# ---------- HJELPEFUNKSJONER ----------
def kommune_navn(kid): return KOMMUNE_NAVN.get(int(kid or 0), "en norsk kommune")

# ---------- GEMINI-KALL ----------
def gemini_extract(text: str, kommune: str) -> dict:
    sys_msg_ex = (
        f"Du er assisterende saksredaktør i {kommune}. "
        "Ignorer header/footer-metadata og ekskluder personer som "
        "kun nevnes i administrative roller. Returner KUN JSON."
    )
    
    ex_resp = extracter.generate_content(
        [sys_msg_ex, text],
        generation_config={"temperature": 0},
        tool_config={
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": ["extract_case_info"],
            }
        },
    )

    # Sjekk om modellen faktisk kalte en funksjon
    if not ex_resp.candidates[0].content.parts[0].function_call:
        raise ValueError("Modellen kalte ikke en funksjon som forventet.")

    # Hent ut argumentene fra funksjonskallet
    args = ex_resp.candidates[0].content.parts[0].function_call.args

    # Bygg en ny, ren ordbok fra responsen
    info = {
        key: list(value)
        if type(value).__name__ == 'RepeatedComposite' else value
        for key, value in args.items()
    }
    
    return info

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
            doc_id = doc["dokument_id"]

            try:
                extracted_data = gemini_extract(text, kommune)
                
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