"""
============================================================
Saksfremlegg → CSV-oversikt med GEMINI-sammendrag
------------------------------------------------------------
* Steg 1 : Gemini Flash 1.5 (tools)  → strukturerte fakta
* Steg 2 : Gemini Pro 1.5 128k       → kort sammendrag
* Steg 3 : Skriv én CSV-linje
============================================================
"""

import csv, json, os, sys, time
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

# ---------- konfigurasjon ---------------------------------------------------

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_EXTRACT = "models/gemini-1.5-flash-latest"
MODEL_SUMMARY = "models/gemini-1.5-pro-latest"

ROOT   = Path(__file__).resolve().parent.parent
INFILE = ROOT / "baseline" / "baseline_documents_cleaned_first300.jsonl"
OUTCSV = ROOT / "summary" / "gemini_summary_results.csv"

KOMMUNE_NAVN = {4601: "Bergen", 5501: "Tromsø", 5536: "Lyngen"}

# ---------- tools-schema (med korrekte enum-typer) -----------------
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
                "description": "3–10 sentrale tema saken berører"
            },
        },
        "required": ["tittel", "hva_saken_gjelder", "tema"],
    },
}

extracter  = genai.GenerativeModel(MODEL_EXTRACT, tools=[SCHEMA])
summariser = genai.GenerativeModel(MODEL_SUMMARY)

# ---------- hjelpefunksjoner -------------------------------------------------
def kommune_navn(kid): return KOMMUNE_NAVN.get(int(kid or 0), "en norsk kommune")
def listify(v): return [] if v is None else (v if isinstance(v, list) else [str(v)])

# ---------- hovedløp ---------------------------------------------------------
def main() -> None:
    if not INFILE.exists():
        sys.exit(f"Fant ikke {INFILE}")

    OUTCSV.parent.mkdir(parents=True, exist_ok=True)

    with INFILE.open(encoding="utf-8") as fin, \
         OUTCSV.open("w", encoding="utf-8", newline="") as fout:

        writer = csv.writer(fout)
        writer.writerow(["dokument_id", "tittel", "sammendrag", "personer", "tema"])

        for ln in fin:
            doc = json.loads(ln)
            text = doc.get("tekst_cleaned") or doc.get("tekst") or ""
            if not text.strip():
                continue

            kommune = kommune_navn(doc.get("kommune"))

            try:
                # =================================================================
                # Steg 1 : element-ekstraksjon
                # =================================================================
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

                # =================================================================
                # Steg 2 : sammendrag
                # =================================================================
                person_liste = info.get("viktige_personer", []) or []
                tema_liste = info.get("tema", []) or []
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
                    "Skriv kun fakta som støttes av dokumentteksten."
                )
                sum_resp = summariser.generate_content(
                    [sys_msg_sum, prompt],
                    generation_config={"temperature": 0.1, "max_output_tokens": 800},
                )
                summary = sum_resp.text.strip()

            except Exception as e:
                print("Hoppet over", doc["dokument_id"], "→", e)
                time.sleep(2)
                continue

            # Skriv til CSV KUN hvis alt i try-blokken var vellykket
            writer.writerow([
                doc["dokument_id"],
                info.get("tittel", "").replace("\n", " ").strip(),
                summary.replace("\n", " ").strip(),
                ", ".join(person_liste),
                ", ".join(tema_liste),
            ])
            print("Suksess:", doc["dokument_id"])
            time.sleep(1)

    print(
        f"\nFerdig – Geminisammendrag lagret i "
        f"{OUTCSV.relative_to(Path.cwd())}"
    )


if __name__ == "__main__":
    main()