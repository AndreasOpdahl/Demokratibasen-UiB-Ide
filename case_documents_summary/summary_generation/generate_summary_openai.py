"""
============================================================
CSV-oversikt med OPENAI-sammendrag
* Steg 1 : GPT-4o-mini(tool)  -> strukturerte fakta   (case_info)
* Steg 2 : GPT-4o(chat)  -> kort sammendrag 
* Steg 3 : Skriv én CSV-linje
============================================================
"""

import csv, json, os, re, sys, time
from collections import OrderedDict
from pathlib import Path

import openai
from dotenv import load_dotenv

# ---------- KONFIGURASJON ----------
ROOT   = Path(__file__).resolve().parent[2]
load_dotenv(ROOT / ".env")

openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL_EXTRACT = "gpt-4o-mini"
MODEL_SUMMARY = "gpt-4o"

INFILE = ROOT / "baseline" / "baseline_case_documents_cleaned_first300.jsonl"
OUTCSV = ROOT / "summary" / "openai_summary_results.csv"

KOMMUNE_NAVN = {4601: "Bergen", 5501: "Tromsø", 5536: "Lyngen"}

# ---------- VERKTØY-SCHEMA ----------
SCHEMA = {
    "name": "extract_case_info",
    "description": (
        "Identifiser nøkkelopplysninger i et saksfremlegg. "
        "Trekk kun ut informasjon om selve saken, overse administrativ metadata som "
        "ofte forekommer på starten og slutten av dokumentet. E.g. header-info som "
        "saksnr, arkivsaksnr, saksbehandler, signaturer osv."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "tittel":                {"type": "string"},
            "hva_saken_gjelder":     {"type": "string"},
            "foreslått_vedtak":      {"type": "string", "description": "kan være tom"},
            "forventede_konsekvenser":{"type": "string", "description": "kan være tom"},
            "viktige_hendelser":     {"type": "array", "items":{"type":"string"}},
            "viktige_tidspunkter":   {"type": "array", "items":{"type":"string"}},
            "viktige_personer":      {"type": "array", "items":{"type":"string"},
                                      "description": "Ekskluder navn som kun fremkommer i signaturer /"
                                      "dokument-headeren med roller som «kommunedirektør» og «saksbehandler», "
                                      "med mindre teksten etterpå viser at personen faktisk har "
                                      "en aktiv rolle i saken."},
            "viktige_organisasjoner":{"type": "array", "items":{"type":"string"}},
            "viktige_steder":        {"type": "array", "items":{"type":"string"}, "description": "kan være tom"},
            "tema": {
                "type": "array",
                "items": {"type":"string"},
                "description": "3-10 sentrale tema/områder saken berører"
            }
        },
        "required": ["tittel", "hva_saken_gjelder", "tema"]
    }
}

TOOL_DEF    = {"type": "function", "function": SCHEMA}
TOOL_CHOICE = {"type": "function", "function": {"name": "extract_case_info"}}

# ---------- HJELPEFUNKSJONER ----------

def kommune_navn(kid) -> str:
    try:
        return KOMMUNE_NAVN[int(kid)]
    except Exception:
        return "en norsk kommune"

def listify(val):
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [str(val)]

def _sanitise_json_string(raw: str) -> str:
    """
    Returnerer 'raw' der
    - alle kontrolltegn (< 0x20) inne i strenger er escaped
    - et "nakent" dobbelt-anførselstegn inne i en streng skrives om til \"
        (gjelder også sekvensen "").
    """
    out, in_str, esc = [], False, False
    it = iter(enumerate(raw))
    for i, ch in it:
        if esc:                       # forrige tegn var '\'
            out.append(ch)
            esc = False
            continue

        if ch == '\\':                # start av escape
            out.append(ch)
            esc = True
            continue

        if ch == '"':                 # anførselstegn
            if in_str:
                # Sjekk om dette egentlig er et "" (ulovlig) som betyr "
                nxt = raw[i + 1] if i + 1 < len(raw) else ''
                if nxt == '"':        # fant ""
                    out.extend(['\\', '"'])      # legg inn \"
                    next(it)                     # hopp over andre "
                    continue
            in_str = not in_str                  # toggl streng-modus
            out.append(ch)
            continue

        if in_str and ord(ch) < 0x20:           # kontrolltegn i streng
            if ch == '\t':
                out.extend(['\\', 't'])
            elif ch == '\n':
                out.extend(['\\', 'n'])
            elif ch == '\r':
                out.extend(['\\', 'r'])
            else:
                out.extend(['\\', 'u', *f'{ord(ch):04x}'])
        else:
            out.append(ch)

    return ''.join(out)

def iter_json_objects(handle):
    buf = ""
    for raw in handle:
        if not raw.strip():
            continue
        buf += raw
        try:
            clean = _sanitise_json_string(buf)
            yield json.loads(clean)
            buf = ""                       # klar for neste objekt
        except json.JSONDecodeError:
            # posten er ikke komplett ennå – les neste linje
            continue

    if buf.strip():
        raise ValueError("Siste JSON-post er ufullstendig")

# ---------- GPT-KALL ----------
def gpt_extract(text: str, kommune: str) -> dict:
    resp = openai.chat.completions.create(
        model       = MODEL_EXTRACT,
        messages    = [
            {
                "role": "system",
                "content": (
                    f"Du er assisterende saksredaktør i {kommune}. "
                    "Ignorere header-metadata og å ekskludere personer som kun nevnes i administrative roller. "
                    "Navn som ikke står i listen {viktige_personer} skal ikke nevnes i sammendraget.\n"
                    "Returner KUN JSON som følger schemaet."
                )
            },
            {"role": "user", "content": text}
        ],
        tools          = [TOOL_DEF],
        tool_choice    = TOOL_CHOICE,
        response_format= {"type": "json_object"},   
        temperature    = 0.1,
        max_tokens     = 4096                       
    )

    raw_json = resp.choices[0].message.tool_calls[0].function.arguments
    return json.loads(raw_json)

def gpt_summary(case_info: dict, kommune: str) -> str:
    news_values = (
        "Nyhetsverdi-eksempler: hypernærhet, lokale konsekvenser, samfunnsengasjement, "
        "løsningsorientering, lokal ansvarlighet, lokal betydning, økonomisk/sosial samfunnsnytte, "
        "kriselederskap, deltakende engasjement.\n\n"
    )
    prompt = (
        f"Du er journalist i en lokalavis som dekker {kommune}. "
        "Lag et sammendrag på maks én ingress + syv setninger.\n"
        "- Ingen punktlister eller parenteser\n"
        "- Ikke nevne andre personer enn: {allowed_names}\n"
        "- Ikke legg til fakta, roller eller konsekvenser som ikke støttes av dokumentteksten.\n"
        f"Nyhetsverdi kan vektlegges når relevant. {news_values}"
        "Nøkkelopplysninger som må bevares:\n"
        f"{json.dumps(case_info, ensure_ascii=False, indent=2)}"
    )
    rsp = openai.chat.completions.create(
        model       = MODEL_SUMMARY,
        messages    = [
            {"role": "system", "content": "Du skriver korte, nøkterne sammendrag. "
            "Du bryr deg kun om informasjon om selve saken, og overser administrativ metadata."
            "Skriv kun fakta som støttes av dokumentteksten. Ikke gjør antakelser."},
            {"role": "user",   "content": prompt}
        ],
        temperature = 0.1,
        max_tokens  = 800
    )
    return rsp.choices[0].message.content.strip()

# ---------- HOVEDLØP ----------
def main():
    if not INFILE.exists():
        sys.exit(f"Fant ikke {INFILE}")

    with INFILE.open(encoding="utf-8") as fin, \
         OUTCSV.open("w", encoding="utf-8", newline="") as fout:

        writer = csv.writer(fout)
        writer.writerow(["dokument_id", "tittel", "sammendrag", "personer", "tema"])

        for doc in iter_json_objects(fin):
            doc_id  = doc.get("dokument_id", "")
            kommune = kommune_navn(doc.get("kommune"))
            text    = doc.get("tekst_cleaned") or doc.get("tekst") or ""
            if not text.strip():
                continue

            try:
                info = gpt_extract(text, kommune)

                # sørg for at felt som skal joines er lister
                info["viktige_personer"] = listify(info.get("viktige_personer"))
                info["tema"]             = listify(info.get("tema"))

                summ = gpt_summary(info, kommune)

            except Exception as e:
                print("Hoppet over", doc_id, e)
                time.sleep(2)
                continue

            writer.writerow([
                doc_id,
                info["tittel"].replace("\n", " ").strip(),
                summ.replace("\n", " ").strip(),
                ", ".join(info["viktige_personer"]),
                ", ".join(info["tema"])
            ])
            print("Suksess:", doc_id)

    print(f"\nFerdig. Sammendrag lagret i {OUTCSV.relative_to(Path.cwd())}")

if __name__ == "__main__":
    main()