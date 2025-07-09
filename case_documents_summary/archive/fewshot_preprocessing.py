"""
Few-shot-rensing av saksfremlegg, fjerner metadata/signatur.
- Leser baseline_case_documents.jsonl
- Bruker GPT-4o-mini med dynamiske batcher (~80 % av 128 k)
- Skriver case_documents_cleaned_fewshot.jsonl
- Eksemplene hentes fra fewshot_examples.jsonl
"""
from __future__ import annotations
import json, os, time, pathlib, openai, tiktoken

from dotenv import load_dotenv
load_dotenv()

# Konfig
MODEL          = "gpt-4o-mini"
WINDOW         = 128_000
FILL_RATIO     = 0.80
TOK_LIMIT      = int(WINDOW * FILL_RATIO)         # ≈ 102 k tokens
RESERVE_OUTPUT = 8_000                            
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

INFILE       = SCRIPT_DIR.parent / "baseline" / "baseline_documents_test.jsonl"
EXAMPLE_FILE = SCRIPT_DIR / "fewshot_examples.jsonl"
OUTFILE      = SCRIPT_DIR / "baseline_case_documents_cleaned_fewshot2.jsonl"


openai.api_key = os.getenv("OPENAI_API_KEY")      
enc            = tiktoken.encoding_for_model(MODEL)
tok            = lambda s: len(enc.encode(s))     # kort alias


# Prompts
SYSTEM_PROMPT = (
    "Du er en grunndig og konservativ tekstredigerer som skal fjerne alle deler av teksten som kun er metadata om "
    "behandlingen av en sak. Metadata kan for eksempel være når saken legges frem, hvem som har forberedt den, "
    "hvem som behandler den videre. Behold bare selve sakens innhold."
)

USER_INSTRUCTION = (
    "Fjern kun tekst som ligger helt i starten eller helt på slutten av dokumentet, og som er ren metadata knyttet til behandling eller publisering av saken.\n\n"
    "Fjern typisk:\n"
    "- Arkivsaksnummer og dokumentkoder\n"
    "- Saksbehandler, kontaktinfo og signaturblokker\n"
    "- Dato-linjer og saksgang\n"
    "- Postadresser, organisasjonsnummer, elektroniske godkjenninger\n"
    "- Vedleggslister og lenker\n\n"
    "- Signaturblokker (f.eks. navn og tittel)\n"
    "Ikke fjern metadata som står midt i dokumentet – det skal beholdes.\n\n"
    "Ikke fjern noe som handler om selve saken, vurderinger, forslag eller vedtak. Vær svært forsiktig: behold alt innhold som kan være relevant for forståelse av saken.\n\n"
    "Returner et JSON-array på formatet:\n"
    "[{\"dokument_id\": ..., \"cleaned\": ...}]"
)

# few-shot-eksempler 
example_msgs: list[dict] = []
with EXAMPLE_FILE.open(encoding="utf-8") as fh:
    for ln in fh:
        ex = json.loads(ln)
        example_msgs += [
            {
                "role": "user",
                "content": json.dumps(
                    {"dokument_id": ex["dokument_id"], "raw": ex["raw"]},
                    ensure_ascii=False,
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {"dokument_id": ex["dokument_id"], "cleaned": ex["cleaned"]},
                    ensure_ascii=False,
                ),
            },
        ]

FIXED_TOKENS = (
    tok(SYSTEM_PROMPT)
    + tok(USER_INSTRUCTION)
    + sum(tok(m["content"]) for m in example_msgs)
    + 20
)

# batching 
def batch_iter():
    batch, used = [], FIXED_TOKENS
    with INFILE.open(encoding="utf-8") as fh:
        for ln in fh:
            doc = json.loads(ln)
            size = tok(doc["tekst"]) + 30
            if used + size + RESERVE_OUTPUT > TOK_LIMIT and batch:
                yield batch
                batch, used = [], FIXED_TOKENS
            batch.append(doc)
            used += size
        if batch:
            yield batch

# OpenAI-kall
def call_llm(batch):
    payload = [{"dokument_id": d["dokument_id"], "raw": d["tekst"]} for d in batch]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_INSTRUCTION},
        {"role": "user", "content": "Her er noen eksempler på hvordan teksten skal renses:"},
    ]

    messages += example_msgs

    messages.append({
        "role": "user",
        "content": "Nå, rens følgende dokumenter:\n"
                   + json.dumps(payload, ensure_ascii=False)
    })

    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
    )

    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print("JSON-feil ved dekoding av modellens svar:")
        print(content)
        raise e

# hovedløp 
OUTFILE.parent.mkdir(parents=True, exist_ok=True)
processed = 0

with OUTFILE.open("w", encoding="utf-8") as fout:
    for batch in batch_iter():
        while True:
            try:
                resp = call_llm(batch)
                break
            except Exception as err:
                print("API-feil, nytt forsøk om 10 s:", err)
                time.sleep(10)

        data = resp

        #  PATCH: håndter {"documents":[...]} 
        if isinstance(data, dict) and "documents" in data:
            data = data["documents"]

        for item in data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

        processed += len(batch)
        print(f"✔ {processed:,} dokumenter renset")

print("Ferdig resultat:", OUTFILE)