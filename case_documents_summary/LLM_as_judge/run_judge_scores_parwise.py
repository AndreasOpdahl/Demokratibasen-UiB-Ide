import json, time, pathlib, csv, os
import openai
from dotenv import load_dotenv

# ── konfig ────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL        = "gpt-4o"
TEMPERATURE  = 0
ROOT         = pathlib.Path(__file__).resolve().parent
IN_PATH      = ROOT / "judge_input_openai_vs_gemini.jsonl"
CSV_OUT      = ROOT / "judge_scores_openai_vs_gemini.csv"

LABEL_A = "openai"   # Sett ønsket label
LABEL_B = "gemini"     # Sett ønsket label 

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
  mulig *nyhetsverdi*, for eksempel  
    – **Hypernærhet** (geografisk / emosjonell nærhet)  
    – **Lokale konsekvenser** (innvirkning på dagliglivet)  
    – **Samfunnsengasjement** (demokratisk deltakelse)  
    – **Løsningsorientering** (hvordan lokalsamfunn løser problemer)  
    – **Lokal ansvarlighet og åpenhet** (myndighetskontroll)  
    – **Lokal betydning** (identitet, historie, kultur)  
    – **Økonomisk og sosial samhørighet** (utvikling, livskvalitet)  
    – **Kriselederskap / resiliens** (håndtering av kriser)  
    – **Deltakende og interaktivt engasjement** (lokale innspill i prosesser)

[Vi er bare interessert i selve saksinnholdet – ignorer prosedyremetadata.]

============================================================
OPPGAVE
------------------------------------------------------------
1. Gi en NUMERISK poengsum (1–5) til hvert sammendrag for hvert kriterium.
2. For hvert kriterium, gi én kort begrunnelse (1–2 setninger).
   Begrunnelsen skal forklare hvorfor du ga forskjellig poengsum.

SVARFORMAT (gyldig JSON – ingen annen tekst):
{{
  "koherens":   {{"A": <int>, "B": <int>, "forklaring": "<tekst>"}},
  "konsistens": {{"A": <int>, "B": <int>, "forklaring": "<tekst>"}},
  "flyt":     {{"A": <int>, "B": <int>, "forklaring": "<tekst>"}},
  "relevans":   {{"A": <int>, "B": <int>, "forklaring": "<tekst>"}},
}}
"""

def llm_score(doc, a, b):
    rsp = openai.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TMPL.format(
                source=doc, summary_a=a, summary_b=b)}
        ]
    )
    return json.loads(rsp.choices[0].message.content)

def main():
    with IN_PATH.open(encoding="utf-8") as fin, \
         CSV_OUT.open("w", encoding="utf-8", newline="") as cout:

        writer = csv.writer(cout)
        # Dynamiske kolonnenavn basert på labels
        header = (
            ["index", "dokument_id"] +
            [f"{crit}_{LABEL_A}" for crit in ["koherens", "konsistens", "flyt", "relevans"]] +
            [f"{crit}_{LABEL_B}" for crit in ["koherens", "konsistens", "flyt", "relevans"]]
        )
        writer.writerow(header)

        for idx, line in enumerate(fin, start=1):
            row = json.loads(line)
            doc_id = row["dokument_id"]

            doc_txt = row.get("document") or row.get("source")
            try:
                scores = llm_score(doc_txt,
                                   row[LABEL_A],
                                   row[LABEL_B])
            except Exception as e:
                print("Failed:", doc_id, e)
                continue

            writer.writerow(
                [idx, doc_id] +
                [scores["koherens"]["A"],   scores["konsistens"]["A"], scores["flyt"]["A"], scores["relevans"]["A"]] +
                [scores["koherens"]["B"],   scores["konsistens"]["B"], scores["flyt"]["B"], scores["relevans"]["B"]]
            )
            print(f"{idx} | Suksess: {doc_id}")
            time.sleep(0.5)

    print("Ferdig. Resultat lagret i:", CSV_OUT.name)

if __name__ == "__main__":
    main()
