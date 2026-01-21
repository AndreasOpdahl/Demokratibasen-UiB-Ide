import openai   
import os
import json
import pandas as pd
from pydantic import BaseModel
from google import genai
from google.genai import types
import anthropic
import re

from dotenv import load_dotenv

load_dotenv()
CSV_FILE_PATH = './models_classify/labelling_sheet_summarized_data_validation.csv'

validation_set = pd.read_csv(CSV_FILE_PATH)

gpt_client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

claude_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


class DokumenttypeKode1(BaseModel):
    kode1: str
    begrunnelse_kode1: str

class DokumenttypeKode2(BaseModel):
    kode2: str
    begrunnelse_kode2: str

# Prompt templates


PROMPT_SYSTEM="""Du er dyktig og hjelpsom assistent i en avisredaksjon.\
Avisen er interessert i åpne offentlige dokumenter som beskriverpolitiske saker,\
og de trenger din hjelp til å organisere dokumentene. Du er en ekspert på å returnere strukturert data som er gitt."""

PROMPT_KODE1 = '''* Avisen ønsker å skille mellom følgende typer politiske møtedokumenter:

     Møteinnkalling: Formell innkalling til et spesifikt politisk møte
som sendes til organets medlemmer. Innkallingen inneholder liste over
sakene som skal behandles.
     Saksliste: Nummerert oversikt over alle saker som skal behandles i
et spesifikt møte.
     Møteplan: Overordnet årsplan som viser alle planlagte møtedatoer
for politiske organer. Vedtas av kommunestyret som et strategisk
planleggingsdokument.
     Saksfremlegg: Administrasjonens utredning av en sak som skal
behandles politisk. Innholder den faktiske og rettslige analysen som gir
grunnlag for beslutning.
     Innstilling: Administrasjonens konkrete anbefaling til det
politiske organet om vedtak i en sak basert på saksframlegget.
     Sakshistorikk: Administrasjonens oversikt over historikken i en sak
som har blitt behandlet over lengre tid.
     Møteprotokoll: Lovpålagt formell dokumentasjon av møtet. Også kalt
møtebok.
     Møtereferat: Uformelt sammendrag av møtets innhold og beslutninger.
Mens protokollen er en fullstendig juridisk dokumentasjon, er referatet
et mer praktisk arbeidsverktøy som fokuserer på konklusjoner og oppfølging.
     Saksprotokoll: Formell dokumentasjon av politisk vedtak fattet
direkte av folkevalgte organer i møter.
     Saksreferat: Uformelt sammendrag av behandlingen av en sak.
Referatet er et praktisk arbeidsverktøy som fokuserer på konklusjoner og
oppfølging.
     Vedlegg: Er dokumenter som inneholder relevant bakgrunnmateriale
for behandling av en sak.

Gi en kort begrunnelse på hvilken type møtedokument dette er.

[Dette er særlig aktuelt for saksfremlegg, -protokoller og -referater.
Kanskje også vedlegg.]
         
Gi svaret i strukturert json format for hvilket type dokument de er(kode1).
Gi svaret med json-formatet:
{{"kode1": "hvilken type politisk møtedokument det er",
"begrunnelse_kode1": "en begrunnelse for kode1",}}

Dokumentet kommer her i et json format:
{{
'tittel': {title}
'fulltekst': {text}
}}
'''

PROMPT_KODE2 = '''* Avisen ønsker å skille mellom ulike typer politiske møtedokumenter:
Det har de allerede gjort og trenger en til type klassifisering av en kode kalt kode2
         
Denne ekstra kategorien for hvilken type dokumentet er, utenom den politiske kategoriseringen.
Noen av disse kategoriene inkluderer:

Reguleringsplaner, 
klager, 
interne notater, 
tekniske rapporter, 
foto, 
rammetillatelser, 
høringsdokumenter, 
øvrige søknader, 
møteplan, 
økonomiske rapporter, 
økonomiske søknader, 
erklæringer, 
kart, 
budsjett, 
inngående brev, 
søknad, 
kunngjøringer, 
revisjonsrapporter, 
administrativt 
vedtak, 
dispensasjoner, 
spørsmål fra/til politikere, 
tegninger, 
samarbeidsavtale, 
forslag, 
byggesaker

Dette er bare noen eksempler men gjerne bruk andre koder dersom du føler det er mer passende.
Gi svaret i strukturert json format for hvilket type dokument det er (ikke politisk, kode2).
Gi svaret med json-formatet:
{{"kode2": "hvilken type dokument det er (ikke politisk type)",
"begrunnelse_kode2": "en begrunnelse for kode2}}

Dokumentet kommer her i et json format:
{{
'tittel': {title}
'fulltekst': {text}
}}
'''

def extract_json(text: str) -> str:
    """
    Extracts and returns the first JSON object (including braces) found in `text`.
    If none is found, returns an empty string.
    """
    # This regex finds a block starting at { and ending at the matching }, non-greedy.
    match = re.search(r'\{.*?\}', text, flags=re.DOTALL)
    return match.group(0) if match else ""

def classify(title: str, text: str, task: str, provider: str = "openai") -> dict:
    # pick prompt + schema
    if task == "kode1":
        prompt = PROMPT_KODE1.format(title=title, text=text)
        schema = DokumenttypeKode1
    else:
        prompt = PROMPT_KODE2.format(title=title, text=text)
        schema = DokumenttypeKode2

    if provider == "openai":
        resp = gpt_client.responses.parse(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            text_format=schema,
        )
        json_response = resp.output_text
    elif provider == "gemini":
        # Gemini Flash 2.5 call
        gem_resp = google_client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=PROMPT_SYSTEM,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=schema, 
            ),
            contents=prompt
        )
        json_response = gem_resp.text
    elif provider == "claude":
        # Claude 3.5 Haiku: prompt the system then the user, ask for JSON only
        claude_resp = claude_client.messages.create(
            model="claude-3-5-haiku-20241022",
            system=PROMPT_SYSTEM,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        full_response = claude_resp.content[0].text
        json_response = extract_json(full_response)
    return json_response

def return_json_dict(json_str):
    dict_data = json.loads(json_str)
    return dict_data

def process_dataframe(df: pd.DataFrame, model_string) -> pd.DataFrame:
    results_kode1 = []
    results_kode2 = []
    for index, row in df.iterrows():
        title, text = row["tittel"], row["fulltekst"]
        output1_json = classify(title, text, task="kode1", provider=model_string)
        output2_json = classify(title, text, task="kode2", provider=model_string)
        output1_dict = return_json_dict(output1_json)
        output2_dict = return_json_dict(output2_json)
        results_kode1.append(output1_dict["kode1"])
        results_kode2.append(output2_dict["kode2"])
        print(f"Processed row: {index}")
        print(f"Processed: {title[:30]}…")

    df[f"kode1_{model_string}"] = results_kode1
    df[f"kode2_{model_string}"] = results_kode2
    return df

if __name__ == "__main__":
    # Usage:
    df = pd.read_csv(CSV_FILE_PATH)
    df = process_dataframe(df, "claude")
    df.to_csv("labelled_with_claude.csv", index=False)
