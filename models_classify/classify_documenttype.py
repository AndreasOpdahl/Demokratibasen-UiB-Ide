import os
import random
from typing import List
import pandas as pd
import json
from pydantic import BaseModel
import openai
from google import genai
from google.genai import types
import anthropic
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

gpt_client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

claude_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

DOCUMENTS_FOLDER_PATH = "./download_texts_from_URLS/dokument_jsons"

PROMPT_SYSTEM="""Du er dyktig og hjelpsom assistent i en avisredaksjon.\
Avisen er interessert i åpne offentlige dokumenter som beskriverpolitiske saker,\
og de trenger din hjelp til å organisere dokumentene. Du er en ekspert på å returnere strukturert data som er gitt."""

PROMPT_WITH_CODE = """
Avisen ønsker å skille mellom ulike typer politiske møtedokumenter:
Det har de allerede gjort og trenger en til type klassifisering av en kode kalt kode2
         
Denne ekstra kategorien er for hvilken type kategori dokumentet er utenom den politiske kategoriseringen.
Her er kategoriene for kode2 med tilhørende beskrivelser:

Administrativt_vedtak: Formell beslutning eller vedtak fattet av administrativ instans eller politisk organ.
Artikkel: Eksternt informasjonsdokument eller nyhetsartikkel om et spesifikt tema.
Bilag: Vedlegg som gir tilleggsinformasjon eller dokumentasjon til et hoveddokument.
Byggesak: Dokumenter knyttet til søknader, tillatelser, avslag, klager, dispensasjoner og vedtak i byggeprosjekter.
Erklæring: Formell skriftlig uttalelse eller bekreftelse, for eksempel nabosamtykke eller standpunktserklæring.
Høringsdokument: Dokumenter som inviterer til høring eller presenterer innspill, endringsforslag og sammendrag.
Internt notat: Internt arbeidsdokument eller kommunikasjon—notater, e-poster og statusoppdateringer.
Kart: Visuelle fremstillinger av geografisk informasjon som situasjonskart eller plankart.
Klage: Formell innvending eller protest mot et vedtak eller handling, inkludert klagesvar.
Handlingsplan: Dokument som beskriver mål, strategier og konkrete tiltak for et bestemt område eller prosjekt.
Internkontroll: Dokumentasjon av prosedyrer og tiltak for å sikre etterlevelse av interne krav.
Møteagenda: Oversikt over saker og temaer som skal behandles i et møte, inkludert saksliste.
Orientering: Skriftlig oppdatering eller informasjon om pågående saker til politiske eller administrative organer.
Planverk: Samlebetegnelse for reguleringsplaner, kommunedelplaner, plankart og tilhørende dokumentasjon.
Spørsmål og svar: Spørsmål fra politikere og tilhørende skriftlige svar eller redegjørelser.
Presentasjon: PowerPoint eller annet presentasjonsdokument med oppsummering av funn og anbefalinger.
Rapport: Generell rapport eller utredning—faglige, tekniske eller økonomiske analyser.
Risikoanalyse: Dokument som identifiserer, evaluerer og håndterer risikoer knyttet til et prosjekt eller område.
Saksdokument: Dokument som presenterer en sak for behandling—bakgrunn, analyse og forslag til vedtak.
Strategisk dokument: Overordnet strategi- eller styringsdokument som setter mål og retning for virksomheten.
Søknad: Formell henvendelse om tillatelse, tilskudd eller godkjenning (forhåndstilsagn, dispensasjon osv.).
Tegninger: Visuelle eller tekniske tegninger—arkitekt-, situasjons- eller diagramtegninger.
Vedtak: Formelle beslutninger fattet av kommunestyre, bystyre eller administrativt organ.
Økonomidokument: Dokumenter med økonomisk innhold—budsjetter, kostnadsberegninger, regnskap og økonomiske søknader.
Rammetillatelser: Dokumentasjon av formelle rammetillatelser for byggeprosjekter.
Dispensasjoner: Søknad om tillatelse til å avvike fra reguleringsbestemmelser eller byggegrenser.
Internasjonal: Dokumenter som omhandler kommunens internasjonale forpliktelser og samarbeid.
Digitalisering: Dokumenter som beskriver digitaliseringsstrategier, prosjekter og tiltak.

Dette er i hovedsak kategoriene som skal brukes men dersom ingen av disse passer så gjerne bruk andre koder.
Gi svaret i strukturert json format for hvilket type dokument det er (ikke politisk, kode2).
Gi svaret med json-formatet:
{{"kode2": "hvilken type dokument det er (ikke politisk type)",
"begrunnelse_kode2": "en begrunnelse for kode2}}

Dokumentet kommer her i et json format:
{{
'tittel': {title}
'fulltekst': {text}
}}

"""

class DokumenttypeKode2(BaseModel):
    kode2: str
    begrunnelse_kode2: str

def return_already_used_json(csv_path):
    used_files = []
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        dokument_id = row["dokument_id"]
        kommune = row["kommune"]
        used_files.append(f"{kommune}_{dokument_id}")
    return used_files

def select_random_json(
    folder_path: str,
    n: int,
    exclude_list: List[str] = None,
    random_state = 45
) -> List[str]:
    """
    Select n random JSON files from `folder_path`, excluding any whose
    basename (without .json) is in `exclude_list`.

    Args:
        folder_path:    Path to the directory containing .json files.
        n:              Number of files to select.
        exclude_list:   List of basenames (no extension) to skip.

    Returns:
        List of file paths to the selected JSON files.

    Raises:
        ValueError: if fewer than n candidate files are available.
    """
    random.seed(random_state)
    exclude = set(exclude_list or [])

    # Gather all .json files
    all_jsons = [
        fname for fname in os.listdir(folder_path)
        if fname.lower().endswith('.json')
    ]

    # Filter out excluded basenames
    candidates = [
        fname for fname in all_jsons
        if os.path.splitext(fname)[0] not in exclude
    ]

    if len(candidates) < n:
        raise ValueError(
            f"Requested {n} files, but only {len(candidates)} available "
            f"after excluding {exclude_list}"
        )

    # Pick n at random
    chosen = random.sample(candidates, n)

    # Return full paths
    return [os.path.join(folder_path, fname) for fname in chosen]

def return_json_dict(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def extract_json(text: str) -> str:
    """
    Extracts and returns the first JSON object (including braces) found in `text`.
    If none is found, returns an empty string.
    """
    # This regex finds a block starting at { and ending at the matching }, non-greedy.
    match = re.search(r'\{.*?\}', text, flags=re.DOTALL)
    return match.group(0) if match else ""

def classify_document(title: str, text: str, provider: str = "openai"):
    prompt = PROMPT_WITH_CODE.format(title=title, text=text)
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

# Function from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

#print(return_already_used_json("./models_classify/sample_documents.csv"))
# … (all your imports and helper definitions up above) …

def worker(provider: str, document_paths: list[str], thread_index: int) -> list[dict]:
    """
    Classify every document in document_paths with the given provider,
    save partial results after each document, and return the final list.
    """
    results = []
    #total_i = len(document_paths)
    pbar = tqdm(total=len(document_paths), desc=provider, position=thread_index)

    # Prepare this provider's output path
    out_dir = "./model_outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"kode2_{provider}.json")

    for idx, path in enumerate(document_paths, start=1):
        doc_id = os.path.splitext(os.path.basename(path))[0]
        doc = return_json_dict(path)
        try:
            raw = classify_document(doc["tittel"], doc["tekst"], provider)
            parsed = json.loads(raw)
            parsed["tittel"] = doc["tittel"]
            parsed["tekst"] = doc["tekst"]
            parsed["dokument_id"] = doc_id



            results.append(parsed)
        except Exception as e:
            error_json = {
                "kode2": "ERROR",
                "begrunnelse_kode2": str(e),
                "tittel": doc["tittel"],
                "tekst": doc["tekst"]
            }
            results.append(error_json)


        # Immediately persist the full list so far
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Update progress bar or simple print
        pbar.update(1)
        #printProgressBar(idx, total_i, f"{provider}", "Completed")
    pbar.close()
    print(f"{provider} DONE — saved {len(results)} items to {out_path}")
    return results

if __name__ == "__main__":
    used_json = return_already_used_json("./models_classify/sample_documents.csv")
    document_paths = select_random_json(DOCUMENTS_FOLDER_PATH,2000, used_json)
    # print(len(random_documents))
    # print(random_documents[4])
    # test_json = return_json_dict(random_documents[4])
    # tittel = test_json["tittel"]
    # tekst = test_json["tekst"]
    # test_respons = classify_document(tittel, tekst, "gemini")
    # print(test_respons)
    providers = ["openai", "gemini", "claude"]
    out_dir = "./model_outputs"
    os.makedirs(out_dir, exist_ok=True)

    # 2) run each provider in its own thread
    with ThreadPoolExecutor(max_workers=len(providers)) as executor:
        futures = {
            executor.submit(worker, prov, document_paths, i): prov
            for i,prov in enumerate(providers)
        }

        for fut in as_completed(futures):
            prov = futures[fut]
            try:
                result_list = fut.result()
                # 3) save to file
                out_path = os.path.join(out_dir, f"kode2_{prov}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result_list, f, ensure_ascii=False, indent=2)
                print(f"{prov} done, saved {len(result_list)} items to {out_path}")
            except Exception as e:
                print(f"{prov} failed: {e}")
