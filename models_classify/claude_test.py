import os
import json
from google import genai
import openai
import anthropic
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel
from structured_documenttype_class import extract_json

load_dotenv()
class DokumenttypeKode2(BaseModel):
    kode2: str
    begrunnelse_kode2: str

class Kode2Liste(BaseModel):
    liste: list[DokumenttypeKode2]


PROMPT_SYSTEM = """Du er en ekspert på å oppsummere kategorier. 
Du får inn en liste med par `[("<en kategori>", "<kort forklaring>")]`. Den korte forklaringen er en forklaring på hvorfor denne koden ble brukt tidligere.
"""
MODEL_PROMPT = """
Her er dataen som er en liste med par med kategori og en forklaring på hvorfor den ble brukt for et dokument:
{pairs}

**Din oppgave**  
- Lag cirka 40 `kode2`-elementer som sammen gir en god dekningsgrad og dekker bredden i dataene.  
- Lag en forklaring til hva denne koden betyr med `forklaring` til hvert valgt `kode`.  
- Output skal være **ren JSON**, en liste av objekter i samme format, slik:

```json
[
  {{ "kode2": "…", "forklaring": "…" }},
  {{ "kode2": "…", "forklaring": "…" }},
  …
]
"""
def create_unique_code(dict_list) -> set:
    kode2_list = [e["kode"] for e in dict_list]
    unique = set(kode2_list)
    return unique

#Function that finds the first "begrunnelse" for a "kode"
def find_begrunnelse(model_list, code):
    for d in model_list:
        if d["kode"] == code:
            return d["begrunnelse"]

def create_kode_begrunnelse_pair(unique_list, model_list):
    unique_begrunnelse = [(unique_kode, find_begrunnelse(model_list, unique_kode)) for unique_kode in unique_list]
    return unique_begrunnelse

gpt_client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

claude_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

def get_kode2_from_model(model, kode_begrunnelse_pair_list):
    prompt = MODEL_PROMPT.format(pairs=str(kode_begrunnelse_pair_list) )
    schema = Kode2Liste
    if model == "openai":
        resp = gpt_client.responses.parse(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            text_format=schema,
        )
        json_response = resp.output_text
    elif model == "claude":
        # Claude 3.5 Haiku: prompt the system then the user, ask for JSON only
        claude_resp = claude_client.messages.create(
            model="claude-3-5-haiku-20241022",
            system=PROMPT_SYSTEM,
            max_tokens=8192,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        full_response = claude_resp.content[0].text
        #json_response = extract_json(full_response)
        json_response = full_response


    elif model == "gemini":
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
    return json_response

def save_json(filename, json_file):
# write all three into one JSON
    with open(f"./models_classify/{filename}.json", "w") as outfile:
        json.dump(json_file, outfile, indent=2)


if __name__ == "__main__":
    with open('./models_classify/kode2.json', 'r') as file:
        kode2_dict = json.load(file)
    
    kode2_claude = kode2_dict["claude"]
    kode2_openai = kode2_dict["openai"]
    kode2_gemini = kode2_dict["gemini"]
    
    claude_unique = create_unique_code(kode2_claude)
    openai_unique = create_unique_code(kode2_openai)
    gemini_unique = create_unique_code(kode2_gemini)

    print(f"Claude unique codes #{len(claude_unique)}")
    print(f"openai unique codes #{len(openai_unique)}")
    print(f"gemini unique codes #{len(gemini_unique)}")

    claude_pair = create_kode_begrunnelse_pair(claude_unique, kode2_claude) 
    openai_pair = create_kode_begrunnelse_pair(openai_unique, kode2_openai) 
    gemini_pair = create_kode_begrunnelse_pair(gemini_unique, kode2_gemini) 
    print(claude_pair[0])   
    #print(str(claude_pair))
    openai_kode2 = get_kode2_from_model("openai", openai_pair)
    claude_kode2 = get_kode2_from_model("claude", claude_pair)
    gemini_kode2 = get_kode2_from_model("gemini", gemini_pair)
    # write all three into one JSON
    save_json("openai_kode2", openai_kode2)
    #save_json("claude", claude_kode2)
    save_json("gemini_kode2", gemini_kode2)
    #Claude doesnt support structured json output, so it's saved as .txt file instead
    with open('./models_classify/claude_kode2.txt', 'w') as file:
        file.write(claude_kode2)

# RESULTS: 
#   Gemini 42
#   Claude 36
#   Openai 
