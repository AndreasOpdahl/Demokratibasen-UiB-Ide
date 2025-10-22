#!/usr/bin/env python3
"""
Script to evaluate newsworthiness of documents using LLM models.
Processes all JSON documents in the documents folder and generates a single JSON output file.
"""

import os
import json
import glob
import openai
import anthropic
from mistralai import Mistral
import cohere
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# System prompt for LLM evaluation
PROMPT_SYSTEM = """Du er dyktig og hjelpsom journalist i en lokal norsk avisredaksjon.

Avisen er interessert i å rapportere om ulike dokumenter som blir publisert av kommunen,
og de trenger din help til å vurdere nyhetsverdien til dokumentene. Du er en ekspert på å forstå deg på offentlige dokumenter og vurdere dem etter gitte nyhetsverdier.
"""

# Base prompt template - the dynamic parts will be filled in
PROMPT_TEMPLATE = """
Den lokale avisredaksjonen har tilgang på en enorm mengde dokumenter som blir
publisert av kommunen. Disse dokumentene er det for mange av for at en vanlig journalist kan lese gjennom og evaluere alle dokumentene.

De trenger derfor noen til å hjelpe med å vurdere nyhetsverdien til disse dokumentene, og gi en samlet poengsum ut av 100 for hvor nyhetsverdig dokumentet er.

Her er de ulike nyhetsverdier som skal brukes:

1. Makteliten
2. Positivitet
3. Underholdning
4. Konflikt
5. Identifikasjon
6. Magnitude

Her følger en forklaring av hver nyhetsverdi:

1. Makteliten innebærer om dokumentet nevner, omtaler eller beskriver en politisk instans, en politisk person eller en politisk sak. Fokuset er som regel rundt den politiske eliten i lokalsamfunnet. Jo mer makteliten omtales, eller jo mer makteliten er en del av dokumentet, jo mer nyhetsverdig er dokumentet.

2. Positivitet innebærer om dokumentet nevner, omtaler eller beskriver en positiv hendelse, en positiv person eller en positiv sak. Jo mer positivt sentimentet av dokumentets innhold er, jo mer nyhetsverdig er dokumentet.

3. Underholdning innebærer om dokumentet nevner, omtaler eller beskriver en aktivitet, et arrangement, eller større begivenhet som er til for lokalsamfunnets gode. Jo større aktiviteten, arrangementet, eller begivenheten er, jo mer nyhetsverdig er dokumentet.

4. Konflikt innebærer om innholdet i dokumentet nevner, beskriver eller inneholder en konflikt. Dette kan være en konflikt mellom 2 eller flere personer, 1 eller flere personer og en annen juridisk enhet, eller en konflikt mellom to eller flere juridiske enheter. Jo større konflikten er, jo mer nyhetsverdig er dokumentet.

5. Identifikasjon omhandler om dokumentet nevner, omtaler eller beskriver en eller flere personer fra lokalområdet eller en lokal organisasjon. Jo mer kjent personen eller organisasjonen er, jo mer nyhetsverdig er dokumentet. Jo mer lokal for området personen eller organisasjonen er, jo mer nyhetsverdig er dokumentet.

6. Magnitude innebærer om størrelse på saken. I dette tilfellet omhandler om hvor mange som er involvert i dokumentet, eller hvor store konsekvensene av innholdet i dokumentet er, hvor stort omfanget av innholdet i dokumentet er

Oppgaven din er som følger:

For hvert dokument, skal du først oppsummere hva dokumentet handler om, og deretter gi dokumentet en vurdering på en skala fra 0-10 for hver nyhetsverdi.

En vurdering på 10 betyr at definisjonen av nyhetsverdien, som gitt over, er fullstendig tilfredsstilles av dokumentet. En vurdering på 0 betyr at definisjonen av nyhetsverdien, som gitt over, ikke tilfredsstilles av dokumentet i det hele tatt.

For hver nyhetsverdi, hvis dokumentet ikke er nyhetsverdig, skal du gi dokumentet en vurdering på 0 for den spesifikke nyhetsverdien. Det samme gjelder hvis teksten ikke eksisterer eller gir mening.

Deretter skal du gi dokumentet en samlet vurdering på en skala fra 0-10, som er summen av alle vurderingene for hver nyhetsverdi.

Gi svaret i strukturert json format for hvilket type dokument det er (ikke politisk type).
Gi svaret med json-formatet:
{{
    "makteliten": 10,
    "positivitet": 10,
    "underholdning": 10,
    "konflikt": 10,
    "identifikasjon": 10,
    "magnitude": 10,
    "samlet_vurdering": 10
}}

Dokumentet kommer her i et json format:
{{
    'title': '{title}',
    'fulltekst': '{text}'
}}
"""


def load_documents():
    """Load all document JSON files from the documents folder"""
    documents_dir = os.path.join(os.path.dirname(__file__), 'documents')
    document_files = glob.glob(os.path.join(documents_dir, '*.json'))

    documents = []
    for file_path in document_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
                documents.append({
                    'id': doc_data.get('dokument_id'),
                    'title': doc_data.get('tittel', ''),
                    'text': doc_data.get('tekst', ''),
                    'doc_type': doc_data.get('doc_type'),
                    'url': doc_data.get('url')
                })
        except Exception as e:
            print(f"Error loading document {file_path}: {str(e)}")

    return documents


def create_prompt(title, text):
    """Create the prompt with the document title and text"""
    # Truncate text if it's too long to avoid token limits
    max_text_length = 4000  # Adjust based on your needs
    if len(text) > max_text_length:
        text = text[:max_text_length] + "..."

    return PROMPT_TEMPLATE.format(title=title, text=text)


def test_openai(prompt):
    #Test OpenAI GPT model with a specific prompt
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"OpenAI Error: {str(e)}"


def test_claude(prompt):
    #Test Anthropic Claude model with a specific prompt
    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            system=PROMPT_SYSTEM,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.content[0].text

    except Exception as e:
        return f"Claude Error: {str(e)}"


def test_mistral(prompt):
    #Test Mistral model with a specific prompt

    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    mistral_model = "mistral-large-latest"

    try:
        client = Mistral(api_key=mistral_api_key)

        response = client.chat.complete(
            model=mistral_model,
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Mistral Error: {str(e)}"

def test_cohere(prompt):
    #Test Cohere model with a specific prompt

    cohere_api_key = os.getenv("COHERE_API_KEY")
    cohere_model = "command-a-03-2025"

    try:
        client = cohere.ClientV2(cohere_api_key)

        response = client.chat(
            model=cohere_model,
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024
        )

        return response.message.content[0].text

    except Exception as e:
        return f"Cohere Error: {str(e)}" 

def parse_llm_response(response_text):
    """Parse the LLM response to extract JSON data"""
    try:
        # Find JSON in the response (LLM might add extra text)
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)

        return None
    except json.JSONDecodeError:
        return None


def process_documents():
    """Process all documents and return results"""
    documents = load_documents()

    if not documents:
        print("No documents found in documents folder")
        return {}

    results = {}

    for i, doc in enumerate(documents, 1):
        print(f"Processing document {i}/{len(documents)}: {doc['id']}")

        prompt = create_prompt(doc['title'], doc['text'])

        # Get evaluations from both models
        gpt_response = test_openai(prompt)
        claude_response = test_claude(prompt)
        mistral_response = test_mistral(prompt)
        cohere_response = test_cohere(prompt)

        # Parse responses
        gpt_data = parse_llm_response(gpt_response)
        claude_data = parse_llm_response(claude_response)
        mistral_data = parse_llm_response(mistral_response)
        cohere_data = parse_llm_response(cohere_response)

        # Store results
        results[doc['id']] = {
            'document_info': {
                'title': doc['title'],
                'doc_type': doc['doc_type'],
                'url': doc['url']
            },
            'gpt_evaluation': gpt_data,
            'claude_evaluation': claude_data,
            'mistral_evaluation': mistral_data,
            'cohere_evaluation': cohere_data,
            'raw_responses': {
                'gpt': gpt_response,
                'claude': claude_response,
                'mistral': mistral_response,
                'cohere': cohere_response
            }
        }

    return results


def save_results(results, filename="llm_evaluation_results.json"):
    """Save results to a JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {filename}")


def main():
    """Main function to run the evaluation"""
    print("Starting LLM evaluation of documents...")

    results = process_documents()

    if results:
        save_results(results)
        print(f"Processed {len(results)} documents successfully")
    else:
        print("No results to save")


if __name__ == "__main__":
    main()
