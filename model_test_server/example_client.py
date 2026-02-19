"""
Example client for testing the model summary server.
"""

import requests
import json
import time
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed

SERVER_URL = "http://localhost:8000"

# Example texts from the dataset
EXAMPLE_TEXTS = [
    {
        "text": """Søknaden gjelder bygging av nytt boligbygg på tomt 123, eiendom 456 i Bergen kommune. 
        Søkeren ønsker å bygge et tre-etasjers boligbygg med totalt 12 leiligheter. 
        Bygget skal ha et areal på 1200 kvadratmeter og skal oppføres i henhold til gjeldende 
        byggeforskrifter. Søkeren har vedlagt tegninger og beregninger som viser at bygget 
        oppfyller alle krav til brannsikkerhet, tilgjengelighet og miljøstandarder. 
        Prosjektet er planlagt å starte i første kvartal 2024 og forventes å være ferdig 
        innen utgangen av 2025.""",
        "doc_type": "saksforelegg",
        "description": "Case presentation (saksforelegg)"
    },
    {
        "text": """Møtet ble åpnet klokken 14:00 av møteleder. Det var til stede 12 medlemmer 
        av kommunestyret. Hovedtemaet for møtet var diskusjon om nytt budsjett for 2024. 
        Det ble presentert et forslag om en budsjettøkning på 5 prosent sammenlignet med 
        forrige år. Flere medlemmer uttrykte bekymring over økningen, mens andre mente 
        at økningen var nødvendig for å opprettholde tjenestenivået. Etter lang diskusjon 
        ble det vedtatt å godkjenne budsjettet med en mindre justering ned til 4 prosent økning. 
        Møtet ble avsluttet klokken 16:30.""",
        "doc_type": "møtereferat",
        "description": "Meeting minutes (møtereferat)"
    },
    {
        "text": """Vedtak: Søknaden om byggetillatelse for tomt 123, eiendom 456 er godkjent. 
        Tillatelsen gis med følgende vilkår: 1) Bygget skal oppføres i henhold til vedlagte 
        tegninger. 2) Alle brannsikkerhetskrav skal oppfylles. 3) Byggearbeidet skal starte 
        innen 12 måneder. 4) Prosjektet skal rapporteres til kommunen hver 6. måned. 
        Vedtaket kan påklages innen 3 uker.""",
        "doc_type": "vedtak",
        "description": "Decision (vedtak)"
    },
    {
        "text": """Dette er en generell tekst om kommunal forvaltning og tjenester. 
        Kommunen tilbyr en rekke tjenester til innbyggerne, inkludert skole, helse, 
        eldreomsorg og kultur. Alle tjenester er tilgjengelige for alle innbyggere 
        uavhengig av bakgrunn eller situasjon. Kommunen jobber kontinuerlig med å 
        forbedre tjenestene og tilpasse dem til innbyggernes behov.""",
        "doc_type": "tekst",
        "description": "General text (tekst)"
    }
]

def test_health():
    """Test the health endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{SERVER_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_summarize(example_index=0):
    """Test the summarize endpoint with a specific example."""
    print("Testing summarize endpoint...")
    
    if example_index >= len(EXAMPLE_TEXTS):
        example_index = 0
    
    example = EXAMPLE_TEXTS[example_index]
    print(f"Using example: {example['description']}")
    
    payload = {
        "text": example["text"],
        "doc_type": example["doc_type"],
        "max_length": 150,  # Reduced for speed
        "min_length": 20,   # Reduced for speed
        "temperature": 0.3,  # Lower temperature for faster, more deterministic generation
        "do_sample": False  # Use greedy decoding for speed
    }
    
    print(f"Request payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    print()
    
    start_time = time.time()
    response = requests.post(f"{SERVER_URL}/summarize", json=payload)
    elapsed = time.time() - start_time
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Summary: {result['summary']}")
        print(f"Processing time (server): {result['processing_time']:.2f}s")
        print(f"Total time (client): {elapsed:.2f}s")
    else:
        print(f"Error: {response.text}")
    print()

if __name__ == "__main__":
    print("=" * 70)
    print("Model Summary Server - Example Client")
    print("=" * 70)
    print()
    
    # Test health
    try:
        test_health()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Is it running?")
        print(f"Expected server at: {SERVER_URL}")
        exit(1)
    
    # Test summarize - choose sequential or parallel
    use_parallel = True  # Set to False for sequential requests
    
    if use_parallel:
        print("Testing with PARALLEL requests...")
        print("=" * 70)
        
        def send_request(example, index):
            """Send a single request and return result."""
            payload = {
                "text": example["text"],
                "doc_type": example["doc_type"],
                "max_length": 150,
                "min_length": 20,
                "temperature": 0.3,
                "do_sample": False
            }
            
            start_time = time.time()
            try:
                response = requests.post(f"{SERVER_URL}/summarize", json=payload, timeout=300)
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "index": index,
                        "example": example,
                        "status": "success",
                        "summary": result['summary'],
                        "server_time": result['processing_time'],
                        "client_time": elapsed
                    }
                else:
                    return {
                        "index": index,
                        "example": example,
                        "status": "error",
                        "error": response.text,
                        "client_time": elapsed
                    }
            except Exception as e:
                return {
                    "index": index,
                    "example": example,
                    "status": "exception",
                    "error": str(e),
                    "client_time": time.time() - start_time
                }
        
        # Send all requests in parallel
        start_total = time.time()
        with ThreadPoolExecutor(max_workers=len(EXAMPLE_TEXTS)) as executor:
            futures = {
                executor.submit(send_request, example, i): (i, example) 
                for i, example in enumerate(EXAMPLE_TEXTS)
            }
            
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        total_time = time.time() - start_total
        
        # Print results in order
        results.sort(key=lambda x: x["index"])
        for result in results:
            i = result["index"] + 1
            example = result["example"]
            print(f"\nExample {i}/{len(EXAMPLE_TEXTS)}: {example['description']}")
            print("-" * 70)
            
            if result["status"] == "success":
                example = result["example"]
                original_text = example['text'].strip()
                
                print(f"\nOriginal Text ({len(original_text)} characters):")
                print("-" * 70)
                # Show full text with word wrapping
                import textwrap
                wrapped_text = textwrap.fill(original_text, width=68, initial_indent="  ", subsequent_indent="  ")
                print(wrapped_text)
                
                print(f"\nGenerated Summary ({len(result['summary'])} characters):")
                print("-" * 70)
                wrapped_summary = textwrap.fill(result['summary'], width=68, initial_indent="  ", subsequent_indent="  ")
                print(wrapped_summary)
                
                print(f"\nProcessing time (server): {result['server_time']:.2f}s")
                print(f"Client time: {result['client_time']:.2f}s")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
        
        print(f"\n{'='*70}")
        print(f"Total time for {len(EXAMPLE_TEXTS)} parallel requests: {total_time:.2f}s")
        print(f"Average time per request: {total_time/len(EXAMPLE_TEXTS):.2f}s")
        print(f"{'='*70}")
    else:
        print("Testing with SEQUENTIAL requests...")
        print("=" * 70)
        
        for i, example in enumerate(EXAMPLE_TEXTS, 1):
            print(f"\n{'='*70}")
            print(f"Example {i}/{len(EXAMPLE_TEXTS)}: {example['description']}")
            print(f"{'='*70}")
            
            payload = {
                "text": example["text"],
                "doc_type": example["doc_type"],
                "max_length": 150,
                "min_length": 20,
                "temperature": 0.3,
                "do_sample": False
            }
            
            start_time = time.time()
            response = requests.post(f"{SERVER_URL}/summarize", json=payload)
            elapsed = time.time() - start_time
            
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Summary: {result['summary']}")
                print(f"Processing time (server): {result['processing_time']:.2f}s")
                print(f"Total time (client): {elapsed:.2f}s")
            else:
                print(f"Error: {response.text}")
            print()
    
    print("=" * 70)
    print("Tests complete!")
    print("=" * 70)
