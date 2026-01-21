"""
============================================================
Sammendrag av ekstraherte data fra flere modeller
* Laster data fra alle JSON-filer som matcher et mønster
* Finner dokumenter som er analysert av alle modeller
* Sammenligner feltene og beregner statistikk
============================================================
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
from collections import defaultdict, Counter
import glob

# ---------- KONFIGURASJON ----------
ROOT = Path(__file__).resolve().parent
EXTRACTED_DATA_DIR = ROOT / "extracted-data"
SUMMARIES_DIR = ROOT / "summaries"

# Felter som skal telles
COUNT_FIELDS = [
    "tema",
    "viktige_hendelser",
    "viktige_tidspunkter",
    "viktige_personer",
    "viktige_organisasjoner",
    "viktige_steder"
]

# ---------- HJELPEFUNKSJONER ----------
def load_schema() -> Dict[str, Any]:
    """Laster schema for å få alle properties."""
    schema_path = ROOT / "create_prompt" / "extraction-202512-schema.json"
    with schema_path.open('r', encoding='utf-8') as f:
        schema_data = json.load(f)
    return schema_data["schema"]["properties"]

def find_matching_directories_by_task(task_name: str) -> List[Path]:
    """
    Finner alle modellmapper i en spesifikk task-mappe.
    
    Args:
        task_name: Task-navn (f.eks. "dataset-202510-all-tokens-extraction-202512")
    
    Returns:
        Liste av Path-objekter til modellmapper
    """
    if not EXTRACTED_DATA_DIR.exists():
        return []
    
    task_dir = EXTRACTED_DATA_DIR / task_name
    if not task_dir.exists() or not task_dir.is_dir():
        return []
    
    matching_dirs = []
    for model_dir in task_dir.iterdir():
        if model_dir.is_dir():
            matching_dirs.append(model_dir)
    
    return sorted(matching_dirs)

def find_matching_directories_from_path(directory_path: Path) -> List[Path]:
    """
    Finner alle modellmapper fra en gitt sti.
    
    Args:
        directory_path: Path til enten en task-mappe eller en modell-mappe
    
    Returns:
        Liste av Path-objekter til modellmapper
    """
    if not directory_path.exists():
        return []
    
    if not directory_path.is_dir():
        return []
    
    # Check if this is a task directory (contains model subdirectories)
    # or a model directory (contains JSON files)
    has_json_files = any(directory_path.glob("*.json"))
    has_subdirs = any(item.is_dir() for item in directory_path.iterdir())
    
    if has_json_files and not has_subdirs:
        # This is a model directory, return it
        return [directory_path]
    elif has_subdirs:
        # This is a task directory, return all model subdirectories
        matching_dirs = []
        for model_dir in directory_path.iterdir():
            if model_dir.is_dir():
                matching_dirs.append(model_dir)
        return sorted(matching_dirs)
    
    return []

def load_json_files(directory: Path) -> Dict[str, Dict]:
    """Laster alle JSON-filer fra en mappe og returnerer {dokument_id: data}."""
    data = {}
    json_files = list(directory.glob("*.json"))
    
    # Filtrer bort prompt_schema.json og system_prompt.json
    json_files = [f for f in json_files if f.name not in ["prompt_schema.json", "system_prompt.json"]]
    
    for json_file in json_files:
        try:
            with json_file.open('r', encoding='utf-8') as f:
                record = json.load(f)
                doc_id = record.get("dokument_id")
                if doc_id:
                    data[doc_id] = record
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  - Advarsel: Kunne ikke laste {json_file.name}: {e}")
            continue
    
    return data

def capitalize_first_letter(text: str) -> str:
    """Kapitaliserer første bokstav i en streng."""
    if not text:
        return text
    return text[0].upper() + text[1:] if len(text) > 1 else text.upper()

def remove_parentheses(text: str) -> str:
    """Fjerner parenteser og innholdet i dem fra en streng."""
    # Fjern parenteser og innhold: (tekst) eller [tekst]
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    return text.strip()

def clean_tidspunkt(text: str) -> str:
    """Renser viktige_tidspunkter ved å fjerne beskrivende tekst etter datoer."""
    # Fjern tekst etter datoer/tidspunkter
    # Mønstre som: "11.09.2025 - møte i..." -> "11.09.2025"
    # "2021-2024 - handlingsprogram..." -> "2021-2024"
    # "11.09.2025: Møtedato..." -> "11.09.2025"
    # "30.06.2026 – Innføring..." -> "30.06.2026" (en-dash)
    # "10.04.2025 kl 11:00" -> "10.04.2025 kl 11:00" (behold klokkeslett)
    
    # Character class for separators: hyphen (-), en-dash (–), em-dash (—), colon (:)
    separators = r'[-–—:]'
    
    # Pattern 1: DD.MM.YYYY kl HH:MM (behold klokkeslett, fjern resten)
    match = re.match(r'(\d{1,2}\.\d{1,2}\.\d{2,4}\s+kl\s+\d{1,2}:\d{2})\s*' + separators + r'.*', text)
    if match:
        return match.group(1).strip()
    
    # Pattern 2: DD.MM.YYYY eller DD.MM.YY (uten klokkeslett, fjern alt etter separator)
    match = re.match(r'(\d{1,2}\.\d{1,2}\.\d{2,4})\s*' + separators + r'.*', text)
    if match:
        return match.group(1).strip()
    
    # Pattern 3: YYYY-YYYY (årrekkevidde)
    match = re.match(r'(\d{4}-\d{4})\s*' + separators + r'.*', text)
    if match:
        return match.group(1).strip()
    
    # Pattern 4: YYYY-MM-DD
    match = re.match(r'(\d{4}-\d{1,2}-\d{1,2})\s*' + separators + r'.*', text)
    if match:
        return match.group(1).strip()
    
    # Pattern 5: DD.-DD. måned YYYY (f.eks. "10.-14. mai 2025")
    # For disse, beholde hele dato-området, men fjern beskrivende tekst etter
    match = re.match(r'(\d{1,2}\.-\d{1,2}\.\s+\w+\s+\d{4})\s*' + separators + r'.*', text)
    if match:
        return match.group(1).strip()
    
    # Hvis ingen mønster matcher, returner originalen
    return text.strip()

def normalize_list(value: Any, field_name: str = None) -> List[str]:
    """Normaliserer verdier til liste av strenger med feltspesifikk behandling."""
    if value is None:
        return []
    
    # Konverter til liste
    if isinstance(value, list):
        items = [str(item).strip() for item in value if item and str(item).strip()]
    else:
        items = [str(value).strip()] if value and str(value).strip() else []
    
    # Apply field-specific normalization
    normalized_items = []
    for item in items:
        # Capitalize first letter for tema and viktige_hendelser
        if field_name in ["tema", "viktige_hendelser"]:
            item = capitalize_first_letter(item)
        
        # Remove parentheses for viktige_personer and viktige_tidspunkter
        if field_name in ["viktige_personer", "viktige_tidspunkter"]:
            item = remove_parentheses(item)
        
        # Clean tidspunkter (remove descriptive text)
        if field_name == "viktige_tidspunkter":
            item = clean_tidspunkt(item)
        
        if item:  # Only add non-empty items after normalization
            normalized_items.append(item)
    
    return normalized_items

def is_acronym_or_name(text: str) -> bool:
    """
    Sjekker om en tekst ser ut som et akronym eller navn.
    Akronymer: alle store bokstaver (minst 2 tegn)
    Navn: starter med stor bokstav, kan inneholde bindestrek eller mellomrom
    """
    if not text or len(text) < 2:
        return False
    
    # Akronym: alle store bokstaver (minst 2)
    if text.isupper() and len(text) >= 2:
        return True
    
    # Navn: starter med stor bokstav, kan ha bindestrek eller mellomrom
    # Eksempler: "John Doe", "Ola Nordmann", "Anne-Lise"
    words = text.split()
    if words and all(word[0].isupper() if word else False for word in words if word):
        # Sjekk om det ser ut som et navn (ikke bare første ord med stor bokstav)
        if len(words) > 1 or '-' in text:
            return True
    
    return False

def normalize_for_comparison(text: str, for_substring: bool = False) -> str:
    """
    Normaliserer tekst for sammenligning.
    
    Args:
        text: Teksten som skal normaliseres
        for_substring: Hvis True, normaliserer også akronymer/navn til lowercase
                      for substring-sjekking. Hvis False, beholder original casing
                      for akronymer/navn (for eksakt matching).
    """
    if for_substring:
        # For substring-sjekking, alltid lowercase (inkludert akronymer/navn)
        return text.lower()
    else:
        # For eksakt matching, behold original casing for akronymer/navn
        if is_acronym_or_name(text):
            return text
        return text.lower()

def count_string_occurrences(all_lists: List[List[str]]) -> Dict[str, Dict[str, int]]:
    """
    Beregner string_count og substr_count for hver unik streng.
    
    string_count: Antall ganger strengen forekommer eksakt.
    substr_count: Antall ganger superstrings (strenger som inneholder denne som substring) 
                  forekommer totalt.
                  For eksempel, hvis "Motorferdsel" forekommer 4 ganger og 
                  "Motorferdsel i utmark" forekommer 2 ganger, så vil 
                  substr_count for "Motorferdsel" være 2 (siden "Motorferdsel i utmark" 
                  inneholder "Motorferdsel").
    
    Args:
        all_lists: Liste av lister med strenger
    
    Returns:
        Dict med {string: {"string_count": int, "substr_count": int}}
    """
    # Få union av alle strenger
    all_strings = set()
    for lst in all_lists:
        all_strings.update(lst)
    
    # Flatten all lists for counting
    all_items = []
    for lst in all_lists:
        all_items.extend(lst)
    
    counts = {}
    
    for string in all_strings:
        string_count = 0
        substr_count = 0
        
        # Normaliser string for sammenligning
        normalized_string = normalize_for_comparison(string)
        
        # Tell eksakte forekomster (case-insensitive, unntatt akronymer/navn)
        for item in all_items:
            normalized_item = normalize_for_comparison(item)
            if normalized_string == normalized_item:
                string_count += 1
        
        # Tell substring-forekomster: finn alle andre strenger som er superstrings
        # (inneholder denne strengen som substring), og tell hvor mange ganger de forekommer totalt.
        # For eksempel, hvis "Motorferdsel" forekommer 4 ganger og 
        # "Motorferdsel i utmark" forekommer 2 ganger, så vil substr_count for
        # "Motorferdsel" være 2 (siden "Motorferdsel i utmark" inneholder "Motorferdsel").
        # For substring-sjekking, bruk lowercase for alle (inkludert akronymer/navn)
        normalized_string_for_substr = normalize_for_comparison(string, for_substring=True)
        
        for other_string in all_strings:
            if other_string == string:
                continue  # Skip eksakt match
            
            # For substring-sjekking, normaliser begge til lowercase
            normalized_other = normalize_for_comparison(other_string, for_substring=True)
            # Sjekk om string er en substring av other_string (other_string er superstring)
            if normalized_string_for_substr in normalized_other:
                # Tell alle forekomster av other_string (bruk eksakt matching)
                for item in all_items:
                    normalized_item = normalize_for_comparison(item, for_substring=False)
                    normalized_other_exact = normalize_for_comparison(other_string, for_substring=False)
                    if normalized_item == normalized_other_exact:
                        substr_count += 1
        
        counts[string] = {
            "string_count": string_count,
            "substr_count": substr_count
        }
    
    return counts

def analyze_document_group(doc_id: str, doc_data_by_model: Dict[str, Dict], schema_properties: Dict[str, Any]) -> Dict[str, Any]:
    """Analyserer en gruppe dokumenter (samme dokument_id fra forskjellige modeller)."""
    # Get list of models in consistent order
    models_list = sorted(doc_data_by_model.keys())
    
    analysis = {
        "dokument_id": doc_id,
        "models": models_list,
        "properties": {},
        "counts": {}
    }
    
    # Hent alle properties fra schema
    for prop_name in schema_properties.keys():
        all_values = []
        # Iterate in the same order as models_list
        for model in models_list:
            data = doc_data_by_model[model]
            try:
                if not isinstance(data, dict):
                    print(f"  - Advarsel: Data for {model} i doc {doc_id} er ikke en dict: {type(data)}")
                    value = None
                else:
                    response = data.get("response", {})
                    if not isinstance(response, dict):
                        # Hvis response ikke er en dict, prøv å behandle den som en liste eller annen struktur
                        response = {}
                    value = response.get(prop_name) if isinstance(response, dict) else None
            except (AttributeError, TypeError) as e:
                print(f"  - Advarsel: Kunne ikke hente {prop_name} for {model} i doc {doc_id}: {e}")
                value = None
            all_values.append(normalize_list(value, field_name=prop_name))
        
        # Lagre alle verdier som liste i samme rekkefølge som models_list
        analysis["properties"][prop_name] = all_values
    
    # Beregn counts for spesifikke felter
    for field in COUNT_FIELDS:
        all_lists = []
        # Iterate in the same order as models_list
        for model in models_list:
            data = doc_data_by_model[model]
            try:
                if not isinstance(data, dict):
                    value = None
                else:
                    response = data.get("response", {})
                    if not isinstance(response, dict):
                        response = {}
                    value = response.get(field) if isinstance(response, dict) else None
            except (AttributeError, TypeError) as e:
                print(f"  - Advarsel: Kunne ikke hente {field} for {model} i doc {doc_id}: {e}")
                value = None
            all_lists.append(normalize_list(value, field_name=field))
        
        counts = count_string_occurrences(all_lists)
        # Convert counts from dict to sorted list: [string, string_count, substr_count]
        # Sort by string_count (descending), then substr_count (descending)
        counts_list = [
            [string, count_info["string_count"], count_info["substr_count"]]
            for string, count_info in counts.items()
        ]
        # Sort by string_count first (descending), then substr_count (descending)
        counts_list.sort(key=lambda x: (-x[1], -x[2]))
        analysis["counts"][field] = counts_list
    
    return analysis

def calculate_coverage_stats(doc_data_by_model: Dict[str, Dict[str, Dict]]) -> Dict[str, Any]:
    """Beregner coverage-statistikk."""
    all_models = set(doc_data_by_model.keys())
    num_models = len(all_models)
    
    # Finn alle dokumenter
    all_doc_ids = set()
    for model_data in doc_data_by_model.values():
        all_doc_ids.update(model_data.keys())
    
    # Beregn coverage for hvert dokument
    doc_coverage = {}
    for doc_id in all_doc_ids:
        models_with_doc = {model for model, data in doc_data_by_model.items() if doc_id in data}
        doc_coverage[doc_id] = models_with_doc
    
    # Tell dokumenter med forskjellig coverage-nivåer
    coverage_counts = defaultdict(int)
    for doc_id, models in doc_coverage.items():
        num_missing = num_models - len(models)
        coverage_counts[num_missing] += 1
    
    # Finn modeller med manglende dokumenter
    model_missing = {}
    for model in all_models:
        model_docs = set(doc_data_by_model[model].keys())
        missing_docs = all_doc_ids - model_docs
        model_missing[model] = {
            "missing_count": len(missing_docs),
            "missing_doc_ids": sorted(list(missing_docs))
        }
    
    return {
        "total_documents": len(all_doc_ids),
        "total_models": num_models,
        "coverage_by_missing": dict(sorted(coverage_counts.items())),
        "model_missing_docs": model_missing
    }

def build_task_name(dataset: str, max_tokens: str, prompt: str) -> str:
    """
    Bygger task-navn fra komponenter.
    
    Format: <dataset>-<max-tokens>-<prompt>
    """
    # Normalize max_tokens: "all" -> "all-tokens", otherwise keep as-is
    if max_tokens.lower() == "all":
        max_tokens_str = "all-tokens"
    else:
        # If it's a number, add "-tokens" suffix
        try:
            int(max_tokens)
            max_tokens_str = f"{max_tokens}-tokens"
        except ValueError:
            # Already has format like "all-tokens" or "2048-tokens"
            max_tokens_str = max_tokens
    
    return f"{dataset}-{max_tokens_str}-{prompt}"

def main():
    parser = argparse.ArgumentParser(
        description="Sammendrag av ekstraherte data fra flere modeller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Eksempler:
  # Bruk prompt, dataset og max-tokens:
  %(prog)s --prompt extraction-202512 --dataset dataset-202510 --max-tokens all
  
  # Bruk direkte sti:
  %(prog)s --extracted-data extracted-data/dataset-202510-all-tokens-extraction-202512
        """
    )
    
    parser.add_argument(
        "--extracted-data",
        type=str,
        help="Direkte sti til task-mappe eller modell-mappe i extracted-data/ (alternativ til --prompt/--dataset/--max-tokens)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt-navn (f.eks. 'extraction-202512'). Krever også --dataset og --max-tokens."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset-navn (f.eks. 'dataset-202510'). Krever også --prompt og --max-tokens."
    )
    parser.add_argument(
        "--max-tokens",
        type=str,
        help="Max tokens (f.eks. 'all' eller '2048'). Krever også --prompt og --dataset."
    )
    
    args = parser.parse_args()
    
    # Validate arguments: either --extracted-data OR (--prompt, --dataset, --max-tokens)
    if args.extracted_data:
        if any([args.prompt, args.dataset, args.max_tokens]):
            parser.error("--extracted-data kan ikke brukes sammen med --prompt, --dataset eller --max-tokens")
    else:
        if not all([args.prompt, args.dataset, args.max_tokens]):
            parser.error("Enten --extracted-data må spesifiseres, eller alle av --prompt, --dataset og --max-tokens må spesifiseres")
    
    # Determine which mode we're in
    task_name = None
    if args.extracted_data:
        # Direct path mode
        directory_path = Path(args.extracted_data)
        if not directory_path.is_absolute():
            directory_path = EXTRACTED_DATA_DIR / args.extracted_data
        
        print(f"Laster ekstraherte data fra '{directory_path}'...")
        matching_dirs = find_matching_directories_from_path(directory_path)
        
        if not matching_dirs:
            print(f"Ingen mapper funnet i '{directory_path}'")
            return
        
        # For output filename, use the directory name
        if directory_path.parent == EXTRACTED_DATA_DIR:
            # It's a task directory
            output_filename = f"{directory_path.name}.json"
            task_name = directory_path.name
        else:
            # It's a model directory or deeper
            output_filename = f"{directory_path.name}.json"
        
    else:
        # Component mode: --prompt, --dataset, --max-tokens
        task_name = build_task_name(args.dataset, args.max_tokens, args.prompt)
        print(f"Laster ekstraherte data for task '{task_name}'...")
        
        matching_dirs = find_matching_directories_by_task(task_name)
        
        if not matching_dirs:
            print(f"Ingen mapper funnet for task '{task_name}' i {EXTRACTED_DATA_DIR}")
            return
        
        # For output filename, use the task name
        output_filename = f"{task_name}.json"
    
    print(f"Fant {len(matching_dirs)} matchende mapper:")
    for dir_path in matching_dirs:
        print(f"  - {dir_path.name}")
    
    # Last alle JSON-filer fra hver mappe
    doc_data_by_model = {}
    for dir_path in matching_dirs:
        model_name = dir_path.name
        print(f"\nLaster data fra {model_name}...")
        data = load_json_files(dir_path)
        doc_data_by_model[model_name] = data
        print(f"  - {len(data)} dokumenter lastet")
    
    # Beregn coverage-statistikk
    print("\n" + "=" * 70)
    print("COVERAGE-STATISTIKK")
    print("=" * 70)
    
    coverage_stats = calculate_coverage_stats(doc_data_by_model)
    
    print(f"\nTotalt antall dokumenter: {coverage_stats['total_documents']}")
    print(f"Totalt antall modeller: {coverage_stats['total_models']}")
    
    print("\nDokumenter etter coverage:")
    for num_missing in sorted(coverage_stats['coverage_by_missing'].keys()):
        count = coverage_stats['coverage_by_missing'][num_missing]
        if num_missing == 0:
            print(f"  - Alle modeller: {count} dokumenter")
        else:
            print(f"  - Alle unntatt {num_missing} modell(er): {count} dokumenter")
    
    print("\nModeller med manglende dokumenter:")
    for model, missing_info in coverage_stats['model_missing_docs'].items():
        print(f"  - {model}: {missing_info['missing_count']} manglende dokumenter")
        if missing_info['missing_count'] > 0 and missing_info['missing_count'] <= 5:
            print(f"    Manglende doc IDs: {', '.join(missing_info['missing_doc_ids'])}")
    
    # Last schema
    schema_properties = load_schema()
    
    # Analyser hver dokumentgruppe
    print("\n" + "=" * 70)
    print("ANALYSERER DOKUMENTGRUPPER")
    print("=" * 70)
    
    all_doc_ids = set()
    for model_data in doc_data_by_model.values():
        all_doc_ids.update(model_data.keys())
    
    analyses = []
    for doc_id in sorted(all_doc_ids):
        # Hent data for denne dokument_id fra alle modeller som har den
        doc_data = {
            model: data[doc_id]
            for model, data in doc_data_by_model.items()
            if doc_id in data
        }
        
        analysis = analyze_document_group(doc_id, doc_data, schema_properties)
        analyses.append(analysis)
    
    print(f"\nAnalysert {len(analyses)} dokumentgrupper")
    
    # Lagre resultater
    SUMMARIES_DIR.mkdir(exist_ok=True)
    
    output_path = SUMMARIES_DIR / output_filename
    
    # Build metadata for output
    if args.extracted_data:
        metadata = {"extracted_data_path": args.extracted_data}
    else:
        metadata = {
            "prompt": args.prompt,
            "dataset": args.dataset,
            "max_tokens": args.max_tokens,
            "task_name": task_name
        }
    
    output_data = {
        "metadata": metadata,
        "coverage_stats": coverage_stats,
        "document_analyses": analyses
    }
    
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultater lagret til: {output_path}")
    print(f"  - Coverage-statistikk")
    print(f"  - {len(analyses)} dokumentanalyser")
    print(f"  - Counts for: {', '.join(COUNT_FIELDS)}")

if __name__ == "__main__":
    main()


