"""
============================================================
Sammenligning av ekstraherte data fra tre modeller
* Laster data fra OpenAI, Claude og Gemini ekstraksjoner
* Finner artikler som er analysert av alle tre
* Sammenligner feltene: viktige_hendelser, viktige_tidspunkter, viktige_personer, 
  viktige_organisasjoner, viktige_steder, tema
* Marker eksplisitt om feltene er identiske eller forskjellige
============================================================
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict

# ---------- KONFIGURASJON ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_extraction"

FILES = {
    "openai": DATA_DIR / "extracted_data_openai.jsonl",
    "claude": DATA_DIR / "extracted_data_claude.jsonl", 
    "gemini": DATA_DIR / "extracted_data_gemini.jsonl"
}

COMPARISON_FIELDS = [
    "viktige_hendelser",
    "viktige_tidspunkter", 
    "viktige_personer",
    "viktige_organisasjoner",
    "viktige_steder",
    "tema"
]

# ---------- HJELPEFUNKSJONER ----------
def load_jsonl(filepath: Path) -> Dict[str, Dict]:
    """Laster JSONL-fil og returnerer {dokument_id: data}."""
    data = {}
    if not filepath.exists():
        print(f"Advarsel: Fant ikke {filepath}")
        return data

    lines = []
    with filepath.open('r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Ignorer siste linje hvis den er ufullstendig
    if lines:
        lines = lines[:-1]  # Fjern siste linje
    
    # print(f"  - Laster {len(lines)} linjer (siste linje ignorert)")
    
    for line in lines:
        if line.strip():
            try:
                record = json.loads(line)
                doc_id = record["dokument_id"]
                data[doc_id] = record
            except json.JSONDecodeError as e:
                print(f"  - Advarsel: Kunne ikke parse linje: {e}")
                continue
    
    return data

def normalize_list(value: Any) -> List[str]:
    """Normaliserer verdier til liste av strenger."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if item]
    return [str(value).strip()] if value else []

def compare_field_values(values: Dict[str, List[str]], field: str) -> Dict[str, Any]:
    """Sammenligner verdier for et felt på tvers av modeller."""
    # Hent verdier for alle modeller
    model_values = {}
    for model, data in values.items():
        if "extracted_data" in data and field in data["extracted_data"]:
            model_values[model] = normalize_list(data["extracted_data"][field])
        else:
            model_values[model] = []
    
    # Sammenlign verdier
    all_values = list(model_values.values())
    unique_values = set(tuple(sorted(v)) for v in all_values)
    
    is_identical = len(unique_values) == 1
    comparison = {
        "field": field,
        "is_identical": is_identical,
        "values": model_values,
        "unique_sets": len(unique_values)
    }
    
    return comparison

def analyze_article(article_data: Dict[str, Dict], doc_id: str) -> Dict[str, Any]:
    """Analyserer en artikkel på tvers av alle modeller."""
    analysis = {
        "dokument_id": doc_id,
        "kommune": article_data.get("openai", {}).get("kommune_navn", "Ukjent"),
        "fields": {},
        "summary": {}
    }
    
    # Sammenlign hvert felt
    for field in COMPARISON_FIELDS:
        field_comparison = compare_field_values(article_data, field)
        analysis["fields"][field] = field_comparison
    
    # Oppsummering
    identical_fields = sum(1 for f in analysis["fields"].values() if f["is_identical"])
    total_fields = len(COMPARISON_FIELDS)
    
    analysis["summary"] = {
        "total_fields": total_fields,
        "identical_fields": identical_fields,
        "different_fields": total_fields - identical_fields,
        "consistency_percentage": round((identical_fields / total_fields) * 100, 1)
    }
    
    return analysis

def write_comparison_report(analyses: List[Dict], output_file: Path):
    """Skriver sammenligningsrapport til fil."""
    with output_file.open('w', encoding='utf-8') as f:
        f.write("Sammenligning av ekstraherte data fra OpenAI, Claude og Gemini\n")
        f.write("=" * 70 + "\n\n")
        
        for analysis in analyses:
            doc_id = analysis["dokument_id"]
            summary = analysis["summary"]
            
            f.write(f"Dokument: {doc_id}\n")
            f.write(f"Kommune: {analysis['kommune']}\n")
            f.write(f"Konsistens: {summary['identical_fields']}/{summary['total_fields']} felter identiske ({summary['consistency_percentage']}%)\n")
            f.write("-" * 50 + "\n")
            
            for field_name, field_data in analysis["fields"].items():
                f.write(f"\n{field_name}:\n")
                f.write(f"  Identisk: {'JA' if field_data['is_identical'] else 'NEI'}\n")
                
                for model, values in field_data["values"].items():
                    f.write(f"  {model.upper()}: {values}\n")
            
            f.write("\n" + "=" * 70 + "\n\n")

def write_csv_summary(analyses: List[Dict], output_file: Path):
    """Skriver CSV-sammendrag av sammenligningen."""
    with output_file.open('w', encoding='utf-8', newline='') as f:
        # Header
        header = ["dokument_id", "kommune", "konsistens_prosent", "identiske_felter", "forskjellige_felter"]
        for field in COMPARISON_FIELDS:
            header.append(f"{field}_identisk")
        
        f.write(",".join(header) + "\n")
        
        # Data
        for analysis in analyses:
            row = [
                analysis["dokument_id"],
                analysis["kommune"],
                str(analysis["summary"]["consistency_percentage"]),
                str(analysis["summary"]["identical_fields"]),
                str(analysis["summary"]["different_fields"])
            ]
            
            for field in COMPARISON_FIELDS:
                is_identical = analysis["fields"][field]["is_identical"]
                row.append("JA" if is_identical else "NEI")
            
            f.write(",".join(row) + "\n")

# ---------- HOVEDLØP ----------
def main():
    print("Laster ekstraherte data...")
    
    # Last alle tre filer
    all_data = {}
    for model, filepath in FILES.items():
        print(f"Laster {model} data fra {filepath.name}...")
        all_data[model] = load_jsonl(filepath)
        print(f"  - {len(all_data[model])} dokumenter lastet")
    
    # Finn dokumenter som er analysert av alle tre modeller
    all_doc_ids = set(all_data["openai"].keys()) & set(all_data["claude"].keys()) & set(all_data["gemini"].keys())
    
    if not all_doc_ids:
        print("Ingen dokumenter funnet som er analysert av alle tre modeller!")
        return
    
    print(f"\nFant {len(all_doc_ids)} dokumenter analysert av alle tre modeller")
    
    # Analyser hver artikkel
    analyses = []
    for doc_id in sorted(all_doc_ids):
        article_data = {model: all_data[model][doc_id] for model in FILES.keys()}
        analysis = analyze_article(article_data, doc_id)
        analyses.append(analysis)
    
    # Skriv rapporter
    report_file = DATA_DIR / "extraction_comparison_report.txt"
    csv_file = DATA_DIR / "extraction_comparison_summary.csv"
    
    write_comparison_report(analyses, report_file)
    write_csv_summary(analyses, csv_file)
    
    print(f"\nRapporter skrevet:")
    print(f"  - Detaljert rapport: {report_file}")
    print(f"  - CSV-sammendrag: {csv_file}")
    
    # Vis oppsummering
    total_articles = len(analyses)
    avg_consistency = sum(a["summary"]["consistency_percentage"] for a in analyses) / total_articles
    
    print(f"\nOppsummering:")
    print(f"  - Totalt antall artikler: {total_articles}")
    print(f"  - Gjennomsnittlig konsistens: {avg_consistency:.1f}%")
    
    # Vis felter med høyest/lavest konsistens
    field_consistency = defaultdict(list)
    for analysis in analyses:
        for field_name, field_data in analysis["fields"].items():
            field_consistency[field_name].append(field_data["is_identical"])
    
    print(f"\nKonsistens per felt:")
    for field in COMPARISON_FIELDS:
        consistency_rate = sum(field_consistency[field]) / len(field_consistency[field]) * 100
        print(f"  - {field}: {consistency_rate:.1f}%")

if __name__ == "__main__":
    main()
