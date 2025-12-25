#!/usr/bin/env python3
"""
Convert extraction files from 2025-08 format to 2025-12 schema format.

Reads extracted_data_gemini.jsonl and extracted_data_openai.jsonl,
converts to the new schema format, and writes individual JSON files.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


# Schema-defined fields (required)
SCHEMA_FIELDS = [
    "hva_saken_gjelder",
    "tema",
    "viktige_hendelser",
    "viktige_tidspunkter",
    "viktige_personer",
    "viktige_organisasjoner",
    "viktige_steder",
]


def normalize_field_value(value: Any, field_name: str) -> Any:
    """Normalize a field value to match schema requirements."""
    if field_name == "hva_saken_gjelder":
        # Must be a string
        if value is None:
            return ""
        return str(value)
    else:
        # Must be an array of strings
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        # If it's a single value, wrap it in a list
        return [str(value)]


def convert_extracted_data(old_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert old extracted_data format to new schema format.
    
    Only includes fields specified in the schema.
    Ensures all required fields are present with appropriate defaults.
    """
    new_data = {}
    
    for field in SCHEMA_FIELDS:
        if field in old_data:
            new_data[field] = normalize_field_value(old_data[field], field)
        else:
            # Provide default for missing required fields
            if field == "hva_saken_gjelder":
                new_data[field] = ""
            else:
                new_data[field] = []
    
    return new_data


def extract_model_name(extraction_model: str) -> str:
    """Extract clean model name from extraction_model field."""
    model_lower = extraction_model.lower()
    
    if "gemini" in model_lower:
        # Extract "gemini-2.5-flash" from "models/gemini-2.5-flash"
        if "models/" in extraction_model:
            return extraction_model.split("models/")[-1]
        return "gemini-2.5-flash"
    elif "gpt" in model_lower or "openai" in model_lower:
        # Extract "gpt-4o-mini" from extraction_model
        if "gpt-4o-mini" in model_lower:
            return "gpt-4o-mini"
        return "gpt-4o-mini"  # default
    else:
        # Return as-is if unrecognized
        return extraction_model


def determine_temperature(model_name: str) -> float:
    """Determine temperature based on model name."""
    model_lower = model_name.lower()
    if "gemini" in model_lower:
        return 0.0
    elif "gpt" in model_lower or "openai" in model_lower:
        return 0.1
    else:
        return 0.0  # default


def determine_output_dir(extraction_model: str) -> str:
    """Determine the output directory based on extraction_model."""
    model_lower = extraction_model.lower()
    
    if "gemini" in model_lower or "models/gemini" in model_lower:
        return "gemini-2.5-flash"
    elif "gpt" in model_lower or "openai" in model_lower:
        return "gpt-4o-mini"
    else:
        # Default fallback
        return "unknown"


def process_jsonl_file(
    input_file: Path,
    base_output_dir: Path,
    file_type: str
) -> tuple[int, int]:
    """
    Process a JSONL file and convert entries.
    
    Returns:
        Tuple of (processed_count, error_count)
    """
    processed_count = 0
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse the JSON line
                entry = json.loads(line)
                
                # Extract fields
                dokument_id = entry.get("dokument_id")
                kommune = entry.get("kommune")
                kommune_navn = entry.get("kommune_navn", "")
                extracted_at = entry.get("extracted_at", "")
                extraction_model = entry.get("extraction_model", "")
                extracted_data = entry.get("extracted_data", {})
                
                if not dokument_id:
                    print(f"Warning: Line {line_num} has no dokument_id, skipping")
                    error_count += 1
                    continue
                
                # Convert extracted_data to new format
                response_data = convert_extracted_data(extracted_data)
                
                # Extract model name and determine temperature
                model_name = extract_model_name(extraction_model)
                temperature = determine_temperature(model_name)
                
                # Build the output structure with metadata
                output_data = {
                    "dokument_id": dokument_id,
                    "kommune_nummer": kommune,
                    "kommune_navn": kommune_navn,
                    "generated_at": extracted_at,
                    "model": model_name,
                    "temperature": temperature,
                    "max_tokens": None,
                    "response": response_data
                }
                
                # Determine output directory
                model_dir = determine_output_dir(extraction_model)
                output_dir = base_output_dir / model_dir
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Write to file
                output_file = output_dir / f"{dokument_id}.json"
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    json.dump(output_data, out_f, ensure_ascii=False, indent=2)
                
                processed_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num} in {input_file}: {e}")
                error_count += 1
            except Exception as e:
                print(f"Error processing line {line_num} in {input_file}: {e}")
                error_count += 1
    
    return processed_count, error_count


def main():
    """Main conversion function."""
    # Define paths
    script_dir = Path(__file__).parent
    input_dir = script_dir.parent / "extractions-202508"
    base_output_dir = script_dir / "extracted-data" / "dataset-202505-all-tokens-extraction-202512"
    
    # Input files
    gemini_file = input_dir / "extracted_data_gemini.jsonl"
    openai_file = input_dir / "extracted_data_openai.jsonl"
    
    # Create output base directory
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    total_errors = 0
    
    # Process Gemini file
    if gemini_file.exists():
        print(f"Processing {gemini_file}...")
        processed, errors = process_jsonl_file(gemini_file, base_output_dir, "gemini")
        total_processed += processed
        total_errors += errors
        print(f"  Processed: {processed}, Errors: {errors}")
    else:
        print(f"Warning: {gemini_file} not found")
    
    # Process OpenAI file
    if openai_file.exists():
        print(f"Processing {openai_file}...")
        processed, errors = process_jsonl_file(openai_file, base_output_dir, "openai")
        total_processed += processed
        total_errors += errors
        print(f"  Processed: {processed}, Errors: {errors}")
    else:
        print(f"Warning: {openai_file} not found")
    
    print(f"\nTotal: {total_processed} processed, {total_errors} errors")
    print(f"Output directory: {base_output_dir}")


if __name__ == "__main__":
    main()

