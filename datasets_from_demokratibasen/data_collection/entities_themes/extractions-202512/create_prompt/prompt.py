"""
Prompt creation for structured data extraction from case documents.
"""
import json
from pathlib import Path


class Prompt:
    """Class for creating prompts for LLM-based structured data extraction."""
    
    def __init__(self, prompt_name: str = "OLD-structured-data-202509"):
        """
        Initialize the Prompt class with a specific prompt name.
        
        Args:
            prompt_name: Name of the prompt to load (default: "OLD-structured-data-202509")
        """
        self.prompt_name = prompt_name
        self._base_path = Path(__file__).parent
        
        # Load prompt template
        prompt_file = self._base_path / f"{prompt_name}-prompt.txt"
        if prompt_file.exists():
            with open(prompt_file, "r", encoding="utf-8") as f:
                self.PROMPT_TEMPLATE = f.read()
        else:
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_file}. "
                f"Expected file: {prompt_name}-prompt.txt"
            )
        
        # Load schema
        schema_file = self._base_path / f"{prompt_name}-schema.json"
        if schema_file.exists():
            with open(schema_file, "r", encoding="utf-8") as f:
                schema_data = json.load(f)
                # Handle both formats:
                # - Old format (OLD-structured-data-202509): 
                #   {"name": "...", "description": "...", "schema": {"name": "...", "parameters": {...}}}
                #   SCHEMA should be schema_data["schema"] (which has "parameters")
                # - New format (structured-data-202512):
                #   {"name": "...", "description": "...", "schema": {"type": "object", "properties": {...}}}
                #   SCHEMA should be the whole schema_data (which has "schema" key with direct JSON schema)
                if "schema" in schema_data:
                    # Check if it's old format (has "parameters" in schema) or new format (has "type" in schema)
                    inner_schema = schema_data["schema"]
                    if "parameters" in inner_schema:
                        # Old format: return the inner schema (which has "parameters")
                        self.SCHEMA = inner_schema
                    else:
                        # New format: return the whole schema_data (which has "schema" key)
                        self.SCHEMA = schema_data
                else:
                    # If no "schema" key, assume the whole file is the schema
                    self.SCHEMA = schema_data
        else:
            raise FileNotFoundError(
                f"Schema file not found: {schema_file}. "
                f"Expected file: {prompt_name}-schema.json"
            )
    
    def get_prompt(self, kommune_navn: str, include_schema: bool = False) -> str:
        """
        Get the prompt with kommune name filled in.
        
        Args:
            kommune_navn: Name of the municipality (kommune)
            include_schema: If True, append the JSON schema to the prompt (useful for models
                          that don't support structured outputs and need schema in prompt)
            
        Returns:
            Prompt with kommune name inserted, optionally with schema appended
        """
        prompt = self.PROMPT_TEMPLATE.replace("<<kommune_navn>>", kommune_navn)
        
        if include_schema:
            # Extract the actual JSON schema for inclusion in prompt
            schema_to_include = self.SCHEMA
            if "schema" in schema_to_include:
                # If SCHEMA has nested "schema" key, extract the inner schema
                inner_schema = schema_to_include["schema"]
                if "type" in inner_schema:
                    # New format: inner schema is the JSON schema
                    schema_to_include = inner_schema
                elif "parameters" in inner_schema:
                    # Old format: extract parameters
                    schema_to_include = inner_schema["parameters"]
            
            # Append schema to prompt
            schema_json = json.dumps(schema_to_include, ensure_ascii=False, indent=2)
            prompt += f"\n\nJSON-skjemaet du skal følge:\n```json\n{schema_json}\n```\n"
            prompt += "\nVIKTIG: Alle felter som er listet i 'required' må inkluderes i JSON-objektet, selv om de er tomme (bruk \"\" for strenger og [] for lister).\n"
        
        return prompt
    
    def get_document_text(self, text: str) -> str:
        """
        Get the document text (user input).
        
        Args:
            text: The document text to extract structured data from
            
        Returns:
            The document text (which is used as user input to the LLM)
        """
        return text

