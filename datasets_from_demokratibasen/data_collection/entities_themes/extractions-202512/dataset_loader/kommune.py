"""
Kommune ID to name translation utilities.

Reads kommune codes from the official KLASS classification file.
"""

import csv
from pathlib import Path
from typing import Dict

# Path to the official KLASS kommune codes CSV file
_CSV_FILE = Path(__file__).parent / "klass-version-1710-codes.csv"

# Cache for the kommune dictionary (loaded on first access)
_KOMMUNENAVN_CACHE: Dict[int, str] = None


def _load_kommunenavn() -> Dict[int, str]:
    """
    Load kommune codes and names from the official KLASS CSV file.
    
    Returns:
        Dictionary mapping kommune number (int) to kommune name (str)
    """
    global _KOMMUNENAVN_CACHE
    
    if _KOMMUNENAVN_CACHE is not None:
        return _KOMMUNENAVN_CACHE
    
    _KOMMUNENAVN_CACHE = {}
    
    if not _CSV_FILE.exists():
        # Fallback to empty dict if file doesn't exist
        return _KOMMUNENAVN_CACHE
    
    try:
        # Try different encodings (CSV files from KLASS may use ISO-8859-1 or Windows-1252)
        encodings = ['utf-8', 'iso-8859-1', 'windows-1252', 'latin1']
        last_error = None
        
        for encoding in encodings:
            try:
                with open(_CSV_FILE, 'r', encoding=encoding) as f:
                    # CSV uses semicolon as delimiter
                    reader = csv.DictReader(f, delimiter=';')
                    for row in reader:
                        code_str = row.get('code', '').strip()
                        name = row.get('name', '').strip()
                        
                        if code_str and name:
                            try:
                                code = int(code_str)
                                _KOMMUNENAVN_CACHE[code] = name
                            except (ValueError, TypeError):
                                # Skip invalid codes
                                continue
                # If we get here, loading succeeded
                break
            except UnicodeDecodeError as e:
                last_error = e
                continue
            except Exception as e:
                last_error = e
                continue
        
        if not _KOMMUNENAVN_CACHE and last_error:
            raise last_error
            
    except Exception as e:
        # If loading fails, return empty dict (will fall back to default)
        import sys
        print(f"Warning: Failed to load kommune codes from {_CSV_FILE}: {e}", file=sys.stderr)
        _KOMMUNENAVN_CACHE = {}
    
    return _KOMMUNENAVN_CACHE


def kommunenavn(kid) -> str:
    """
    Translate kommune ID to kommune name.
    
    Args:
        kid: Kommune ID (int, str, or other numeric type)
        
    Returns:
        Kommune name as string, or "en norsk kommune" if ID is unknown or 9999 (Uoppgitt)
    """
    kommunenavn_dict = _load_kommunenavn()
    
    try:
        kommune_id = int(kid)
        # Treat 9999 (Uoppgitt/Unspecified) as unknown and return default
        if kommune_id == 9999:
            return "en norsk kommune"
        return kommunenavn_dict.get(kommune_id, "en norsk kommune")
    except (ValueError, TypeError):
        return "en norsk kommune"


# Expose the dictionary for backward compatibility
# Load it when module is imported
KOMMUNENAVN: Dict[int, str] = _load_kommunenavn()

