"""
Dataset adapter for Bergen 2017-2023 format.

This adapter handles the JSON format from Kommunebasen-Bergen/ekstraher_tekster_og_typer/dok_tekster,
which uses the following fields:
- dok_id: document identifier string
- kommune: municipality code (four-digit string, e.g., "4601")
- url: source URL
- dok_type: document type (e.g., "case_minutes", "case_presentation")
- dok_tittel: document title
- filformat: file format (e.g., "doc", "pdf") - not used in normalization
- tekst: extracted text content
"""

from typing import Dict, Any, Optional

# Support both relative and absolute imports for testing
try:
    from .kommune import kommunenavn
except ImportError:
    from kommune import kommunenavn


# Document types to include when loading documents
DOCUMENT_TYPES_TO_INFER = [
    "meeting_agenda",
    "meeting_minutes",
    "case_presentation",
    "case_minutes",
]


class DatasetAdapterBergen2017_2023:
    """
    Adapter class for translating Bergen 2017-2023 format to normalized format.
    
    The Bergen format uses field names that match the normalized format:
    - dok_id -> dok_id
    - dok_type -> dok_type
    - dok_tittel -> dok_tittel
    - tekst -> tekst
    - kommune (string) -> kommune_nummer (int)
    - kommune (string) -> kommune_navn (translated via kommunenavn function)
    """
    
    @staticmethod
    def get_dok_id(doc: Dict[str, Any]) -> str:
        """Extract document ID."""
        return doc.get("dok_id", "")
    
    @staticmethod
    def get_tekst(doc: Dict[str, Any]) -> str:
        """Extract text content."""
        return doc.get("tekst", "")
    
    @staticmethod
    def get_kommune_nummer(doc: Dict[str, Any]) -> Optional[int]:
        """Extract kommune number (four digits) from document."""
        kommune_value = doc.get("kommune")
        if kommune_value is None:
            return None
        try:
            return int(kommune_value)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def get_kommune_navn(doc: Dict[str, Any]) -> str:
        """Extract and translate kommune number to kommune name."""
        kommune_nummer = DatasetAdapterBergen2017_2023.get_kommune_nummer(doc)
        if kommune_nummer is None:
            return "en norsk kommune"
        return kommunenavn(kommune_nummer)
    
    @staticmethod
    def get_url(doc: Dict[str, Any]) -> Optional[str]:
        """Extract URL if present."""
        return doc.get("url")
    
    @staticmethod
    def get_dok_type(doc: Dict[str, Any]) -> Optional[str]:
        """Extract document type."""
        return doc.get("dok_type")
    
    @staticmethod
    def get_dok_tittel(doc: Dict[str, Any]) -> Optional[str]:
        """Extract document title."""
        return doc.get("dok_tittel")
    
    @classmethod
    def should_include_document(cls, doc: Dict[str, Any]) -> bool:
        """
        Check if a document should be included based on its document type.
        
        Args:
            doc: Raw document dictionary
            
        Returns:
            True if the document should be included, False otherwise
        """
        doc_type = cls.get_dok_type(doc)
        if doc_type is None:
            return False
        return doc_type in DOCUMENT_TYPES_TO_INFER
    
    @classmethod
    def normalize(cls, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a document by extracting and translating all fields.
        
        Returns a dictionary with normalized field names:
        - dok_id: str
        - kommune_nummer: Optional[int] (four-digit kommune number from input)
        - kommune_navn: str (kommune name translated from number)
        - tekst: str
        - url: Optional[str]
        - dok_type: Optional[str]
        - dok_tittel: Optional[str]
        """
        return {
            "dok_id": cls.get_dok_id(doc),
            "kommune_nummer": cls.get_kommune_nummer(doc),
            "kommune_navn": cls.get_kommune_navn(doc),
            "tekst": cls.get_tekst(doc),
            "url": cls.get_url(doc),
            "dok_type": cls.get_dok_type(doc),
            "dok_tittel": cls.get_dok_tittel(doc),
        }

