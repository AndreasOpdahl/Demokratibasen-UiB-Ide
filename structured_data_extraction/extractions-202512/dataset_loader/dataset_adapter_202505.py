"""
Dataset adapter for 2025-05 format (field name translation).

Supports backward compatibility by accepting both old and new field names.
"""

from typing import Dict, Any, Optional
from .kommune import kommunenavn


class DatasetAdapter202505:
    """
    Adapter class for translating field names from various formats to a normalized format.
    
    Supports backward compatibility by accepting both old and new field names:
    - dok_id or dokument_id -> dok_id
    - tekst or text -> tekst
    - dok_type or doc_type -> dok_type
    - dok_tittel or tittel -> dok_tittel
    """
    
    @staticmethod
    def get_dok_id(doc: Dict[str, Any]) -> str:
        """Extract document ID, supporting both new and old field names."""
        return doc.get("dok_id") or doc.get("dokument_id", "")
    
    @staticmethod
    def get_tekst(doc: Dict[str, Any]) -> str:
        """Extract text content, preferring 'tekst' over 'text'."""
        return doc.get("tekst") or doc.get("text", "")
    
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
        kommune_nummer = DatasetAdapter202505.get_kommune_nummer(doc)
        if kommune_nummer is None:
            return "en norsk kommune"
        return kommunenavn(kommune_nummer)
    
    @staticmethod
    def get_url(doc: Dict[str, Any]) -> Optional[str]:
        """Extract URL if present."""
        return doc.get("url")
    
    @staticmethod
    def get_dok_type(doc: Dict[str, Any]) -> Optional[str]:
        """Extract document type, supporting both new and old field names."""
        return doc.get("dok_type") or doc.get("doc_type")
    
    @staticmethod
    def get_dok_tittel(doc: Dict[str, Any]) -> Optional[str]:
        """Extract document title, supporting both new and old field names."""
        return doc.get("dok_tittel") or doc.get("tittel")
    
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

