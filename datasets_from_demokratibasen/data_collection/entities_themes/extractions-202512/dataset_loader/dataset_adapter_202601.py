"""
Dataset adapter for 2026-01 cleaned text summary datasets (149978_text_summary_examples_*).

These files have the same logical structure as the 2025-10 processed_data/text_summary
datasets handled by DatasetAdapter202510:

- input: document text content
- metadata: dictionary with dokument_id, kommune (four-digit number), doc_type, etc.

This adapter mirrors DatasetAdapter202510 as closely as possible so it can be used
with the unified DatasetLoader and analyse_dataset tooling.
"""

from typing import Dict, Any, Optional
from .kommune import kommunenavn


class DatasetAdapter202601:
    """
    Adapter class for translating 2026-01 cleaned text summary examples
    (149978_text_summary_examples_*.jsonl) to the normalized format.

    Handles structure:
    - input -> tekst
    - metadata.dokument_id -> dok_id
    - metadata.kommune -> kommune_nummer (four-digit number from input, variable name)
    - metadata.kommune -> kommune_navn (translated to name via kommunenavn function)
    - metadata.doc_type -> dok_type
    - metadata.tittel or metadata.dok_tittel -> dok_tittel
    """

    @staticmethod
    def get_dok_id(doc: Dict[str, Any]) -> str:
        """Extract document ID from metadata.dokument_id."""
        metadata = doc.get("metadata", {})
        if isinstance(metadata, dict):
            return metadata.get("dokument_id", "")
        return ""

    @staticmethod
    def get_tekst(doc: Dict[str, Any]) -> str:
        """Extract text content from 'input' field."""
        return doc.get("input", "")

    @staticmethod
    def get_kommune_nummer(doc: Dict[str, Any]) -> Optional[int]:
        """Extract kommune number (four digits) from metadata.kommune."""
        metadata = doc.get("metadata", {})
        if isinstance(metadata, dict):
            kommune_value = metadata.get("kommune")
            if kommune_value is None:
                return None
            try:
                return int(kommune_value)
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def get_kommune_navn(doc: Dict[str, Any]) -> str:
        """Translate kommune number to kommune name using kommunenavn."""
        kommune_nummer = DatasetAdapter202601.get_kommune_nummer(doc)
        if kommune_nummer is None:
            return "en norsk kommune"
        return kommunenavn(kommune_nummer)

    @staticmethod
    def get_url(doc: Dict[str, Any]) -> Optional[str]:
        """Extract URL if present in metadata."""
        metadata = doc.get("metadata", {})
        if isinstance(metadata, dict):
            return metadata.get("url")
        return None

    @staticmethod
    def get_dok_type(doc: Dict[str, Any]) -> Optional[str]:
        """Extract document type from metadata.doc_type."""
        metadata = doc.get("metadata", {})
        if isinstance(metadata, dict):
            return metadata.get("doc_type")
        return None

    @staticmethod
    def get_dok_tittel(doc: Dict[str, Any]) -> Optional[str]:
        """Extract document title if present in metadata."""
        metadata = doc.get("metadata", {})
        if isinstance(metadata, dict):
            return metadata.get("tittel") or metadata.get("dok_tittel")
        return None

    @classmethod
    def normalize(cls, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a 149978_text_summary_examples_* document to standard format.

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

