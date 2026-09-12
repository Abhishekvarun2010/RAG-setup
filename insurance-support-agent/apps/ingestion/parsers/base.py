"""
Base parser interface, protocols, and document data models for document ingestion.

This module defines:
- ParsedPage: Single page extracted content (for paged formats like PDF/Word).
- ParsedDocument: Standardized representation of an ingested document before chunking.
- DocumentParser: @runtime_checkable Protocol defining the parser interface.
- BaseParser: Abstract base class with default extension-matching logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Union, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from apps.ingestion.models import DocumentType, LineOfBusiness, SourceType


# =====================================================================
# Parsed Document Models
# =====================================================================

class ParsedPage(BaseModel):
    """
    Represents text and metadata extracted from a single page of a document.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    page_number: int = Field(
        ...,
        ge=1,
        description="1-indexed page number in the original document.",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Textual content extracted from this page.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional page-specific metadata (e.g. headers, footers).",
    )


class ParsedDocument(BaseModel):
    """
    Standardized in-memory representation of an ingested document
    produced by a DocumentParser before chunking.
    """

    model_config = ConfigDict(
        use_enum_values=False,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    # Core required fields
    content: str = Field(
        ...,
        min_length=1,
        description="Full extracted textual content of the document.",
    )
    document_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier of the document (e.g. 'DOC-KB-FAQ', 'DOC-C-1000-ADJ').",
    )
    document_type: DocumentType = Field(
        ...,
        description="Classification of the document (e.g. policy_contract, customer_faq, fnol).",
    )
    source_type: SourceType = Field(
        ...,
        description="Source format or domain category (e.g. pdf, docx, markdown, kb, policy).",
    )

    # Optional source provenance & structural attributes
    source_uri: Optional[str] = Field(
        default=None,
        description="File path or URI where the original document is located.",
    )
    title: Optional[str] = Field(
        default=None,
        description="Extracted document title or heading, if available.",
    )
    pages: List[ParsedPage] = Field(
        default_factory=list,
        description="Optional list of page-by-page extractions (for paged PDF/DOCX documents).",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary document metadata extracted during parsing.",
    )

    # Optional insurance domain entity identifiers
    policy_id: Optional[str] = Field(
        default=None,
        description="Associated policy ID, if resolved during parsing.",
    )
    claim_id: Optional[str] = Field(
        default=None,
        description="Associated claim ID, if resolved during parsing.",
    )
    policyholder_id: Optional[str] = Field(
        default=None,
        description="Associated policyholder ID, if resolved during parsing.",
    )
    line_of_business: Optional[LineOfBusiness] = Field(
        default=None,
        description="Associated line of business, if resolved during parsing.",
    )

    # Ingestion timestamp
    parsed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the document was parsed.",
    )

    @property
    def page_count(self) -> int:
        """Return the number of parsed pages, or 1 if unpaged."""
        return len(self.pages) if self.pages else 1


# =====================================================================
# Parser Interface & Abstract Base Class
# =====================================================================

@runtime_checkable
class DocumentParser(Protocol):
    """
    Protocol defining the interface for document parsers.
    
    Any parser class that implements `parse` and `can_parse`
    will satisfy this protocol (structural subtyping / duck typing).
    Using @runtime_checkable allows `isinstance(parser, DocumentParser)` checks.
    """

    def parse(self, file_path: Union[str, Path], **kwargs: Any) -> ParsedDocument:
        """
        Parse a document file from disk into a standardized ParsedDocument.
        
        Args:
            file_path: Path or string URI to the document file.
            **kwargs: Parser-specific options.
            
        Returns:
            ParsedDocument containing extracted content and metadata.
        """
        ...

    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """
        Check whether this parser supports the given file.
        
        Args:
            file_path: Path or string URI to the candidate document.
            
        Returns:
            True if this parser can handle the file, False otherwise.
        """
        ...


class BaseParser(ABC):
    """
    Abstract base class providing default extension-matching logic.
    
    Subclasses should define `supported_extensions` and implement `parse()`.
    """

    supported_extensions: Sequence[str] = ()

    @abstractmethod
    def parse(self, file_path: Union[str, Path], **kwargs: Any) -> ParsedDocument:
        """Parse a document file from disk into a standardized ParsedDocument."""
        pass

    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """
        Check if the file's extension matches any extension in `supported_extensions`.
        Handles extensions both with and without leading dots case-insensitively.
        """
        ext = Path(file_path).suffix.lower()
        if not ext:
            return False
        normalized_supported = {
            e.lower() if e.startswith(".") else f".{e.lower()}"
            for e in self.supported_extensions
        }
        return ext in normalized_supported
