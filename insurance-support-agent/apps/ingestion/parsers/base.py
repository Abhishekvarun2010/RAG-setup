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

from pydantic import BaseModel, ConfigDict, Field, model_validator

from apps.ingestion.models import DocumentType, LineOfBusiness, SourceType


# =====================================================================
# Parsed Document Models
# =====================================================================

class ParsedPage(BaseModel):
    """
    Represents text and metadata extracted from a single page of a document.
    """

    model_config = ConfigDict(
        str_strip_whitespace=False,  # Preserve structure, line breaks, and column alignment
        validate_assignment=True,
        extra="forbid",
    )

    page_number: int = Field(
        ...,
        ge=1,
        description="1-indexed page number in the original document.",
    )
    content: str = Field(
        default="",
        description="Textual content extracted from this page (can be empty for image/scanned pages).",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional page-specific metadata (e.g. dimensions, headers).",
    )


class ParsedDocument(BaseModel):
    """
    Standardized in-memory representation of an ingested document
    produced by a DocumentParser before chunking.
    """

    model_config = ConfigDict(
        use_enum_values=False,
        str_strip_whitespace=False,  # Preserve layout structure
        validate_assignment=True,
        extra="forbid",
    )

    # Text content fields (synonymous and kept synchronized; empty string for scanned PDFs)
    raw_text: str = Field(
        default="",
        description="Full raw extracted textual content of the document (empty if scanned/unextractable).",
    )
    content: str = Field(
        default="",
        description="Extracted textual content of the document (synchronized with raw_text).",
    )

    # Core required identifiers
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

    # Source provenance & structural attributes
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
    total_pages: int = Field(
        default=0,
        description="Total page count of the source document.",
    )
    has_text: bool = Field(
        default=False,
        description="Whether the document contains extractable text (False for scanned documents).",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata extracted during parsing (e.g. total_pages, parser, has_text).",
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

    @classmethod
    def _sync_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Synchronize raw_text/content, has_text, and total_pages."""
        # 1. Sync raw_text and content
        raw = data.get("raw_text")
        cnt = data.get("content")
        if raw is not None and cnt is None:
            data["content"] = raw
        elif cnt is not None and raw is None:
            data["raw_text"] = cnt
        elif raw is None and cnt is None:
            data["raw_text"] = ""
            data["content"] = ""

        # 2. Compute has_text
        text_val = data.get("raw_text") or data.get("content") or ""
        computed_has_text = bool(text_val.strip())
        if "has_text" not in data:
            data["has_text"] = computed_has_text

        # 3. Compute total_pages
        pages = data.get("pages") or []
        if "total_pages" not in data or data["total_pages"] == 0:
            data["total_pages"] = len(pages)

        # 4. Populate metadata dict
        meta = data.setdefault("metadata", {})
        if "has_text" not in meta:
            meta["has_text"] = data["has_text"]
        if "total_pages" not in meta:
            meta["total_pages"] = data["total_pages"]

        return data

    @property
    def page_count(self) -> int:
        """Return the number of parsed pages, or 1 if unpaged."""
        if self.total_pages > 0:
            return self.total_pages
        return len(self.pages) if self.pages else 1

    @model_validator(mode="before")
    @classmethod
    def _run_sync_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = cls._sync_fields(data)
        return data


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
