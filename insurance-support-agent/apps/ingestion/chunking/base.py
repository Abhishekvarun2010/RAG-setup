"""
Base interfaces and common utilities for document-type-aware chunking.

Defines:
- Chunker: Runtime-checkable Protocol for document chunking strategies.
- BaseChunker: Abstract base class implementing shared chunk creation,
  metadata inheritance, and token sizing logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable
from uuid import uuid4

from apps.ingestion.models import (
    Chunk,
    DocumentType,
    LineOfBusiness,
    SourceType,
    Status,
)
from apps.ingestion.structure.models import StructuredDocument


@runtime_checkable
class Chunker(Protocol):
    """
    Protocol defining the interface for document chunkers.

    Any class implementing `chunk(document: StructuredDocument, **kwargs) -> list[Chunk]`
    satisfies this protocol.
    """

    def chunk(self, document: StructuredDocument, **kwargs: Any) -> List[Chunk]:
        """
        Segment a StructuredDocument into semantically coherent, searchable Chunk objects.

        Args:
            document: StructuredDocument containing typed structural blocks.
            **kwargs: Strategy-specific chunking options.

        Returns:
            List of validated Chunk objects.
        """
        ...


class BaseChunker(ABC):
    """
    Abstract base class for document chunkers.

    Provides common functionality for token budget estimation, domain metadata
    inheritance, and Chunk instance creation.
    """

    def __init__(self, max_tokens: int = 500, token_overlap: int = 0):
        """
        Initialize the base chunker.

        Args:
            max_tokens: Target maximum tokens per chunk (default 500 tokens ~ 2000 chars).
            token_overlap: Optional token overlap between split chunks.
        """
        self.max_tokens = max_tokens
        self.token_overlap = token_overlap

    @abstractmethod
    def chunk(self, document: StructuredDocument, **kwargs: Any) -> List[Chunk]:
        """Subclasses must implement the document-type-specific chunking algorithm."""
        pass

    def estimate_tokens(self, text: str) -> int:
        """
        Heuristic token count estimation.
        Approximately 4 characters per token for standard English text with punctuation.
        """
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _create_chunk(
        self,
        document: StructuredDocument,
        content: str,
        section: Optional[str] = None,
        page_number: Optional[int] = None,
        chunk_index: Optional[int] = None,
        **extra_kwargs: Any,
    ) -> Chunk:
        """
        Construct a validated Chunk, inheriting identifiers and entity metadata
        from the parent StructuredDocument.

        Args:
            document: Parent StructuredDocument.
            content: Textual content of the chunk.
            section: Name of the governing section or heading.
            page_number: 1-indexed page number where the chunk appears.
            chunk_index: 1-indexed sequence number within the document.
            **extra_kwargs: Optional overrides for Chunk fields.
        """
        doc_meta = document.metadata or {}

        # Resolve DocumentType
        raw_doc_type = extra_kwargs.get("document_type") or doc_meta.get("document_type")
        if isinstance(raw_doc_type, DocumentType):
            doc_type = raw_doc_type
        elif isinstance(raw_doc_type, str):
            try:
                doc_type = DocumentType(raw_doc_type)
            except ValueError:
                doc_type = DocumentType.OTHER
        else:
            doc_type = DocumentType.OTHER

        # Resolve SourceType
        raw_src_type = extra_kwargs.get("source_type") or doc_meta.get("source_type")
        if isinstance(raw_src_type, SourceType):
            src_type = raw_src_type
        elif isinstance(raw_src_type, str):
            try:
                src_type = SourceType(raw_src_type)
            except ValueError:
                src_type = SourceType.PDF
        else:
            src_type = SourceType.PDF

        # Resolve LineOfBusiness
        raw_lob = extra_kwargs.get("line_of_business") or doc_meta.get("line_of_business")
        if isinstance(raw_lob, LineOfBusiness):
            lob = raw_lob
        elif isinstance(raw_lob, str):
            try:
                lob = LineOfBusiness(raw_lob)
            except ValueError:
                lob = None
        else:
            lob = None

        # Resolve other domain entities
        policy_id = extra_kwargs.get("policy_id") or doc_meta.get("policy_id")
        claim_id = extra_kwargs.get("claim_id") or doc_meta.get("claim_id")
        policyholder_id = extra_kwargs.get("policyholder_id") or doc_meta.get("policyholder_id")
        source_uri = extra_kwargs.get("source_uri") or doc_meta.get("source_uri")

        # Generate chunk_id
        if chunk_index is not None:
            chunk_id = f"{document.document_id}#c{chunk_index}"
        else:
            chunk_id = f"chk_{uuid4().hex[:12]}"

        clean_content = content.strip()
        if not clean_content:
            clean_content = "[Empty Section]"

        return Chunk(
            chunk_id=chunk_id,
            content=clean_content,
            document_id=document.document_id,
            document_type=doc_type,
            source_type=src_type,
            policy_id=policy_id,
            claim_id=claim_id,
            policyholder_id=policyholder_id,
            line_of_business=lob,
            section=section,
            page_number=page_number,
            source_uri=source_uri,
        )
