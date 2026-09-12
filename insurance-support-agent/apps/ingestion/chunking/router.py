"""
Chunking router that dispatches documents to the appropriate chunking strategy based on DocumentType.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from apps.ingestion.chunking.base import Chunker
from apps.ingestion.chunking.claims import ClaimsChunker
from apps.ingestion.chunking.faq import FAQChunker
from apps.ingestion.chunking.policy import PolicyChunker
from apps.ingestion.models import Chunk, DocumentType
from apps.ingestion.structure.models import StructuredDocument


class ChunkingRouter:
    """
    Router that directs a StructuredDocument to its specialized ChunkingStrategy
    according to its DocumentType.

    Routing Map:
    - Policy (contracts, declarations, endorsements, schedules) -> PolicyChunker
    - Customer FAQ & KB -> FAQChunker
    - Claims (FNOL, adjuster reports, estimates, letters) -> ClaimsChunker
    """

    def __init__(
        self,
        policy_chunker: Optional[Chunker] = None,
        faq_chunker: Optional[Chunker] = None,
        claims_chunker: Optional[Chunker] = None,
    ):
        self.policy_chunker = policy_chunker or PolicyChunker()
        self.faq_chunker = faq_chunker or FAQChunker()
        self.claims_chunker = claims_chunker or ClaimsChunker()

    def get_chunker(self, document_type: Optional[DocumentType | str]) -> Chunker:
        """Resolve the appropriate chunker for a given DocumentType."""
        if document_type is None:
            return self.policy_chunker

        # Normalize string to enum if needed
        if isinstance(document_type, str):
            try:
                doc_type = DocumentType(document_type)
            except ValueError:
                doc_type = DocumentType.OTHER
        else:
            doc_type = document_type

        # 1. Policy Documents
        if doc_type in {
            DocumentType.POLICY_CONTRACT,
            DocumentType.POLICY_DECLARATIONS,
            DocumentType.POLICY_ENDORSEMENTS,
            DocumentType.POLICY_SCHEDULE,
        }:
            return self.policy_chunker

        # 2. FAQ Documents
        if doc_type in {
            DocumentType.CUSTOMER_FAQ,
        }:
            return self.faq_chunker

        # 3. Claims Documents
        if doc_type in {
            DocumentType.FNOL,
            DocumentType.FNOL_SCANNED,
            DocumentType.ADJUSTER_REPORT,
            DocumentType.ESTIMATE,
            DocumentType.SETTLEMENT_LETTER,
            DocumentType.SETTLEMENT_LETTER_SCANNED,
            DocumentType.DENIAL_LETTER,
            DocumentType.DENIAL_LETTER_SCANNED,
            DocumentType.ACCIDENT_STATEMENT,
            DocumentType.ACCIDENT_STATEMENT_SCANNED,
            DocumentType.POLICE_REPORT,
        }:
            return self.claims_chunker

        # Fallback default
        return self.policy_chunker

    def route_and_chunk(
        self,
        document: StructuredDocument,
        document_type: Optional[DocumentType | str] = None,
        **kwargs: Any,
    ) -> List[Chunk]:
        """
        Determine document type, route to the matched chunker, and return generated Chunks.

        Args:
            document: StructuredDocument with classified structural blocks.
            document_type: Optional explicit DocumentType override.
            **kwargs: Extra attributes passed into chunker.

        Returns:
            List of generated Chunk instances.
        """
        # Resolve document_type priority: explicit param > metadata['document_type'] > None
        resolved_type = document_type or document.metadata.get("document_type")
        chunker = self.get_chunker(resolved_type)
        return chunker.chunk(document, document_type=resolved_type, **kwargs)
