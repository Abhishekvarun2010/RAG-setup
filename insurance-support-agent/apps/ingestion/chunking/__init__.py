"""
Document-type-aware chunking package for insurance documents.

Provides specialized chunkers and routing logic:
- Chunker: Protocol for all chunking strategies.
- BaseChunker: Abstract base with token sizing and chunk creation.
- PolicyChunker: Section-aware chunker for policies, contracts, declarations, schedules.
- FAQChunker: Question-Answer pairing strategy with question repetition on splits.
- ClaimsChunker: Section-preserving strategy for claims, adjuster reports, estimates.
- ChunkingRouter: Dispatches StructuredDocuments to appropriate chunkers by DocumentType.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apps.ingestion.chunking.base import BaseChunker, Chunker
    from apps.ingestion.chunking.claims import ClaimsChunker
    from apps.ingestion.chunking.faq import FAQChunker
    from apps.ingestion.chunking.policy import PolicyChunker
    from apps.ingestion.chunking.router import ChunkingRouter

__all__ = [
    "BaseChunker",
    "Chunker",
    "ChunkingRouter",
    "ClaimsChunker",
    "FAQChunker",
    "PolicyChunker",
]


def __getattr__(name: str):
    if name in {"BaseChunker", "Chunker"}:
        import apps.ingestion.chunking.base as base
        return getattr(base, name)
    if name == "PolicyChunker":
        import apps.ingestion.chunking.policy as policy
        return getattr(policy, name)
    if name == "FAQChunker":
        import apps.ingestion.chunking.faq as faq
        return getattr(faq, name)
    if name == "ClaimsChunker":
        import apps.ingestion.chunking.claims as claims
        return getattr(claims, name)
    if name == "ChunkingRouter":
        import apps.ingestion.chunking.router as router
        return getattr(router, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
