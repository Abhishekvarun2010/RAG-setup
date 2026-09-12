"""
Ingestion module for document processing, parsing, and chunking.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apps.ingestion.models import (
        AccessLevel,
        Chunk,
        DocumentFormat,
        DocumentType,
        LineOfBusiness,
        SourceType,
        Status,
    )
    from apps.ingestion.parsers import (
        BaseParser,
        DocumentParser,
        ParsedDocument,
        ParsedPage,
        PDFParser,
    )
    from apps.ingestion.structure import (
        BlockType,
        StructuredBlock,
        StructuredDocument,
        StructureExtractor,
    )
    from apps.ingestion.chunking import (
        BaseChunker,
        Chunker,
        ChunkingRouter,
        ClaimsChunker,
        FAQChunker,
        PolicyChunker,
    )
    from apps.ingestion.embeddings import (
        BaseEmbeddingProvider,
        EmbeddingConnectionError,
        EmbeddingError,
        EmbeddingModelNotFoundError,
        EmbeddingProvider,
        OllamaEmbeddingProvider,
    )

__all__ = [
    "AccessLevel",
    "BaseChunker",
    "BaseEmbeddingProvider",
    "BaseParser",
    "BlockType",
    "Chunk",
    "Chunker",
    "ChunkingRouter",
    "ClaimsChunker",
    "DocumentFormat",
    "DocumentParser",
    "DocumentType",
    "EmbeddingConnectionError",
    "EmbeddingError",
    "EmbeddingModelNotFoundError",
    "EmbeddingProvider",
    "FAQChunker",
    "LineOfBusiness",
    "OllamaEmbeddingProvider",
    "ParsedDocument",
    "ParsedPage",
    "PDFParser",
    "PolicyChunker",
    "SourceType",
    "Status",
    "StructuredBlock",
    "StructuredDocument",
    "StructureExtractor",
]


def __getattr__(name: str):
    if name in {
        "AccessLevel",
        "Chunk",
        "DocumentFormat",
        "DocumentType",
        "LineOfBusiness",
        "SourceType",
        "Status",
    }:
        import apps.ingestion.models as models
        return getattr(models, name)
    if name in {
        "BaseParser",
        "DocumentParser",
        "ParsedDocument",
        "ParsedPage",
        "PDFParser",
    }:
        import apps.ingestion.parsers as parsers
        return getattr(parsers, name)
    if name in {
        "BlockType",
        "StructuredBlock",
        "StructuredDocument",
        "StructureExtractor",
    }:
        import apps.ingestion.structure as structure
        return getattr(structure, name)
    if name in {
        "BaseChunker",
        "Chunker",
        "ChunkingRouter",
        "ClaimsChunker",
        "FAQChunker",
        "PolicyChunker",
    }:
        import apps.ingestion.chunking as chunking
        return getattr(chunking, name)
    if name in {
        "BaseEmbeddingProvider",
        "EmbeddingConnectionError",
        "EmbeddingError",
        "EmbeddingModelNotFoundError",
        "EmbeddingProvider",
        "OllamaEmbeddingProvider",
    }:
        import apps.ingestion.embeddings as embeddings
        return getattr(embeddings, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
