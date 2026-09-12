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

__all__ = [
    "AccessLevel",
    "BaseParser",
    "BlockType",
    "Chunk",
    "DocumentFormat",
    "DocumentParser",
    "DocumentType",
    "LineOfBusiness",
    "ParsedDocument",
    "ParsedPage",
    "PDFParser",
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
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
