"""
Ingestion module for document processing, parsing, and chunking.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apps.ingestion.models import (
        AccessLevel,
        Chunk,
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
    )

__all__ = [
    "AccessLevel",
    "BaseParser",
    "Chunk",
    "DocumentParser",
    "DocumentType",
    "LineOfBusiness",
    "ParsedDocument",
    "ParsedPage",
    "SourceType",
    "Status",
]


def __getattr__(name: str):
    if name in {
        "AccessLevel",
        "Chunk",
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
    }:
        import apps.ingestion.parsers.base as parsers_base
        return getattr(parsers_base, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
