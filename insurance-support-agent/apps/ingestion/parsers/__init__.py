"""
Document parsers package for the insurance ingestion pipeline.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apps.ingestion.parsers.base import (
        BaseParser,
        DocumentParser,
        ParsedDocument,
        ParsedPage,
    )
    from apps.ingestion.parsers.pdf import PDFParser

__all__ = [
    "BaseParser",
    "DocumentParser",
    "ParsedDocument",
    "ParsedPage",
    "PDFParser",
]


def __getattr__(name: str):
    if name in {"BaseParser", "DocumentParser", "ParsedDocument", "ParsedPage"}:
        import apps.ingestion.parsers.base as base
        return getattr(base, name)
    if name == "PDFParser":
        import apps.ingestion.parsers.pdf as pdf
        return getattr(pdf, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
