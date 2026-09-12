"""
Structure extraction package for document ingestion.

Provides models and extractors to identify structural elements (headings, paragraphs,
tables, lists) from parsed documents and raw formats.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apps.ingestion.structure.extractor import StructureExtractor
    from apps.ingestion.structure.models import (
        BlockType,
        StructuredBlock,
        StructuredDocument,
    )

__all__ = [
    "BlockType",
    "StructureExtractor",
    "StructuredBlock",
    "StructuredDocument",
]


def __getattr__(name: str):
    if name in {"BlockType", "StructuredBlock", "StructuredDocument"}:
        import apps.ingestion.structure.models as models
        return getattr(models, name)
    if name == "StructureExtractor":
        import apps.ingestion.structure.extractor as extractor
        return getattr(extractor, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
