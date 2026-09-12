"""
Data models representing the structural decomposition of documents.

This module defines:
- BlockType: Enumeration of supported structural block classifications.
- StructuredBlock: A single structural unit (e.g. heading, paragraph) with content,
  page number, and layout/font metadata.
- StructuredDocument: A collection of ordered structural blocks representing a full document.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class BlockType(str, Enum):
    """
    Classification of structural blocks within a document.
    
    Inheriting from (str, Enum) ensures clean JSON serialization and
    type-safe comparisons (e.g. `block.block_type == BlockType.HEADING`
    or `block.block_type == "heading"`).
    """

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"


class StructuredBlock(BaseModel):
    """
    Represents an atomic structural unit extracted from a document page.

    Attributes:
        block_type: Classification of the block (heading, paragraph, table, list).
        content: Extracted text content of the block.
        page_number: 1-indexed page number where the block appears.
        metadata: Block-specific layout, bounding box, font, and positioning attributes.
    """

    model_config = ConfigDict(
        use_enum_values=False,
        str_strip_whitespace=False,
        validate_assignment=True,
        extra="forbid",
    )

    block_type: BlockType = Field(
        ...,
        description="Type of structural block (heading, paragraph, table, list).",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Textual content of this structural block.",
    )
    page_number: int = Field(
        ...,
        ge=1,
        description="1-indexed page number in the original document.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Layout and font metadata (e.g. bbox, font_name, font_size, is_bold).",
    )


class StructuredDocument(BaseModel):
    """
    Standardized structural representation of an ingested document containing
    an ordered sequence of structural blocks.

    Attributes:
        document_id: Unique identifier for the parent document.
        blocks: Ordered list of structural blocks across all pages.
        metadata: Document-level metadata (e.g. source_uri, total_pages, parser).
    """

    model_config = ConfigDict(
        use_enum_values=False,
        str_strip_whitespace=False,
        validate_assignment=True,
        extra="forbid",
    )

    document_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier of the document (e.g. 'DOC-MOT-0000001-DEC').",
    )
    blocks: List[StructuredBlock] = Field(
        default_factory=list,
        description="Ordered list of structural blocks extracted from the document.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document-level metadata (e.g. total_pages, source_uri, parser).",
    )

    @property
    def page_count(self) -> int:
        """Return total page count inferred from metadata or blocks."""
        if "total_pages" in self.metadata and isinstance(self.metadata["total_pages"], int):
            return self.metadata["total_pages"]
        if self.blocks:
            return max(block.page_number for block in self.blocks)
        return 0

    @property
    def headings(self) -> List[StructuredBlock]:
        """Return all heading blocks in the document."""
        return [b for b in self.blocks if b.block_type == BlockType.HEADING]

    @property
    def paragraphs(self) -> List[StructuredBlock]:
        """Return all paragraph blocks in the document."""
        return [b for b in self.blocks if b.block_type == BlockType.PARAGRAPH]

    @property
    def tables(self) -> List[StructuredBlock]:
        """Return all table blocks in the document."""
        return [b for b in self.blocks if b.block_type == BlockType.TABLE]

    @property
    def lists(self) -> List[StructuredBlock]:
        """Return all list blocks in the document."""
        return [b for b in self.blocks if b.block_type == BlockType.LIST]

    def get_blocks_by_page(self, page_number: int) -> List[StructuredBlock]:
        """Return all blocks located on the specified 1-indexed page number."""
        return [b for b in self.blocks if b.page_number == page_number]
