"""
Structure extraction layer for document ingestion.

Extracts layout-aware structural blocks (headings, paragraphs, tables, lists) from
parsed documents or directly from PDF files using PyMuPDF (fitz) layout and table engines.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure project root is on sys.path if run directly
_current = Path(__file__).resolve()
for _parent in [_current.parents[2], _current.parents[3]]:
    if (_parent / "apps").exists() and str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))

import pymupdf

from apps.ingestion.parsers.base import ParsedDocument, ParsedPage
from apps.ingestion.structure.models import BlockType, StructuredBlock, StructuredDocument

# Regex matching common bullet markers, middle dots, dashes, or numbered/lettered lists
LIST_PATTERN = re.compile(
    r"^\s*([•·–—\*\-▪▫◦\u2022\u00b7]|\d+[\.\)]|[a-zA-Z][\.\)]|\([a-zA-Z0-9]+\))\s+"
)


class StructureExtractor:
    """
    Extracts structural blocks (headings, paragraphs, tables, lists) from documents
    using layout, table analysis, and typography signals from PyMuPDF.

    Workflow:
        PDF / ParsedDocument
           │
           ├── extract tables via PyMuPDF table finder
           │      ↓
           │   TABLE blocks (with headers & rows preserved in metadata)
           │
           └── extract text blocks (excluding table bounding boxes)
                  ↓
               classify as HEADING, PARAGRAPH, or LIST
                  ↓
           sort all blocks by reading order (vertical position)
                  ↓
        StructuredDocument (containing ordered StructuredBlock items)
    """

    def extract(
        self,
        source: Union[ParsedDocument, str, Path],
        document_id: Optional[str] = None,
        **kwargs: Any,
    ) -> StructuredDocument:
        """
        Extract structural blocks from a ParsedDocument or a PDF file path.

        Args:
            source: Either an existing ParsedDocument or a Path/str to a PDF file.
            document_id: Optional explicit document identifier (overrides source).
            **kwargs: Additional metadata to merge into StructuredDocument.metadata.

        Returns:
            StructuredDocument containing classified StructuredBlock items.
        """
        if isinstance(source, ParsedDocument):
            return self._extract_from_parsed_document(source, explicit_id=document_id, **kwargs)
        else:
            return self._extract_from_pdf_path(source, explicit_id=document_id, **kwargs)

    def _extract_from_parsed_document(
        self,
        parsed_doc: ParsedDocument,
        explicit_id: Optional[str] = None,
        **kwargs: Any,
    ) -> StructuredDocument:
        """Extract structure from a ParsedDocument, using disk PDF if available or text fallback."""
        doc_id = explicit_id or parsed_doc.document_id
        source_path = Path(parsed_doc.source_uri) if parsed_doc.source_uri else None

        domain_metadata = {
            "document_type": parsed_doc.document_type,
            "source_type": parsed_doc.source_type,
            "policy_id": parsed_doc.policy_id,
            "claim_id": parsed_doc.claim_id,
            "policyholder_id": parsed_doc.policyholder_id,
            "line_of_business": parsed_doc.line_of_business,
        }

        # If a real PDF file exists at source_uri, leverage high-fidelity PyMuPDF layout analysis
        if source_path and source_path.exists() and source_path.suffix.lower() == ".pdf":
            doc = self._extract_from_pdf_path(source_path, explicit_id=doc_id, **kwargs)
            # Merge ParsedDocument metadata and domain fields
            merged_metadata = {
                **domain_metadata,
                **parsed_doc.metadata,
                **doc.metadata,
                **kwargs.get("metadata", {}),
            }
            doc.metadata = merged_metadata
            return doc

        # Fallback: extract structure from in-memory ParsedDocument pages / raw text
        fallback_doc = self._extract_from_text_pages(parsed_doc, explicit_id=doc_id, **kwargs)
        fallback_doc.metadata = {**domain_metadata, **fallback_doc.metadata}
        return fallback_doc

    def _extract_from_pdf_path(
        self,
        file_path: Union[str, Path],
        explicit_id: Optional[str] = None,
        **kwargs: Any,
    ) -> StructuredDocument:
        """Extract structure directly from a PDF file using PyMuPDF layout and table engines."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {path.name}")

        doc_id = explicit_id or f"DOC-{path.stem.upper()}"
        pdf_doc = pymupdf.open(str(path))
        total_pages = len(pdf_doc)
        all_blocks: List[StructuredBlock] = []

        try:
            for page_idx, page in enumerate(pdf_doc):
                page_num = page_idx + 1
                page_blocks = self._extract_blocks_from_page(page, page_num=page_num)
                all_blocks.extend(page_blocks)
        finally:
            pdf_doc.close()

        metadata: Dict[str, Any] = {
            "source_uri": str(path),
            "total_pages": total_pages,
            "total_blocks": len(all_blocks),
            "extractor": "StructureExtractor",
            **kwargs.get("metadata", {}),
        }

        return StructuredDocument(
            document_id=doc_id,
            blocks=all_blocks,
            metadata=metadata,
        )

    def _extract_blocks_from_page(
        self,
        page: pymupdf.Page,
        page_num: int,
    ) -> List[StructuredBlock]:
        """Inspect PyMuPDF tables and text blocks on a single page, classifying and ordering them."""
        # 1. Extract tables first
        table_blocks, table_bboxes = self._extract_tables_from_page(page, page_num=page_num)

        # 2. Extract text blocks, suppressing any text that falls inside a detected table
        page_dict = page.get_text("dict")
        blocks_data = page_dict.get("blocks", [])
        text_blocks: List[StructuredBlock] = []

        for block in blocks_data:
            # PyMuPDF block type 0 is text (type 1 is image)
            if block.get("type") != 0:
                continue

            b_bbox = block.get("bbox", (0, 0, 0, 0))

            # Suppress text blocks located inside detected table bounding boxes
            if self._is_inside_table(b_bbox, table_bboxes):
                continue

            lines = block.get("lines", [])
            if not lines:
                continue

            # Extract full text lines while preserving line breaks
            line_strings: List[str] = []
            all_spans: List[Dict[str, Any]] = []

            for line in lines:
                spans = line.get("spans", [])
                line_text = "".join(span.get("text", "") for span in spans).rstrip()
                if line_text.strip():
                    line_strings.append(line_text)
                all_spans.extend(spans)

            full_text = "\n".join(line_strings).strip()
            if not full_text:
                continue

            # Typography signals from the first line / primary span
            first_line_spans = lines[0].get("spans", [])
            primary_span = first_line_spans[0] if first_line_spans else (all_spans[0] if all_spans else {})

            font_name = primary_span.get("font", "")
            font_size = float(primary_span.get("size", 10.0))
            flags = int(primary_span.get("flags", 0))

            # Bold detection: PyMuPDF flag bit 4 (value 16) or 'bold' in font name
            is_bold = bool(flags & 16) or ("bold" in font_name.lower())

            # Detect mixed font weights across lines (e.g. Key in bold, Value in regular text)
            has_mixed_weight = False
            if len(lines) > 1:
                has_regular = any("bold" not in s.get("font", "").lower() for l in lines[1:] for s in l.get("spans", []))
                if is_bold and has_regular:
                    has_mixed_weight = True

            bbox = [round(coord, 2) for coord in b_bbox]

            # Classification: LIST vs HEADING vs PARAGRAPH
            if self._is_list_block(line_strings):
                block_type = BlockType.LIST
                block_metadata: Dict[str, Any] = {
                    "bbox": bbox,
                    "font_name": font_name,
                    "font_size": round(font_size, 1),
                    "is_bold": is_bold,
                    "line_count": len(line_strings),
                    "item_count": sum(1 for l in line_strings if LIST_PATTERN.match(l)),
                }
            else:
                # Heading heuristic:
                # 1. Compact: <= 2 lines, <= 80 chars total, does not end with sentence period
                # 2. Emphasized: bold (without being a mixed key-value pair) OR significantly larger font (>= 12.0)
                is_compact = len(line_strings) <= 2 and len(full_text) <= 80 and not full_text.endswith(".")
                is_prominent = (font_size >= 12.0) or (is_bold and not has_mixed_weight)

                if is_compact and is_prominent:
                    block_type = BlockType.HEADING
                else:
                    block_type = BlockType.PARAGRAPH

                block_metadata = {
                    "bbox": bbox,
                    "font_name": font_name,
                    "font_size": round(font_size, 1),
                    "is_bold": is_bold,
                    "line_count": len(line_strings),
                }

            text_blocks.append(
                StructuredBlock(
                    block_type=block_type,
                    content=full_text,
                    page_number=page_num,
                    metadata=block_metadata,
                )
            )

        # 3. Merge table and text blocks and sort by vertical reading order (y0, then x0)
        all_page_blocks = table_blocks + text_blocks
        all_page_blocks.sort(
            key=lambda b: (
                b.metadata.get("bbox", [0, 0, 0, 0])[1],
                b.metadata.get("bbox", [0, 0, 0, 0])[0],
            )
        )

        return all_page_blocks

    def _extract_tables_from_page(
        self,
        page: pymupdf.Page,
        page_num: int,
    ) -> Tuple[List[StructuredBlock], List[Tuple[float, float, float, float]]]:
        """
        Detect and extract tables from a PDF page using PyMuPDF's table engine.

        Returns:
            Tuple of (list of StructuredBlock with block_type=TABLE, list of table bbox tuples).
        """
        table_blocks: List[StructuredBlock] = []
        table_bboxes: List[Tuple[float, float, float, float]] = []

        try:
            tabs = page.find_tables()
        except Exception:
            return table_blocks, table_bboxes

        for table in tabs.tables:
            raw_data = table.extract()
            if not raw_data or len(raw_data) < 1:
                continue

            # Clean and normalize cells
            cleaned_rows: List[List[str]] = []
            for row in raw_data:
                cleaned_row = [str(cell or "").strip().replace("\n", " ") for cell in row]
                # Keep row if it has at least one non-empty cell
                if any(cleaned_row):
                    cleaned_rows.append(cleaned_row)

            if not cleaned_rows:
                continue

            headers = cleaned_rows[0]
            data_rows = cleaned_rows[1:]

            # Format as clean Markdown table string
            markdown_content = self._format_markdown_table(headers, data_rows)

            bbox_coords = [round(c, 2) for c in table.bbox]
            table_bboxes.append(table.bbox)

            metadata: Dict[str, Any] = {
                "bbox": bbox_coords,
                "headers": headers,
                "rows": data_rows,
                "row_count": len(cleaned_rows),
                "col_count": len(headers),
            }

            table_blocks.append(
                StructuredBlock(
                    block_type=BlockType.TABLE,
                    content=markdown_content,
                    page_number=page_num,
                    metadata=metadata,
                )
            )

        return table_blocks, table_bboxes

    @staticmethod
    def _is_inside_table(
        block_bbox: Sequence[float],
        table_bboxes: List[Tuple[float, float, float, float]],
        tolerance: float = 4.0,
    ) -> bool:
        """Check if a text block's center point is situated within any table's bounding box."""
        if not table_bboxes or len(block_bbox) < 4:
            return False
        bx0, by0, bx1, by1 = block_bbox[:4]
        cx = (bx0 + bx1) / 2.0
        cy = (by0 + by1) / 2.0

        for tx0, ty0, tx1, ty1 in table_bboxes:
            if (tx0 - tolerance <= cx <= tx1 + tolerance) and (ty0 - tolerance <= cy <= ty1 + tolerance):
                return True
        return False

    @staticmethod
    def _is_list_block(lines: List[str]) -> bool:
        """Determine if a group of lines forms an ordered or bulleted list."""
        if not lines:
            return False
        bullet_count = sum(1 for line in lines if LIST_PATTERN.match(line.strip()))
        if len(lines) == 1:
            return bool(LIST_PATTERN.match(lines[0].strip()))
        # If multi-line, at least 2 lines or > 50% must match list patterns
        return bullet_count >= 2 or (bullet_count / len(lines) >= 0.5)

    @staticmethod
    def _format_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
        """Format 2D table grid into a standard Markdown table."""
        col_count = max(len(headers), max((len(r) for r in rows), default=0))
        if col_count == 0:
            return ""

        h_padded = headers + [""] * (col_count - len(headers))
        lines = [
            "| " + " | ".join(h_padded) + " |",
            "| " + " | ".join(["---"] * col_count) + " |",
        ]
        for row in rows:
            r_padded = row + [""] * (col_count - len(row))
            lines.append("| " + " | ".join(r_padded) + " |")

        return "\n".join(lines)

    def _extract_from_text_pages(
        self,
        parsed_doc: ParsedDocument,
        explicit_id: Optional[str] = None,
        **kwargs: Any,
    ) -> StructuredDocument:
        """Fallback extraction for in-memory or text-only ParsedDocuments without PDF backing."""
        all_blocks: List[StructuredBlock] = []

        if parsed_doc.pages:
            for page in parsed_doc.pages:
                blocks = self._extract_blocks_from_raw_text(page.content, page_num=page.page_number)
                all_blocks.extend(blocks)
        elif parsed_doc.raw_text:
            all_blocks = self._extract_blocks_from_raw_text(parsed_doc.raw_text, page_num=1)

        metadata: Dict[str, Any] = {
            **parsed_doc.metadata,
            "total_blocks": len(all_blocks),
            "extractor": "StructureExtractorFallback",
            **kwargs.get("metadata", {}),
        }

        return StructuredDocument(
            document_id=explicit_id or parsed_doc.document_id,
            blocks=all_blocks,
            metadata=metadata,
        )

    def _extract_blocks_from_raw_text(
        self,
        text: str,
        page_num: int,
    ) -> List[StructuredBlock]:
        """Heuristic splitting and classification of raw text chunks (supporting tables and lists)."""
        blocks: List[StructuredBlock] = []
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]

        for chunk in chunks:
            lines = [l.strip() for l in chunk.split("\n") if l.strip()]
            if not lines:
                continue

            # Check for Markdown Table
            if len(lines) >= 2 and "|" in lines[0] and any("---" in l for l in lines[1:3]):
                headers = [c.strip() for c in lines[0].strip("|").split("|")]
                data_lines = [l for l in lines[1:] if not re.match(r"^\|?(\s*:?-+:?\s*\|?)+$", l)]
                data_rows = [[c.strip() for c in l.strip("|").split("|")] for l in data_lines]
                blocks.append(
                    StructuredBlock(
                        block_type=BlockType.TABLE,
                        content=chunk,
                        page_number=page_num,
                        metadata={
                            "headers": headers,
                            "rows": data_rows,
                            "row_count": len(lines),
                            "col_count": len(headers),
                        },
                    )
                )
                continue

            # Check for List
            if self._is_list_block(lines):
                blocks.append(
                    StructuredBlock(
                        block_type=BlockType.LIST,
                        content=chunk,
                        page_number=page_num,
                        metadata={
                            "line_count": len(lines),
                            "item_count": sum(1 for l in lines if LIST_PATTERN.match(l)),
                        },
                    )
                )
                continue

            # Heading heuristic for plain text:
            # - Starts with '#' (markdown heading), OR
            # - Single short line (<= 60 chars), no trailing period, title/all-caps or titled
            first_line = lines[0]
            is_heading = False

            if first_line.startswith("#"):
                is_heading = True
                chunk = chunk.lstrip("#").strip()
            elif len(lines) == 1 and len(chunk) <= 60 and not chunk.endswith((".", ":", ";")):
                if chunk.isupper() or chunk.istitle() or any(w in chunk.lower() for w in ["declarations", "coverage", "schedule", "exclusions"]):
                    is_heading = True

            block_type = BlockType.HEADING if is_heading else BlockType.PARAGRAPH
            blocks.append(
                StructuredBlock(
                    block_type=block_type,
                    content=chunk,
                    page_number=page_num,
                    metadata={"line_count": len(lines)},
                )
            )

        return blocks
