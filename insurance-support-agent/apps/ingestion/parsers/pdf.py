"""
PDF document parser implementation using PyMuPDF (fitz).

Extracts page-by-page text, preserves 1-indexed page numbers, retains
layout and line-break structure, and compiles unified raw_text.
Detects whether the document has extractable text (has_text) for routing to OCR.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

# Ensure project root is on sys.path if run directly as a script
_current = Path(__file__).resolve()
for _parent in [_current.parents[3], _current.parents[4]]:
    if (_parent / "apps").exists() and str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))

import pymupdf

from apps.ingestion.models import DocumentFormat, DocumentType, LineOfBusiness, SourceType
from apps.ingestion.parsers.base import BaseParser, ParsedDocument, ParsedPage


class PDFParser(BaseParser):
    """
    Parser for Adobe PDF (.pdf) documents using PyMuPDF.

    Responsibilities:
    - Accepts a PDF file path.
    - Validates file existence and extension.
    - Iterates through every page in the PDF document.
    - Extracts textual content from each page using `page.get_text()` with conservative normalization.
    - Preserves layout/structure without aggressive cleaning (does NOT replace newlines with spaces).
    - Assembles `raw_text = "\\n\\n".join(page_texts).strip()`, leaving it empty for scanned PDFs.
    - Populates metadata with `total_pages`, `parser`, and `has_text = bool(raw_text.strip())`.
    - Returns a standardized `ParsedDocument`.
    """

    supported_extensions: Sequence[str] = (".pdf",)

    def parse(
        self,
        file_path: Union[str, Path],
        document_id: Optional[str] = None,
        document_type: Optional[DocumentType] = None,
        source_type: Optional[SourceType] = None,
        **kwargs: Any,
    ) -> ParsedDocument:
        """
        Parse a PDF file into a standardized ParsedDocument.

        Args:
            file_path: Absolute or relative Path/str to the PDF file.
            document_id: Optional explicit document identifier.
            document_type: Optional explicit DocumentType.
            source_type: Optional explicit SourceType (defaults to SourceType.PDF).
            **kwargs: Extra attributes passed into ParsedDocument.metadata.

        Returns:
            ParsedDocument containing extracted pages, raw_text, and metadata.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {path.name}")

        # Open document with PyMuPDF
        pdf_doc = pymupdf.open(str(path))
        total_pages = len(pdf_doc)

        pages: List[ParsedPage] = []
        page_texts: List[str] = []

        try:
            for page_idx, page in enumerate(pdf_doc):
                page_number = page_idx + 1

                # Extract text using PyMuPDF text layout
                raw_page_text = page.get_text()

                # Conservative normalization decision:
                # - Normalize carriage returns (\r\n and \r -> \n)
                # - Strip trailing whitespace on individual lines
                # - Do NOT replace \n with space; preserve line breaks, indents, columns, tables
                normalized_lines = [line.rstrip() for line in raw_page_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
                normalized_page_text = "\n".join(normalized_lines).strip("\n")

                pages.append(
                    ParsedPage(
                        page_number=page_number,
                        content=normalized_page_text,
                        metadata={
                            "page_width": round(page.rect.width, 2),
                            "page_height": round(page.rect.height, 2),
                        },
                    )
                )

                if normalized_page_text.strip():
                    page_texts.append(normalized_page_text)
        finally:
            pdf_doc.close()

        # Compile raw_text: join pages with double newline, but leave empty if no text found.
        # Scanned PDFs legitimately have raw_text = "" so downstream pipelines can route to OCR.
        raw_text = "\n\n".join(page_texts).strip()

        # Detection for extractable text
        has_text = bool(raw_text.strip())

        # Infer metadata (doc_id, doc_type, etc.) from filename/path if not explicitly provided
        doc_id, doc_type, src_type, policy_id, claim_id, lob = self._infer_metadata(
            path, document_id, document_type, source_type
        )

        metadata: dict[str, Any] = {
            "total_pages": total_pages,
            "parser": "PDFParser",
            "has_text": has_text,
            "file_size_bytes": path.stat().st_size,
            "file_name": path.name,
            **kwargs.get("metadata", {}),
        }

        return ParsedDocument(
            content=raw_text,
            raw_text=raw_text,
            document_id=doc_id,
            document_type=doc_type,
            source_type=src_type,
            source_uri=str(path),
            pages=pages,
            total_pages=total_pages,
            has_text=has_text,
            metadata=metadata,
            policy_id=policy_id,
            claim_id=claim_id,
            line_of_business=lob,
        )

    @staticmethod
    def _infer_metadata(
        path: Path,
        explicit_id: Optional[str],
        explicit_doc_type: Optional[DocumentType],
        explicit_src_type: Optional[SourceType],
    ) -> Tuple[str, DocumentType, SourceType, Optional[str], Optional[str], Optional[LineOfBusiness]]:
        """Infer document identifiers and types from file name patterns if not supplied."""
        stem = path.stem

        # 1. SourceType (default to PDF)
        src_type = explicit_src_type or SourceType.PDF

        # 2. Extract entity IDs via regex
        policy_match = re.search(r"(MOT-\d+|COM-\d+|HH-\d+)", stem)
        policy_id = policy_match.group(1) if policy_match else None

        claim_match = re.search(r"(C-\d+)", stem)
        claim_id = claim_match.group(1) if claim_match else None

        # 3. Infer LineOfBusiness from policy_id prefix
        lob = None
        if policy_id:
            if policy_id.startswith("MOT-"):
                lob = LineOfBusiness.PERSONAL_AUTO
            elif policy_id.startswith("HH-"):
                lob = LineOfBusiness.HOMEOWNERS
            elif policy_id.startswith("COM-"):
                lob = LineOfBusiness.BOP

        # 4. Infer DocumentType from filename tokens
        if explicit_doc_type:
            doc_type = explicit_doc_type
        else:
            lower_stem = stem.lower()
            if "declarations" in lower_stem or lower_stem.endswith("-dec"):
                doc_type = DocumentType.POLICY_DECLARATIONS
            elif "endorsements" in lower_stem:
                doc_type = DocumentType.POLICY_ENDORSEMENTS
            elif "schedule" in lower_stem:
                doc_type = DocumentType.POLICY_SCHEDULE
            elif "contract" in lower_stem:
                doc_type = DocumentType.POLICY_CONTRACT
            elif "adjuster-report" in lower_stem:
                doc_type = DocumentType.ADJUSTER_REPORT
            elif "estimate" in lower_stem:
                doc_type = DocumentType.ESTIMATE
            elif "settlement-letter" in lower_stem:
                doc_type = DocumentType.SETTLEMENT_LETTER
            elif "denial-letter" in lower_stem:
                doc_type = DocumentType.DENIAL_LETTER
            elif "fnol" in lower_stem:
                doc_type = DocumentType.FNOL
            elif "police-report" in lower_stem:
                doc_type = DocumentType.POLICE_REPORT
            elif "accident-statement" in lower_stem:
                doc_type = DocumentType.ACCIDENT_STATEMENT
            elif "id-card" in lower_stem:
                doc_type = DocumentType.ID_CARD
            else:
                doc_type = DocumentType.POLICY_CONTRACT if policy_id else DocumentType.OTHER

        # 5. Document ID
        if explicit_id:
            doc_id = explicit_id
        else:
            # Match standard Strata corpus naming convention DOC-<STEM>
            suffix_map = {
                DocumentType.POLICY_DECLARATIONS: "DEC",
                DocumentType.POLICY_ENDORSEMENTS: "ENDORSEMENTS",
                DocumentType.POLICY_SCHEDULE: "SCHEDULE",
                DocumentType.POLICY_CONTRACT: "CONTRACT",
                DocumentType.ADJUSTER_REPORT: "ADJ",
                DocumentType.ESTIMATE: "ESTIMATE",
                DocumentType.SETTLEMENT_LETTER: "SETTLEMENT",
                DocumentType.DENIAL_LETTER: "DENIAL",
                DocumentType.FNOL: "FNOL",
            }
            if policy_id and doc_type in suffix_map:
                doc_id = f"DOC-{policy_id}-{suffix_map[doc_type]}"
            elif claim_id and doc_type in suffix_map:
                doc_id = f"DOC-{claim_id}-{suffix_map[doc_type]}"
            else:
                doc_id = f"DOC-{stem.upper()}"

        return doc_id, doc_type, src_type, policy_id, claim_id, lob


if __name__ == "__main__":
    # Quick run on an actual policy PDF from the corpus
    sample_path = Path("insurance-support-agent/data/raw/docs/policy/MOT-0000001-declarations.pdf")
    if not sample_path.exists():
        sample_path = Path("data/raw/docs/policy/MOT-0000001-declarations.pdf")

    parser = PDFParser()
    doc = parser.parse(sample_path)

    print("==================================================")
    print("PDF PARSER EXECUTION ON ACTUAL POLICY PDF")
    print("==================================================")
    print(f"document_id:      {doc.document_id}")
    print(f"source_type:      {doc.source_type.value if hasattr(doc.source_type, 'value') else doc.source_type}")
    print(f"total_pages:      {doc.metadata.get('total_pages', doc.total_pages)}")
    print(f"has_text:         {doc.metadata.get('has_text', doc.has_text)}")
    print("--------------------------------------------------")
    print("first 500 characters:")
    print("--------------------------------------------------")
    print(doc.raw_text[:500])
    print("--------------------------------------------------")
    print("first page content:")
    print("--------------------------------------------------")
    first_page_text = doc.pages[0].content if doc.pages else "[No pages]"
    print(first_page_text)
    print("==================================================")
