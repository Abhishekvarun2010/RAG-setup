"""
Unit tests for ParsedDocument data model, DocumentParser protocol, BaseParser, and PDFParser.
"""
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union
import pymupdf
import pytest
from pydantic import ValidationError

from apps.ingestion.models import DocumentType, LineOfBusiness, SourceType
from apps.ingestion.parsers import (
    BaseParser,
    DocumentParser,
    ParsedDocument,
    ParsedPage,
    PDFParser,
)


# =====================================================================
# 1. Valid ParsedDocument Creation Tests
# =====================================================================

def test_create_valid_minimal_parsed_document():
    """Verify that a valid ParsedDocument can be created with required fields."""
    doc = ParsedDocument(
        content="Customer FAQ content regarding claims process.",
        document_id="DOC-KB-FAQ",
        document_type=DocumentType.CUSTOMER_FAQ,
        source_type=SourceType.KB,
    )

    assert doc.content == "Customer FAQ content regarding claims process."
    assert doc.raw_text == "Customer FAQ content regarding claims process."
    assert doc.has_text is True
    assert doc.document_id == "DOC-KB-FAQ"
    assert doc.document_type == DocumentType.CUSTOMER_FAQ
    assert doc.source_type == SourceType.KB
    assert doc.source_uri is None
    assert doc.title is None
    assert doc.pages == []
    assert doc.page_count == 1
    assert doc.metadata["has_text"] is True
    assert doc.policy_id is None
    assert doc.claim_id is None
    assert doc.policyholder_id is None
    assert doc.line_of_business is None
    assert isinstance(doc.parsed_at, datetime)


def test_create_valid_rich_parsed_document_with_pages():
    """Verify creating a ParsedDocument with multiple pages, metadata, and entity IDs."""
    now = datetime.now(timezone.utc)
    pages = [
        ParsedPage(page_number=1, content="Page 1: Policy Declarations Header"),
        ParsedPage(page_number=2, content="Page 2: Coverage Limits and Deductibles"),
    ]
    doc = ParsedDocument(
        content="Page 1: Policy Declarations Header\n\nPage 2: Coverage Limits and Deductibles",
        document_id="DOC-MOT-0000001-DEC",
        document_type=DocumentType.POLICY_DECLARATIONS,
        source_type=SourceType.PDF,
        source_uri="docs/policy/MOT-0000001-declarations.pdf",
        title="Motor Policy Declarations",
        pages=pages,
        metadata={"author": "Meridian Mutual SE"},
        policy_id="MOT-0000001",
        policyholder_id="PH-00053",
        line_of_business=LineOfBusiness.PERSONAL_AUTO,
        parsed_at=now,
    )

    assert doc.document_id == "DOC-MOT-0000001-DEC"
    assert doc.title == "Motor Policy Declarations"
    assert len(doc.pages) == 2
    assert doc.total_pages == 2
    assert doc.page_count == 2
    assert doc.has_text is True
    assert doc.metadata["has_text"] is True
    assert doc.metadata["total_pages"] == 2
    assert doc.pages[0].page_number == 1
    assert doc.pages[1].page_number == 2
    assert doc.policy_id == "MOT-0000001"
    assert doc.policyholder_id == "PH-00053"
    assert doc.line_of_business == LineOfBusiness.PERSONAL_AUTO
    assert doc.parsed_at == now


def test_scanned_document_allows_empty_text_and_sets_has_text_false():
    """
    Verify that a scanned document with no extractable text legitimately
    has raw_text = '' and has_text = False (without fake placeholder text).
    """
    doc = ParsedDocument(
        raw_text="",
        document_id="DOC-SCAN-001",
        document_type=DocumentType.FNOL_SCANNED,
        source_type=SourceType.PDF,
    )

    assert doc.raw_text == ""
    assert doc.content == ""
    assert doc.has_text is False
    assert doc.metadata["has_text"] is False


# =====================================================================
# 2. Required Fields Validation Tests
# =====================================================================

@pytest.mark.parametrize(
    "missing_field_kwargs",
    [
        # Missing document_id
        {
            "content": "Valid text content",
            "document_type": DocumentType.CUSTOMER_FAQ,
            "source_type": SourceType.KB,
        },
        # Missing document_type
        {
            "content": "Valid text content",
            "document_id": "DOC-1",
            "source_type": SourceType.KB,
        },
        # Missing source_type
        {
            "content": "Valid text content",
            "document_id": "DOC-1",
            "document_type": DocumentType.CUSTOMER_FAQ,
        },
    ],
)
def test_required_fields_validation(missing_field_kwargs):
    """Verify that omitting required identity/classification fields raises a ValidationError."""
    with pytest.raises(ValidationError):
        ParsedDocument(**missing_field_kwargs)


def test_extra_fields_forbidden():
    """Verify that unexpected extra fields raise a ValidationError."""
    with pytest.raises(ValidationError):
        ParsedDocument(
            content="Some text",
            document_id="DOC-1",
            document_type=DocumentType.CUSTOMER_FAQ,
            source_type=SourceType.KB,
            unexpected_field="disallowed",
        )


def test_parsed_page_page_number_ge_1():
    """Verify that ParsedPage enforces 1-indexed page numbering."""
    with pytest.raises(ValidationError):
        ParsedPage(page_number=0, content="Invalid page number")


# =====================================================================
# 3. source_type Representation Tests
# =====================================================================

@pytest.mark.parametrize(
    "source_input, expected_enum",
    [
        (SourceType.PDF, SourceType.PDF),
        (SourceType.DOCX, SourceType.DOCX),
        (SourceType.MARKDOWN, SourceType.MARKDOWN),
        (SourceType.KB, SourceType.KB),
        ("pdf", SourceType.PDF),
        ("docx", SourceType.DOCX),
        ("markdown", SourceType.MARKDOWN),
        ("claim", SourceType.CLAIM),
        ("policy", SourceType.POLICY),
    ],
)
def test_source_type_represented_properly(source_input, expected_enum):
    """Verify that source_type accepts both SourceType enums and valid string values."""
    doc = ParsedDocument(
        content="Valid sample document content.",
        document_id="DOC-100",
        document_type=DocumentType.POLICY_CONTRACT,
        source_type=source_input,
    )
    assert doc.source_type == expected_enum
    assert isinstance(doc.source_type, SourceType)


def test_invalid_source_type_rejected():
    """Verify that an invalid source_type raises a ValidationError."""
    with pytest.raises(ValidationError):
        ParsedDocument(
            content="Valid content",
            document_id="DOC-100",
            document_type=DocumentType.POLICY_CONTRACT,
            source_type="unsupported_source_format_xyz",
        )


# =====================================================================
# 4. Parser Interface / Protocol Tests
# =====================================================================

class MockMarkdownParser(BaseParser):
    """A concrete parser subclassing BaseParser to test interface conformance."""

    supported_extensions = (".md", ".markdown")

    def parse(self, file_path: Union[str, Path], **kwargs: Any) -> ParsedDocument:
        path = Path(file_path)
        return ParsedDocument(
            content=f"Parsed content from {path.name}",
            document_id=f"DOC-{path.stem.upper()}",
            document_type=DocumentType.CUSTOMER_FAQ,
            source_type=SourceType.MARKDOWN,
            source_uri=str(path),
        )


class DuckTypedParser:
    """A parser that does NOT subclass BaseParser but implements the protocol."""

    def parse(self, file_path: Union[str, Path], **kwargs: Any) -> ParsedDocument:
        return ParsedDocument(
            content="Duck-typed parse result",
            document_id="DOC-DUCK",
            document_type=DocumentType.OTHER,
            source_type=SourceType.PDF,
        )

    def can_parse(self, file_path: Union[str, Path]) -> bool:
        return str(file_path).endswith(".pdf")


class IncompleteParser:
    """A parser missing `can_parse`, failing protocol conformance."""

    def parse(self, file_path: Union[str, Path], **kwargs: Any) -> ParsedDocument:
        return ParsedDocument(
            content="Incomplete",
            document_id="DOC-INCOMPLETE",
            document_type=DocumentType.OTHER,
            source_type=SourceType.PDF,
        )


def test_base_parser_subclass_conforms_to_interface():
    """Verify that a subclass of BaseParser conforms to DocumentParser protocol."""
    parser = MockMarkdownParser()
    assert isinstance(parser, DocumentParser)
    assert isinstance(parser, BaseParser)

    assert parser.can_parse("document.md") is True
    assert parser.can_parse("DOCUMENT.MD") is True
    assert parser.can_parse("guide.markdown") is True
    assert parser.can_parse("report.pdf") is False
    assert parser.can_parse("contract.docx") is False

    doc = parser.parse("data/raw/docs/kb/customer-faq.md")
    assert isinstance(doc, ParsedDocument)
    assert doc.document_id == "DOC-CUSTOMER-FAQ"
    assert doc.source_type == SourceType.MARKDOWN


def test_duck_typed_parser_conforms_to_protocol():
    """Verify that duck-typing works with DocumentParser protocol."""
    parser = DuckTypedParser()
    assert isinstance(parser, DocumentParser)
    assert parser.can_parse("file.pdf") is True
    assert parser.can_parse("file.docx") is False

    doc = parser.parse("file.pdf")
    assert isinstance(doc, ParsedDocument)
    assert doc.document_id == "DOC-DUCK"


def test_incomplete_parser_fails_protocol_check():
    """Verify that a class missing required methods does NOT conform to DocumentParser."""
    parser = IncompleteParser()
    assert not isinstance(parser, DocumentParser)


# =====================================================================
# 5. PDFParser Unit Tests
# =====================================================================

def test_pdf_parser_conforms_to_interface():
    """Verify PDFParser satisfies DocumentParser protocol and BaseParser."""
    parser = PDFParser()
    assert isinstance(parser, DocumentParser)
    assert isinstance(parser, BaseParser)
    assert parser.can_parse("declarations.pdf") is True
    assert parser.can_parse("DECLARATIONS.PDF") is True
    assert parser.can_parse("document.docx") is False


def test_pdf_parser_on_actual_policy_pdf():
    """Verify PDFParser parses an actual policy PDF from the corpus correctly."""
    policy_pdf = Path("insurance-support-agent/data/raw/docs/policy/MOT-0000001-declarations.pdf")
    if not policy_pdf.exists():
        policy_pdf = Path("data/raw/docs/policy/MOT-0000001-declarations.pdf")

    assert policy_pdf.exists(), f"Policy PDF not found: {policy_pdf}"

    parser = PDFParser()
    doc = parser.parse(policy_pdf)

    # 1. Document identifiers
    assert doc.document_id == "DOC-MOT-0000001-DEC"
    assert doc.source_type == SourceType.PDF
    assert doc.document_type == DocumentType.POLICY_DECLARATIONS
    assert doc.policy_id == "MOT-0000001"
    assert doc.line_of_business == LineOfBusiness.PERSONAL_AUTO

    # 2. Page and text detection
    assert doc.total_pages == 1
    assert doc.has_text is True
    assert doc.metadata["has_text"] is True
    assert doc.metadata["total_pages"] == 1
    assert doc.metadata["parser"] == "PDFParser"

    # 3. Content and layout preservation (contains line breaks, not replaced with spaces)
    assert "MOT-0000001" in doc.raw_text
    assert "Robin Hardy" in doc.raw_text
    assert "\n" in doc.raw_text  # Layout structure preserved
    assert len(doc.pages) == 1
    assert doc.pages[0].page_number == 1
    assert "Peugeot 208" in doc.pages[0].content


def test_pdf_parser_on_empty_scanned_pdf(tmp_path):
    """
    Verify that an empty or image-only PDF without extractable text
    results in raw_text = '' and has_text = False (no fake text inserted).
    """
    blank_pdf_path = tmp_path / "blank_scanned.pdf"

    # Create an empty 1-page PDF using PyMuPDF
    blank_doc = pymupdf.open()
    blank_doc.new_page(width=595, height=842)  # Blank A4 page
    blank_doc.save(str(blank_pdf_path))
    blank_doc.close()

    parser = PDFParser()
    doc = parser.parse(blank_pdf_path)

    # Verify no fake text is placed into raw_text
    assert doc.raw_text == ""
    assert doc.content == ""
    assert doc.has_text is False
    assert doc.metadata["has_text"] is False
    assert doc.total_pages == 1
    assert len(doc.pages) == 1
    assert doc.pages[0].content == ""
