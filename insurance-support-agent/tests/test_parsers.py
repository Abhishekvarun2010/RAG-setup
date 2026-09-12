"""
Unit tests for ParsedDocument data model, DocumentParser protocol, and BaseParser.
"""
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union
import pytest
from pydantic import ValidationError

from apps.ingestion.models import DocumentType, LineOfBusiness, SourceType
from apps.ingestion.parsers import (
    BaseParser,
    DocumentParser,
    ParsedDocument,
    ParsedPage,
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
    assert doc.document_id == "DOC-KB-FAQ"
    assert doc.document_type == DocumentType.CUSTOMER_FAQ
    assert doc.source_type == SourceType.KB
    assert doc.source_uri is None
    assert doc.title is None
    assert doc.pages == []
    assert doc.page_count == 1  # Unpaged documents report page_count=1
    assert doc.metadata == {}
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
        metadata={"author": "Meridian Mutual SE", "page_count": 2},
        policy_id="MOT-0000001",
        policyholder_id="PH-00053",
        line_of_business=LineOfBusiness.PERSONAL_AUTO,
        parsed_at=now,
    )

    assert doc.document_id == "DOC-MOT-0000001-DEC"
    assert doc.title == "Motor Policy Declarations"
    assert len(doc.pages) == 2
    assert doc.page_count == 2
    assert doc.pages[0].page_number == 1
    assert doc.pages[1].page_number == 2
    assert doc.policy_id == "MOT-0000001"
    assert doc.policyholder_id == "PH-00053"
    assert doc.line_of_business == LineOfBusiness.PERSONAL_AUTO
    assert doc.metadata["author"] == "Meridian Mutual SE"
    assert doc.parsed_at == now


# =====================================================================
# 2. Required Fields Validation Tests
# =====================================================================

@pytest.mark.parametrize(
    "missing_field_kwargs",
    [
        # Missing content
        {
            "document_id": "DOC-1",
            "document_type": DocumentType.CUSTOMER_FAQ,
            "source_type": SourceType.KB,
        },
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
    """Verify that omitting any of the 4 core required fields raises a ValidationError."""
    with pytest.raises(ValidationError):
        ParsedDocument(**missing_field_kwargs)


@pytest.mark.parametrize("empty_content", ["", "   "])
def test_empty_content_rejected(empty_content):
    """Verify that empty or whitespace-only content is rejected."""
    with pytest.raises(ValidationError):
        ParsedDocument(
            content=empty_content,
            document_id="DOC-1",
            document_type=DocumentType.CUSTOMER_FAQ,
            source_type=SourceType.KB,
        )


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
    # Protocol runtime check
    assert isinstance(parser, DocumentParser)
    assert isinstance(parser, BaseParser)

    # Extension checking
    assert parser.can_parse("document.md") is True
    assert parser.can_parse("DOCUMENT.MD") is True
    assert parser.can_parse("guide.markdown") is True
    assert parser.can_parse("report.pdf") is False
    assert parser.can_parse("contract.docx") is False

    # Parsing execution
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
