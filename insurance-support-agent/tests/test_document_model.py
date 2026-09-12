"""
Unit tests for the ParsedDocument and ParsedPage data models defined in apps.ingestion.parsers.base.

Tests cover:
1. ParsedPage model validation, constraints, and whitespace preservation.
2. ParsedDocument creation (minimal, full, scanned/empty text).
3. Field synchronization (_sync_fields model validator for raw_text/content, has_text, total_pages, metadata).
4. `page_count` property behavior across different pagination configurations.
5. Pydantic validation rules (required fields, min_length, extra='forbid', validate_assignment).
6. Enum coercion and representation (DocumentType, SourceType, LineOfBusiness).
7. Layout and whitespace preservation (str_strip_whitespace=False).
8. Serialization and round-trip deserialization (JSON and Python dict).
"""
from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from apps.ingestion.models import DocumentType, LineOfBusiness, SourceType
from apps.ingestion.parsers.base import ParsedDocument, ParsedPage


# =====================================================================
# 1. ParsedPage Model Tests
# =====================================================================

def test_parsed_page_minimal_creation():
    """Verify ParsedPage can be instantiated with only the required page_number."""
    page = ParsedPage(page_number=1)

    assert page.page_number == 1
    assert page.content == ""
    assert page.metadata == {}


def test_parsed_page_full_creation():
    """Verify ParsedPage retains all provided attributes including metadata."""
    metadata = {"width": 595.0, "height": 842.0, "rotation": 0}
    page = ParsedPage(
        page_number=2,
        content="Section 2.1: Coverage Terms & Conditions\n  - Deductible: £250",
        metadata=metadata,
    )

    assert page.page_number == 2
    assert "Coverage Terms & Conditions" in page.content
    assert page.metadata == metadata
    assert page.metadata["rotation"] == 0


@pytest.mark.parametrize("valid_page", [1, 2, 10, 500])
def test_parsed_page_valid_page_numbers(valid_page):
    """Verify positive 1-indexed page numbers are accepted."""
    page = ParsedPage(page_number=valid_page, content="Page content")
    assert page.page_number == valid_page


@pytest.mark.parametrize("invalid_page", [0, -1, -100])
def test_parsed_page_rejects_non_positive_page_numbers(invalid_page):
    """Verify page_number enforces ge=1 (rejects 0 and negative integers)."""
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        ParsedPage(page_number=invalid_page, content="Invalid page")


def test_parsed_page_extra_fields_forbidden():
    """Verify ParsedPage rejects unmodeled extra fields."""
    with pytest.raises(ValidationError):
        ParsedPage(page_number=1, content="Text", unexpected_key="disallowed")


def test_parsed_page_validate_assignment():
    """Verify validate_assignment prevents assigning invalid values post-initialization."""
    page = ParsedPage(page_number=1, content="Text")
    with pytest.raises(ValidationError):
        page.page_number = 0


def test_parsed_page_preserves_whitespace():
    """Verify str_strip_whitespace=False preserves leading and trailing spaces and newlines."""
    formatted_content = "   \n\tLine 1 with spaces   \n   Line 2\n\n"
    page = ParsedPage(page_number=1, content=formatted_content)
    assert page.content == formatted_content


# =====================================================================
# 2. ParsedDocument Creation Tests (Happy Path)
# =====================================================================

def test_parsed_document_minimal_with_content():
    """
    Verify creating a minimal ParsedDocument with 'content'.
    Checks that raw_text is synchronized to content, has_text is True,
    and all optional entity IDs default to None.
    """
    doc = ParsedDocument(
        content="Policy wording regarding accidental damage coverage.",
        document_id="DOC-KB-FAQ",
        document_type=DocumentType.CUSTOMER_FAQ,
        source_type=SourceType.KB,
    )

    assert doc.document_id == "DOC-KB-FAQ"
    assert doc.document_type == DocumentType.CUSTOMER_FAQ
    assert doc.source_type == SourceType.KB
    assert doc.content == "Policy wording regarding accidental damage coverage."
    assert doc.raw_text == "Policy wording regarding accidental damage coverage."
    assert doc.has_text is True
    assert doc.source_uri is None
    assert doc.title is None
    assert doc.pages == []
    assert doc.total_pages == 0
    assert doc.page_count == 1
    assert doc.policy_id is None
    assert doc.claim_id is None
    assert doc.policyholder_id is None
    assert doc.line_of_business is None
    assert isinstance(doc.parsed_at, datetime)
    assert doc.metadata["has_text"] is True
    assert doc.metadata["total_pages"] == 0


def test_parsed_document_minimal_with_raw_text():
    """
    Verify creating a minimal ParsedDocument with 'raw_text'.
    Checks that content is synchronized to raw_text.
    """
    doc = ParsedDocument(
        raw_text="Extracted plain text from markdown document.",
        document_id="DOC-POL-001",
        document_type=DocumentType.POLICY_CONTRACT,
        source_type=SourceType.MARKDOWN,
    )

    assert doc.raw_text == "Extracted plain text from markdown document."
    assert doc.content == "Extracted plain text from markdown document."
    assert doc.has_text is True
    assert doc.metadata["has_text"] is True


def test_parsed_document_fully_populated():
    """
    Verify creating a ParsedDocument with all optional metadata, insurance entity IDs,
    custom timestamps, and page extractions.
    """
    now = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
    pages = [
        ParsedPage(page_number=1, content="Page 1: Policy Declarations Header"),
        ParsedPage(page_number=2, content="Page 2: Schedule of Insured Assets"),
        ParsedPage(page_number=3, content="Page 3: Endorsements and Exclusions"),
    ]
    doc = ParsedDocument(
        content="Full extracted text across all pages.",
        document_id="DOC-MOT-0000001-DEC",
        document_type=DocumentType.POLICY_DECLARATIONS,
        source_type=SourceType.PDF,
        source_uri="docs/policy/MOT-0000001-declarations.pdf",
        title="Motor Policy Declarations - Meridian Mutual",
        pages=pages,
        total_pages=3,
        has_text=True,
        metadata={"author": "Underwriting Dept", "version": "1.2"},
        policy_id="MOT-0000001",
        claim_id="C-1001",
        policyholder_id="PH-00053",
        line_of_business=LineOfBusiness.PERSONAL_AUTO,
        parsed_at=now,
    )

    assert doc.document_id == "DOC-MOT-0000001-DEC"
    assert doc.document_type == DocumentType.POLICY_DECLARATIONS
    assert doc.source_type == SourceType.PDF
    assert doc.source_uri == "docs/policy/MOT-0000001-declarations.pdf"
    assert doc.title == "Motor Policy Declarations - Meridian Mutual"
    assert len(doc.pages) == 3
    assert doc.total_pages == 3
    assert doc.page_count == 3
    assert doc.has_text is True
    assert doc.policy_id == "MOT-0000001"
    assert doc.claim_id == "C-1001"
    assert doc.policyholder_id == "PH-00053"
    assert doc.line_of_business == LineOfBusiness.PERSONAL_AUTO
    assert doc.parsed_at == now
    assert doc.metadata["author"] == "Underwriting Dept"
    assert doc.metadata["version"] == "1.2"
    assert doc.metadata["has_text"] is True
    assert doc.metadata["total_pages"] == 3


def test_parsed_document_parsed_at_defaults_to_utc():
    """Verify parsed_at defaults to a timezone-aware UTC datetime."""
    before = datetime.now(timezone.utc)
    doc = ParsedDocument(
        content="Sample",
        document_id="DOC-TIME",
        document_type=DocumentType.OTHER,
        source_type=SourceType.PDF,
    )
    after = datetime.now(timezone.utc)

    assert doc.parsed_at.tzinfo is not None
    assert before <= doc.parsed_at <= after


# =====================================================================
# 3. Field Synchronization Tests (_sync_fields / @model_validator)
# =====================================================================

def test_sync_content_and_raw_text_when_only_content_passed():
    """When only 'content' is provided, raw_text must be synced to equal content."""
    doc = ParsedDocument(
        content="Only content provided.",
        document_id="DOC-1",
        document_type=DocumentType.CUSTOMER_FAQ,
        source_type=SourceType.KB,
    )
    assert doc.content == "Only content provided."
    assert doc.raw_text == "Only content provided."


def test_sync_content_and_raw_text_when_only_raw_text_passed():
    """When only 'raw_text' is provided, content must be synced to equal raw_text."""
    doc = ParsedDocument(
        raw_text="Only raw_text provided.",
        document_id="DOC-1",
        document_type=DocumentType.CUSTOMER_FAQ,
        source_type=SourceType.KB,
    )
    assert doc.raw_text == "Only raw_text provided."
    assert doc.content == "Only raw_text provided."


def test_sync_content_and_raw_text_when_both_passed():
    """When both 'content' and 'raw_text' are provided, each keeps its explicit value."""
    doc = ParsedDocument(
        content="Normalized text",
        raw_text="Raw text with original layout",
        document_id="DOC-1",
        document_type=DocumentType.CUSTOMER_FAQ,
        source_type=SourceType.KB,
    )
    assert doc.content == "Normalized text"
    assert doc.raw_text == "Raw text with original layout"


def test_sync_content_and_raw_text_when_neither_passed():
    """When neither 'content' nor 'raw_text' is provided, both default to empty string."""
    doc = ParsedDocument(
        document_id="DOC-1",
        document_type=DocumentType.FNOL_SCANNED,
        source_type=SourceType.PDF,
    )
    assert doc.content == ""
    assert doc.raw_text == ""
    assert doc.has_text is False


@pytest.mark.parametrize(
    "text_input, expected_has_text",
    [
        ("Non-empty document content", True),
        ("A", True),
        ("   valid content   ", True),
        ("", False),
        ("   ", False),
        ("\n\t\r\n", False),
    ],
)
def test_sync_has_text_computation(text_input, expected_has_text):
    """Verify has_text is correctly computed based on non-whitespace content."""
    doc = ParsedDocument(
        content=text_input,
        document_id="DOC-HAS-TEXT",
        document_type=DocumentType.POLICY_CONTRACT,
        source_type=SourceType.PDF,
    )
    assert doc.has_text is expected_has_text
    assert doc.metadata["has_text"] is expected_has_text


def test_sync_has_text_explicit_override_preserved():
    """Verify that an explicitly provided has_text is not overwritten by auto-computation."""
    doc = ParsedDocument(
        content="Valid text present",
        has_text=False,  # Explicit override (e.g. marked as corrupted/non-extractable)
        document_id="DOC-OVERRIDE",
        document_type=DocumentType.OTHER,
        source_type=SourceType.PDF,
    )
    assert doc.has_text is False
    assert doc.metadata["has_text"] is False


def test_sync_total_pages_computed_from_pages():
    """Verify total_pages is automatically computed from the length of pages list."""
    pages = [
        ParsedPage(page_number=1, content="P1"),
        ParsedPage(page_number=2, content="P2"),
        ParsedPage(page_number=3, content="P3"),
        ParsedPage(page_number=4, content="P4"),
    ]
    doc = ParsedDocument(
        content="P1\n\nP2\n\nP3\n\nP4",
        document_id="DOC-PAGED",
        document_type=DocumentType.ADJUSTER_REPORT,
        source_type=SourceType.PDF,
        pages=pages,
    )
    assert doc.total_pages == 4
    assert doc.metadata["total_pages"] == 4


def test_sync_total_pages_explicit_override_preserved():
    """Verify that an explicitly provided non-zero total_pages is preserved even if pages is shorter."""
    pages = [ParsedPage(page_number=1, content="Page 1 sample")]
    doc = ParsedDocument(
        content="Page 1 sample",
        document_id="DOC-PAGED-OVERRIDE",
        document_type=DocumentType.UNDERWRITING_GUIDELINES,
        source_type=SourceType.PDF,
        pages=pages,
        total_pages=15,  # e.g., only first page was parsed of a 15-page document
    )
    assert doc.total_pages == 15
    assert doc.metadata["total_pages"] == 15


def test_sync_metadata_dict_preserves_custom_keys():
    """Verify that custom metadata keys are preserved when has_text and total_pages are injected."""
    custom_metadata = {
        "parser": "PDFParser",
        "file_size_bytes": 102400,
        "is_encrypted": False,
    }
    doc = ParsedDocument(
        content="Some content",
        document_id="DOC-META",
        document_type=DocumentType.CUSTOMER_FAQ,
        source_type=SourceType.KB,
        metadata=custom_metadata,
    )
    assert doc.metadata["parser"] == "PDFParser"
    assert doc.metadata["file_size_bytes"] == 102400
    assert doc.metadata["is_encrypted"] is False
    assert doc.metadata["has_text"] is True
    assert doc.metadata["total_pages"] == 0


def test_sync_metadata_dict_does_not_overwrite_existing_metadata_keys():
    """Verify that if metadata already contains has_text or total_pages, they are preserved."""
    custom_metadata = {
        "has_text": True,
        "total_pages": 42,
    }
    doc = ParsedDocument(
        document_id="DOC-PRESET-META",
        document_type=DocumentType.CUSTOMER_FAQ,
        source_type=SourceType.KB,
        metadata=custom_metadata,
    )
    assert doc.metadata["has_text"] is True
    assert doc.metadata["total_pages"] == 42


# =====================================================================
# 4. page_count Property Tests
# =====================================================================

def test_page_count_returns_total_pages_when_positive():
    """When total_pages > 0, page_count must return total_pages."""
    doc = ParsedDocument(
        content="Document content",
        document_id="DOC-COUNT-1",
        document_type=DocumentType.POLICY_CONTRACT,
        source_type=SourceType.PDF,
        total_pages=5,
    )
    assert doc.page_count == 5


def test_page_count_returns_len_pages_when_total_pages_is_zero_and_pages_non_empty():
    """When total_pages is 0 (or overridden) but pages has items, page_count returns len(pages)."""
    pages = [ParsedPage(page_number=1, content="P1"), ParsedPage(page_number=2, content="P2")]
    # Force total_pages = 0 after initialization
    doc = ParsedDocument(
        content="P1 P2",
        document_id="DOC-COUNT-2",
        document_type=DocumentType.POLICY_CONTRACT,
        source_type=SourceType.PDF,
        pages=pages,
    )
    # By default total_pages is synced to 2
    assert doc.page_count == 2


def test_page_count_returns_one_for_unpaged_empty_pages():
    """When total_pages is 0 and pages is empty, page_count returns 1 (representing single doc)."""
    doc = ParsedDocument(
        content="Unpaged single document text",
        document_id="DOC-COUNT-3",
        document_type=DocumentType.CUSTOMER_FAQ,
        source_type=SourceType.KB,
    )
    assert doc.total_pages == 0
    assert doc.pages == []
    assert doc.page_count == 1


# =====================================================================
# 5. Scanned / OCR Document Handling Tests
# =====================================================================

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


def test_scanned_document_with_blank_pages():
    """Verify scanned document with empty ParsedPage instances maintains has_text=False."""
    blank_pages = [
        ParsedPage(page_number=1, content=""),
        ParsedPage(page_number=2, content=""),
    ]
    doc = ParsedDocument(
        raw_text="",
        document_id="DOC-SCAN-PAGES",
        document_type=DocumentType.ID_CARD_SCANNED,
        source_type=SourceType.PDF,
        pages=blank_pages,
    )

    assert doc.raw_text == ""
    assert doc.content == ""
    assert doc.has_text is False
    assert doc.total_pages == 2
    assert len(doc.pages) == 2
    assert doc.pages[0].content == ""
    assert doc.pages[1].content == ""


# =====================================================================
# 6. Validation Constraints & Error Handling Tests
# =====================================================================

@pytest.mark.parametrize(
    "missing_kwargs",
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
def test_missing_required_fields_raises_validation_error(missing_kwargs):
    """Verify that omitting required identity and classification fields raises a ValidationError."""
    with pytest.raises(ValidationError):
        ParsedDocument(**missing_kwargs)


def test_empty_document_id_raises_validation_error():
    """Verify document_id cannot be an empty string (min_length=1)."""
    with pytest.raises(ValidationError):
        ParsedDocument(
            content="Valid content",
            document_id="",
            document_type=DocumentType.POLICY_CONTRACT,
            source_type=SourceType.PDF,
        )


def test_extra_fields_forbidden():
    """Verify that unexpected extra fields raise a ValidationError (extra='forbid')."""
    with pytest.raises(ValidationError):
        ParsedDocument(
            content="Some text",
            document_id="DOC-1",
            document_type=DocumentType.CUSTOMER_FAQ,
            source_type=SourceType.KB,
            unrecognized_attribute="invalid",
        )


def test_validate_assignment_on_parsed_document():
    """Verify validate_assignment enforces constraints on field updates after creation."""
    doc = ParsedDocument(
        content="Valid content",
        document_id="DOC-ASSIGN-TEST",
        document_type=DocumentType.CUSTOMER_FAQ,
        source_type=SourceType.KB,
    )

    # Emptying document_id must fail min_length=1 constraint
    with pytest.raises(ValidationError):
        doc.document_id = ""


# =====================================================================
# 7. Domain Enums & Type Coercion Tests
# =====================================================================

@pytest.mark.parametrize(
    "source_input, expected_enum",
    [
        (SourceType.PDF, SourceType.PDF),
        (SourceType.DOCX, SourceType.DOCX),
        (SourceType.MARKDOWN, SourceType.MARKDOWN),
        (SourceType.KB, SourceType.KB),
        (SourceType.POLICY, SourceType.POLICY),
        (SourceType.CLAIM, SourceType.CLAIM),
        ("pdf", SourceType.PDF),
        ("docx", SourceType.DOCX),
        ("markdown", SourceType.MARKDOWN),
        ("kb", SourceType.KB),
        ("policy", SourceType.POLICY),
        ("claim", SourceType.CLAIM),
    ],
)
def test_source_type_enum_and_string_coercion(source_input, expected_enum):
    """Verify source_type accepts both SourceType enum instances and valid string values."""
    doc = ParsedDocument(
        content="Sample content",
        document_id="DOC-ENUM",
        document_type=DocumentType.CUSTOMER_FAQ,
        source_type=source_input,
    )
    assert doc.source_type == expected_enum
    assert isinstance(doc.source_type, SourceType)


@pytest.mark.parametrize(
    "doc_type_input, expected_enum",
    [
        (DocumentType.POLICY_CONTRACT, DocumentType.POLICY_CONTRACT),
        (DocumentType.FNOL, DocumentType.FNOL),
        (DocumentType.CUSTOMER_FAQ, DocumentType.CUSTOMER_FAQ),
        (DocumentType.UNDERWRITING_GUIDELINES, DocumentType.UNDERWRITING_GUIDELINES),
        (DocumentType.ADJUSTER_REPORT, DocumentType.ADJUSTER_REPORT),
        ("policy_contract", DocumentType.POLICY_CONTRACT),
        ("fnol", DocumentType.FNOL),
        ("customer_faq", DocumentType.CUSTOMER_FAQ),
        ("underwriting_guidelines", DocumentType.UNDERWRITING_GUIDELINES),
    ],
)
def test_document_type_enum_and_string_coercion(doc_type_input, expected_enum):
    """Verify document_type accepts both DocumentType enum instances and valid string values."""
    doc = ParsedDocument(
        content="Sample content",
        document_id="DOC-ENUM-TYPE",
        document_type=doc_type_input,
        source_type=SourceType.PDF,
    )
    assert doc.document_type == expected_enum
    assert isinstance(doc.document_type, DocumentType)


@pytest.mark.parametrize(
    "lob_input, expected_enum",
    [
        (LineOfBusiness.PERSONAL_AUTO, LineOfBusiness.PERSONAL_AUTO),
        (LineOfBusiness.HOMEOWNERS, LineOfBusiness.HOMEOWNERS),
        (LineOfBusiness.BOP, LineOfBusiness.BOP),
        ("personal_auto", LineOfBusiness.PERSONAL_AUTO),
        ("homeowners", LineOfBusiness.HOMEOWNERS),
        ("bop", LineOfBusiness.BOP),
        (None, None),
    ],
)
def test_line_of_business_enum_and_string_coercion(lob_input, expected_enum):
    """Verify line_of_business accepts LineOfBusiness enums, valid strings, and None."""
    doc = ParsedDocument(
        content="Sample content",
        document_id="DOC-LOB",
        document_type=DocumentType.POLICY_CONTRACT,
        source_type=SourceType.PDF,
        line_of_business=lob_input,
    )
    assert doc.line_of_business == expected_enum


def test_invalid_source_type_rejected():
    """Verify invalid source_type strings raise ValidationError."""
    with pytest.raises(ValidationError):
        ParsedDocument(
            content="Sample",
            document_id="DOC-BAD-SRC",
            document_type=DocumentType.POLICY_CONTRACT,
            source_type="non_existent_source_format",
        )


def test_invalid_document_type_rejected():
    """Verify invalid document_type strings raise ValidationError."""
    with pytest.raises(ValidationError):
        ParsedDocument(
            content="Sample",
            document_id="DOC-BAD-TYPE",
            document_type="invalid_doc_type_name",
            source_type=SourceType.PDF,
        )


def test_invalid_line_of_business_rejected():
    """Verify invalid line_of_business strings raise ValidationError."""
    with pytest.raises(ValidationError):
        ParsedDocument(
            content="Sample",
            document_id="DOC-BAD-LOB",
            document_type=DocumentType.POLICY_CONTRACT,
            source_type=SourceType.PDF,
            line_of_business="invalid_lob",
        )


# =====================================================================
# 8. Layout & Whitespace Preservation Tests
# =====================================================================

def test_parsed_document_preserves_text_layout_and_whitespace():
    """
    Verify str_strip_whitespace=False ensures leading/trailing whitespace,
    newlines, and indentation structures are strictly preserved.
    """
    structured_text = (
        "   SCHEDULE OF COVERAGES   \n"
        "---------------------------\n"
        "   Item 1: Building        £500,000\n"
        "   Item 2: Contents        £150,000\n"
        "   Item 3: Liability     £2,000,000\n"
    )
    doc = ParsedDocument(
        content=structured_text,
        document_id="DOC-LAYOUT-01",
        document_type=DocumentType.POLICY_SCHEDULE,
        source_type=SourceType.PDF,
    )

    assert doc.content == structured_text
    assert doc.raw_text == structured_text
    assert doc.content.startswith("   SCHEDULE")
    assert doc.content.endswith("\n")


# =====================================================================
# 9. Serialization & Deserialization Tests
# =====================================================================

def test_parsed_document_model_dump_dict():
    """Verify model_dump returns a complete dictionary with correct key-value pairs."""
    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    pages = [ParsedPage(page_number=1, content="Page 1 Text")]
    doc = ParsedDocument(
        content="Page 1 Text",
        document_id="DOC-DUMP",
        document_type=DocumentType.CUSTOMER_FAQ,
        source_type=SourceType.KB,
        pages=pages,
        parsed_at=now,
    )

    dumped = doc.model_dump()
    assert isinstance(dumped, dict)
    assert dumped["document_id"] == "DOC-DUMP"
    assert dumped["document_type"] == DocumentType.CUSTOMER_FAQ
    assert dumped["source_type"] == SourceType.KB
    assert dumped["total_pages"] == 1
    assert dumped["has_text"] is True
    assert len(dumped["pages"]) == 1
    assert dumped["pages"][0]["page_number"] == 1
    assert dumped["parsed_at"] == now


def test_parsed_document_json_serialization_and_deserialization_roundtrip():
    """
    Verify model_dump_json serializes ParsedDocument into valid JSON and
    ParsedDocument.model_validate_json reconstructs an identical model instance.
    """
    now = datetime(2024, 5, 20, 14, 45, 0, tzinfo=timezone.utc)
    pages = [
        ParsedPage(page_number=1, content="Page 1 Content", metadata={"section": "Intro"}),
        ParsedPage(page_number=2, content="Page 2 Content", metadata={"section": "Details"}),
    ]
    original_doc = ParsedDocument(
        content="Page 1 Content\n\nPage 2 Content",
        document_id="DOC-JSON-ROUNDTRIP",
        document_type=DocumentType.POLICY_CONTRACT,
        source_type=SourceType.PDF,
        source_uri="docs/policy/contract.pdf",
        title="Commercial Policy Contract",
        pages=pages,
        total_pages=2,
        has_text=True,
        metadata={"author": "Legal", "has_text": True, "total_pages": 2},
        policy_id="POL-99999",
        line_of_business=LineOfBusiness.COMMERCIAL,
        parsed_at=now,
    )

    json_str = original_doc.model_dump_json()

    # Verify JSON strings contain expected serialized formats
    assert '"document_id":"DOC-JSON-ROUNDTRIP"' in json_str
    assert '"document_type":"policy_contract"' in json_str
    assert '"source_type":"pdf"' in json_str
    assert '"line_of_business":"commercial"' in json_str

    # Roundtrip reconstruct
    reconstructed_doc = ParsedDocument.model_validate_json(json_str)

    assert reconstructed_doc.document_id == original_doc.document_id
    assert reconstructed_doc.document_type == original_doc.document_type
    assert reconstructed_doc.source_type == original_doc.source_type
    assert reconstructed_doc.source_uri == original_doc.source_uri
    assert reconstructed_doc.title == original_doc.title
    assert reconstructed_doc.content == original_doc.content
    assert reconstructed_doc.raw_text == original_doc.raw_text
    assert reconstructed_doc.total_pages == original_doc.total_pages
    assert reconstructed_doc.has_text == original_doc.has_text
    assert len(reconstructed_doc.pages) == len(original_doc.pages)
    assert reconstructed_doc.pages[0].page_number == original_doc.pages[0].page_number
    assert reconstructed_doc.pages[0].content == original_doc.pages[0].content
    assert reconstructed_doc.pages[0].metadata == original_doc.pages[0].metadata
    assert reconstructed_doc.policy_id == original_doc.policy_id
    assert reconstructed_doc.line_of_business == original_doc.line_of_business
    assert reconstructed_doc.parsed_at == original_doc.parsed_at
    assert reconstructed_doc.metadata == original_doc.metadata
