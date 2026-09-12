"""
Unit tests for the Chunk Pydantic model and supporting domain enums.

Uses pytest with table-driven parameterized testing (@pytest.mark.parametrize)
to verify data validation, normalization, and edge cases.
"""
from datetime import date, datetime, timezone
import pytest
from pydantic import ValidationError

from apps.ingestion.models import (
    AccessLevel,
    Chunk,
    DocumentType,
    LineOfBusiness,
    SourceType,
    Status,
)


# =====================================================================
# Happy Path / Creation Tests
# =====================================================================

def test_create_minimal_kb_chunk():
    """
    Verifies that a chunk can be created with only the required fields.
    
    In a Knowledge Base document (like an FAQ), policy_id, claim_id,
    and policyholder_id do not exist and should default to None.
    """
    chunk = Chunk(
        content="We offer Motor, Household, and Commercial insurance.",
        document_id="DOC-KB-FAQ",
        document_type=DocumentType.CUSTOMER_FAQ,
        source_type=SourceType.KB,
        source_uri="docs/kb/customer-faq.md",
    )

    # Core required fields
    assert chunk.content == "We offer Motor, Household, and Commercial insurance."
    assert chunk.chunk_id.startswith("chk_")  # Auto-generated UUID prefix
    assert chunk.document_id == "DOC-KB-FAQ"
    assert chunk.document_type == DocumentType.CUSTOMER_FAQ
    assert chunk.source_type == SourceType.KB

    # Entity IDs should default to None for KB docs
    assert chunk.policy_id is None
    assert chunk.claim_id is None
    assert chunk.policyholder_id is None
    assert chunk.line_of_business is None
    assert chunk.product is None
    assert chunk.version is None
    assert chunk.status is None

    # Temporal and structural metadata should default to None
    assert chunk.effective_from is None
    assert chunk.effective_to is None
    assert chunk.section is None
    assert chunk.page_number is None

    # Security and provenance defaults
    assert chunk.access_control == ["internal"]
    assert chunk.source_uri == "docs/kb/customer-faq.md"
    assert isinstance(chunk.ingested_at, datetime)


def test_create_fully_populated_policy_chunk():
    """
    Verifies creating a chunk with every metadata field populated.
    
    Simulates a chunk extracted from page 1 of a Policy Declarations PDF.
    """
    now = datetime.now(timezone.utc)
    chunk = Chunk(
        chunk_id="chk_mot_001_p1",
        content="Declarations for Peugeot 208 under policy MOT-0000001.",
        document_id="DOC-MOT-0000001-DEC",
        document_type=DocumentType.POLICY_DECLARATIONS,
        source_type=SourceType.POLICY,
        policy_id="MOT-0000001",
        claim_id=None,
        policyholder_id="PH-00053",
        line_of_business=LineOfBusiness.PERSONAL_AUTO,
        product="Personal Auto Policy",
        version="2.0",
        status=Status.ACTIVE,
        effective_from=date(2023, 11, 29),
        effective_to=date(2024, 11, 28),
        section="Declarations",
        page_number=1,
        access_control=[AccessLevel.AGENT, "policyholder"],
        source_uri="docs/policy/MOT-0000001-declarations.pdf",
        ingested_at=now,
    )

    # Identifiers
    assert chunk.chunk_id == "chk_mot_001_p1"
    assert chunk.policy_id == "MOT-0000001"
    assert chunk.claim_id is None
    assert chunk.policyholder_id == "PH-00053"

    # Business classifications
    assert chunk.line_of_business == LineOfBusiness.PERSONAL_AUTO
    assert chunk.status == Status.ACTIVE

    # Dates and pagination
    assert chunk.effective_from == date(2023, 11, 29)
    assert chunk.effective_to == date(2024, 11, 28)
    assert chunk.page_number == 1

    # Access control and provenance
    assert chunk.access_control == ["agent", "policyholder"]
    assert chunk.ingested_at == now


def test_create_claim_chunk():
    """
    Verifies creating a chunk for a Claim FNOL document.
    
    Claim documents have a claim_id and link back to a policy_id and policyholder_id.
    """
    chunk = Chunk(
        chunk_id="chk_c1000_fnol",
        content="Slip and fall reported at commercial property.",
        document_id="DOC-C-1000-FNOL",
        document_type=DocumentType.FNOL,
        source_type=SourceType.CLAIM,
        claim_id="C-1000",
        policy_id="COM-0000077",
        policyholder_id="PH-00029",
        line_of_business=LineOfBusiness.BOP,
        status=Status.CLOSED,
        section="Loss Description",
        page_number=1,
        access_control=["adjuster", "internal"],
        source_uri="docs/claim/C-1000-fnol.pdf",
    )

    assert chunk.claim_id == "C-1000"
    assert chunk.policy_id == "COM-0000077"
    assert chunk.line_of_business == LineOfBusiness.BOP
    assert chunk.status == Status.CLOSED


# =====================================================================
# Validation & Normalization Tests
# =====================================================================

@pytest.mark.parametrize(
    "raw_access, expected",
    [
        # Case 1: Single string should be wrapped into a 1-element list
        ("public", ["public"]),
        # Case 2: Standard list of strings should be preserved
        (["agent", "adjuster"], ["agent", "adjuster"]),
        # Case 3: AccessLevel enum should be unwrapped to its string value
        ([AccessLevel.ADMIN], ["admin"]),
        # Case 4: None should be normalized to an empty list
        (None, []),
    ],
)
def test_access_control_normalization(raw_access, expected):
    """
    Tests that the @field_validator normalizes different input types
    (string, list, enum, or None) into a uniform List[str].
    """
    chunk = Chunk(
        content="Access test notice",
        document_id="DOC-1",
        document_type=DocumentType.CUSTOMER_FAQ,
        source_type=SourceType.KB,
        access_control=raw_access,
    )
    assert chunk.access_control == expected


def test_date_validation_error():
    """
    Tests that the model validator rejects dates where effective_to
    is earlier than effective_from (chronological ordering violation).
    """
    with pytest.raises(ValidationError, match="effective_to"):
        Chunk(
            content="Invalid date range",
            document_id="DOC-1",
            document_type=DocumentType.POLICY_CONTRACT,
            source_type=SourceType.POLICY,
            effective_from=date(2024, 1, 1),
            effective_to=date(2023, 1, 1),  # Invalid: expires before it starts
        )


@pytest.mark.parametrize("invalid_page", [0, -1, -10])
def test_page_number_ge_1(invalid_page):
    """
    Tests that page_number enforces 1-indexed pagination (ge=1).
    Zero and negative values must raise a ValidationError.
    """
    with pytest.raises(ValidationError):
        Chunk(
            content="Invalid page content",
            document_id="DOC-1",
            document_type=DocumentType.POLICY_CONTRACT,
            source_type=SourceType.POLICY,
            page_number=invalid_page,
        )


@pytest.mark.parametrize("empty_content", ["", "   "])
def test_empty_content_rejected(empty_content):
    """
    Tests that chunk content cannot be an empty string or whitespace only.
    Guaranteed by min_length=1 and str_strip_whitespace=True.
    """
    with pytest.raises(ValidationError):
        Chunk(
            content=empty_content,
            document_id="DOC-1",
            document_type=DocumentType.CUSTOMER_FAQ,
            source_type=SourceType.KB,
        )


def test_extra_fields_forbidden():
    """
    Tests that unexpected extra fields raise a ValidationError.
    Guaranteed by ConfigDict(extra='forbid') to prevent silent typos.
    """
    with pytest.raises(ValidationError):
        Chunk(
            content="Valid content",
            document_id="DOC-1",
            document_type=DocumentType.CUSTOMER_FAQ,
            source_type=SourceType.KB,
            unrecognized_field="illegal",
        )


# =====================================================================
# Serialization Tests
# =====================================================================

def test_json_serialization():
    """
    Tests that model_dump_json properly serializes custom enums and date objects
    into standard ISO strings.
    """
    chunk = Chunk(
        chunk_id="chk_test_1",
        content="Test content",
        document_id="DOC-TEST",
        document_type=DocumentType.CUSTOMER_FAQ,
        source_type=SourceType.KB,
        effective_from=date(2024, 1, 1),
    )
    json_str = chunk.model_dump_json()

    # Enums serialize to their string values
    assert '"document_type":"customer_faq"' in json_str
    assert '"source_type":"kb"' in json_str

    # Date serializes to ISO-8601 YYYY-MM-DD
    assert '"effective_from":"2024-01-01"' in json_str
