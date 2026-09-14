"""
Tests for FieldRedactor: PII scrubbing, role-based metadata field stripping,
and admin bypass behavior.
"""
import pytest

from apps.ingestion.models import AccessLevel
from apps.auth.models import SecurityContext
from apps.auth.redaction import FieldRedactor, PII_PATTERNS, ROLE_REDACTED_FIELDS
from apps.retrieval.models import RetrievalMethod, RetrievalResult


def _make_result(
    chunk_id: str = "test-chunk-1",
    content: str = "Test content",
    metadata: dict = None,
) -> RetrievalResult:
    """Helper to create a RetrievalResult for testing."""
    return RetrievalResult(
        chunk_id=chunk_id,
        content=content,
        score=1.0,
        retrieval_method=RetrievalMethod.HYBRID_RRF,
        metadata=metadata or {},
    )


# =====================================================================
# PII Scrubbing
# =====================================================================

class TestPIIScrubbing:
    """Test regex-based PII removal from content."""

    @pytest.fixture
    def redactor(self) -> FieldRedactor:
        return FieldRedactor()

    @pytest.fixture
    def policyholder_ctx(self) -> SecurityContext:
        return SecurityContext(
            user_id="PH-00029",
            roles=[AccessLevel.POLICYHOLDER],
        )

    def test_iban_redaction(self, redactor: FieldRedactor, policyholder_ctx: SecurityContext):
        result = _make_result(content="Payment to IBAN DE89370400440532013000")
        redacted = redactor.redact([result], policyholder_ctx)
        assert "[REDACTED-IBAN]" in redacted[0].content
        assert "DE89370400440532013000" not in redacted[0].content

    def test_ssn_redaction(self, redactor: FieldRedactor, policyholder_ctx: SecurityContext):
        result = _make_result(content="SSN: 123-45-6789")
        redacted = redactor.redact([result], policyholder_ctx)
        assert "[REDACTED-SSN]" in redacted[0].content
        assert "123-45-6789" not in redacted[0].content

    def test_email_redaction(self, redactor: FieldRedactor, policyholder_ctx: SecurityContext):
        result = _make_result(content="Contact john.doe@example.com for details")
        redacted = redactor.redact([result], policyholder_ctx)
        assert "[REDACTED-EMAIL]" in redacted[0].content
        assert "john.doe@example.com" not in redacted[0].content

    def test_credit_card_redaction(self, redactor: FieldRedactor, policyholder_ctx: SecurityContext):
        result = _make_result(content="Card number: 4111-1111-1111-1111")
        redacted = redactor.redact([result], policyholder_ctx)
        assert "[REDACTED-CC]" in redacted[0].content
        assert "4111-1111-1111-1111" not in redacted[0].content

    def test_phone_redaction(self, redactor: FieldRedactor, policyholder_ctx: SecurityContext):
        result = _make_result(content="Call +1-555-123-4567")
        redacted = redactor.redact([result], policyholder_ctx)
        assert "[REDACTED-PHONE]" in redacted[0].content

    def test_no_pii_unchanged(self, redactor: FieldRedactor, policyholder_ctx: SecurityContext):
        result = _make_result(content="The policy covers fire and water damage.")
        redacted = redactor.redact([result], policyholder_ctx)
        assert redacted[0].content == "The policy covers fire and water damage."

    def test_multiple_pii_in_content(self, redactor: FieldRedactor, policyholder_ctx: SecurityContext):
        result = _make_result(
            content="Contact john@example.com, SSN: 123-45-6789, IBAN GB29NWBK60161331926819"
        )
        redacted = redactor.redact([result], policyholder_ctx)
        content = redacted[0].content
        assert "[REDACTED-EMAIL]" in content
        assert "[REDACTED-SSN]" in content
        assert "[REDACTED-IBAN]" in content

    def test_pii_disabled(self, policyholder_ctx: SecurityContext):
        redactor = FieldRedactor(redact_pii=False)
        result = _make_result(content="SSN: 123-45-6789")
        redacted = redactor.redact([result], policyholder_ctx)
        assert "123-45-6789" in redacted[0].content


# =====================================================================
# Role-Based Metadata Stripping
# =====================================================================

class TestMetadataStripping:
    """Test role-based removal of sensitive metadata fields."""

    @pytest.fixture
    def redactor(self) -> FieldRedactor:
        return FieldRedactor(redact_pii=False)  # Disable PII for metadata-focused tests

    def test_policyholder_sees_no_internal_notes(self, redactor: FieldRedactor):
        ctx = SecurityContext(user_id="PH-001", roles=[AccessLevel.POLICYHOLDER])
        result = _make_result(
            content="Policy details",
            metadata={
                "policy_id": "COM-0000077",
                "internal_notes": "High risk client",
                "reserve_amount": "50000",
                "adjuster_assessment": "Major loss",
                "underwriting_score": "78",
            },
        )
        redacted = redactor.redact([result], ctx)
        meta = redacted[0].metadata
        assert "policy_id" in meta  # Non-restricted field preserved
        assert "internal_notes" not in meta
        assert "reserve_amount" not in meta
        assert "adjuster_assessment" not in meta
        assert "underwriting_score" not in meta
        assert "_redacted_fields" in meta

    def test_agent_keeps_internal_notes(self, redactor: FieldRedactor):
        ctx = SecurityContext(user_id="AGT-001", roles=[AccessLevel.AGENT])
        result = _make_result(
            content="Policy details",
            metadata={
                "internal_notes": "High risk client",
                "reserve_amount": "50000",
                "underwriting_score": "78",
            },
        )
        redacted = redactor.redact([result], ctx)
        meta = redacted[0].metadata
        assert "internal_notes" in meta  # Agent can see internal_notes
        assert "reserve_amount" not in meta
        assert "underwriting_score" not in meta

    def test_underwriter_keeps_most_fields(self, redactor: FieldRedactor):
        ctx = SecurityContext(user_id="UW-001", roles=[AccessLevel.UNDERWRITER])
        result = _make_result(
            content="Policy details",
            metadata={
                "internal_notes": "High risk client",
                "reserve_amount": "50000",
                "underwriting_score": "78",
                "agent_commission": "10%",
            },
        )
        redacted = redactor.redact([result], ctx)
        meta = redacted[0].metadata
        assert "internal_notes" in meta
        assert "reserve_amount" in meta
        assert "underwriting_score" in meta
        assert "agent_commission" not in meta  # Underwriter can't see agent commission

    def test_no_fields_to_strip(self, redactor: FieldRedactor):
        ctx = SecurityContext(user_id="AGT-001", roles=[AccessLevel.AGENT])
        result = _make_result(
            content="Policy details",
            metadata={"policy_id": "COM-0000077"},
        )
        redacted = redactor.redact([result], ctx)
        meta = redacted[0].metadata
        assert "policy_id" in meta
        assert "_redacted_fields" not in meta  # Nothing was redacted


# =====================================================================
# Admin Bypass
# =====================================================================

class TestAdminBypass:
    """Test that admin users see unredacted content."""

    def test_admin_no_pii_redaction(self):
        redactor = FieldRedactor()
        ctx = SecurityContext(user_id="ADMIN-001", roles=[AccessLevel.ADMIN])
        result = _make_result(
            content="SSN: 123-45-6789, IBAN DE89370400440532013000",
            metadata={
                "internal_notes": "Secret info",
                "reserve_amount": "50000",
            },
        )
        redacted = redactor.redact([result], ctx)
        # Admin sees everything raw
        assert "123-45-6789" in redacted[0].content
        assert "internal_notes" in redacted[0].metadata
        assert "reserve_amount" in redacted[0].metadata

    def test_admin_returns_copy(self):
        redactor = FieldRedactor()
        ctx = SecurityContext(user_id="ADMIN-001", roles=[AccessLevel.ADMIN])
        results = [_make_result(content="Test")]
        redacted = redactor.redact(results, ctx)
        assert len(redacted) == len(results)


# =====================================================================
# Non-mutating behavior
# =====================================================================

class TestNonMutating:
    """Test that redaction doesn't mutate the original results."""

    def test_original_unchanged(self):
        redactor = FieldRedactor()
        ctx = SecurityContext(user_id="PH-001", roles=[AccessLevel.POLICYHOLDER])
        original_content = "SSN: 123-45-6789"
        original_meta = {"internal_notes": "Secret", "policy_id": "POL-001"}
        result = _make_result(content=original_content, metadata=dict(original_meta))

        redacted = redactor.redact([result], ctx)

        # Original should be unchanged
        assert result.content == original_content
        assert "internal_notes" in result.metadata

        # Redacted should be different
        assert "[REDACTED-SSN]" in redacted[0].content
        assert "internal_notes" not in redacted[0].metadata


# =====================================================================
# Multi-role redaction
# =====================================================================

class TestMultiRoleRedaction:
    """Test redaction behavior when user has multiple roles (least restrictive wins)."""

    def test_agent_and_adjuster_intersection(self):
        redactor = FieldRedactor(redact_pii=False)
        ctx = SecurityContext(
            user_id="MULTI-001",
            roles=[AccessLevel.AGENT, AccessLevel.ADJUSTER],
        )
        result = _make_result(
            content="Details",
            metadata={
                "reserve_amount": "50000",  # Agent restricted, Adjuster allowed
                "underwriting_score": "78",  # Agent restricted, Adjuster restricted
                "agent_commission": "10%",  # Agent allowed, Adjuster restricted
                "internal_notes": "Note",  # Both allowed
            },
        )
        redacted = redactor.redact([result], ctx)
        meta = redacted[0].metadata
        # Only fields restricted by BOTH roles should be stripped (intersection)
        # Agent restricted: reserve_amount, underwriting_score, underwriting_notes, loss_ratio
        # Adjuster restricted: underwriting_score, underwriting_notes, agent_commission
        # Intersection: underwriting_score, underwriting_notes
        assert "internal_notes" in meta
        assert "underwriting_score" not in meta  # Both restrict this
