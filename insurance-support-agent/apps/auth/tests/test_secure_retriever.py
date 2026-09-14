"""
Tests for SecureRetriever: mandatory filter injection, filter merging,
post-retrieval authorization, redaction integration, and audit logging.
"""
import pytest
from datetime import date
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, patch

from apps.ingestion.models import AccessLevel
from apps.auth.audit import InMemoryAuditLogger
from apps.auth.models import SecurityContext
from apps.auth.redaction import FieldRedactor
from apps.auth.secure_retriever import SecureRetriever, _merge_filters
from apps.retrieval.models import RetrievalFilters, RetrievalMethod, RetrievalResult


# =====================================================================
# Helpers
# =====================================================================

def _make_result(
    chunk_id: str = "test-chunk",
    content: str = "Test content",
    metadata: Optional[Dict[str, Any]] = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        content=content,
        score=1.0,
        retrieval_method=RetrievalMethod.HYBRID_RRF,
        metadata=metadata or {},
    )


def _mock_retriever(results: List[RetrievalResult] = None) -> MagicMock:
    """Create a mock HybridRetriever that returns specified results."""
    mock = MagicMock()
    mock.retrieve.return_value = results or []
    mock.retrieve_bm25.return_value = results or []
    mock.retrieve_vector.return_value = results or []
    return mock


# =====================================================================
# Filter Merging
# =====================================================================

class TestFilterMerging:
    """Test _merge_filters behavior."""

    def test_no_user_filters(self):
        mandatory = RetrievalFilters(
            access_control=["public", "policyholder"],
            policy_id="COM-0000077",
        )
        merged = _merge_filters(mandatory, None)
        assert merged.access_control == ["public", "policyholder"]
        assert merged.policy_id == "COM-0000077"

    def test_user_filters_narrowing(self):
        mandatory = RetrievalFilters(
            access_control=["public", "policyholder"],
        )
        user_supplied = RetrievalFilters(
            document_type="estimate",
        )
        merged = _merge_filters(mandatory, user_supplied)
        # Mandatory wins for access_control
        assert merged.access_control == ["public", "policyholder"]
        # User-supplied adds narrowing field
        assert merged.document_type == "estimate"

    def test_mandatory_overrides_user_on_conflict(self):
        mandatory = RetrievalFilters(
            access_control=["public", "policyholder"],
            policy_id="COM-0000077",
        )
        user_supplied = RetrievalFilters(
            policy_id="COM-9999999",  # User tries to widen
        )
        merged = _merge_filters(mandatory, user_supplied)
        # Mandatory wins
        assert merged.policy_id == "COM-0000077"

    def test_access_control_narrowing(self):
        mandatory = RetrievalFilters(
            access_control=["public", "policyholder", "internal", "agent"],
        )
        user_supplied = RetrievalFilters(
            access_control=["public"],  # User narrows to public only
        )
        merged = _merge_filters(mandatory, user_supplied)
        # Intersection: user can narrow
        assert merged.access_control == ["public"]

    def test_access_control_widening_rejected(self):
        mandatory = RetrievalFilters(
            access_control=["public", "policyholder"],
        )
        user_supplied = RetrievalFilters(
            access_control=["public", "policyholder", "admin"],  # User tries to add admin
        )
        merged = _merge_filters(mandatory, user_supplied)
        # admin is not in mandatory set, so it's filtered out
        assert "admin" not in merged.access_control

    def test_user_dict_filters(self):
        mandatory = RetrievalFilters(
            access_control=["public", "policyholder"],
        )
        user_dict = {"document_type": "estimate"}
        merged = _merge_filters(mandatory, user_dict)
        assert merged.access_control == ["public", "policyholder"]
        assert merged.document_type == "estimate"

    def test_invalid_user_dict_ignored(self):
        mandatory = RetrievalFilters(
            access_control=["public"],
        )
        user_dict = {"invalid_field": "value"}
        merged = _merge_filters(mandatory, user_dict)
        assert merged.access_control == ["public"]


# =====================================================================
# SecureRetriever Core Behavior
# =====================================================================

class TestSecureRetriever:
    """Test SecureRetriever retrieval with mandatory security enforcement."""

    def test_mandatory_filters_applied(self):
        """Verify mandatory filters are passed to the underlying retriever."""
        mock_retriever = _mock_retriever([
            _make_result("c1", metadata={"access_control": "policyholder", "policy_id": "COM-0000077"}),
        ])
        ctx = SecurityContext(
            user_id="PH-00029",
            roles=[AccessLevel.POLICYHOLDER],
            allowed_policy_ids=["COM-0000077"],
        )
        secure = SecureRetriever(retriever=mock_retriever)
        secure.retrieve("test query", context=ctx)

        # Verify retrieve was called
        mock_retriever.retrieve.assert_called_once()
        call_kwargs = mock_retriever.retrieve.call_args
        filters = call_kwargs.kwargs.get("filters") or call_kwargs[1].get("filters")
        assert filters is not None
        assert filters.access_control is not None
        assert "policyholder" in filters.access_control
        assert filters.policy_id == "COM-0000077"

    def test_post_retrieval_filters_unauthorized_chunks(self):
        """Chunks that fail can_access_chunk are removed post-retrieval."""
        mock_retriever = _mock_retriever([
            _make_result("c1", metadata={"access_control": "policyholder", "policy_id": "COM-0000077"}),
            _make_result("c2", metadata={"access_control": "internal", "policy_id": "COM-9999999"}),
        ])
        ctx = SecurityContext(
            user_id="PH-00029",
            roles=[AccessLevel.POLICYHOLDER],
            allowed_policy_ids=["COM-0000077"],
        )
        secure = SecureRetriever(retriever=mock_retriever)
        results = secure.retrieve("test query", context=ctx)

        # Only the authorized chunk should remain
        assert len(results) == 1
        assert results[0].chunk_id == "c1"

    def test_admin_sees_everything(self):
        """Admin should see all chunks without filtering, including untagged files."""
        mock_retriever = _mock_retriever([
            _make_result("c1", metadata={"access_control": "restricted"}),
            _make_result("c2", metadata={"access_control": "confidential"}),
            _make_result("c3", metadata={"access_control": "admin"}),
            _make_result("c4", metadata={}),  # No access_control metadata at all
            _make_result("c5", metadata={"policy_id": "OTHER-POL", "claim_id": "OTHER-CLM"}),
        ])
        ctx = SecurityContext(user_id="ADMIN-001", roles=[AccessLevel.ADMIN])
        secure = SecureRetriever(retriever=mock_retriever)
        results = secure.retrieve("test query", context=ctx)
        assert len(results) == 5

        # Verify underlying retriever was called with NO filter restrictions
        call_kwargs = mock_retriever.retrieve.call_args
        filters = call_kwargs.kwargs.get("filters") or call_kwargs[1].get("filters")
        assert filters.to_opensearch_filter() is None

    def test_admin_voluntary_filters_honored(self):
        """Admin can voluntarily apply filters if she chooses to."""
        mock_retriever = _mock_retriever([])
        ctx = SecurityContext(user_id="ADMIN-001", roles=[AccessLevel.ADMIN])
        secure = SecureRetriever(retriever=mock_retriever)
        secure.retrieve(
            "test query",
            context=ctx,
            filters={"policy_id": "COM-0000077", "document_type": "policy"},
        )

        call_kwargs = mock_retriever.retrieve.call_args
        filters = call_kwargs.kwargs.get("filters") or call_kwargs[1].get("filters")
        assert filters.policy_id == "COM-0000077"
        assert filters.document_type == "policy"
        assert filters.access_control is None

    def test_with_as_of_date(self):
        """Verify as_of_date is forwarded to mandatory filters."""
        mock_retriever = _mock_retriever([])
        ctx = SecurityContext(
            user_id="PH-00029",
            roles=[AccessLevel.POLICYHOLDER],
        )
        secure = SecureRetriever(retriever=mock_retriever)
        secure.retrieve("test query", context=ctx, as_of_date=date(2024, 3, 15))

        call_kwargs = mock_retriever.retrieve.call_args
        filters = call_kwargs.kwargs.get("filters") or call_kwargs[1].get("filters")
        assert filters.as_of_date == date(2024, 3, 15)


# =====================================================================
# Redaction Integration
# =====================================================================

class TestSecureRetrieverRedaction:
    """Test SecureRetriever with FieldRedactor integration."""

    def test_pii_redacted_in_results(self):
        mock_retriever = _mock_retriever([
            _make_result(
                "c1",
                content="Contact john@example.com for SSN: 123-45-6789",
                metadata={"access_control": "policyholder"},
            ),
        ])
        ctx = SecurityContext(user_id="PH-00029", roles=[AccessLevel.POLICYHOLDER])
        redactor = FieldRedactor()
        secure = SecureRetriever(retriever=mock_retriever, redactor=redactor)
        results = secure.retrieve("test query", context=ctx)

        assert len(results) == 1
        assert "[REDACTED-EMAIL]" in results[0].content
        assert "[REDACTED-SSN]" in results[0].content
        assert "john@example.com" not in results[0].content

    def test_metadata_fields_stripped(self):
        mock_retriever = _mock_retriever([
            _make_result(
                "c1",
                content="Policy details",
                metadata={
                    "access_control": "policyholder",
                    "policy_id": "COM-0000077",
                    "internal_notes": "Secret info",
                    "reserve_amount": "50000",
                },
            ),
        ])
        ctx = SecurityContext(user_id="PH-00029", roles=[AccessLevel.POLICYHOLDER])
        redactor = FieldRedactor()
        secure = SecureRetriever(retriever=mock_retriever, redactor=redactor)
        results = secure.retrieve("test query", context=ctx)

        meta = results[0].metadata
        assert "policy_id" in meta
        assert "internal_notes" not in meta
        assert "reserve_amount" not in meta


# =====================================================================
# Audit Logging Integration
# =====================================================================

class TestSecureRetrieverAudit:
    """Test SecureRetriever with audit logging integration."""

    def test_audit_event_logged(self):
        mock_retriever = _mock_retriever([
            _make_result("c1", metadata={"access_control": "policyholder"}),
        ])
        ctx = SecurityContext(user_id="PH-00029", roles=[AccessLevel.POLICYHOLDER])
        audit = InMemoryAuditLogger()
        secure = SecureRetriever(retriever=mock_retriever, audit_logger=audit)
        secure.retrieve("what does my policy cover?", context=ctx)

        assert len(audit.events) == 1
        event = audit.events[0]
        assert event.user_id == "PH-00029"
        assert event.query == "what does my policy cover?"
        assert "policyholder" in event.roles
        assert "c1" in event.chunks_returned

    def test_audit_captures_filtered_chunks(self):
        mock_retriever = _mock_retriever([
            _make_result("c1", metadata={"access_control": "policyholder"}),
            _make_result("c2", metadata={"access_control": "admin"}),  # Will be filtered
        ])
        ctx = SecurityContext(user_id="PH-00029", roles=[AccessLevel.POLICYHOLDER])
        audit = InMemoryAuditLogger()
        secure = SecureRetriever(retriever=mock_retriever, audit_logger=audit)
        results = secure.retrieve("query", context=ctx)

        assert len(results) == 1
        event = audit.events[0]
        assert event.chunks_filtered_out == 1

    def test_audit_captures_redacted_fields(self):
        mock_retriever = _mock_retriever([
            _make_result(
                "c1",
                content="Details",
                metadata={
                    "access_control": "policyholder",
                    "internal_notes": "Secret",
                },
            ),
        ])
        ctx = SecurityContext(user_id="PH-00029", roles=[AccessLevel.POLICYHOLDER])
        redactor = FieldRedactor(redact_pii=False)
        audit = InMemoryAuditLogger()
        secure = SecureRetriever(
            retriever=mock_retriever,
            redactor=redactor,
            audit_logger=audit,
        )
        secure.retrieve("query", context=ctx)

        event = audit.events[0]
        assert "internal_notes" in event.redacted_fields

    def test_no_audit_when_logger_absent(self):
        """When no audit logger is configured, retrieval should still work."""
        mock_retriever = _mock_retriever([
            _make_result("c1", metadata={"access_control": "policyholder"}),
        ])
        ctx = SecurityContext(user_id="PH-00029", roles=[AccessLevel.POLICYHOLDER])
        secure = SecureRetriever(retriever=mock_retriever)
        results = secure.retrieve("query", context=ctx)
        assert len(results) == 1  # Should work without audit logger


# =====================================================================
# Cross-Role Isolation
# =====================================================================

class TestCrossRoleIsolation:
    """Test that different roles see different document sets."""

    def _make_corpus(self) -> List[RetrievalResult]:
        return [
            _make_result("public-1", metadata={"access_control": "public"}),
            _make_result("internal-1", metadata={"access_control": "internal"}),
            _make_result("policyholder-1", metadata={"access_control": "policyholder", "policy_id": "POL-001"}),
            _make_result("adjuster-1", metadata={"access_control": "adjuster"}),
            _make_result("admin-1", metadata={"access_control": "admin"}),
        ]

    def test_policyholder_isolation(self):
        mock = _mock_retriever(self._make_corpus())
        ctx = SecurityContext(
            user_id="PH-001",
            roles=[AccessLevel.POLICYHOLDER],
            allowed_policy_ids=["POL-001"],
        )
        secure = SecureRetriever(retriever=mock)
        results = secure.retrieve("query", context=ctx)
        chunk_ids = {r.chunk_id for r in results}
        assert "public-1" in chunk_ids
        assert "policyholder-1" in chunk_ids
        assert "internal-1" not in chunk_ids
        assert "adjuster-1" not in chunk_ids
        assert "admin-1" not in chunk_ids

    def test_agent_visibility(self):
        mock = _mock_retriever(self._make_corpus())
        ctx = SecurityContext(user_id="AGT-001", roles=[AccessLevel.AGENT])
        secure = SecureRetriever(retriever=mock)
        results = secure.retrieve("query", context=ctx)
        chunk_ids = {r.chunk_id for r in results}
        assert "public-1" in chunk_ids
        assert "internal-1" in chunk_ids
        assert "policyholder-1" in chunk_ids
        assert "adjuster-1" not in chunk_ids
        assert "admin-1" not in chunk_ids

    def test_admin_sees_all(self):
        mock = _mock_retriever(self._make_corpus())
        ctx = SecurityContext(user_id="ADMIN-001", roles=[AccessLevel.ADMIN])
        secure = SecureRetriever(retriever=mock)
        results = secure.retrieve("query", context=ctx)
        assert len(results) == 5
