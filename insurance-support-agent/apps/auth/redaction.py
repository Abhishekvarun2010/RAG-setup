"""
Field-level content redaction for role-based information security.

Strips PII patterns and role-inappropriate metadata from retrieved chunks
before they are passed to the Agent Runtime / LLM context.
"""
from __future__ import annotations

import copy
import logging
import re
from typing import Any, Dict, List, Optional, Protocol, Set, runtime_checkable

from apps.ingestion.models import AccessLevel
from apps.auth.models import SecurityContext
from apps.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)


# =====================================================================
# Protocol
# =====================================================================

@runtime_checkable
class Redactor(Protocol):
    """Protocol defining the interface for content redactors."""

    def redact(
        self,
        results: List[RetrievalResult],
        context: SecurityContext,
    ) -> List[RetrievalResult]:
        """
        Apply redaction rules to a list of retrieval results
        based on the user's security context.
        """
        ...


# =====================================================================
# PII Patterns
# =====================================================================

# Common EU/insurance PII patterns
PII_PATTERNS: List[Dict[str, Any]] = [
    {
        "name": "IBAN",
        "pattern": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b"),
        "replacement": "[REDACTED-IBAN]",
    },
    {
        "name": "credit_card",
        "pattern": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "replacement": "[REDACTED-CC]",
    },
    {
        "name": "ssn_us",
        "pattern": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "replacement": "[REDACTED-SSN]",
    },
    {
        "name": "email",
        "pattern": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "replacement": "[REDACTED-EMAIL]",
    },
    {
        "name": "phone_intl",
        "pattern": re.compile(r"\+\d{1,3}[\s-]?\(?\d{1,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}\b"),
        "replacement": "[REDACTED-PHONE]",
    },
]

# =====================================================================
# Role-Based Metadata Field Restrictions
# =====================================================================

# Fields that should be stripped from metadata for each role.
# Admin sees everything, so is not listed.
ROLE_REDACTED_FIELDS: Dict[AccessLevel, Set[str]] = {
    AccessLevel.POLICYHOLDER: {
        "internal_notes",
        "reserve_amount",
        "adjuster_assessment",
        "underwriting_score",
        "underwriting_notes",
        "agent_commission",
        "loss_ratio",
    },
    AccessLevel.AGENT: {
        "reserve_amount",
        "underwriting_score",
        "underwriting_notes",
        "loss_ratio",
    },
    AccessLevel.ADJUSTER: {
        "underwriting_score",
        "underwriting_notes",
        "agent_commission",
    },
    AccessLevel.UNDERWRITER: {
        "agent_commission",
    },
}


# =====================================================================
# Concrete Redactor
# =====================================================================

class FieldRedactor:
    """
    Applies field-level and PII redaction to retrieval results.

    Features:
    - Regex-based PII scrubbing from chunk content (IBAN, SSN, email, phone, CC).
    - Role-based metadata field stripping (e.g. policyholders can't see reserve amounts).
    - Admin bypass: no redaction applied for admin users.
    - Non-destructive: creates copies of results, never mutates originals.
    """

    def __init__(
        self,
        pii_patterns: Optional[List[Dict[str, Any]]] = None,
        role_redacted_fields: Optional[Dict[AccessLevel, Set[str]]] = None,
        redact_pii: bool = True,
    ) -> None:
        self._pii_patterns = pii_patterns if pii_patterns is not None else PII_PATTERNS
        self._role_redacted_fields = (
            role_redacted_fields if role_redacted_fields is not None else ROLE_REDACTED_FIELDS
        )
        self._redact_pii = redact_pii

    def redact(
        self,
        results: List[RetrievalResult],
        context: SecurityContext,
    ) -> List[RetrievalResult]:
        """
        Apply redaction rules to retrieval results based on the user's security context.

        Args:
            results: List of RetrievalResult instances to redact.
            context: The authenticated user's SecurityContext.

        Returns:
            New list of RetrievalResult instances with redacted content and metadata.
        """
        if context.is_admin:
            return list(results)

        redacted: List[RetrievalResult] = []
        for result in results:
            redacted.append(self._redact_single(result, context))
        return redacted

    def _redact_single(
        self,
        result: RetrievalResult,
        context: SecurityContext,
    ) -> RetrievalResult:
        """Redact a single RetrievalResult."""
        content = result.content
        metadata = copy.deepcopy(result.metadata)
        redacted_fields: List[str] = []

        # 1. PII scrubbing from content
        if self._redact_pii:
            content = self._scrub_pii(content)

        # 2. Role-based metadata field stripping
        fields_to_strip: Set[str] = set()
        for role in context.roles:
            role_fields = self._role_redacted_fields.get(role)
            if role_fields is not None:
                fields_to_strip.update(role_fields)

        # If user has multiple roles, only strip fields that ALL roles would strip
        # (most permissive wins). But for simplicity in v1, we take the intersection
        # of restricted fields across all roles (least restrictive).
        if len(context.roles) > 1:
            role_field_sets = [
                self._role_redacted_fields.get(role, set()) for role in context.roles
            ]
            fields_to_strip = set.intersection(*role_field_sets) if role_field_sets else set()

        for field in fields_to_strip:
            if field in metadata:
                del metadata[field]
                redacted_fields.append(field)

        if redacted_fields:
            metadata["_redacted_fields"] = redacted_fields

        return RetrievalResult(
            chunk_id=result.chunk_id,
            content=content,
            score=result.score,
            retrieval_method=result.retrieval_method,
            metadata=metadata,
        )

    def _scrub_pii(self, text: str) -> str:
        """Apply PII regex patterns to scrub sensitive data from text."""
        for pattern_def in self._pii_patterns:
            text = pattern_def["pattern"].sub(pattern_def["replacement"], text)
        return text
