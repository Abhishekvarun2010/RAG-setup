"""
Core identity, authorization, and exception models for the security layer.

SecurityContext represents an authenticated user's identity and permissions,
and translates them into mandatory OpenSearch retrieval filters that cannot
be bypassed by application code.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from apps.ingestion.models import AccessLevel
from apps.retrieval.models import RetrievalFilters


# =====================================================================
# Domain Exceptions
# =====================================================================

class AuthenticationError(Exception):
    """Raised when a JWT token is invalid, expired, or tampered."""
    pass


class AuthorizationError(Exception):
    """Raised when an authenticated user lacks permission for an operation."""
    pass


# =====================================================================
# Role → Access Level Mapping
# =====================================================================

ROLE_ACCESS_LEVELS: Dict[AccessLevel, List[str]] = {
    AccessLevel.POLICYHOLDER: [
        "public",
        "policyholder",
    ],
    AccessLevel.AGENT: [
        "public",
        "policyholder",
        "internal",
        "agent",
    ],
    AccessLevel.ADJUSTER: [
        "public",
        "internal",
        "agent",
        "adjuster",
    ],
    AccessLevel.UNDERWRITER: [
        "public",
        "internal",
        "agent",
        "adjuster",
        "underwriter",
    ],
    AccessLevel.ADMIN: [
        "public",
        "internal",
        "confidential",
        "restricted",
        "policyholder",
        "agent",
        "adjuster",
        "underwriter",
        "admin",
    ],
}


# =====================================================================
# SecurityContext
# =====================================================================

class SecurityContext(BaseModel):
    """
    Represents an authenticated user's identity and authorization scope.

    Constructed from JWT claims after token validation. Used by SecureRetriever
    to inject mandatory, non-bypassable OpenSearch filters into every retrieval.
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    user_id: str = Field(
        ...,
        min_length=1,
        description="Authenticated user identifier (e.g. 'PH-00029', 'AGT-001').",
    )
    roles: List[AccessLevel] = Field(
        ...,
        min_length=1,
        description="User's roles from JWT (e.g. [AccessLevel.POLICYHOLDER]).",
    )
    allowed_policy_ids: Optional[List[str]] = Field(
        default=None,
        description="Policy IDs this user may access. None = unrestricted (admin/underwriter).",
    )
    allowed_claim_ids: Optional[List[str]] = Field(
        default=None,
        description="Claim IDs this user may access. None = unrestricted.",
    )
    allowed_policyholder_ids: Optional[List[str]] = Field(
        default=None,
        description="Policyholder IDs (for agents managing multiple clients). None = unrestricted.",
    )

    @property
    def is_admin(self) -> bool:
        """Check if the user has admin privileges."""
        return AccessLevel.ADMIN in self.roles

    @property
    def permitted_access_levels(self) -> List[str]:
        """
        Compute the union of access_control tiers permitted across all roles.
        Admin gets everything. Unknown roles get only 'public'.
        """
        levels: set[str] = set()
        for role in self.roles:
            levels.update(ROLE_ACCESS_LEVELS.get(role, ["public"]))
        return sorted(levels)

    def to_mandatory_filters(
        self,
        as_of_date: Optional[date] = None,
    ) -> RetrievalFilters:
        """
        Convert identity and authorization scope into mandatory RetrievalFilters.

        These filters are injected by SecureRetriever into every OpenSearch query
        and cannot be bypassed or overridden by the caller.

        For entity-scoped users (policyholder, agent), the filters restrict
        retrieval to only their allowed policy/claim/policyholder IDs.
        For unrestricted users (admin, underwriter), entity filters are omitted.
        """
        kwargs: Dict[str, Any] = {}

        if as_of_date is not None:
            kwargs["as_of_date"] = as_of_date

        if self.is_admin:
            # Admin has NO restrictions whatsoever — can view any files
            return RetrievalFilters(**kwargs)

        # For non-admin roles, enforce permitted access_control tiers
        kwargs["access_control"] = self.permitted_access_levels

        # Entity-level restrictions (only for scoped users)
        # If the user has exactly one allowed policy/claim/policyholder,
        # we can set it as a direct term filter. Multiple values require
        # a more complex terms filter, but RetrievalFilters currently
        # supports single-value fields, so we apply the first if available.
        # For multi-entity access, the post-retrieval can_access_chunk() check
        # serves as the secondary enforcement layer.
        if self.allowed_policy_ids is not None and len(self.allowed_policy_ids) == 1:
            kwargs["policy_id"] = self.allowed_policy_ids[0]

        if self.allowed_claim_ids is not None and len(self.allowed_claim_ids) == 1:
            kwargs["claim_id"] = self.allowed_claim_ids[0]

        if self.allowed_policyholder_ids is not None and len(self.allowed_policyholder_ids) == 1:
            kwargs["policyholder_id"] = self.allowed_policyholder_ids[0]

        return RetrievalFilters(**kwargs)

    def can_access_chunk(self, chunk_metadata: Dict[str, Any]) -> bool:
        """
        Post-retrieval defense-in-depth check.

        Validates that a retrieved chunk's metadata falls within
        this user's authorization scope. Used as a secondary enforcement
        after OpenSearch filtering (which handles the primary filtering).
        """
        if self.is_admin:
            return True

        # Check access_control tier
        chunk_access = chunk_metadata.get("access_control")
        if chunk_access is not None:
            permitted = set(self.permitted_access_levels)
            if isinstance(chunk_access, list):
                if not any(a in permitted for a in chunk_access):
                    return False
            elif isinstance(chunk_access, str):
                if chunk_access not in permitted:
                    return False

        # Check entity-level restrictions
        if self.allowed_policy_ids is not None:
            chunk_policy = chunk_metadata.get("policy_id")
            if chunk_policy is not None and chunk_policy not in self.allowed_policy_ids:
                return False

        if self.allowed_claim_ids is not None:
            chunk_claim = chunk_metadata.get("claim_id")
            if chunk_claim is not None and chunk_claim not in self.allowed_claim_ids:
                return False

        if self.allowed_policyholder_ids is not None:
            chunk_ph = chunk_metadata.get("policyholder_id")
            if chunk_ph is not None and chunk_ph not in self.allowed_policyholder_ids:
                return False

        return True
