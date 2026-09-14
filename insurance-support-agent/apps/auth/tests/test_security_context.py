"""
Tests for SecurityContext model, RBAC mapping, mandatory filter generation,
and post-retrieval authorization checks.
"""
import pytest
from datetime import date

from apps.ingestion.models import AccessLevel
from apps.auth.models import SecurityContext, ROLE_ACCESS_LEVELS


# =====================================================================
# SecurityContext construction
# =====================================================================

class TestSecurityContextCreation:
    """Test SecurityContext instantiation and validation."""

    def test_minimal_context(self):
        ctx = SecurityContext(
            user_id="PH-00029",
            roles=[AccessLevel.POLICYHOLDER],
        )
        assert ctx.user_id == "PH-00029"
        assert ctx.roles == [AccessLevel.POLICYHOLDER]
        assert ctx.allowed_policy_ids is None
        assert ctx.allowed_claim_ids is None
        assert ctx.allowed_policyholder_ids is None

    def test_full_context(self):
        ctx = SecurityContext(
            user_id="PH-00029",
            roles=[AccessLevel.POLICYHOLDER],
            allowed_policy_ids=["COM-0000077"],
            allowed_claim_ids=["C-1000"],
            allowed_policyholder_ids=["PH-00029"],
        )
        assert ctx.allowed_policy_ids == ["COM-0000077"]
        assert ctx.allowed_claim_ids == ["C-1000"]
        assert ctx.allowed_policyholder_ids == ["PH-00029"]

    def test_admin_context(self):
        ctx = SecurityContext(
            user_id="ADMIN-001",
            roles=[AccessLevel.ADMIN],
        )
        assert ctx.is_admin is True

    def test_non_admin_context(self):
        ctx = SecurityContext(
            user_id="AGT-001",
            roles=[AccessLevel.AGENT],
        )
        assert ctx.is_admin is False

    def test_empty_user_id_rejected(self):
        with pytest.raises(Exception):
            SecurityContext(
                user_id="",
                roles=[AccessLevel.POLICYHOLDER],
            )

    def test_empty_roles_rejected(self):
        with pytest.raises(Exception):
            SecurityContext(
                user_id="PH-00029",
                roles=[],
            )

    def test_multiple_roles(self):
        ctx = SecurityContext(
            user_id="MULTI-001",
            roles=[AccessLevel.AGENT, AccessLevel.ADJUSTER],
        )
        assert len(ctx.roles) == 2


# =====================================================================
# permitted_access_levels
# =====================================================================

class TestPermittedAccessLevels:
    """Test that each role maps to the correct access tiers."""

    def test_policyholder_levels(self):
        ctx = SecurityContext(user_id="PH-001", roles=[AccessLevel.POLICYHOLDER])
        levels = ctx.permitted_access_levels
        assert set(levels) == {"public", "policyholder"}

    def test_agent_levels(self):
        ctx = SecurityContext(user_id="AGT-001", roles=[AccessLevel.AGENT])
        levels = ctx.permitted_access_levels
        assert set(levels) == {"public", "policyholder", "internal", "agent"}

    def test_adjuster_levels(self):
        ctx = SecurityContext(user_id="ADJ-001", roles=[AccessLevel.ADJUSTER])
        levels = ctx.permitted_access_levels
        assert set(levels) == {"public", "internal", "agent", "adjuster"}

    def test_underwriter_levels(self):
        ctx = SecurityContext(user_id="UW-001", roles=[AccessLevel.UNDERWRITER])
        levels = ctx.permitted_access_levels
        assert set(levels) == {"public", "internal", "agent", "adjuster", "underwriter"}

    def test_admin_levels(self):
        ctx = SecurityContext(user_id="ADMIN-001", roles=[AccessLevel.ADMIN])
        levels = ctx.permitted_access_levels
        expected = {"public", "internal", "confidential", "restricted", "policyholder",
                    "agent", "adjuster", "underwriter", "admin"}
        assert set(levels) == expected

    def test_multi_role_union(self):
        """Multiple roles should produce union of their access levels."""
        ctx = SecurityContext(
            user_id="MULTI-001",
            roles=[AccessLevel.POLICYHOLDER, AccessLevel.AGENT],
        )
        levels = ctx.permitted_access_levels
        # Union of policyholder and agent
        expected = {"public", "policyholder", "internal", "agent"}
        assert set(levels) == expected


# =====================================================================
# to_mandatory_filters()
# =====================================================================

class TestMandatoryFilters:
    """Test SecurityContext.to_mandatory_filters()."""

    def test_policyholder_single_policy(self):
        ctx = SecurityContext(
            user_id="PH-00029",
            roles=[AccessLevel.POLICYHOLDER],
            allowed_policy_ids=["COM-0000077"],
            allowed_claim_ids=["C-1000"],
        )
        filters = ctx.to_mandatory_filters()
        assert filters.access_control is not None
        assert set(filters.access_control) == {"public", "policyholder"}
        assert filters.policy_id == "COM-0000077"
        assert filters.claim_id == "C-1000"

    def test_admin_no_entity_restrictions(self):
        ctx = SecurityContext(
            user_id="ADMIN-001",
            roles=[AccessLevel.ADMIN],
        )
        filters = ctx.to_mandatory_filters()
        # Admin has NO restrictions whatsoever — she can view any files
        assert filters.access_control is None
        assert filters.policy_id is None
        assert filters.claim_id is None
        assert filters.policyholder_id is None
        assert filters.to_opensearch_filter() is None

    def test_admin_with_entity_ids_still_unrestricted(self):
        """Even if entity IDs are present, admin is never restricted by them."""
        ctx = SecurityContext(
            user_id="ADMIN-001",
            roles=[AccessLevel.ADMIN],
            allowed_policy_ids=["POL-999"],
            allowed_claim_ids=["CLM-999"],
            allowed_policyholder_ids=["PH-999"],
        )
        filters = ctx.to_mandatory_filters()
        assert filters.access_control is None
        assert filters.policy_id is None
        assert filters.claim_id is None
        assert filters.policyholder_id is None
        assert filters.to_opensearch_filter() is None

    def test_agent_multiple_policies_no_term_filter(self):
        """Multiple entity IDs don't set a single-value term filter."""
        ctx = SecurityContext(
            user_id="AGT-001",
            roles=[AccessLevel.AGENT],
            allowed_policy_ids=["POL-001", "POL-002"],
        )
        filters = ctx.to_mandatory_filters()
        # Multiple policies means policy_id term filter is NOT set
        assert filters.policy_id is None

    def test_with_as_of_date(self):
        ctx = SecurityContext(
            user_id="PH-00029",
            roles=[AccessLevel.POLICYHOLDER],
        )
        filters = ctx.to_mandatory_filters(as_of_date=date(2024, 3, 15))
        assert filters.as_of_date == date(2024, 3, 15)

    def test_filters_produce_opensearch_dsl(self):
        ctx = SecurityContext(
            user_id="PH-00029",
            roles=[AccessLevel.POLICYHOLDER],
            allowed_policy_ids=["COM-0000077"],
        )
        filters = ctx.to_mandatory_filters()
        dsl = filters.to_opensearch_filter()
        assert dsl is not None
        assert "bool" in dsl
        assert "filter" in dsl["bool"]


# =====================================================================
# can_access_chunk()
# =====================================================================

class TestCanAccessChunk:
    """Test post-retrieval authorization checks."""

    def test_admin_can_access_everything(self):
        ctx = SecurityContext(user_id="ADMIN-001", roles=[AccessLevel.ADMIN])
        assert ctx.can_access_chunk({"access_control": "restricted", "policy_id": "ANY"}) is True

    def test_policyholder_allowed_policy(self):
        ctx = SecurityContext(
            user_id="PH-00029",
            roles=[AccessLevel.POLICYHOLDER],
            allowed_policy_ids=["COM-0000077"],
        )
        assert ctx.can_access_chunk({
            "access_control": "policyholder",
            "policy_id": "COM-0000077",
        }) is True

    def test_policyholder_denied_other_policy(self):
        ctx = SecurityContext(
            user_id="PH-00029",
            roles=[AccessLevel.POLICYHOLDER],
            allowed_policy_ids=["COM-0000077"],
        )
        assert ctx.can_access_chunk({
            "access_control": "policyholder",
            "policy_id": "COM-9999999",
        }) is False

    def test_policyholder_denied_internal_doc(self):
        ctx = SecurityContext(
            user_id="PH-00029",
            roles=[AccessLevel.POLICYHOLDER],
        )
        assert ctx.can_access_chunk({
            "access_control": "internal",
        }) is False

    def test_agent_allowed_internal(self):
        ctx = SecurityContext(
            user_id="AGT-001",
            roles=[AccessLevel.AGENT],
        )
        assert ctx.can_access_chunk({"access_control": "internal"}) is True

    def test_chunk_with_list_access_control(self):
        ctx = SecurityContext(
            user_id="PH-00029",
            roles=[AccessLevel.POLICYHOLDER],
        )
        # Chunk has multiple access levels, one matches
        assert ctx.can_access_chunk({
            "access_control": ["public", "internal"],
        }) is True

    def test_chunk_with_list_access_control_none_match(self):
        ctx = SecurityContext(
            user_id="PH-00029",
            roles=[AccessLevel.POLICYHOLDER],
        )
        assert ctx.can_access_chunk({
            "access_control": ["internal", "adjuster"],
        }) is False

    def test_no_access_control_on_chunk(self):
        """Chunks without access_control metadata should pass (no restriction)."""
        ctx = SecurityContext(
            user_id="PH-00029",
            roles=[AccessLevel.POLICYHOLDER],
        )
        assert ctx.can_access_chunk({"policy_id": "COM-0000077"}) is True

    def test_claim_restriction(self):
        ctx = SecurityContext(
            user_id="PH-00029",
            roles=[AccessLevel.POLICYHOLDER],
            allowed_claim_ids=["C-1000"],
        )
        assert ctx.can_access_chunk({"claim_id": "C-1000"}) is True
        assert ctx.can_access_chunk({"claim_id": "C-9999"}) is False

    def test_policyholder_id_restriction(self):
        ctx = SecurityContext(
            user_id="AGT-001",
            roles=[AccessLevel.AGENT],
            allowed_policyholder_ids=["PH-00029", "PH-00030"],
        )
        assert ctx.can_access_chunk({"policyholder_id": "PH-00029"}) is True
        assert ctx.can_access_chunk({"policyholder_id": "PH-99999"}) is False


# =====================================================================
# ROLE_ACCESS_LEVELS completeness
# =====================================================================

class TestRoleAccessLevels:
    """Ensure ROLE_ACCESS_LEVELS covers all expected roles."""

    def test_all_roles_have_mappings(self):
        expected_roles = {
            AccessLevel.POLICYHOLDER,
            AccessLevel.AGENT,
            AccessLevel.ADJUSTER,
            AccessLevel.UNDERWRITER,
            AccessLevel.ADMIN,
        }
        assert expected_roles.issubset(set(ROLE_ACCESS_LEVELS.keys()))

    def test_public_always_included(self):
        """Every role should have access to 'public' documents."""
        for role, levels in ROLE_ACCESS_LEVELS.items():
            assert "public" in levels, f"Role '{role}' missing 'public' access"

    def test_admin_has_all_levels(self):
        admin_levels = set(ROLE_ACCESS_LEVELS[AccessLevel.ADMIN])
        all_access_values = {al.value for al in AccessLevel}
        assert admin_levels == all_access_values
