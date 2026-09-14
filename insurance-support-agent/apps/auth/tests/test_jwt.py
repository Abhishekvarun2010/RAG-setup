"""
Tests for JWTValidator: token validation, SecurityContext construction,
and error handling for invalid/expired/tampered tokens.
"""
import time

import jwt
import pytest

from apps.ingestion.models import AccessLevel
from apps.auth.jwt import JWTValidator
from apps.auth.models import AuthenticationError


SECRET_KEY = "test-secret-key-for-insurance-agent"
ISSUER = "insurance-agent-test"
AUDIENCE = "insurance-api"


@pytest.fixture
def validator() -> JWTValidator:
    return JWTValidator(
        secret_key=SECRET_KEY,
        algorithm="HS256",
        issuer=ISSUER,
        audience=AUDIENCE,
    )


@pytest.fixture
def simple_validator() -> JWTValidator:
    """Validator without issuer/audience constraints."""
    return JWTValidator(secret_key=SECRET_KEY)


def _make_token(
    payload: dict,
    secret: str = SECRET_KEY,
    algorithm: str = "HS256",
) -> str:
    """Helper to create a raw JWT token."""
    return jwt.encode(payload, secret, algorithm=algorithm)


# =====================================================================
# Valid Token Tests
# =====================================================================

class TestValidTokens:
    """Test successful JWT validation and SecurityContext construction."""

    def test_policyholder_token(self, validator: JWTValidator):
        now = int(time.time())
        token = _make_token({
            "sub": "PH-00029",
            "roles": ["policyholder"],
            "policy_ids": ["COM-0000077"],
            "claim_ids": ["C-1000"],
            "iat": now,
            "exp": now + 3600,
            "iss": ISSUER,
            "aud": AUDIENCE,
        })
        ctx = validator.validate(token)
        assert ctx.user_id == "PH-00029"
        assert ctx.roles == [AccessLevel.POLICYHOLDER]
        assert ctx.allowed_policy_ids == ["COM-0000077"]
        assert ctx.allowed_claim_ids == ["C-1000"]

    def test_agent_token(self, validator: JWTValidator):
        now = int(time.time())
        token = _make_token({
            "sub": "AGT-001",
            "roles": ["agent"],
            "policyholder_ids": ["PH-00029", "PH-00030"],
            "iat": now,
            "exp": now + 3600,
            "iss": ISSUER,
            "aud": AUDIENCE,
        })
        ctx = validator.validate(token)
        assert ctx.user_id == "AGT-001"
        assert ctx.roles == [AccessLevel.AGENT]
        assert ctx.allowed_policyholder_ids == ["PH-00029", "PH-00030"]

    def test_admin_token(self, validator: JWTValidator):
        now = int(time.time())
        token = _make_token({
            "sub": "ADMIN-001",
            "roles": ["admin"],
            "iat": now,
            "exp": now + 3600,
            "iss": ISSUER,
            "aud": AUDIENCE,
        })
        ctx = validator.validate(token)
        assert ctx.user_id == "ADMIN-001"
        assert ctx.is_admin is True
        assert ctx.allowed_policy_ids is None

    def test_multi_role_token(self, validator: JWTValidator):
        now = int(time.time())
        token = _make_token({
            "sub": "MULTI-001",
            "roles": ["agent", "adjuster"],
            "iat": now,
            "exp": now + 3600,
            "iss": ISSUER,
            "aud": AUDIENCE,
        })
        ctx = validator.validate(token)
        assert len(ctx.roles) == 2
        assert AccessLevel.AGENT in ctx.roles
        assert AccessLevel.ADJUSTER in ctx.roles

    def test_token_without_entity_restrictions(self, simple_validator: JWTValidator):
        now = int(time.time())
        token = _make_token({
            "sub": "UW-001",
            "roles": ["underwriter"],
            "iat": now,
            "exp": now + 3600,
        })
        ctx = simple_validator.validate(token)
        assert ctx.user_id == "UW-001"
        assert ctx.allowed_policy_ids is None
        assert ctx.allowed_claim_ids is None


# =====================================================================
# Invalid Token Tests
# =====================================================================

class TestInvalidTokens:
    """Test JWT validation failure cases."""

    def test_expired_token(self, validator: JWTValidator):
        now = int(time.time())
        token = _make_token({
            "sub": "PH-00029",
            "roles": ["policyholder"],
            "iat": now - 7200,
            "exp": now - 3600,  # Expired 1 hour ago
            "iss": ISSUER,
            "aud": AUDIENCE,
        })
        with pytest.raises(AuthenticationError, match="expired"):
            validator.validate(token)

    def test_tampered_token(self, validator: JWTValidator):
        now = int(time.time())
        token = _make_token(
            {
                "sub": "PH-00029",
                "roles": ["policyholder"],
                "iat": now,
                "exp": now + 3600,
                "iss": ISSUER,
                "aud": AUDIENCE,
            },
            secret="wrong-secret-key",
        )
        with pytest.raises(AuthenticationError, match="malformed|tampered"):
            validator.validate(token)

    def test_missing_subject(self, simple_validator: JWTValidator):
        now = int(time.time())
        token = _make_token({
            "roles": ["policyholder"],
            "iat": now,
            "exp": now + 3600,
        })
        with pytest.raises(AuthenticationError, match="sub"):
            simple_validator.validate(token)

    def test_missing_roles(self, simple_validator: JWTValidator):
        now = int(time.time())
        token = _make_token({
            "sub": "PH-00029",
            "iat": now,
            "exp": now + 3600,
        })
        with pytest.raises(AuthenticationError, match="roles"):
            simple_validator.validate(token)

    def test_invalid_role_values(self, simple_validator: JWTValidator):
        now = int(time.time())
        token = _make_token({
            "sub": "PH-00029",
            "roles": ["nonexistent_role"],
            "iat": now,
            "exp": now + 3600,
        })
        with pytest.raises(AuthenticationError, match="no recognized roles"):
            simple_validator.validate(token)

    def test_empty_token(self, validator: JWTValidator):
        with pytest.raises(AuthenticationError, match="Empty"):
            validator.validate("")

    def test_whitespace_token(self, validator: JWTValidator):
        with pytest.raises(AuthenticationError, match="Empty"):
            validator.validate("   ")

    def test_garbage_token(self, validator: JWTValidator):
        with pytest.raises(AuthenticationError):
            validator.validate("not.a.valid.jwt.token")

    def test_wrong_issuer(self, validator: JWTValidator):
        now = int(time.time())
        token = _make_token({
            "sub": "PH-00029",
            "roles": ["policyholder"],
            "iat": now,
            "exp": now + 3600,
            "iss": "wrong-issuer",
            "aud": AUDIENCE,
        })
        with pytest.raises(AuthenticationError, match="issuer"):
            validator.validate(token)

    def test_wrong_audience(self, validator: JWTValidator):
        now = int(time.time())
        token = _make_token({
            "sub": "PH-00029",
            "roles": ["policyholder"],
            "iat": now,
            "exp": now + 3600,
            "iss": ISSUER,
            "aud": "wrong-audience",
        })
        with pytest.raises(AuthenticationError, match="audience"):
            validator.validate(token)


# =====================================================================
# Token Creation (convenience helper)
# =====================================================================

class TestTokenCreation:
    """Test the create_token convenience method."""

    def test_roundtrip(self, simple_validator: JWTValidator):
        token = simple_validator.create_token(
            user_id="PH-00029",
            roles=["policyholder"],
            policy_ids=["COM-0000077"],
            claim_ids=["C-1000"],
        )
        ctx = simple_validator.validate(token)
        assert ctx.user_id == "PH-00029"
        assert ctx.roles == [AccessLevel.POLICYHOLDER]
        assert ctx.allowed_policy_ids == ["COM-0000077"]
        assert ctx.allowed_claim_ids == ["C-1000"]

    def test_roundtrip_admin(self, simple_validator: JWTValidator):
        token = simple_validator.create_token(
            user_id="ADMIN-001",
            roles=["admin"],
        )
        ctx = simple_validator.validate(token)
        assert ctx.is_admin is True

    def test_roundtrip_with_issuer_audience(self, validator: JWTValidator):
        token = validator.create_token(
            user_id="AGT-001",
            roles=["agent"],
        )
        ctx = validator.validate(token)
        assert ctx.user_id == "AGT-001"
