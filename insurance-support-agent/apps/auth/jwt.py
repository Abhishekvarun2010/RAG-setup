"""
JWT token validation and SecurityContext construction.

Validates JWT tokens (HS256 by default), extracts identity claims,
and builds a SecurityContext for downstream authorization enforcement.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import jwt

from apps.ingestion.models import AccessLevel
from apps.auth.models import AuthenticationError, SecurityContext

logger = logging.getLogger(__name__)


class JWTValidator:
    """
    Validates JWT tokens and constructs SecurityContext instances.

    Supports HS256 (symmetric) for development and RS256 (asymmetric)
    for production IdP integration (Keycloak, Auth0, Cognito).

    Expected JWT payload structure:
        {
            "sub": "PH-00029",
            "roles": ["policyholder"],
            "policy_ids": ["COM-0000077"],
            "claim_ids": ["C-1000"],
            "policyholder_ids": ["PH-00029"],
            "iat": 1726300000,
            "exp": 1726303600
        }
    """

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        issuer: Optional[str] = None,
        audience: Optional[str] = None,
    ) -> None:
        self._secret_key = secret_key
        self._algorithm = algorithm
        self._issuer = issuer
        self._audience = audience

    @property
    def algorithm(self) -> str:
        return self._algorithm

    def validate(self, token: str) -> SecurityContext:
        """
        Decode and validate a JWT token, returning a SecurityContext.

        Raises:
            AuthenticationError: If the token is invalid, expired, or tampered.
        """
        if not token or not token.strip():
            raise AuthenticationError("Empty or missing JWT token.")

        try:
            decode_options: Dict[str, Any] = {}
            decode_kwargs: Dict[str, Any] = {
                "algorithms": [self._algorithm],
            }
            if self._issuer:
                decode_kwargs["issuer"] = self._issuer
            if self._audience:
                decode_kwargs["audience"] = self._audience

            payload = jwt.decode(
                token.strip(),
                self._secret_key,
                **decode_kwargs,
            )
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("JWT token has expired.")
        except jwt.InvalidIssuerError:
            raise AuthenticationError("JWT token has an invalid issuer.")
        except jwt.InvalidAudienceError:
            raise AuthenticationError("JWT token has an invalid audience.")
        except jwt.DecodeError as e:
            raise AuthenticationError(f"JWT token is malformed or tampered: {e}")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid JWT token: {e}")

        return self._build_context(payload)

    def _build_context(self, payload: Dict[str, Any]) -> SecurityContext:
        """
        Extract identity claims from decoded JWT payload and construct SecurityContext.
        """
        # Subject (user ID) is required
        user_id = payload.get("sub")
        if not user_id:
            raise AuthenticationError("JWT token missing required 'sub' (subject) claim.")

        # Roles are required
        raw_roles = payload.get("roles")
        if not raw_roles or not isinstance(raw_roles, list):
            raise AuthenticationError("JWT token missing or invalid 'roles' claim.")

        roles: List[AccessLevel] = []
        for role_str in raw_roles:
            try:
                roles.append(AccessLevel(role_str))
            except ValueError:
                logger.warning(f"Unknown role '{role_str}' in JWT for user '{user_id}', skipping.")

        if not roles:
            raise AuthenticationError(
                f"JWT token for user '{user_id}' contains no recognized roles."
            )

        # Optional entity restrictions
        allowed_policy_ids = payload.get("policy_ids")
        allowed_claim_ids = payload.get("claim_ids")
        allowed_policyholder_ids = payload.get("policyholder_ids")

        return SecurityContext(
            user_id=str(user_id),
            roles=roles,
            allowed_policy_ids=allowed_policy_ids,
            allowed_claim_ids=allowed_claim_ids,
            allowed_policyholder_ids=allowed_policyholder_ids,
        )

    def create_token(
        self,
        user_id: str,
        roles: List[str],
        policy_ids: Optional[List[str]] = None,
        claim_ids: Optional[List[str]] = None,
        policyholder_ids: Optional[List[str]] = None,
        expires_in: int = 3600,
    ) -> str:
        """
        Create a signed JWT token (convenience method for testing and development).

        Args:
            user_id: The subject identifier.
            roles: List of role strings (e.g. ["policyholder"]).
            policy_ids: Optional list of allowed policy IDs.
            claim_ids: Optional list of allowed claim IDs.
            policyholder_ids: Optional list of allowed policyholder IDs.
            expires_in: Token TTL in seconds (default: 1 hour).
        """
        import time

        now = int(time.time())
        payload: Dict[str, Any] = {
            "sub": user_id,
            "roles": roles,
            "iat": now,
            "exp": now + expires_in,
        }
        if policy_ids is not None:
            payload["policy_ids"] = policy_ids
        if claim_ids is not None:
            payload["claim_ids"] = claim_ids
        if policyholder_ids is not None:
            payload["policyholder_ids"] = policyholder_ids
        if self._issuer:
            payload["iss"] = self._issuer
        if self._audience:
            payload["aud"] = self._audience

        return jwt.encode(payload, self._secret_key, algorithm=self._algorithm)
