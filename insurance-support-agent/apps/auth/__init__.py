"""
Authentication and authorization module for the insurance support agent.

Provides JWT-based authentication, SecurityContext-driven authorization,
role-based document filtering, field-level redaction, and audit logging.
"""
from apps.auth.audit import AuditEvent, AuditLogger, FileAuditLogger, InMemoryAuditLogger
from apps.auth.jwt import JWTValidator
from apps.auth.models import (
    AuthenticationError,
    AuthorizationError,
    ROLE_ACCESS_LEVELS,
    SecurityContext,
)
from apps.auth.redaction import FieldRedactor, Redactor
from apps.auth.secure_retriever import SecureRetriever

__all__ = [
    "AuditEvent",
    "AuditLogger",
    "AuthenticationError",
    "AuthorizationError",
    "FieldRedactor",
    "FileAuditLogger",
    "InMemoryAuditLogger",
    "JWTValidator",
    "ROLE_ACCESS_LEVELS",
    "Redactor",
    "SecureRetriever",
    "SecurityContext",
]
