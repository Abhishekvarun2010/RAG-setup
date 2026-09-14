"""
Compliance audit logging for retrieval operations.

Records structured audit events for every retrieval operation,
including who queried, what filters were applied, which chunks
were returned, and what was redacted.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# =====================================================================
# Audit Event Model
# =====================================================================

class AuditEvent(BaseModel):
    """Structured record of a retrieval operation for compliance logging."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the retrieval event.",
    )
    user_id: str = Field(
        ...,
        description="Authenticated user identifier.",
    )
    roles: List[str] = Field(
        ...,
        description="User's roles at time of query.",
    )
    query: str = Field(
        ...,
        description="The search query submitted.",
    )
    filters_applied: Dict[str, Any] = Field(
        default_factory=dict,
        description="OpenSearch filters that were applied (including mandatory security filters).",
    )
    chunks_returned: List[str] = Field(
        default_factory=list,
        description="List of chunk IDs returned to the user.",
    )
    chunks_filtered_out: int = Field(
        default=0,
        description="Number of chunks removed by post-retrieval authorization checks.",
    )
    redacted_fields: List[str] = Field(
        default_factory=list,
        description="Metadata fields that were redacted from results.",
    )
    retrieval_method: Optional[str] = Field(
        default=None,
        description="The retrieval method used (hybrid_rrf, reranked, etc.).",
    )


# =====================================================================
# Protocol
# =====================================================================

@runtime_checkable
class AuditLogger(Protocol):
    """Protocol defining the interface for audit loggers."""

    def log_retrieval(self, event: AuditEvent) -> None:
        """Log a retrieval audit event."""
        ...


# =====================================================================
# File-Based Audit Logger
# =====================================================================

class FileAuditLogger:
    """
    Appends structured JSON-line audit events to a log file.

    Each line in the file is a complete JSON object representing
    one retrieval event, enabling efficient log parsing and analysis.
    """

    def __init__(self, log_path: str = "audit/retrieval_audit.jsonl") -> None:
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def log_path(self) -> Path:
        return self._log_path

    def log_retrieval(self, event: AuditEvent) -> None:
        """Append a retrieval audit event as a JSON line to the log file."""
        try:
            line = event.model_dump_json() + "\n"
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            logger.error(f"Failed to write audit event for user '{event.user_id}': {e}")


# =====================================================================
# In-Memory Audit Logger (for testing)
# =====================================================================

class InMemoryAuditLogger:
    """In-memory audit logger that stores events in a list for testing."""

    def __init__(self) -> None:
        self.events: List[AuditEvent] = []

    def log_retrieval(self, event: AuditEvent) -> None:
        """Store the audit event in memory."""
        self.events.append(event)

    def clear(self) -> None:
        """Clear all stored events."""
        self.events.clear()
