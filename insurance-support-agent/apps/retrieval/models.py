"""
Domain models for the retrieval layer.

Decouples the rest of the application and the Agent Runtime from infrastructure-specific
search payloads (e.g. OpenSearch hit dictionaries).
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class RetrievalMethod(str, Enum):
    """Search / retrieval strategy utilized to surface candidate chunks."""
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID_RRF = "hybrid_rrf"


class RetrievalFilters(BaseModel):
    """
    Structured metadata filters applied natively inside OpenSearch queries.
    """
    model_config = ConfigDict(
        use_enum_values=True,
        str_strip_whitespace=True,
        extra="forbid",
    )

    policy_id: Optional[str] = Field(default=None, description="Filter by policy ID (e.g. 'COM-0000077')")
    claim_id: Optional[str] = Field(default=None, description="Filter by claim ID (e.g. 'C-1000')")
    policyholder_id: Optional[str] = Field(default=None, description="Filter by policyholder ID (e.g. 'PH-00029')")
    document_type: Optional[str] = Field(default=None, description="Filter by document type (e.g. 'estimate')")
    line_of_business: Optional[str] = Field(default=None, description="Filter by line of business (e.g. 'commercial')")
    product: Optional[str] = Field(default=None, description="Filter by product name")
    status: Optional[str] = Field(default=None, description="Filter by status (e.g. 'closed', 'active')")
    section: Optional[str] = Field(default=None, description="Filter by section title (e.g. 'Line Items')")
    access_control: Optional[List[str]] = Field(default=None, description="Filter by permitted access levels")
    effective_from: Optional[date] = Field(default=None, description="Effective date range lower bound")
    effective_to: Optional[date] = Field(default=None, description="Effective date range upper bound")

    def to_opensearch_filter(self) -> Optional[Dict[str, Any]]:
        """
        Build an OpenSearch bool filter clause.
        Returns None if no filters are active.
        """
        clauses: List[Dict[str, Any]] = []

        keyword_fields = [
            ("policy_id", self.policy_id),
            ("claim_id", self.claim_id),
            ("policyholder_id", self.policyholder_id),
            ("document_type", self.document_type),
            ("line_of_business", self.line_of_business),
            ("product", self.product),
            ("status", self.status),
            ("section", self.section),
        ]
        for field_name, value in keyword_fields:
            if value is not None:
                str_val = value.value if isinstance(value, Enum) else str(value)
                clauses.append({"term": {field_name: str_val}})

        if self.access_control:
            clauses.append({"terms": {"access_control": self.access_control}})

        if self.effective_from or self.effective_to:
            range_clause: Dict[str, Any] = {}
            if self.effective_from:
                range_clause["gte"] = self.effective_from.isoformat()
            if self.effective_to:
                range_clause["lte"] = self.effective_to.isoformat()
            clauses.append({"range": {"effective_from": range_clause}})

        if not clauses:
            return None

        return {"bool": {"filter": clauses}}


class RetrievalResult(BaseModel):
    """
    Standardized domain candidate returned by the retrieval layer.
    """
    model_config = ConfigDict(
        use_enum_values=False,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    chunk_id: str = Field(
        ...,
        min_length=1,
        description="Unique chunk identifier (e.g. 'DOC-C-1000-ESTIMATE#c3').",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Extracted textual content of the chunk.",
    )
    score: float = Field(
        ...,
        description="Relevance or fusion score (higher is more relevant).",
    )
    retrieval_method: RetrievalMethod = Field(
        ...,
        description="The retrieval mechanism that produced this candidate.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Associated metadata (document_id, line_of_business, page_number, etc.).",
    )

    @classmethod
    def from_opensearch_hit(
        cls,
        hit: Dict[str, Any],
        method: RetrievalMethod,
        custom_score: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Construct a clean RetrievalResult from a raw OpenSearch hit dictionary:
        {
            "_id": "...",
            "_score": 1.234,
            "_source": {...}
        }
        Excludes the large raw embedding array from candidate metadata.
        """
        source: Dict[str, Any] = hit.get("_source", {})
        chunk_id = hit.get("_id") or source.get("chunk_id", "")
        content = source.get("content", "")

        raw_score = hit.get("_score")
        if custom_score is not None:
            score = float(custom_score)
        elif raw_score is not None:
            score = float(raw_score)
        else:
            score = 0.0

        # Filter out content (already in top-level) and embedding (unnecessary payload)
        meta = {
            k: v for k, v in source.items()
            if k not in ("content", "embedding")
        }

        return cls(
            chunk_id=chunk_id,
            content=content,
            score=score,
            retrieval_method=method,
            metadata=meta,
        )
