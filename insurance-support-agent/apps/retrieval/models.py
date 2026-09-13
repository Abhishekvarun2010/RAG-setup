"""
Domain models for the retrieval layer.

Decouples the rest of the application and the Agent Runtime from infrastructure-specific
search payloads (e.g. OpenSearch hit dictionaries).
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class RetrievalMethod(str, Enum):
    """Search / retrieval strategy utilized to surface candidate chunks."""
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID_RRF = "hybrid_rrf"


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
