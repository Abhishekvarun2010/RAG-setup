"""
Cross-Encoder Reranking layer for rescoring retrieved candidates.

Uses BAAI/bge-reranker-v2-m3 (via sentence-transformers) to calculate deep
query-document cross-attention relevance scores, improving precision of
fused hybrid search candidates.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Protocol, Sequence, runtime_checkable

from apps.retrieval.models import RetrievalMethod, RetrievalResult

logger = logging.getLogger(__name__)


# =====================================================================
# Protocol Definition
# =====================================================================

@runtime_checkable
class Reranker(Protocol):
    """
    Protocol defining the interface for rerankers.
    """

    @property
    def model_name(self) -> str:
        """The identifier or tag of the underlying cross-encoder model."""
        ...

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Rescore candidates based on joint attention with the query.
        """
        ...


# =====================================================================
# Concrete Cross-Encoder Implementation
# =====================================================================

class CrossEncoderReranker:
    """
    Reranker powered by sentence-transformers CrossEncoder.

    Features:
    - Lazy loading: Weights are loaded upon first call to `.rerank()`.
    - Dependency injection: Supports custom model instances for unit testing.
    - Provenance tracking: Preserves previous RRF/BM25/vector scores and methods in metadata.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        batch_size: int = 16,
        model: Optional[Any] = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> Optional[str]:
        return self._device

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _ensure_model_loaded(self) -> Any:
        """Loads the sentence_transformers CrossEncoder model if not already loaded."""
        if self._model is None:
            logger.info(f"Loading CrossEncoder model: {self._model_name}...")
            from sentence_transformers import CrossEncoder

            kwargs: dict[str, Any] = {}
            if self._device is not None:
                kwargs["device"] = self._device

            self._model = CrossEncoder(self._model_name, **kwargs)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Rescore candidates using the cross-encoder model against the query.

        Args:
            query: The user query string.
            candidates: Sequence of RetrievalResult instances to rescore.
            top_k: Optional number of top results to return after reranking.

        Returns:
            List of RetrievalResult objects rescored and sorted descending.
        """
        if not candidates:
            return []

        if not query or not query.strip():
            results = list(candidates)
            return results[:top_k] if top_k is not None else results

        model = self._ensure_model_loaded()
        cleaned_query = query.strip()

        # Build sentence pairs for CrossEncoder: (query, passage)
        pairs = [(cleaned_query, c.content) for c in candidates]

        raw_scores = model.predict(pairs, batch_size=self._batch_size)

        rescored_results: List[RetrievalResult] = []
        for orig_candidate, raw_score in zip(candidates, raw_scores):
            # Normalize float score
            score_val = float(raw_score)

            merged_meta = dict(orig_candidate.metadata)
            merged_meta["original_score"] = orig_candidate.score
            merged_meta["original_retrieval_method"] = orig_candidate.retrieval_method.value
            merged_meta["reranker_model"] = self._model_name

            rescored_results.append(
                RetrievalResult(
                    chunk_id=orig_candidate.chunk_id,
                    content=orig_candidate.content,
                    score=round(score_val, 6),
                    retrieval_method=RetrievalMethod.RERANKED,
                    metadata=merged_meta,
                )
            )

        # Sort descending by cross-encoder score
        rescored_results.sort(key=lambda r: r.score, reverse=True)

        if top_k is not None:
            rescored_results = rescored_results[:top_k]

        return rescored_results
