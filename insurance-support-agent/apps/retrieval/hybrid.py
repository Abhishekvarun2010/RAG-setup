"""
Hybrid retrieval orchestration combining Lexical (BM25) and Semantic (Vector) search.

Executes BM25 search and dense vector retrieval in parallel using a ThreadPoolExecutor
and fuses the candidate lists via Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Any, Dict, List, Optional, Union

from apps.ingestion.embeddings.base import EmbeddingProvider
from apps.ingestion.indexing.base import VectorStore
from apps.retrieval.fusion import reciprocal_rank_fusion
from apps.retrieval.models import RetrievalFilters, RetrievalMethod, RetrievalResult
from apps.retrieval.rerank import Reranker

logger = logging.getLogger(__name__)


def _normalize_filter(
    filters: Optional[Union[RetrievalFilters, Dict[str, Any]]]
) -> Optional[Dict[str, Any]]:
    """Normalize RetrievalFilters model or dict into OpenSearch bool filter dict."""
    if filters is None:
        return None
    if isinstance(filters, RetrievalFilters):
        return filters.to_opensearch_filter()
    return filters


class HybridRetriever:
    """
    Hybrid retriever orchestrating parallel BM25 and dense vector search
    with Reciprocal Rank Fusion (RRF) and optional Cross-Encoder reranking.

    Flow:
        query + filters
         │
         ├───────────────────────┐
         ▼                       ▼
      [Thread 1]              [Thread 2]
     BM25 + Filter        Embed Query + Filter
         │                       │
         │                       ▼
         │                 Vector search (k-NN)
         │                       │
         └───────────┬───────────┘
                     ▼
             RRF Candidate Pool (top fusion_k)
                     ▼
           Cross-Encoder Reranker
                     ▼
             RetrievalResult[]
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        reranker: Optional[Reranker] = None,
        max_workers: int = 2,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.max_workers = max_workers

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Union[RetrievalFilters, Dict[str, Any]]] = None,
        rrf_k: int = 60,
        vector_k: int = 10,
        bm25_k: int = 10,
        fusion_k: int = 10,
        rerank: bool = True,
    ) -> List[RetrievalResult]:
        """
        Primary entrypoint: executes parallel hybrid retrieval with metadata filters,
        Reciprocal Rank Fusion, and optional Cross-Encoder reranking.
        """
        return self.retrieve_hybrid(
            query=query,
            top_k=top_k,
            filters=filters,
            rrf_k=rrf_k,
            vector_k=vector_k,
            bm25_k=bm25_k,
            fusion_k=fusion_k,
            rerank=rerank,
        )

    def retrieve_bm25(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Union[RetrievalFilters, Dict[str, Any]]] = None,
    ) -> List[RetrievalResult]:
        """
        Execute BM25 keyword search with OpenSearch metadata filters.
        """
        if not query or not query.strip():
            return []

        opensearch_filter = _normalize_filter(filters)
        hits: List[Dict[str, Any]] = self.vector_store.search_text(
            query_text=query.strip(),
            size=top_k,
            filters=opensearch_filter,
        )

        return [
            RetrievalResult.from_opensearch_hit(hit, method=RetrievalMethod.BM25)
            for hit in hits
        ]

    def retrieve_vector(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Union[RetrievalFilters, Dict[str, Any]]] = None,
    ) -> List[RetrievalResult]:
        """
        Execute dense vector search with OpenSearch metadata filters.
        """
        if not query or not query.strip():
            return []

        opensearch_filter = _normalize_filter(filters)
        query_vector = self.embedding_provider.embed(query.strip())
        hits: List[Dict[str, Any]] = self.vector_store.search_knn(
            query_vector=query_vector,
            k=top_k,
            filters=opensearch_filter,
        )

        return [
            RetrievalResult.from_opensearch_hit(hit, method=RetrievalMethod.VECTOR)
            for hit in hits
        ]

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Union[RetrievalFilters, Dict[str, Any]]] = None,
        rrf_k: int = 60,
        vector_k: int = 10,
        bm25_k: int = 10,
        fusion_k: int = 10,
        rerank: bool = True,
    ) -> List[RetrievalResult]:
        """
        Execute Parallel Hybrid Search using Reciprocal Rank Fusion (RRF) and Cross-Encoder Reranker.

        Runs BM25 and (Embed Query -> Vector Search) concurrently in separate worker threads,
        applying OpenSearch metadata filters to both branches, fuses candidates via RRF,
        and optionally reranks top fusion candidates with a Cross-Encoder model.
        """
        if not query or not query.strip():
            return []

        cleaned_query = query.strip()
        normalized_filters = _normalize_filter(filters)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_bm25 = executor.submit(
                self.retrieve_bm25, cleaned_query, bm25_k, normalized_filters
            )
            future_vector = executor.submit(
                self.retrieve_vector, cleaned_query, vector_k, normalized_filters
            )
            bm25_results = future_bm25.result()
            vector_results = future_vector.result()

        # If reranking is enabled and a reranker is provided, pool top fusion_k candidates
        fuse_top_k = fusion_k if (self.reranker is not None and rerank) else top_k

        fused_candidates = reciprocal_rank_fusion(
            ranked_lists=[bm25_results, vector_results],
            rrf_k=rrf_k,
            top_k=fuse_top_k,
        )

        if self.reranker is not None and rerank and fused_candidates:
            return self.reranker.rerank(
                query=cleaned_query,
                candidates=fused_candidates,
                top_k=top_k,
            )

        return fused_candidates[:top_k]


# Alias for backward compatibility
RetrievalService = HybridRetriever

