"""
Hybrid retrieval orchestration combining Lexical (BM25) and Semantic (Vector) search.

Executes BM25 search and dense vector retrieval in parallel using a ThreadPoolExecutor
and fuses the candidate lists via Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Any, Dict, List, Optional

from apps.ingestion.embeddings.base import EmbeddingProvider
from apps.ingestion.indexing.base import VectorStore
from apps.retrieval.fusion import reciprocal_rank_fusion
from apps.retrieval.models import RetrievalMethod, RetrievalResult

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever orchestrating parallel BM25 and dense vector search
    with Reciprocal Rank Fusion (RRF).

    Flow:
        query
         │
         ├───────────────────────┐
         ▼                       ▼
      [Thread 1]              [Thread 2]
        BM25                 Embed Query
         │                       │
         │                       ▼
         │                 Vector search (k-NN)
         │                       │
         └───────────┬───────────┘
                     ▼
                    RRF
                     ▼
             RetrievalResult[]
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        max_workers: int = 2,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.max_workers = max_workers

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        vector_k: int = 10,
        bm25_k: int = 10,
        vector_filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Primary entrypoint: executes parallel hybrid retrieval.
        """
        return self.retrieve_hybrid(
            query=query,
            top_k=top_k,
            rrf_k=rrf_k,
            vector_k=vector_k,
            bm25_k=bm25_k,
            vector_filters=vector_filters,
        )

    def retrieve_bm25(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Execute BM25 keyword search and return standardized candidates.
        """
        if not query or not query.strip():
            return []

        hits: List[Dict[str, Any]] = self.vector_store.search_text(
            query_text=query.strip(),
            size=top_k,
        )

        return [
            RetrievalResult.from_opensearch_hit(hit, method=RetrievalMethod.BM25)
            for hit in hits
        ]

    def retrieve_vector(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Execute dense vector search by embedding the query and querying k-NN.
        """
        if not query or not query.strip():
            return []

        query_vector = self.embedding_provider.embed(query.strip())
        hits: List[Dict[str, Any]] = self.vector_store.search_knn(
            query_vector=query_vector,
            k=top_k,
            filters=filters,
        )

        return [
            RetrievalResult.from_opensearch_hit(hit, method=RetrievalMethod.VECTOR)
            for hit in hits
        ]

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        vector_k: int = 10,
        bm25_k: int = 10,
        vector_filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Execute Parallel Hybrid Search using Reciprocal Rank Fusion (RRF).

        Runs BM25 and (Embed Query -> Vector Search) concurrently in separate worker threads,
        then fuses both candidate lists using RRF.
        """
        if not query or not query.strip():
            return []

        cleaned_query = query.strip()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_bm25 = executor.submit(self.retrieve_bm25, cleaned_query, bm25_k)
            future_vector = executor.submit(
                self.retrieve_vector, cleaned_query, vector_k, vector_filters
            )
            bm25_results = future_bm25.result()
            vector_results = future_vector.result()

        return reciprocal_rank_fusion(
            ranked_lists=[bm25_results, vector_results],
            rrf_k=rrf_k,
            top_k=top_k,
        )


# Alias for backward compatibility
RetrievalService = HybridRetriever
