"""
Unit and integration tests for the Retrieval Layer:
- RetrievalResult and RetrievalMethod models
- OpenSearch hit parsing and embedding stripping
- Reciprocal Rank Fusion (RRF) algorithm and mathematical scoring
- RetrievalService (BM25, Vector, Hybrid)
"""
from typing import Any, Dict, List, Optional
import pytest

from apps.retrieval import (
    HybridRetriever,
    RetrievalMethod,
    RetrievalResult,
    RetrievalService,
    reciprocal_rank_fusion,
)


# =====================================================================
# Fixtures and Mock Classes
# =====================================================================

class MockVectorStore:
    """Mock VectorStore implementing search_text and search_knn."""
    def __init__(self, text_hits: Optional[List[Dict[str, Any]]] = None, knn_hits: Optional[List[Dict[str, Any]]] = None):
        self.text_hits = text_hits or []
        self.knn_hits = knn_hits or []
        self.last_text_query = None
        self.last_knn_vector = None

    @property
    def index_name(self) -> str:
        return "mock_index"

    def search_text(self, query_text: str, size: int = 5) -> List[Dict[str, Any]]:
        self.last_text_query = query_text
        return self.text_hits[:size]

    def search_knn(self, query_vector: List[float], k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        self.last_knn_vector = query_vector
        return self.knn_hits[:k]


class MockEmbeddingProvider:
    """Mock EmbeddingProvider returning static vector."""
    def __init__(self, vector: Optional[List[float]] = None):
        self.vector = vector or [0.1] * 1024
        self.last_embed_text = None

    @property
    def dimension(self) -> int:
        return len(self.vector)

    @property
    def model_name(self) -> str:
        return "mock-bge-m3"

    def embed(self, text: str) -> List[float]:
        self.last_embed_text = text
        return self.vector


def make_raw_hit(chunk_id: str, content: str, score: float, section: str = "Details") -> Dict[str, Any]:
    return {
        "_id": chunk_id,
        "_score": score,
        "_source": {
            "chunk_id": chunk_id,
            "document_id": "DOC-C-1000",
            "content": content,
            "section": section,
            "claim_id": "C-1000",
            "page_number": 1,
            "embedding": [0.05] * 1024,  # Raw embedding that must be stripped
        },
    }


# =====================================================================
# 1. Model Tests
# =====================================================================

def test_retrieval_result_creation():
    """Verify RetrievalResult properties and validation."""
    result = RetrievalResult(
        chunk_id="chunk-1",
        content="Coverage for collision damage.",
        score=0.88,
        retrieval_method=RetrievalMethod.VECTOR,
        metadata={"section": "Collision", "claim_id": "C-1000"},
    )
    assert result.chunk_id == "chunk-1"
    assert result.content == "Coverage for collision damage."
    assert result.score == 0.88
    assert result.retrieval_method == RetrievalMethod.VECTOR
    assert result.metadata["section"] == "Collision"


def test_from_opensearch_hit_strips_embedding_and_maps_fields():
    """Verify from_opensearch_hit extracts fields and strips raw 1024-dim embedding."""
    raw_hit = make_raw_hit("DOC#c3", "Line Items repair estimate", 1.25, section="Line Items")
    candidate = RetrievalResult.from_opensearch_hit(raw_hit, method=RetrievalMethod.BM25)

    assert candidate.chunk_id == "DOC#c3"
    assert candidate.content == "Line Items repair estimate"
    assert candidate.score == 1.25
    assert candidate.retrieval_method == RetrievalMethod.BM25
    assert candidate.metadata["section"] == "Line Items"
    assert candidate.metadata["claim_id"] == "C-1000"
    assert "embedding" not in candidate.metadata


def test_from_opensearch_hit_with_custom_score():
    """Verify custom score override for fusion or normalization."""
    raw_hit = make_raw_hit("DOC#c1", "Claim Details", 0.5)
    candidate = RetrievalResult.from_opensearch_hit(raw_hit, method=RetrievalMethod.VECTOR, custom_score=0.99)
    assert candidate.score == 0.99


# =====================================================================
# 2. Reciprocal Rank Fusion (RRF) Tests
# =====================================================================

def test_reciprocal_rank_fusion_empty():
    """Verify empty rankings return empty list."""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_reciprocal_rank_fusion_single_list():
    """Verify single list RRF assigns 1 / (rrf_k + rank)."""
    item1 = RetrievalResult(chunk_id="c1", content="Doc 1", score=10.0, retrieval_method=RetrievalMethod.BM25)
    item2 = RetrievalResult(chunk_id="c2", content="Doc 2", score=5.0, retrieval_method=RetrievalMethod.BM25)

    fused = reciprocal_rank_fusion([[item1, item2]], rrf_k=60)
    assert len(fused) == 2
    assert fused[0].chunk_id == "c1"
    assert fused[0].retrieval_method == RetrievalMethod.HYBRID_RRF
    assert fused[0].score == round(1.0 / 61, 6)
    assert fused[1].chunk_id == "c2"
    assert fused[1].score == round(1.0 / 62, 6)


def test_reciprocal_rank_fusion_two_lists_boosts_overlap():
    """
    Verify RRF mathematical combination:
    List 1 (BM25):   [c1, c2]
    List 2 (Vector): [c2, c3]

    Scores with rrf_k=60:
    c2: rank 2 in L1 (1/62) + rank 1 in L2 (1/61) = 0.016129 + 0.016393 = 0.032522
    c1: rank 1 in L1 (1/61)                       = 0.016393
    c3: rank 2 in L2 (1/62)                       = 0.016129

    c2 must win #1 because it appeared in both lists.
    """
    c1 = RetrievalResult(chunk_id="c1", content="Text 1", score=2.0, retrieval_method=RetrievalMethod.BM25)
    c2_bm25 = RetrievalResult(chunk_id="c2", content="Text 2", score=1.5, retrieval_method=RetrievalMethod.BM25)
    c2_vec = RetrievalResult(chunk_id="c2", content="Text 2", score=0.9, retrieval_method=RetrievalMethod.VECTOR)
    c3 = RetrievalResult(chunk_id="c3", content="Text 3", score=0.8, retrieval_method=RetrievalMethod.VECTOR)

    fused = reciprocal_rank_fusion([[c1, c2_bm25], [c2_vec, c3]], rrf_k=60)
    assert len(fused) == 3

    # Rank 1: c2
    assert fused[0].chunk_id == "c2"
    expected_c2_score = round(1.0 / 62 + 1.0 / 61, 6)
    assert fused[0].score == expected_c2_score
    assert fused[0].metadata["rrf_ranks"] == {"bm25": 2, "vector": 1}

    # Rank 2: c1
    assert fused[1].chunk_id == "c1"
    assert fused[1].score == round(1.0 / 61, 6)

    # Rank 3: c3
    assert fused[2].chunk_id == "c3"
    assert fused[2].score == round(1.0 / 62, 6)


def test_reciprocal_rank_fusion_top_k_cutoff():
    """Verify top_k limits returned items."""
    c1 = RetrievalResult(chunk_id="c1", content="Text 1", score=1.0, retrieval_method=RetrievalMethod.BM25)
    c2 = RetrievalResult(chunk_id="c2", content="Text 2", score=0.5, retrieval_method=RetrievalMethod.BM25)
    fused = reciprocal_rank_fusion([[c1, c2]], top_k=1)
    assert len(fused) == 1
    assert fused[0].chunk_id == "c1"


# =====================================================================
# 3. RetrievalService Tests
# =====================================================================

def test_retrieve_bm25():
    """Verify retrieve_bm25 calls search_text and maps to RetrievalResult."""
    hit = make_raw_hit("c1", "Water damage repair", 2.45)
    store = MockVectorStore(text_hits=[hit])
    provider = MockEmbeddingProvider()

    service = RetrievalService(vector_store=store, embedding_provider=provider)
    results = service.retrieve_bm25("water damage", top_k=3)

    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert results[0].content == "Water damage repair"
    assert results[0].score == 2.45
    assert results[0].retrieval_method == RetrievalMethod.BM25
    assert store.last_text_query == "water damage"


def test_retrieve_vector():
    """Verify retrieve_vector embeds query and calls search_knn."""
    hit = make_raw_hit("c2", "Dehumidifier rental fees", 0.89)
    store = MockVectorStore(knn_hits=[hit])
    provider = MockEmbeddingProvider()

    service = RetrievalService(vector_store=store, embedding_provider=provider)
    results = service.retrieve_vector("equipment fees", top_k=2)

    assert len(results) == 1
    assert results[0].chunk_id == "c2"
    assert results[0].score == 0.89
    assert results[0].retrieval_method == RetrievalMethod.VECTOR
    assert provider.last_embed_text == "equipment fees"
    assert store.last_knn_vector == provider.vector


def test_retrieve_hybrid():
    """Verify retrieve_hybrid executes both pipelines and fuses with RRF."""
    hit1 = make_raw_hit("c1", "BM25 only match", 3.0)
    hit2 = make_raw_hit("c2", "Both match", 2.0)
    hit3 = make_raw_hit("c2", "Both match", 0.95)
    hit4 = make_raw_hit("c3", "Vector only match", 0.80)

    store = MockVectorStore(text_hits=[hit1, hit2], knn_hits=[hit3, hit4])
    provider = MockEmbeddingProvider()

    service = RetrievalService(vector_store=store, embedding_provider=provider)
    fused = service.retrieve_hybrid("insurance claim query", top_k=2)

    assert len(fused) == 2
    # c2 appeared in both BM25 and Vector -> should win rank 1
    assert fused[0].chunk_id == "c2"
    assert fused[0].retrieval_method == RetrievalMethod.HYBRID_RRF


def test_empty_query_handling():
    """Verify empty or whitespace query returns empty list without calling store."""
    store = MockVectorStore(text_hits=[make_raw_hit("c1", "Text", 1.0)])
    provider = MockEmbeddingProvider()
    service = RetrievalService(vector_store=store, embedding_provider=provider)

    assert service.retrieve_bm25("") == []
    assert service.retrieve_bm25("   ") == []
    assert service.retrieve_vector("") == []
    assert service.retrieve_hybrid("") == []
    assert store.last_text_query is None
    assert store.last_knn_vector is None


def test_hybrid_retriever_executes_in_parallel():
    """Verify BM25 and Vector search run concurrently in separate worker threads."""
    import threading
    import time

    thread_ids = set()

    class ParallelCheckVectorStore(MockVectorStore):
        def search_text(self, query_text: str, size: int = 5):
            thread_ids.add(threading.get_ident())
            time.sleep(0.02)
            return [make_raw_hit("c1", "BM25 hit", 2.0)]

        def search_knn(self, query_vector, k: int = 5, filters=None):
            thread_ids.add(threading.get_ident())
            time.sleep(0.02)
            return [make_raw_hit("c2", "Vector hit", 0.9)]

    store = ParallelCheckVectorStore()
    provider = MockEmbeddingProvider()
    retriever = HybridRetriever(vector_store=store, embedding_provider=provider)

    results = retriever.retrieve("parallel test query", top_k=2)

    assert len(results) == 2
    # Verify both branches executed in distinct worker threads
    assert len(thread_ids) == 2
    # Verify main thread was not one of the worker threads
    assert threading.get_ident() not in thread_ids

