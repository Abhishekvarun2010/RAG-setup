"""
Unit and integration tests for the Cross-Encoder Reranking layer:
- Protocol compliance (Reranker)
- CrossEncoderReranker rescoring, sorting, and provenance preservation
- Lazy model loading behavior
- Edge cases (empty query, empty candidates, top_k slicing)
- HybridRetriever integration with reranking enabled / disabled
"""
from typing import Any, Dict, List, Optional
import pytest

from apps.retrieval import (
    CrossEncoderReranker,
    HybridRetriever,
    Reranker,
    RetrievalMethod,
    RetrievalResult,
)


# =====================================================================
# Mock Classes
# =====================================================================

class MockCrossEncoderModel:
    """Mock CrossEncoder for fast, deterministic unit testing without neural weights."""

    def __init__(self, scores: Optional[List[float]] = None):
        self.scores = scores or []
        self.last_pairs: Optional[List[tuple]] = None
        self.last_batch_size: Optional[int] = None

    def predict(self, pairs: List[tuple], batch_size: int = 16) -> List[float]:
        self.last_pairs = pairs
        self.last_batch_size = batch_size
        if self.scores:
            return self.scores[:len(pairs)]
        # Default mock scoring based on length or index
        return [float(i + 1) * 0.1 for i in range(len(pairs))]


class MockReranker:
    """Mock Reranker conforming to Reranker protocol."""

    def __init__(self, model_name: str = "mock-reranker"):
        self._model_name = model_name
        self.last_query: Optional[str] = None
        self.last_candidates: Optional[List[RetrievalResult]] = None
        self.last_top_k: Optional[int] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        self.last_query = query
        self.last_candidates = list(candidates)
        self.last_top_k = top_k

        # Reverse order of candidates as a dummy reranking effect
        results = []
        for idx, c in enumerate(reversed(candidates)):
            meta = dict(c.metadata)
            meta["original_score"] = c.score
            meta["original_retrieval_method"] = c.retrieval_method.value
            results.append(
                RetrievalResult(
                    chunk_id=c.chunk_id,
                    content=c.content,
                    score=1.0 - (idx * 0.1),
                    retrieval_method=RetrievalMethod.RERANKED,
                    metadata=meta,
                )
            )
        return results[:top_k] if top_k is not None else results


class MockVectorStore:
    """Mock VectorStore for retriever testing."""

    def __init__(self, text_hits=None, knn_hits=None):
        self.text_hits = text_hits or []
        self.knn_hits = knn_hits or []

    @property
    def index_name(self) -> str:
        return "mock_index"

    def search_text(self, query_text: str, size: int = 5, filters=None):
        return self.text_hits[:size]

    def search_knn(self, query_vector, k: int = 5, filters=None):
        return self.knn_hits[:k]


class MockEmbeddingProvider:
    """Mock EmbeddingProvider."""

    @property
    def dimension(self) -> int:
        return 1024

    @property
    def model_name(self) -> str:
        return "mock-bge-m3"

    def embed(self, text: str):
        return [0.1] * 1024


def make_candidate(chunk_id: str, content: str, score: float, method: RetrievalMethod = RetrievalMethod.HYBRID_RRF) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        content=content,
        score=score,
        retrieval_method=method,
        metadata={"section": "Summary", "claim_id": "C-1000"},
    )


def make_raw_hit(chunk_id: str, content: str, score: float) -> Dict[str, Any]:
    return {
        "_id": chunk_id,
        "_score": score,
        "_source": {
            "chunk_id": chunk_id,
            "document_id": "DOC-C-1000",
            "content": content,
            "section": "General",
            "claim_id": "C-1000",
            "page_number": 1,
        },
    }


# =====================================================================
# 1. Protocol Verification
# =====================================================================

def test_reranker_protocol_compliance():
    """Verify CrossEncoderReranker implements the Reranker runtime-checkable protocol."""
    reranker = CrossEncoderReranker()
    assert isinstance(reranker, Reranker)
    assert reranker.model_name == "BAAI/bge-reranker-v2-m3"


# =====================================================================
# 2. CrossEncoderReranker Unit Tests
# =====================================================================

def test_cross_encoder_rerank_rescores_and_sorts():
    """Verify candidates are rescored and sorted in descending order according to model predictions."""
    mock_model = MockCrossEncoderModel(scores=[0.15, 0.95, 0.60])
    reranker = CrossEncoderReranker(model=mock_model)

    candidates = [
        make_candidate("c1", "First chunk text", score=0.033),
        make_candidate("c2", "Second chunk text with key answer", score=0.032),
        make_candidate("c3", "Third chunk text partial match", score=0.031),
    ]

    results = reranker.rerank(query="What is the net amount?", candidates=candidates)

    # Expected order: c2 (0.95) -> c3 (0.60) -> c1 (0.15)
    assert len(results) == 3
    assert results[0].chunk_id == "c2"
    assert results[0].score == 0.95
    assert results[0].retrieval_method == RetrievalMethod.RERANKED

    assert results[1].chunk_id == "c3"
    assert results[1].score == 0.60

    assert results[2].chunk_id == "c1"
    assert results[2].score == 0.15

    # Check pair format passed to model
    assert mock_model.last_pairs == [
        ("What is the net amount?", "First chunk text"),
        ("What is the net amount?", "Second chunk text with key answer"),
        ("What is the net amount?", "Third chunk text partial match"),
    ]


def test_cross_encoder_rerank_metadata_provenance():
    """Verify previous score and retrieval_method are saved in metadata under original_* keys."""
    mock_model = MockCrossEncoderModel(scores=[0.85])
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3", model=mock_model)

    candidate = make_candidate("c1", "Sample text", score=0.033, method=RetrievalMethod.HYBRID_RRF)
    candidate.metadata["existing_field"] = "keep_me"

    results = reranker.rerank(query="test query", candidates=[candidate])

    assert len(results) == 1
    meta = results[0].metadata
    assert meta["existing_field"] == "keep_me"
    assert meta["original_score"] == 0.033
    assert meta["original_retrieval_method"] == "hybrid_rrf"
    assert meta["reranker_model"] == "BAAI/bge-reranker-v2-m3"


def test_cross_encoder_rerank_top_k_truncation():
    """Verify top_k limits the number of returned results after rescoring."""
    mock_model = MockCrossEncoderModel(scores=[0.1, 0.9, 0.5])
    reranker = CrossEncoderReranker(model=mock_model)

    candidates = [
        make_candidate("c1", "text 1", 0.01),
        make_candidate("c2", "text 2", 0.02),
        make_candidate("c3", "text 3", 0.03),
    ]

    results = reranker.rerank(query="query", candidates=candidates, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "c2"
    assert results[1].chunk_id == "c3"


def test_cross_encoder_rerank_empty_candidates():
    """Verify empty candidate list returns empty without invoking model."""
    mock_model = MockCrossEncoderModel()
    reranker = CrossEncoderReranker(model=mock_model)

    results = reranker.rerank(query="query", candidates=[])
    assert results == []
    assert mock_model.last_pairs is None


def test_cross_encoder_rerank_empty_query():
    """Verify empty query returns original candidates unmodified up to top_k."""
    mock_model = MockCrossEncoderModel()
    reranker = CrossEncoderReranker(model=mock_model)

    candidates = [
        make_candidate("c1", "text 1", 0.9),
        make_candidate("c2", "text 2", 0.8),
    ]

    results = reranker.rerank(query="   ", candidates=candidates, top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert mock_model.last_pairs is None


def test_cross_encoder_lazy_loading(monkeypatch):
    """Verify CrossEncoder is only instantiated upon calling rerank(), not on __init__."""
    loaded = []

    class DummyCrossEncoder:
        def __init__(self, model_name, **kwargs):
            loaded.append(model_name)

        def predict(self, pairs, **kwargs):
            return [0.7] * len(pairs)

    import sentence_transformers
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", DummyCrossEncoder)

    reranker = CrossEncoderReranker(model_name="test-model")
    assert len(loaded) == 0  # Not loaded yet

    candidates = [make_candidate("c1", "text", 0.5)]
    results = reranker.rerank(query="test", candidates=candidates)

    assert len(loaded) == 1
    assert loaded[0] == "test-model"
    assert len(results) == 1


# =====================================================================
# 3. HybridRetriever + Reranker Integration Tests
# =====================================================================

def test_hybrid_retriever_with_reranker_enabled():
    """Verify HybridRetriever fuses candidates with fusion_k, then delegates to reranker."""
    text_hits = [
        make_raw_hit("c1", "Text hit 1", 2.0),
        make_raw_hit("c2", "Text hit 2", 1.5),
    ]
    knn_hits = [
        make_raw_hit("c2", "KNN hit 2", 0.95),
        make_raw_hit("c3", "KNN hit 3", 0.85),
    ]

    store = MockVectorStore(text_hits=text_hits, knn_hits=knn_hits)
    provider = MockEmbeddingProvider()
    mock_reranker = MockReranker()

    retriever = HybridRetriever(
        vector_store=store,
        embedding_provider=provider,
        reranker=mock_reranker,
    )

    results = retriever.retrieve(
        query="what is deductible?",
        top_k=2,
        fusion_k=5,
        rerank=True,
    )

    # Reranker was called with fusion_k candidates and requested top_k
    assert mock_reranker.last_query == "what is deductible?"
    assert mock_reranker.last_top_k == 2
    assert len(mock_reranker.last_candidates) == 3  # c1, c2, c3 fused

    assert len(results) == 2
    assert all(r.retrieval_method == RetrievalMethod.RERANKED for r in results)


def test_hybrid_retriever_rerank_false_skips_reranker():
    """Verify rerank=False bypasses the reranker even if one is configured."""
    text_hits = [make_raw_hit("c1", "Text hit 1", 2.0)]
    knn_hits = [make_raw_hit("c1", "KNN hit 1", 0.9)]

    store = MockVectorStore(text_hits=text_hits, knn_hits=knn_hits)
    provider = MockEmbeddingProvider()
    mock_reranker = MockReranker()

    retriever = HybridRetriever(
        vector_store=store,
        embedding_provider=provider,
        reranker=mock_reranker,
    )

    results = retriever.retrieve(
        query="what is deductible?",
        top_k=1,
        rerank=False,
    )

    # Reranker should NOT have been invoked
    assert mock_reranker.last_query is None
    assert len(results) == 1
    assert results[0].retrieval_method == RetrievalMethod.HYBRID_RRF


def test_hybrid_retriever_without_reranker():
    """Verify HybridRetriever defaults cleanly to RRF when reranker is None."""
    text_hits = [make_raw_hit("c1", "Text hit 1", 2.0)]
    knn_hits = [make_raw_hit("c2", "KNN hit 2", 0.9)]

    store = MockVectorStore(text_hits=text_hits, knn_hits=knn_hits)
    provider = MockEmbeddingProvider()

    retriever = HybridRetriever(
        vector_store=store,
        embedding_provider=provider,
        reranker=None,
    )

    results = retriever.retrieve(query="what is deductible?", top_k=2)

    assert len(results) == 2
    assert all(r.retrieval_method == RetrievalMethod.HYBRID_RRF for r in results)
