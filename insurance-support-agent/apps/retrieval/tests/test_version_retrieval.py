"""
Unit tests for Version-Aware and Effective-Date-Aware Retrieval:
- Temporal point-in-time OpenSearch filter compilation (as_of_date)
- Explicit version term filtering (version)
- Factory methods (RetrievalFilters.for_date, RetrievalFilters.current)
- HybridRetriever integration passing temporal and version parameters
- Behavioral verification: why as_of_date correctly retrieves historical versions (v1)
  where status="active" would erroneously retrieve current (v3)
"""
from datetime import date
from typing import Any, Dict, List, Optional
import pytest

from apps.retrieval import (
    HybridRetriever,
    RetrievalFilters,
    RetrievalMethod,
    RetrievalResult,
)


# =====================================================================
# Mock Classes
# =====================================================================

class MockVectorStore:
    """Mock VectorStore capturing filter payloads for assertion."""

    def __init__(self, hits: Optional[List[Dict[str, Any]]] = None):
        self.hits = hits or []
        self.last_text_filters: Optional[Dict[str, Any]] = None
        self.last_knn_filters: Optional[Dict[str, Any]] = None

    @property
    def index_name(self) -> str:
        return "mock_index"

    def search_text(self, query_text: str, size: int = 5, filters=None):
        self.last_text_filters = filters
        return self.hits[:size]

    def search_knn(self, query_vector, k: int = 5, filters=None):
        self.last_knn_filters = filters
        return self.hits[:k]


class MockEmbeddingProvider:
    """Mock EmbeddingProvider."""

    @property
    def dimension(self) -> int:
        return 1024

    @property
    def model_name(self) -> str:
        return "mock-bge-m3"

    def embed(self, text: str):
        return [0.05] * 1024


def make_raw_hit(chunk_id: str, content: str, score: float, version: str = "v1") -> Dict[str, Any]:
    return {
        "_id": chunk_id,
        "_score": score,
        "_source": {
            "chunk_id": chunk_id,
            "document_id": "DOC-COM-0000077",
            "content": content,
            "section": "Coverage Terms",
            "policy_id": "COM-0000077",
            "version": version,
            "page_number": 1,
        },
    }


# =====================================================================
# 1. DSL Filter Generation Tests
# =====================================================================

def test_temporal_filter_dsl_march_2024():
    """
    Verify point-in-time as_of_date produces the correct OpenSearch boolean clauses:
    - effective_from <= as_of_date
    - effective_to >= as_of_date OR effective_to is null
    """
    filters = RetrievalFilters(
        policy_id="COM-0000077",
        as_of_date=date(2024, 3, 15),
    )
    os_filter = filters.to_opensearch_filter()
    assert os_filter is not None
    clauses = os_filter["bool"]["filter"]

    # Term policy_id
    term_clauses = [c for c in clauses if "term" in c]
    assert len(term_clauses) == 1
    assert term_clauses[0]["term"]["policy_id"] == "COM-0000077"

    # Range effective_from <= 2024-03-15
    range_clauses = [c for c in clauses if "range" in c]
    assert len(range_clauses) == 1
    assert range_clauses[0]["range"]["effective_from"]["lte"] == "2024-03-15"

    # Should clause: effective_to >= 2024-03-15 OR must_not exists effective_to
    should_bool_clauses = [c for c in clauses if "bool" in c and "should" in c["bool"]]
    assert len(should_bool_clauses) == 1
    should_list = should_bool_clauses[0]["bool"]["should"]
    assert should_bool_clauses[0]["bool"]["minimum_should_match"] == 1

    # Sub-clause 1: range effective_to >= 2024-03-15
    assert should_list[0]["range"]["effective_to"]["gte"] == "2024-03-15"

    # Sub-clause 2: must_not exists effective_to
    assert should_list[1]["bool"]["must_not"]["exists"]["field"] == "effective_to"


def test_explicit_version_filter_dsl():
    """Verify explicit version filter compiles into a keyword term filter."""
    filters = RetrievalFilters(
        policy_id="COM-0000077",
        version="v2",
    )
    os_filter = filters.to_opensearch_filter()
    assert os_filter is not None
    clauses = os_filter["bool"]["filter"]

    term_map = {list(c["term"].keys())[0]: list(c["term"].values())[0] for c in clauses if "term" in c}
    assert term_map["policy_id"] == "COM-0000077"
    assert term_map["version"] == "v2"


def test_retrieval_filters_convenience_factories():
    """Verify for_date and current factory methods."""
    f_hist = RetrievalFilters.for_date(date(2024, 3, 15), policy_id="COM-0000077")
    assert f_hist.as_of_date == date(2024, 3, 15)
    assert f_hist.policy_id == "COM-0000077"

    f_curr = RetrievalFilters.current(policy_id="COM-0000077")
    assert f_curr.as_of_date == date.today()
    assert f_curr.policy_id == "COM-0000077"


# =====================================================================
# 2. HybridRetriever Integration with Temporal / Version Filtering
# =====================================================================

def test_hybrid_retriever_passes_as_of_date_to_store():
    """Verify HybridRetriever passes as_of_date to both BM25 and Vector search branches."""
    store = MockVectorStore(
        hits=[make_raw_hit("c1", "Policy coverage v1", 1.5, version="v1")]
    )
    provider = MockEmbeddingProvider()
    retriever = HybridRetriever(vector_store=store, embedding_provider=provider)

    results = retriever.retrieve(
        query="What is covered?",
        top_k=1,
        filters=RetrievalFilters(policy_id="COM-0000077"),
        as_of_date=date(2024, 3, 15),
    )

    assert len(results) == 1
    # Check that both branches received the temporal filter
    for filters in (store.last_text_filters, store.last_knn_filters):
        assert filters is not None
        clauses = filters["bool"]["filter"]
        range_clause = [c for c in clauses if "range" in c][0]
        assert range_clause["range"]["effective_from"]["lte"] == "2024-03-15"


def test_hybrid_retriever_passes_version_override_to_store():
    """Verify HybridRetriever passes version override to both branches."""
    store = MockVectorStore(
        hits=[make_raw_hit("c2", "Policy coverage v2", 1.5, version="v2")]
    )
    provider = MockEmbeddingProvider()
    retriever = HybridRetriever(vector_store=store, embedding_provider=provider)

    results = retriever.retrieve(
        query="What is covered?",
        top_k=1,
        filters=RetrievalFilters(policy_id="COM-0000077"),
        version="v2",
    )

    assert len(results) == 1
    for filters in (store.last_text_filters, store.last_knn_filters):
        assert filters is not None
        clauses = filters["bool"]["filter"]
        term_map = {list(c["term"].keys())[0]: list(c["term"].values())[0] for c in clauses if "term" in c}
        assert term_map["version"] == "v2"


# =====================================================================
# 3. Behavioral Comparison: status="active" vs as_of_date
# =====================================================================

def test_status_active_vs_as_of_date_distinction():
    """
    Demonstrate that status='active' creates a static term filter on 'status',
    whereas as_of_date creates a dynamic temporal validity window.
    """
    # 1. Naive query with status="active"
    naive_filters = RetrievalFilters(policy_id="COM-0000077", status="active")
    naive_dsl = naive_filters.to_opensearch_filter()
    naive_terms = [c["term"] for c in naive_dsl["bool"]["filter"] if "term" in c]
    assert {"status": "active"} in naive_terms
    # Naive query has no range constraints on effective dates
    assert not any("range" in c for c in naive_dsl["bool"]["filter"])

    # 2. Historical query with as_of_date
    historical_filters = RetrievalFilters(policy_id="COM-0000077", as_of_date=date(2024, 3, 15))
    historical_dsl = historical_filters.to_opensearch_filter()
    hist_terms = [c["term"] for c in historical_dsl["bool"]["filter"] if "term" in c]
    # Historical query does NOT restrict to status="active" (since v1 is expired/superseded)
    assert {"status": "active"} not in hist_terms
    # Historical query DOES enforce temporal bounds
    assert any("range" in c for c in historical_dsl["bool"]["filter"])
