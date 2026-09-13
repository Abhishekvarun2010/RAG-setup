"""
Unit and integration tests for the OpenSearch indexing and vector retrieval layer.

Tests:
1. VectorStore protocol conformance.
2. prepare_document attaches vector and serializes fields.
3. index_chunk with mock HTTP transport.
4. index_batch bulk formatting, refresh parameter, and error handling.
5. get_by_id successful hit and 404 not found.
6. search_knn query structure and hit parsing.
7. Connection errors and query errors.
"""
import json
from typing import List
import httpx
import pytest

from apps.ingestion import (
    BaseVectorStore,
    Chunk,
    DocumentType,
    OpenSearchConnectionError,
    OpenSearchError,
    OpenSearchIndexer,
    OpenSearchIndexingError,
    OpenSearchQueryError,
    SourceType,
    VectorStore,
)


def make_mock_client(handler) -> httpx.Client:
    """Create an httpx.Client with custom mock transport."""
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport, base_url="http://mock-opensearch:9200")


def make_sample_chunk(chunk_id: str = "DOC-C-1000-ESTIMATE#c1") -> Chunk:
    return Chunk(
        content="Damage estimate for claim C-1000. Total repairs €22,950.00.",
        chunk_id=chunk_id,
        document_id="DOC-C-1000-ESTIMATE",
        document_type=DocumentType.ESTIMATE,
        source_type=SourceType.PDF,
        claim_id="C-1000",
        section="Estimate — Claim C-1000",
        page_number=1,
    )


# =====================================================================
# 1. Protocol Conformance Tests
# =====================================================================

def test_indexer_protocol_conformance():
    """Verify OpenSearchIndexer satisfies VectorStore protocol."""
    indexer = OpenSearchIndexer()
    assert isinstance(indexer, VectorStore)
    assert isinstance(indexer, BaseVectorStore)
    assert indexer.index_name == "insurance_documents"


def test_prepare_document_serializes_and_attaches_vector():
    """Verify prepare_document produces valid schema dict with embedding."""
    indexer = OpenSearchIndexer()
    chunk = make_sample_chunk()
    vector = [0.1] * 1024

    doc = indexer.prepare_document(chunk, vector)
    assert doc["chunk_id"] == "DOC-C-1000-ESTIMATE#c1"
    assert doc["document_id"] == "DOC-C-1000-ESTIMATE"
    assert doc["document_type"] == "estimate"
    assert doc["claim_id"] == "C-1000"
    assert doc["embedding"] == vector
    assert "ingested_at" in doc


def test_prepare_document_validates_vector():
    """Verify prepare_document rejects invalid vector inputs."""
    indexer = OpenSearchIndexer()
    chunk = make_sample_chunk()

    with pytest.raises(ValueError):
        indexer.prepare_document(chunk, [])

    with pytest.raises(ValueError):
        indexer.prepare_document(chunk, None)  # type: ignore


# =====================================================================
# 2. Mock Transport Tests (Indexing & Retrieval)
# =====================================================================

def test_index_chunk_success():
    """Verify single chunk indexing with URL-encoding and refresh flag."""
    captured_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(201, json={"result": "created", "_id": "DOC-C-1000-ESTIMATE#c1"})

    client = make_mock_client(handler)
    indexer = OpenSearchIndexer(client=client)

    chunk = make_sample_chunk()
    res = indexer.index_chunk(chunk, [0.05] * 1024, refresh=True)

    assert res["result"] == "created"
    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req.method == "PUT"
    assert "DOC-C-1000-ESTIMATE%23c1" in str(req.url)
    assert "refresh=true" in str(req.url)


def test_index_batch_bulk_success():
    """Verify bulk indexing formats ndjson with action and document pairs."""
    captured_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, json={
            "errors": False,
            "items": [
                {"index": {"_id": "c1", "status": 201}},
                {"index": {"_id": "c2", "status": 201}},
            ]
        })

    client = make_mock_client(handler)
    indexer = OpenSearchIndexer(client=client)

    c1 = make_sample_chunk("c1")
    c2 = make_sample_chunk("c2")
    vectors = [[0.1] * 1024, [0.2] * 1024]

    count = indexer.index_batch([c1, c2], vectors, refresh=True)
    assert count == 2
    assert len(captured_requests) == 1

    req = captured_requests[0]
    assert req.headers["Content-Type"] == "application/x-ndjson"
    lines = req.content.decode("utf-8").strip().split("\n")
    assert len(lines) == 4  # 2 actions + 2 documents


def test_index_batch_error_handling():
    """Verify bulk indexing raises OpenSearchIndexingError if item errors occur."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "errors": True,
            "items": [
                {"index": {"_id": "c1", "error": {"type": "mapper_parsing_exception", "reason": "failed"}}}
            ]
        })

    client = make_mock_client(handler)
    indexer = OpenSearchIndexer(client=client)

    with pytest.raises(OpenSearchIndexingError, match="Bulk indexing encountered errors"):
        indexer.index_batch([make_sample_chunk("c1")], [[0.1] * 1024])


def test_get_by_id_found_and_not_found():
    """Verify get_by_id handles 200 found and 404 not found cleanly."""
    def handler(request: httpx.Request) -> httpx.Response:
        if "c1" in str(request.url):
            return httpx.Response(200, json={"_id": "c1", "_source": {"content": "Found"}})
        return httpx.Response(404, json={"found": False})

    client = make_mock_client(handler)
    indexer = OpenSearchIndexer(client=client)

    doc = indexer.get_by_id("c1")
    assert doc is not None
    assert doc["_id"] == "c1"
    assert doc["_source"]["content"] == "Found"

    missing = indexer.get_by_id("c999")
    assert missing is None


def test_search_knn_success():
    """Verify k-NN search builds query and returns hits list."""
    captured_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, json={
            "hits": {
                "hits": [
                    {
                        "_id": "c1",
                        "_score": 0.92,
                        "_source": {"content": "Emergency water extraction"},
                    }
                ]
            }
        })

    client = make_mock_client(handler)
    indexer = OpenSearchIndexer(client=client)

    hits = indexer.search_knn([0.0] * 1024, k=3)
    assert len(hits) == 1
    assert hits[0]["_id"] == "c1"
    assert hits[0]["_score"] == 0.92

    req = captured_requests[0]
    payload = json.loads(req.content.decode("utf-8"))
    assert payload["query"]["knn"]["embedding"]["k"] == 3


def test_connection_error_surfaced():
    """Verify connection refusal maps to OpenSearchConnectionError."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    client = make_mock_client(handler)
    indexer = OpenSearchIndexer(client=client)

    with pytest.raises(OpenSearchConnectionError, match="Cannot connect to OpenSearch"):
        indexer.get_by_id("c1")


