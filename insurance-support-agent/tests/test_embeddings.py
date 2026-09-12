"""
Unit and integration tests for the Embedding Generation Layer.

Tests:
1. EmbeddingProvider protocol conformance.
2. One text produces a vector (dimension and float types).
3. Batch produces one vector per input text.
4. Empty and whitespace input handling.
5. Provider errors surfaced (connection failure, timeout, model not found, server error).
6. Live integration test with local Ollama if running.
"""
import json
from typing import List
import httpx
import pytest

from apps.ingestion import (
    BaseEmbeddingProvider,
    EmbeddingConnectionError,
    EmbeddingError,
    EmbeddingModelNotFoundError,
    EmbeddingProvider,
    OllamaEmbeddingProvider,
)


# =====================================================================
# Mock Transport Fixtures
# =====================================================================

def make_mock_client(handler) -> httpx.Client:
    """Create an httpx.Client with custom mock transport."""
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport, base_url="http://mock-ollama:11434")


# =====================================================================
# 1. Protocol Conformance Tests
# =====================================================================

def test_ollama_provider_conforms_to_protocol():
    """Verify OllamaEmbeddingProvider conforms to EmbeddingProvider protocol."""
    provider = OllamaEmbeddingProvider()
    assert isinstance(provider, EmbeddingProvider)
    assert isinstance(provider, BaseEmbeddingProvider)
    assert provider.model_name == "bge-m3"
    assert provider.dimension == 1024


def test_custom_duck_typed_provider_conforms():
    """Verify custom duck-typed provider conforms to EmbeddingProvider protocol."""
    class CustomProvider:
        @property
        def dimension(self) -> int:
            return 768

        @property
        def model_name(self) -> str:
            return "custom-model"

        def embed(self, text: str) -> List[float]:
            return [0.1] * 768

        def embed_batch(self, texts: List[str]) -> List[List[float]]:
            return [[0.1] * 768 for _ in texts]

    assert isinstance(CustomProvider(), EmbeddingProvider)


# =====================================================================
# 2. Single Text Produces Vector Tests
# =====================================================================

def test_embed_one_text_produces_vector():
    """✓ Test that a single text input produces a valid dense vector."""
    fake_dim = 1024
    fake_vector = [0.012 * (i % 10) for i in range(fake_dim)]

    def handler(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content)
        assert data["model"] == "bge-m3"
        assert data["input"] == ["Named insured: Robin Hardy"]
        return httpx.Response(200, json={"embeddings": [fake_vector]})

    client = make_mock_client(handler)
    provider = OllamaEmbeddingProvider(client=client)

    vector = provider.embed("Named insured: Robin Hardy")

    assert isinstance(vector, list)
    assert len(vector) == fake_dim
    assert all(isinstance(val, float) for val in vector)
    assert vector == fake_vector


# =====================================================================
# 3. Batch Produces One Vector Per Input Tests
# =====================================================================

def test_embed_batch_produces_one_vector_per_input():
    """✓ Test that batch input produces exactly one vector per input text."""
    inputs = [
        "Policy Declarations",
        "Coverage limit €100,000",
        "Deductible €1,000",
    ]
    fake_vectors = [[0.1 * (j + 1) for _ in range(1024)] for j in range(len(inputs))]

    def handler(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content)
        assert data["input"] == inputs
        return httpx.Response(200, json={"embeddings": fake_vectors})

    client = make_mock_client(handler)
    provider = OllamaEmbeddingProvider(client=client)

    vectors = provider.embed_batch(inputs)

    assert len(vectors) == 3
    for vec in vectors:
        assert len(vec) == 1024
        assert all(isinstance(x, float) for x in vec)


def test_embed_batch_with_batch_size_chunking():
    """Verify that batches larger than batch_size are correctly partitioned."""
    inputs = [f"Text item {i}" for i in range(7)]
    requests_received = []

    def handler(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content)
        requests_received.append(data["input"])
        batch_vecs = [[0.5] * 1024 for _ in data["input"]]
        return httpx.Response(200, json={"embeddings": batch_vecs})

    client = make_mock_client(handler)
    # Set batch_size=3, so 7 items should result in 3 requests (3, 3, 1)
    provider = OllamaEmbeddingProvider(batch_size=3, client=client)

    vectors = provider.embed_batch(inputs)

    assert len(vectors) == 7
    assert len(requests_received) == 3
    assert len(requests_received[0]) == 3
    assert len(requests_received[1]) == 3
    assert len(requests_received[2]) == 1


# =====================================================================
# 4. Empty Input Handling Tests
# =====================================================================

def test_empty_batch_returns_empty_list():
    """✓ Verify that empty input sequence returns empty list without calling network."""
    provider = OllamaEmbeddingProvider()
    assert provider.embed_batch([]) == []


def test_embed_empty_string_raises_value_error():
    """✓ Verify that empty string input raises ValueError."""
    provider = OllamaEmbeddingProvider()
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        provider.embed("")

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        provider.embed("   \n\t  ")


def test_embed_batch_with_empty_item_raises_value_error():
    """✓ Verify that whitespace-only item within a batch raises ValueError."""
    provider = OllamaEmbeddingProvider()
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        provider.embed_batch(["Valid text", "   "])


def test_embed_invalid_type_raises_type_error():
    """Verify that non-string inputs raise TypeError."""
    provider = OllamaEmbeddingProvider()
    with pytest.raises(TypeError, match="to be a string"):
        provider.embed(12345)  # type: ignore

    with pytest.raises(TypeError, match="must be a string"):
        provider.embed_batch([None])  # type: ignore


# =====================================================================
# 5. Provider Errors Surfaced Tests
# =====================================================================

def test_connection_refused_error_surfaced():
    """✓ Verify that network connection failure raises EmbeddingConnectionError."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused by target machine")

    client = make_mock_client(handler)
    provider = OllamaEmbeddingProvider(client=client)

    with pytest.raises(EmbeddingConnectionError, match="Cannot connect to Ollama server"):
        provider.embed("Test text")


def test_connection_timeout_error_surfaced():
    """✓ Verify that network timeout raises EmbeddingConnectionError."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("Request timed out after 30s")

    client = make_mock_client(handler)
    provider = OllamaEmbeddingProvider(client=client)

    with pytest.raises(EmbeddingConnectionError, match="Cannot connect to Ollama server"):
        provider.embed("Test text")


def test_model_not_found_error_surfaced():
    """✓ Verify that missing model (HTTP 404) raises EmbeddingModelNotFoundError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"error": "model 'bge-m3' not found, try pulling it first"},
        )

    client = make_mock_client(handler)
    provider = OllamaEmbeddingProvider(model_name="bge-m3", client=client)

    with pytest.raises(EmbeddingModelNotFoundError, match="was not found on Ollama server"):
        provider.embed("Test text")


def test_server_error_surfaced():
    """✓ Verify that server 500 error raises EmbeddingError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Ollama GPU OOM Error")

    client = make_mock_client(handler)
    provider = OllamaEmbeddingProvider(client=client)

    with pytest.raises(EmbeddingError, match="Ollama server returned HTTP 500"):
        provider.embed("Test text")


# =====================================================================
# 6. Live Local Ollama Integration Test
# =====================================================================

def test_live_local_ollama_if_running():
    """
    Optional live integration test: runs against local Ollama if reachable
    and verifies that bge-m3 generates a real 1024-dimension embedding.
    """
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=1.0)
        if r.status_code != 200:
            pytest.skip("Local Ollama not reachable")
        models = [m.get("name", "") for m in r.json().get("models", [])]
        if not any("bge-m3" in m for m in models):
            pytest.skip("bge-m3 model not installed in local Ollama")
    except Exception:
        pytest.skip("Local Ollama not running")

    provider = OllamaEmbeddingProvider(model_name="bge-m3")
    vector = provider.embed("Policy coverage: Bodily injury up to €100,000")

    assert isinstance(vector, list)
    assert len(vector) == 1024
    assert any(x != 0.0 for x in vector)
