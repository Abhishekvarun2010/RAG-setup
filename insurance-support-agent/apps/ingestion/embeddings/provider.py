"""
Concrete implementation of EmbeddingProvider using local or remote Ollama instances.

Default model: bge-m3 (1024-dimensional dense vector embeddings).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import httpx

from apps.ingestion.embeddings.base import (
    BaseEmbeddingProvider,
    EmbeddingConnectionError,
    EmbeddingError,
    EmbeddingModelNotFoundError,
)


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider connecting to an Ollama server via HTTP.

    Features:
    - Default model: 'bge-m3' (1024 dimensions).
    - Uses Ollama's native batch embedding endpoint (`/api/embed`).
    - Robust error mapping (connection errors, timeouts, missing models).
    - Configurable timeout, batch size, and custom httpx.Client injection.
    """

    KNOWN_DIMENSIONS: Dict[str, int] = {
        "bge-m3": 1024,
        "bge-m3:latest": 1024,
        "nomic-embed-text": 768,
        "nomic-embed-text:latest": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    }

    def __init__(
        self,
        model_name: str = "bge-m3",
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
        batch_size: int = 32,
        dimension: Optional[int] = None,
        client: Optional[httpx.Client] = None,
    ):
        """
        Initialize the Ollama embedding provider.

        Args:
            model_name: Name or tag of the Ollama embedding model (default: 'bge-m3').
            base_url: Base URL where Ollama is listening (default: 'http://localhost:11434').
            timeout: HTTP request timeout in seconds.
            batch_size: Maximum texts per single batch request.
            dimension: Optional explicit vector dimension (inferred if known).
            client: Optional pre-configured httpx.Client (e.g. for testing/mocking).
        """
        self._model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = max(1, batch_size)
        self._dimension = dimension or self.KNOWN_DIMENSIONS.get(model_name, 1024)
        self._client = client

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_client(self) -> httpx.Client:
        """Return the injected client or instantiate a new client."""
        if self._client is not None:
            return self._client
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Generate dense vector embeddings for a sequence of text strings.

        Args:
            texts: Sequence of non-empty text strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        if not texts:
            return []

        # Validate inputs
        for idx, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(f"Item at index {idx} must be a string, got {type(t).__name__}")
            if not t.strip():
                raise ValueError(f"Item at index {idx} is empty or whitespace-only.")

        all_embeddings: List[List[float]] = []
        client = self._get_client()
        should_close = self._client is None

        try:
            # Process in sub-batches
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                sub_embeddings = self._send_embed_request(client, batch)
                all_embeddings.extend(sub_embeddings)
        finally:
            if should_close:
                client.close()

        return all_embeddings

    def _send_embed_request(self, client: httpx.Client, batch: List[str]) -> List[List[float]]:
        """Send a batch embedding request to Ollama's /api/embed endpoint."""
        url = f"{self.base_url}/api/embed"
        payload = {
            "model": self.model_name,
            "input": batch,
        }

        try:
            response = client.post(url, json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise EmbeddingConnectionError(
                f"Cannot connect to Ollama server at {self.base_url}. Is Ollama running? Error: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise EmbeddingConnectionError(
                f"Timeout ({self.timeout}s) waiting for embeddings from Ollama at {self.base_url}."
            ) from e
        except httpx.RequestError as e:
            raise EmbeddingConnectionError(f"HTTP request failed: {e}") from e

        # Handle HTTP Status
        if response.status_code == 404:
            err_msg = response.text.lower()
            if "not found" in err_msg or "model" in err_msg:
                raise EmbeddingModelNotFoundError(
                    f"Model '{self.model_name}' was not found on Ollama server at {self.base_url}. "
                    f"Run `ollama pull {self.model_name}` to install it."
                )
            raise EmbeddingError(f"Endpoint not found: {response.text}")
        elif response.status_code != 200:
            raise EmbeddingError(
                f"Ollama server returned HTTP {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except Exception as e:
            raise EmbeddingError(f"Failed to parse JSON response from Ollama: {e}") from e

        embeddings = data.get("embeddings")
        if embeddings is None:
            # Fallback for single embedding responses
            single_emb = data.get("embedding")
            if single_emb and len(batch) == 1:
                embeddings = [single_emb]
            else:
                raise EmbeddingError(f"Ollama response missing 'embeddings' field: {data}")

        if len(embeddings) != len(batch):
            raise EmbeddingError(
                f"Mismatched embedding count: sent {len(batch)} inputs, received {len(embeddings)} vectors."
            )

        # Update dimension dynamically if observed
        if embeddings and len(embeddings[0]) > 0:
            self._dimension = len(embeddings[0])

        return embeddings
