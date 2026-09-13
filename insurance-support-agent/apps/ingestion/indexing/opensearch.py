"""
OpenSearch implementation of the VectorStore protocol.

Provides bulk indexing, individual chunk indexing, ID retrieval, and k-NN vector search
over HTTP REST endpoints using httpx.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence
import urllib.parse

import httpx

from apps.ingestion.indexing.base import (
    BaseVectorStore,
    OpenSearchConnectionError,
    OpenSearchError,
    OpenSearchIndexingError,
    OpenSearchQueryError,
)
from apps.ingestion.models import Chunk

logger = logging.getLogger(__name__)


class OpenSearchIndexer(BaseVectorStore):
    """
    Client and vector store manager for OpenSearch.

    Communicates with the OpenSearch REST API to index chunks, retrieve them by ID,
    and perform k-NN dense vector searches.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:9200",
        index_name: str = "insurance_documents",
        client: Optional[httpx.Client] = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(index_name=index_name)
        self.endpoint = endpoint.rstrip("/")
        self._client = client or httpx.Client(
            base_url=self.endpoint,
            timeout=timeout,
        )

    def count(self) -> int:
        """Return total document count in the index."""
        try:
            resp = self._client.get(f"/{self.index_name}/_count")
            if resp.status_code != 200:
                raise OpenSearchQueryError(
                    f"OpenSearch _count returned HTTP {resp.status_code}: {resp.text}"
                )
            return resp.json().get("count", 0)
        except httpx.ConnectError as e:
            raise OpenSearchConnectionError(f"Cannot connect to OpenSearch at {self.endpoint}: {e}") from e

    def index_chunk(
        self,
        chunk: Chunk,
        vector: List[float],
        refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Index a single Chunk with its dense vector embedding.
        """
        doc = self.prepare_document(chunk, vector)
        doc_id = urllib.parse.quote(chunk.chunk_id, safe="")
        url = f"/{self.index_name}/_doc/{doc_id}"
        params = {"refresh": "true"} if refresh else {}

        try:
            resp = self._client.put(url, json=doc, params=params)
            if resp.status_code not in (200, 201):
                raise OpenSearchIndexingError(
                    f"Failed to index chunk '{chunk.chunk_id}'. "
                    f"Status: {resp.status_code}, body: {resp.text}"
                )
            return resp.json()
        except httpx.ConnectError as e:
            raise OpenSearchConnectionError(f"Cannot connect to OpenSearch at {self.endpoint}: {e}") from e

    def index_batch(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[List[float]],
        refresh: bool = True,
    ) -> int:
        """
        Index a batch of chunks and embeddings using OpenSearch bulk API.

        Returns the number of documents successfully indexed.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks but {len(vectors)} vectors"
            )
        if not chunks:
            return 0

        lines: List[str] = []
        for chunk, vector in zip(chunks, vectors):
            action = {"index": {"_index": self.index_name, "_id": chunk.chunk_id}}
            doc = self.prepare_document(chunk, vector)
            lines.append(json.dumps(action))
            lines.append(json.dumps(doc))

        ndjson_body = "\n".join(lines) + "\n"
        params = {"refresh": "true"} if refresh else {}

        try:
            resp = self._client.post(
                "/_bulk",
                content=ndjson_body,
                params=params,
                headers={"Content-Type": "application/x-ndjson"},
            )
            if resp.status_code != 200:
                raise OpenSearchIndexingError(
                    f"Bulk indexing failed with HTTP {resp.status_code}: {resp.text}"
                )

            data = resp.json()
            if data.get("errors"):
                failed_items = [
                    item for item in data.get("items", [])
                    if item.get("index", {}).get("error")
                ]
                raise OpenSearchIndexingError(
                    f"Bulk indexing encountered errors on {len(failed_items)} items: {failed_items[:3]}"
                )

            return len(data.get("items", []))
        except httpx.ConnectError as e:
            raise OpenSearchConnectionError(f"Cannot connect to OpenSearch at {self.endpoint}: {e}") from e

    def get_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single document by its chunk_id. Returns None if not found.
        """
        doc_id = urllib.parse.quote(chunk_id, safe="")
        url = f"/{self.index_name}/_doc/{doc_id}"

        try:
            resp = self._client.get(url)
            if resp.status_code == 404:
                return None
            if resp.status_code != 200:
                raise OpenSearchQueryError(
                    f"Failed to get document '{chunk_id}'. Status: {resp.status_code}: {resp.text}"
                )
            return resp.json()
        except httpx.ConnectError as e:
            raise OpenSearchConnectionError(f"Cannot connect to OpenSearch at {self.endpoint}: {e}") from e

    def search_knn(
        self,
        query_vector: List[float],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform a k-NN approximate nearest neighbor search against the 'embedding' field.

        Returns list of hit dictionaries containing '_id', '_score', and '_source'.
        """
        knn_clause: Dict[str, Any] = {
            "embedding": {
                "vector": query_vector,
                "k": k,
            }
        }
        if filters:
            knn_clause["embedding"]["filter"] = filters

        body = {
            "size": k,
            "query": {
                "knn": knn_clause
            },
        }

        try:
            resp = self._client.post(f"/{self.index_name}/_search", json=body)
            if resp.status_code != 200:
                raise OpenSearchQueryError(
                    f"k-NN search failed with HTTP {resp.status_code}: {resp.text}"
                )
            data = resp.json()
            return data.get("hits", {}).get("hits", [])
        except httpx.ConnectError as e:
            raise OpenSearchConnectionError(f"Cannot connect to OpenSearch at {self.endpoint}: {e}") from e

    def search_text(
        self,
        query_text: str,
        size: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Standard BM25 full-text keyword search over the 'content' field.
        """
        body = {
            "size": size,
            "query": {
                "match": {
                    "content": query_text
                }
            },
        }

        try:
            resp = self._client.post(f"/{self.index_name}/_search", json=body)
            if resp.status_code != 200:
                raise OpenSearchQueryError(
                    f"Text search failed with HTTP {resp.status_code}: {resp.text}"
                )
            data = resp.json()
            return data.get("hits", {}).get("hits", [])
        except httpx.ConnectError as e:
            raise OpenSearchConnectionError(f"Cannot connect to OpenSearch at {self.endpoint}: {e}") from e
