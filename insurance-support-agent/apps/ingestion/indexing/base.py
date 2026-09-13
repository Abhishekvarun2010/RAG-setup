"""
Base abstractions and protocol definitions for document/chunk indexing and vector search.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from apps.ingestion.models import Chunk


# =====================================================================
# Domain Exceptions
# =====================================================================

class OpenSearchError(Exception):
    """Base exception for all OpenSearch indexing and search errors."""


class OpenSearchConnectionError(OpenSearchError):
    """Raised when the OpenSearch cluster cannot be reached."""


class OpenSearchIndexingError(OpenSearchError):
    """Raised when document insertion or bulk indexing fails."""


class OpenSearchQueryError(OpenSearchError):
    """Raised when a query or k-NN search fails."""


# =====================================================================
# Protocol Definition
# =====================================================================

@runtime_checkable
class VectorStore(Protocol):
    """
    Protocol defining the interface for vector and metadata indexing stores.
    """

    @property
    def index_name(self) -> str:
        """The name of the target index."""
        ...

    def index_chunk(
        self,
        chunk: Chunk,
        vector: List[float],
        refresh: bool = False,
    ) -> Dict[str, Any]:
        """Index a single chunk along with its dense vector embedding."""
        ...

    def index_batch(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[List[float]],
        refresh: bool = True,
    ) -> int:
        """Index a batch of chunks and vectors in bulk. Returns count indexed."""
        ...

    def get_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its chunk_id."""
        ...

    def search_knn(
        self,
        query_vector: List[float],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform a k-NN approximate nearest neighbor vector search."""
        ...

    def search_text(
        self,
        query_text: str,
        size: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform a full-text BM25 keyword search with optional filters."""
        ...


class BaseVectorStore(ABC):
    """
    Abstract base class providing shared helpers and validation for vector stores.
    """

    def __init__(self, index_name: str = "insurance_documents") -> None:
        self._index_name = index_name

    @property
    def index_name(self) -> str:
        return self._index_name

    def prepare_document(self, chunk: Chunk, vector: List[float]) -> Dict[str, Any]:
        """
        Serialize a Chunk model to JSON-compatible dict and attach the embedding vector.
        """
        if not vector or not isinstance(vector, list):
            raise ValueError(f"Expected non-empty list of floats for vector, got {type(vector)}")
        
        doc = chunk.model_dump(mode="json")
        doc["embedding"] = vector
        return doc
