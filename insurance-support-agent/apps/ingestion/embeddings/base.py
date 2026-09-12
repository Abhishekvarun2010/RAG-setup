"""
Base interfaces, protocols, and exceptions for embedding generation.

Defines:
- EmbeddingError, EmbeddingConnectionError, EmbeddingModelNotFoundError: Domain exceptions.
- EmbeddingProvider: @runtime_checkable Protocol defining the embedding provider interface.
- BaseEmbeddingProvider: Abstract base class providing common input validation and helpers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Protocol, Sequence, runtime_checkable

from apps.ingestion.models import Chunk


# =====================================================================
# Domain Exceptions
# =====================================================================

class EmbeddingError(Exception):
    """Base exception for all embedding generation errors."""
    pass


class EmbeddingConnectionError(EmbeddingError):
    """Raised when unable to connect to the embedding server or when requests time out."""
    pass


class EmbeddingModelNotFoundError(EmbeddingError):
    """Raised when the requested embedding model does not exist or has not been pulled."""
    pass


# =====================================================================
# Embedding Provider Protocol
# =====================================================================

@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Protocol defining the interface for dense embedding generation providers.

    Any class implementing `embed`, `embed_batch`, `dimension`, and `model_name`
    satisfies this protocol (structural subtyping / duck typing).
    """

    @property
    def dimension(self) -> int:
        """The dimensionality of the output vectors (e.g. 1024 for bge-m3)."""
        ...

    @property
    def model_name(self) -> str:
        """Name or tag of the underlying embedding model."""
        ...

    def embed(self, text: str) -> List[float]:
        """
        Generate a dense vector embedding for a single text string.

        Args:
            text: Non-empty text string to embed.

        Returns:
            List of floats representing the dense vector embedding.
        """
        ...

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Generate dense vector embeddings for a sequence of text strings.

        Args:
            texts: Sequence of text strings to embed.

        Returns:
            List of dense vectors, exactly one per input text.
        """
        ...


# =====================================================================
# Abstract Base Class
# =====================================================================

class BaseEmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    Provides default implementations for single-text embedding via embed_batch
    and chunk embedding convenience helpers.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the output vectors."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the underlying embedding model."""
        pass

    @abstractmethod
    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate dense vector embeddings for a batch of texts."""
        pass

    def embed(self, text: str) -> List[float]:
        """
        Generate a dense vector embedding for a single text string.
        Validates input and delegates to embed_batch.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected text to be a string, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("Cannot generate embedding for empty or whitespace-only text.")

        results = self.embed_batch([text])
        if not results:
            raise EmbeddingError("Provider returned no embedding vector.")
        return results[0]

    def embed_chunks(self, chunks: Sequence[Chunk]) -> List[List[float]]:
        """
        Convenience method to generate embeddings directly from Chunk objects.

        Args:
            chunks: Sequence of Chunk objects.

        Returns:
            List of embedding vectors corresponding to each chunk's content.
        """
        if not chunks:
            return []
        texts = [c.content for c in chunks]
        return self.embed_batch(texts)
