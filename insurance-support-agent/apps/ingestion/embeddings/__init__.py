"""
Embeddings package for dense vector generation.

Provides:
- EmbeddingProvider: Protocol defining embedding generation interface.
- BaseEmbeddingProvider: Abstract base class with input validation.
- OllamaEmbeddingProvider: Concrete provider connecting to Ollama (default: bge-m3).
- Domain exceptions: EmbeddingError, EmbeddingConnectionError, EmbeddingModelNotFoundError.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apps.ingestion.embeddings.base import (
        BaseEmbeddingProvider,
        EmbeddingConnectionError,
        EmbeddingError,
        EmbeddingModelNotFoundError,
        EmbeddingProvider,
    )
    from apps.ingestion.embeddings.provider import OllamaEmbeddingProvider

__all__ = [
    "BaseEmbeddingProvider",
    "EmbeddingConnectionError",
    "EmbeddingError",
    "EmbeddingModelNotFoundError",
    "EmbeddingProvider",
    "OllamaEmbeddingProvider",
]


def __getattr__(name: str):
    if name in {
        "BaseEmbeddingProvider",
        "EmbeddingConnectionError",
        "EmbeddingError",
        "EmbeddingModelNotFoundError",
        "EmbeddingProvider",
    }:
        import apps.ingestion.embeddings.base as base
        return getattr(base, name)
    if name == "OllamaEmbeddingProvider":
        import apps.ingestion.embeddings.provider as provider
        return getattr(provider, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
