"""
Retrieval module for insurance support agent.

Provides decoupled retrieval models, candidate fusion (RRF), and parallel hybrid retrieval services.
"""
from apps.retrieval.fusion import reciprocal_rank_fusion
from apps.retrieval.hybrid import HybridRetriever, RetrievalService
from apps.retrieval.models import RetrievalFilters, RetrievalMethod, RetrievalResult

__all__ = [
    "HybridRetriever",
    "RetrievalFilters",
    "RetrievalMethod",
    "RetrievalResult",
    "RetrievalService",
    "reciprocal_rank_fusion",
]
