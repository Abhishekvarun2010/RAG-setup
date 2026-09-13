"""
Fusion algorithms for combining multiple ranked retrieval result lists.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from apps.retrieval.models import RetrievalMethod, RetrievalResult


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievalResult]],
    rrf_k: int = 60,
    top_k: Optional[int] = None,
) -> List[RetrievalResult]:
    """
    Combines multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_score(d) = sum_{list in ranked_lists} (1 / (rrf_k + rank(d)))
    where rank(d) is 1-indexed (1 for 1st place, 2 for 2nd place, etc.).

    Args:
        ranked_lists: A sequence of ranked lists (e.g. [bm25_results, vector_results]).
        rrf_k: Smoothing constant to control impact of top-ranked vs lower-ranked items (default 60).
        top_k: Optional cutoff to return only the top K merged results.

    Returns:
        List of fused RetrievalResult instances sorted by RRF score descending.
    """
    if not ranked_lists:
        return []

    # Map chunk_id -> { "score": float, "result": RetrievalResult, "ranks": dict, "scores": dict }
    fused_entries: Dict[str, Dict] = {}

    for results in ranked_lists:
        for rank, item in enumerate(results, start=1):
            chunk_id = item.chunk_id
            if chunk_id not in fused_entries:
                fused_entries[chunk_id] = {
                    "rrf_score": 0.0,
                    "representative_result": item,
                    "ranks": {},
                    "original_scores": {},
                }

            method_key = item.retrieval_method.value
            fused_entries[chunk_id]["ranks"][method_key] = rank
            fused_entries[chunk_id]["original_scores"][method_key] = item.score

            reciprocal_rank = 1.0 / (rrf_k + rank)
            fused_entries[chunk_id]["rrf_score"] += reciprocal_rank

    # Sort candidates by combined RRF score descending (stable sort)
    sorted_entries = sorted(
        fused_entries.values(),
        key=lambda entry: entry["rrf_score"],
        reverse=True,
    )

    if top_k is not None:
        sorted_entries = sorted_entries[:top_k]

    # Build new RetrievalResult objects with HYBRID_RRF method
    fused_results: List[RetrievalResult] = []
    for entry in sorted_entries:
        orig = entry["representative_result"]
        # Merge RRF details into metadata without mutating original item's metadata
        merged_metadata = dict(orig.metadata)
        merged_metadata["rrf_ranks"] = entry["ranks"]
        merged_metadata["rrf_component_scores"] = entry["original_scores"]

        fused_results.append(
            RetrievalResult(
                chunk_id=orig.chunk_id,
                content=orig.content,
                score=round(entry["rrf_score"], 6),
                retrieval_method=RetrievalMethod.HYBRID_RRF,
                metadata=merged_metadata,
            )
        )

    return fused_results
