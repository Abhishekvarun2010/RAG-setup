"""
Security-enforced retrieval wrapper.

SecureRetriever wraps HybridRetriever to enforce mandatory, non-bypassable
authorization filters derived from the authenticated user's SecurityContext.
It also applies post-retrieval validation, field-level redaction, and audit logging.

IMPORTANT: This is NOT a subclass of HybridRetriever. It wraps it.
There is no way to call retrieve() without a SecurityContext.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional, Union

from apps.auth.audit import AuditEvent, AuditLogger
from apps.auth.models import AuthorizationError, SecurityContext
from apps.auth.redaction import Redactor
from apps.retrieval.hybrid import HybridRetriever
from apps.retrieval.models import RetrievalFilters, RetrievalResult

logger = logging.getLogger(__name__)


def _merge_filters(
    mandatory: RetrievalFilters,
    user_supplied: Optional[Union[RetrievalFilters, Dict[str, Any]]],
) -> RetrievalFilters:
    """
    Merge user-supplied filters with mandatory security filters.

    Mandatory filters always win on conflicts. User-supplied filters
    can only narrow the scope — never widen it.
    """
    if user_supplied is None:
        return mandatory

    # Normalize user-supplied filters to a RetrievalFilters instance
    if isinstance(user_supplied, dict):
        try:
            user_filters = RetrievalFilters(**user_supplied)
        except Exception:
            # If user supplied raw dict that doesn't fit the model, ignore it
            logger.warning("User-supplied filter dict could not be parsed; using mandatory filters only.")
            return mandatory
    else:
        user_filters = user_supplied

    # Start with mandatory as base, overlay user-supplied for fields
    # that mandatory did NOT set (mandatory always wins)
    merged_data: Dict[str, Any] = {}

    mandatory_dict = mandatory.model_dump(exclude_none=True)
    user_dict = user_filters.model_dump(exclude_none=True)

    # Start with user fields
    merged_data.update(user_dict)
    # Override with mandatory (mandatory always wins)
    merged_data.update(mandatory_dict)

    # Special handling for access_control: use mandatory's, since it represents
    # the maximum set of tiers this user is allowed to see. If user tries to
    # supply a broader set, mandatory still wins. If user supplies a narrower
    # set (subset), we allow the narrower set.
    mandatory_ac = mandatory_dict.get("access_control")
    user_ac = user_dict.get("access_control")
    if mandatory_ac is not None and user_ac is not None:
        # User can only narrow: take the intersection
        permitted_set = set(mandatory_ac)
        narrowed = [ac for ac in user_ac if ac in permitted_set]
        merged_data["access_control"] = narrowed if narrowed else mandatory_ac

    return RetrievalFilters(**merged_data)


class SecureRetriever:
    """
    Security-enforced retrieval wrapper around HybridRetriever.

    Enforces:
    1. Mandatory filters from SecurityContext (cannot be bypassed).
    2. Post-retrieval authorization validation (defense-in-depth).
    3. Field-level redaction (if Redactor is configured).
    4. Audit logging (if AuditLogger is configured).
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        redactor: Optional[Redactor] = None,
        audit_logger: Optional[AuditLogger] = None,
    ) -> None:
        self._retriever = retriever
        self._redactor = redactor
        self._audit_logger = audit_logger

    @property
    def retriever(self) -> HybridRetriever:
        """Access the underlying HybridRetriever (read-only)."""
        return self._retriever

    def retrieve(
        self,
        query: str,
        context: SecurityContext,
        top_k: int = 5,
        filters: Optional[Union[RetrievalFilters, Dict[str, Any]]] = None,
        as_of_date: Optional[date] = None,
        version: Optional[str] = None,
        rrf_k: int = 60,
        vector_k: int = 10,
        bm25_k: int = 10,
        fusion_k: int = 10,
        rerank: bool = True,
    ) -> List[RetrievalResult]:
        """
        Execute secure hybrid retrieval.

        Steps:
        1. Build mandatory security filters from SecurityContext.
        2. Merge with any user-supplied filters (mandatory wins on conflicts).
        3. Delegate to HybridRetriever.retrieve() with merged filters.
        4. Post-retrieval validation: verify each result against SecurityContext.
        5. Apply redaction if configured.
        6. Log audit event if configured.
        7. Return sanitized results.

        Args:
            query: The search query text.
            context: Authenticated user's SecurityContext (REQUIRED).
            top_k: Number of final results to return.
            filters: Optional additional filters from the caller (narrowing only).
            as_of_date: Point-in-time date for version-aware retrieval.
            version: Specific document version to retrieve.
            rrf_k: RRF constant (default 60).
            vector_k: Number of vector candidates to retrieve.
            bm25_k: Number of BM25 candidates to retrieve.
            fusion_k: Number of RRF candidates before reranking.
            rerank: Whether to apply cross-encoder reranking.

        Returns:
            List of redacted and authorized RetrievalResult instances.
        """
        # 1. Build mandatory security filters
        mandatory_filters = context.to_mandatory_filters(as_of_date=as_of_date)

        # 2. Merge with user-supplied filters
        merged_filters = _merge_filters(mandatory_filters, filters)

        logger.debug(
            f"SecureRetriever [{context.user_id}]: "
            f"mandatory={mandatory_filters.model_dump(exclude_none=True)}, "
            f"merged={merged_filters.model_dump(exclude_none=True)}"
        )

        # 3. Delegate to HybridRetriever
        raw_results = self._retriever.retrieve(
            query=query,
            top_k=top_k,
            filters=merged_filters,
            as_of_date=as_of_date,
            version=version,
            rrf_k=rrf_k,
            vector_k=vector_k,
            bm25_k=bm25_k,
            fusion_k=fusion_k,
            rerank=rerank,
        )

        # 4. Post-retrieval authorization validation (defense-in-depth)
        authorized_results: List[RetrievalResult] = []
        filtered_count = 0
        for result in raw_results:
            if context.can_access_chunk(result.metadata):
                authorized_results.append(result)
            else:
                filtered_count += 1
                logger.warning(
                    f"SecureRetriever [{context.user_id}]: "
                    f"Post-retrieval filter removed chunk '{result.chunk_id}' — "
                    f"chunk metadata did not pass authorization check."
                )

        # 5. Apply redaction
        redacted_fields: List[str] = []
        if self._redactor is not None:
            authorized_results = self._redactor.redact(authorized_results, context)
            # Collect redacted field names from metadata
            for r in authorized_results:
                redacted_fields.extend(r.metadata.get("_redacted_fields", []))

        # 6. Audit logging
        if self._audit_logger is not None:
            event = AuditEvent(
                user_id=context.user_id,
                roles=[r.value for r in context.roles],
                query=query,
                filters_applied=merged_filters.model_dump(exclude_none=True),
                chunks_returned=[r.chunk_id for r in authorized_results],
                chunks_filtered_out=filtered_count,
                redacted_fields=list(set(redacted_fields)),
                retrieval_method="hybrid_rrf",
            )
            self._audit_logger.log_retrieval(event)

        return authorized_results

    def retrieve_bm25(
        self,
        query: str,
        context: SecurityContext,
        top_k: int = 5,
        filters: Optional[Union[RetrievalFilters, Dict[str, Any]]] = None,
        as_of_date: Optional[date] = None,
        version: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Secure BM25-only retrieval."""
        mandatory_filters = context.to_mandatory_filters(as_of_date=as_of_date)
        merged_filters = _merge_filters(mandatory_filters, filters)
        results = self._retriever.retrieve_bm25(
            query=query,
            top_k=top_k,
            filters=merged_filters,
            version=version,
        )
        return self._post_process(results, context, query, merged_filters, "bm25")

    def retrieve_vector(
        self,
        query: str,
        context: SecurityContext,
        top_k: int = 5,
        filters: Optional[Union[RetrievalFilters, Dict[str, Any]]] = None,
        as_of_date: Optional[date] = None,
        version: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Secure vector-only retrieval."""
        mandatory_filters = context.to_mandatory_filters(as_of_date=as_of_date)
        merged_filters = _merge_filters(mandatory_filters, filters)
        results = self._retriever.retrieve_vector(
            query=query,
            top_k=top_k,
            filters=merged_filters,
            version=version,
        )
        return self._post_process(results, context, query, merged_filters, "vector")

    def _post_process(
        self,
        results: List[RetrievalResult],
        context: SecurityContext,
        query: str,
        filters: RetrievalFilters,
        method: str,
    ) -> List[RetrievalResult]:
        """Shared post-retrieval processing: auth check, redaction, audit."""
        authorized = [r for r in results if context.can_access_chunk(r.metadata)]
        filtered_count = len(results) - len(authorized)

        redacted_fields: List[str] = []
        if self._redactor is not None:
            authorized = self._redactor.redact(authorized, context)
            for r in authorized:
                redacted_fields.extend(r.metadata.get("_redacted_fields", []))

        if self._audit_logger is not None:
            event = AuditEvent(
                user_id=context.user_id,
                roles=[r.value for r in context.roles],
                query=query,
                filters_applied=filters.model_dump(exclude_none=True),
                chunks_returned=[r.chunk_id for r in authorized],
                chunks_filtered_out=filtered_count,
                redacted_fields=list(set(redacted_fields)),
                retrieval_method=method,
            )
            self._audit_logger.log_retrieval(event)

        return authorized
