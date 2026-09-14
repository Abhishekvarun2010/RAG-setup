"""
End-to-End RAG Question-Answering Flow.

Orchestrates the complete pipeline:
1. Authentication & SecurityContext resolution (JWTValidator)
2. Security-enforced hybrid retrieval & cross-encoder reranking (SecureRetriever)
3. Grounded prompt assembly with strict anti-hallucination guardrails (PromptBuilder)
4. LLM inference and reasoning extraction with Qwen3 8B (OllamaLLM)
5. Citation extraction and response structuring (QAResponse)
"""
from __future__ import annotations

from datetime import date
import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Union

from apps.agent.llm import LLMProvider, OllamaLLM
from apps.agent.models import Citation, QARequest, QAResponse
from apps.agent.prompt import PromptBuilder
from apps.auth.jwt import JWTValidator
from apps.auth.models import AuthenticationError, SecurityContext
from apps.auth.secure_retriever import SecureRetriever
from apps.retrieval.models import RetrievalFilters, RetrievalResult

logger = logging.getLogger(__name__)


class RAGQuestionAnsweringFlow:
    """
    Complete Question-Answering flow combining security-enforced retrieval and LLM generation.
    """

    # Matches inline citations like [DOC-C-1000-ESTIMATE#c3] or [COM-0000077-v1#c1]
    CITATION_REGEX = re.compile(r"\[([A-Za-z0-9_\-#]+)\]")

    def __init__(
        self,
        retriever: SecureRetriever,
        llm: Optional[LLMProvider] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        jwt_validator: Optional[JWTValidator] = None,
    ) -> None:
        """
        Initialize the RAG QA Flow.

        Args:
            retriever: SecureRetriever instance enforcing mandatory security filters & redaction.
            llm: LLMProvider instance (defaults to OllamaLLM with qwen3:8b).
            prompt_builder: PromptBuilder instance (defaults to standard grounded prompt builder).
            jwt_validator: Optional JWTValidator for authenticating raw bearer tokens.
        """
        self._retriever = retriever
        self._llm = llm or OllamaLLM(model_name="qwen3:8b")
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._jwt_validator = jwt_validator

    @property
    def retriever(self) -> SecureRetriever:
        return self._retriever

    @property
    def llm(self) -> LLMProvider:
        return self._llm

    @property
    def prompt_builder(self) -> PromptBuilder:
        return self._prompt_builder

    def _resolve_security_context(
        self,
        security_context: Optional[SecurityContext],
        jwt_token: Optional[str],
    ) -> SecurityContext:
        """Resolve SecurityContext from direct instance or JWT token."""
        if security_context is not None:
            return security_context

        if jwt_token is not None:
            if self._jwt_validator is None:
                raise AuthenticationError(
                    "JWT token provided but no JWTValidator is configured on RAGQuestionAnsweringFlow."
                )
            return self._jwt_validator.validate(jwt_token)

        raise AuthenticationError(
            "Authentication required: Must provide either 'security_context' or 'jwt_token'."
        )

    def _extract_citations(
        self,
        answer_text: str,
        retrieved_chunks: Sequence[RetrievalResult],
    ) -> List[Citation]:
        """
        Extract cited chunk IDs from the LLM answer and cross-reference them
        with the retrieved chunks.
        """
        chunk_map = {c.chunk_id: c for c in retrieved_chunks}
        found_ids = self.CITATION_REGEX.findall(answer_text)

        citations: List[Citation] = []
        seen_ids = set()

        for chunk_id in found_ids:
            if chunk_id in seen_ids:
                continue
            if chunk_id in chunk_map:
                matched_chunk = chunk_map[chunk_id]
                metadata = matched_chunk.metadata or {}
                source_uri = metadata.get("source_uri")
                page_number = metadata.get("page_number")
                doc_id = metadata.get("document_id")
                doc_type = metadata.get("document_type")
                section = metadata.get("section")

                file_link = None
                markdown_link = None
                if source_uri:
                    if source_uri.startswith("file://") or source_uri.startswith("http://") or source_uri.startswith("https://"):
                        base_link = source_uri
                    else:
                        base_link = f"file://{source_uri}"
                    
                    file_link = f"{base_link}#page={page_number}" if page_number else base_link
                    label = doc_id or chunk_id
                    if page_number:
                        label += f" (p. {page_number})"
                    markdown_link = f"[{label}]({file_link})"

                citations.append(
                    Citation(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        document_type=doc_type,
                        section=section,
                        page_number=page_number,
                        source_uri=source_uri,
                        file_link=file_link,
                        markdown_link=markdown_link,
                        snippet=matched_chunk.content[:160].strip() if matched_chunk.content else None,
                    )
                )
                seen_ids.add(chunk_id)

        return citations

    def answer(
        self,
        query: str,
        security_context: Optional[SecurityContext] = None,
        jwt_token: Optional[str] = None,
        filters: Optional[Union[RetrievalFilters, Dict[str, Any]]] = None,
        as_of_date: Optional[date] = None,
        top_k: int = 5,
        temperature: float = 0.0,
        include_thinking: bool = True,
        rerank: bool = True,
    ) -> QAResponse:
        """
        Execute the complete RAG question-answering workflow.

        Args:
            query: User inquiry string.
            security_context: Authenticated SecurityContext.
            jwt_token: Optional JWT bearer token string.
            filters: Optional user-supplied filters (can only narrow scope).
            as_of_date: Point-in-time date for temporal retrieval.
            top_k: Number of retrieved chunks to include in LLM context.
            temperature: LLM sampling temperature.
            include_thinking: Whether to include reasoning trace in response.
            rerank: Whether to apply cross-encoder reranking.

        Returns:
            Grounded QAResponse with answer, verified citations, passages, and metrics.
        """
        start_time = time.perf_counter()

        # 1. Resolve security identity
        context = self._resolve_security_context(security_context, jwt_token)

        # 2. Retrieve authorized and redacted passages
        retrieved_chunks = self._retriever.retrieve(
            query=query,
            context=context,
            top_k=top_k,
            filters=filters,
            as_of_date=as_of_date,
            rerank=rerank,
        )

        # 3. Handle empty retrieval
        if not retrieved_chunks:
            elapsed = round(time.perf_counter() - start_time, 4)
            return QAResponse(
                query=query,
                answer=(
                    "Based on the available records and your access permissions, "
                    "I do not have enough information to answer this question."
                ),
                citations=[],
                retrieved_chunks=[],
                thinking=None,
                model=self._llm.model_name,
                latency_seconds=elapsed,
                usage={"prompt_tokens": 0, "completion_tokens": 0},
                security_user_id=context.user_id,
            )

        # 4. Assemble grounded chat messages
        messages = self._prompt_builder.build_messages(
            query=query,
            chunks=retrieved_chunks,
            as_of_date=as_of_date,
        )

        # 5. Generate completion via LLM
        llm_response = self._llm.chat(
            messages=messages,
            temperature=temperature,
        )

        # 6. Extract and cross-reference citations
        citations = self._extract_citations(llm_response.content, retrieved_chunks)

        elapsed = round(time.perf_counter() - start_time, 4)

        return QAResponse(
            query=query,
            answer=llm_response.content,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            thinking=llm_response.thinking if include_thinking else None,
            model=llm_response.model,
            latency_seconds=elapsed,
            usage={
                "prompt_tokens": llm_response.prompt_tokens,
                "completion_tokens": llm_response.completion_tokens,
                "duration_seconds": llm_response.duration_seconds,
            },
            security_user_id=context.user_id,
        )

    def answer_request(self, request: QARequest) -> QAResponse:
        """Convenience method accepting a validated QARequest object."""
        return self.answer(
            query=request.query,
            security_context=request.security_context,
            jwt_token=request.jwt_token,
            filters=request.filters,
            as_of_date=request.as_of_date,
            top_k=request.top_k,
            temperature=request.temperature,
            include_thinking=request.include_thinking,
        )
