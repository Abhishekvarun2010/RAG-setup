"""
Prompt engineering and context assembly for Insurance RAG Question-Answering.

Enforces strict factual groundedness, anti-hallucination guardrails,
inline citation references ([chunk_id]), and temporal point-in-time awareness.
"""
from __future__ import annotations

from datetime import date
from typing import List, Optional, Sequence

from apps.agent.models import ChatMessage
from apps.retrieval.models import RetrievalResult


DEFAULT_SYSTEM_PROMPT = """You are an expert, precise, and compliant Insurance Support Assistant.
Your responsibility is to provide accurate, factual, and strictly grounded answers to insurance inquiries (claims, policies, coverage terms, and adjuster reports).

STRICT GROUNDING & ANTI-HALLUCINATION RULES:
1. Answer the question STRICTLY and ONLY using the provided Context passages.
2. Do NOT extrapolate, speculate, or introduce external knowledge, unmentioned coverage clauses, or fabricated financial numbers.
3. If the provided context does not contain sufficient facts to answer the user's question, you MUST explicitly state:
   "Based on the provided records, I do not have enough information to answer this question."
4. Do NOT disclose or guess any information that is marked as [REDACTED].

CITATION RULES:
1. You MUST cite your sources for every factual assertion, dollar amount, date, cause of loss, or coverage decision.
2. Format citations using the exact Chunk ID inside square brackets ([chunk_id]) immediately following the supported statement, for example:
   - "The net claim payable amount after the €5,000 deductible is €22,950.00 [DOC-C-1000-ESTIMATE#c3]."
   - "The cause of loss was identified as a slip-and-fall incident [DOC-C-1000-ADJ#c2]."
3. If multiple passages support a statement, include each chunk ID, e.g. [DOC-C-1000-ESTIMATE#c2][DOC-C-1000-ADJ#c2].
4. Never invent chunk IDs. Only use the Chunk IDs explicitly listed in the Context.
"""


class PromptBuilder:
    """
    Constructs grounded system and user prompts with structured context blocks.
    """

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.system_prompt = system_prompt

    def build_system_prompt(self, as_of_date: Optional[date] = None) -> str:
        """
        Build the system prompt, optionally augmenting it with point-in-time instructions.
        """
        base_prompt = self.system_prompt.strip()
        if as_of_date is not None:
            temporal_instruction = (
                f"\n\nTEMPORAL POINT-IN-TIME INSTRUCTION:\n"
                f"The user's query is evaluated as of date: {as_of_date.isoformat()}. "
                f"Reference only the policy terms, endorsements, and claim statuses that were effective on this date."
            )
            return base_prompt + temporal_instruction
        return base_prompt

    def format_context(self, chunks: Sequence[RetrievalResult]) -> str:
        """
        Format retrieved chunks into structured context blocks for the prompt.
        """
        if not chunks:
            return "No relevant context passages available."

        passages: List[str] = []
        for idx, chunk in enumerate(chunks, 1):
            metadata = chunk.metadata or {}
            doc_id = metadata.get("document_id") or "UNKNOWN"
            doc_type = metadata.get("document_type") or "general"
            section = metadata.get("section") or "General"
            policy_id = metadata.get("policy_id")
            claim_id = metadata.get("claim_id")

            header_lines = [
                f"--- Passage {idx} ---",
                f"Chunk ID: {chunk.chunk_id}",
                f"Document ID: {doc_id} (Type: {doc_type})",
                f"Section: {section}",
            ]
            if policy_id:
                header_lines.append(f"Policy ID: {policy_id}")
            if claim_id:
                header_lines.append(f"Claim ID: {claim_id}")
            page_number = metadata.get("page_number")
            if page_number:
                header_lines.append(f"Page: {page_number}")
            source_uri = metadata.get("source_uri")
            if source_uri:
                header_lines.append(f"Source File: {source_uri}")

            content_body = chunk.content.strip() if chunk.content else "[Empty Content]"
            block = "\n".join(header_lines) + f"\nContent:\n{content_body}"
            passages.append(block)

        return "\n\n".join(passages)

    def build_user_message(
        self,
        query: str,
        chunks: Sequence[RetrievalResult],
        as_of_date: Optional[date] = None,
    ) -> str:
        """
        Assemble the user message containing context passages and the user's question.
        """
        context_text = self.format_context(chunks)
        date_line = f" (As-Of Date: {as_of_date.isoformat()})" if as_of_date else ""

        return (
            f"Context Passages:\n"
            f"================\n"
            f"{context_text}\n"
            f"================\n\n"
            f"Question{date_line}:\n"
            f"{query.strip()}\n\n"
            f"Provide a clear, grounded answer with inline citations [chunk_id]."
        )

    def build_messages(
        self,
        query: str,
        chunks: Sequence[RetrievalResult],
        as_of_date: Optional[date] = None,
    ) -> List[ChatMessage]:
        """
        Build the complete sequence of ChatMessages ready for an LLM provider.
        """
        sys_msg = ChatMessage(
            role="system",
            content=self.build_system_prompt(as_of_date=as_of_date),
        )
        user_msg = ChatMessage(
            role="user",
            content=self.build_user_message(query=query, chunks=chunks, as_of_date=as_of_date),
        )
        return [sys_msg, user_msg]
