"""
Unit tests for PromptBuilder and context formatting.
"""
from datetime import date
import pytest

from apps.agent.prompt import DEFAULT_SYSTEM_PROMPT, PromptBuilder
from apps.retrieval.models import RetrievalMethod, RetrievalResult


def test_build_system_prompt_default():
    builder = PromptBuilder()
    sys_prompt = builder.build_system_prompt()
    assert "STRICT GROUNDING & ANTI-HALLUCINATION RULES" in sys_prompt
    assert "CITATION RULES" in sys_prompt
    assert "[chunk_id]" in sys_prompt


def test_build_system_prompt_with_as_of_date():
    builder = PromptBuilder()
    target_date = date(2024, 3, 15)
    sys_prompt = builder.build_system_prompt(as_of_date=target_date)
    assert "TEMPORAL POINT-IN-TIME INSTRUCTION" in sys_prompt
    assert "2024-03-15" in sys_prompt


def test_format_context_empty():
    builder = PromptBuilder()
    formatted = builder.format_context([])
    assert "No relevant context passages available." in formatted


def test_format_context_multiple_chunks():
    builder = PromptBuilder()
    chunks = [
        RetrievalResult(
            chunk_id="DOC-C-1000-ESTIMATE#c3",
            content="| Net claim payable | €22,950.00 |",
            score=0.95,
            retrieval_method=RetrievalMethod.RERANKED,
            metadata={
                "document_id": "DOC-C-1000-ESTIMATE",
                "document_type": "estimate",
                "section": "Line Items",
                "claim_id": "C-1000",
                "policy_id": "COM-0000077",
            },
        ),
        RetrievalResult(
            chunk_id="DOC-C-1000-ADJ#c2",
            content="Cause of loss: slip-and-fall",
            score=0.88,
            retrieval_method=RetrievalMethod.RERANKED,
            metadata={
                "document_id": "DOC-C-1000-ADJ",
                "document_type": "adjuster_report",
                "section": "Claim & Assignment",
                "claim_id": "C-1000",
            },
        ),
    ]

    context_str = builder.format_context(chunks)
    assert "--- Passage 1 ---" in context_str
    assert "Chunk ID: DOC-C-1000-ESTIMATE#c3" in context_str
    assert "Document ID: DOC-C-1000-ESTIMATE (Type: estimate)" in context_str
    assert "Section: Line Items" in context_str
    assert "Claim ID: C-1000" in context_str
    assert "Policy ID: COM-0000077" in context_str
    assert "| Net claim payable | €22,950.00 |" in context_str

    assert "--- Passage 2 ---" in context_str
    assert "Chunk ID: DOC-C-1000-ADJ#c2" in context_str
    assert "Cause of loss: slip-and-fall" in context_str


def test_build_messages_structure():
    builder = PromptBuilder()
    chunk = RetrievalResult(
        chunk_id="TEST#c1",
        content="Test coverage details",
        score=0.8,
        retrieval_method=RetrievalMethod.HYBRID_RRF,
        metadata={"document_id": "TEST-DOC"},
    )
    messages = builder.build_messages("What is covered?", [chunk], as_of_date=date(2025, 1, 1))

    assert len(messages) == 2
    assert messages[0].role == "system"
    assert "2025-01-01" in messages[0].content
    assert messages[1].role == "user"
    assert "TEST#c1" in messages[1].content
    assert "What is covered?" in messages[1].content
