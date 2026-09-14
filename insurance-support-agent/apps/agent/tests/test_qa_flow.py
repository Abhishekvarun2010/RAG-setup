"""
Unit tests for RAGQuestionAnsweringFlow.
"""
from datetime import date
from typing import Any, List, Optional
from unittest.mock import MagicMock
import pytest

from apps.agent.llm import LLMProvider
from apps.agent.models import ChatMessage, LLMResponse, QARequest
from apps.agent.qa_flow import RAGQuestionAnsweringFlow
from apps.auth.jwt import JWTValidator
from apps.auth.models import (
    AuthenticationError,
    SecurityContext,
)
from apps.ingestion.models import AccessLevel
from apps.auth.secure_retriever import SecureRetriever
from apps.retrieval.models import RetrievalMethod, RetrievalResult


class MockLLM(LLMProvider):
    def __init__(self, answer: str, thinking: Optional[str] = "Mock thinking"):
        self._answer = answer
        self._thinking = thinking
        self.last_messages: Optional[List[ChatMessage]] = None

    @property
    def model_name(self) -> str:
        return "mock-qwen3"

    def chat(self, messages, temperature=0.0, **kwargs) -> LLMResponse:
        self.last_messages = list(messages)
        return LLMResponse(
            content=self._answer,
            thinking=self._thinking,
            model="mock-qwen3",
            prompt_tokens=100,
            completion_tokens=40,
            duration_seconds=0.25,
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        return self.chat([ChatMessage(role="user", content=prompt)])


@pytest.fixture
def sample_context():
    return SecurityContext(
        user_id="adjuster-42",
        roles=[AccessLevel.ADJUSTER],
        allowed_claim_ids=["C-1000"],
    )


@pytest.fixture
def sample_retrieval_chunks():
    return [
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
                "source_uri": "/data/docs/claim/C-1000-estimate.pdf",
                "page_number": 1,
            },
        ),
        RetrievalResult(
            chunk_id="DOC-C-1000-ADJ#c2",
            content="Cause of loss: slip-and-fall incident on wet floor.",
            score=0.89,
            retrieval_method=RetrievalMethod.RERANKED,
            metadata={
                "document_id": "DOC-C-1000-ADJ",
                "document_type": "adjuster_report",
                "section": "Claim & Assignment",
                "claim_id": "C-1000",
                "source_uri": "/data/docs/claim/C-1000-adjuster.pdf",
                "page_number": 2,
            },
        ),
    ]


def test_qa_flow_successful_answer(sample_context, sample_retrieval_chunks):
    mock_retriever = MagicMock(spec=SecureRetriever)
    mock_retriever.retrieve.return_value = sample_retrieval_chunks

    answer_text = (
        "The net claim payable amount for claim C-1000 is €22,950.00 [DOC-C-1000-ESTIMATE#c3]. "
        "The reported cause of loss was a slip-and-fall incident [DOC-C-1000-ADJ#c2]."
    )
    mock_llm = MockLLM(answer=answer_text, thinking="Calculated net payable after deductible.")

    flow = RAGQuestionAnsweringFlow(
        retriever=mock_retriever,
        llm=mock_llm,
    )

    response = flow.answer(
        query="What is the net claim payable and cause of loss for C-1000?",
        security_context=sample_context,
    )

    assert response.query == "What is the net claim payable and cause of loss for C-1000?"
    assert response.answer == answer_text
    assert response.model == "mock-qwen3"
    assert response.thinking == "Calculated net payable after deductible."
    assert len(response.retrieved_chunks) == 2
    assert response.security_user_id == "adjuster-42"
    assert response.latency_seconds >= 0

    # Verify extracted citations and links
    assert len(response.citations) == 2
    assert response.citations[0].chunk_id == "DOC-C-1000-ESTIMATE#c3"
    assert response.citations[0].document_id == "DOC-C-1000-ESTIMATE"
    assert response.citations[0].document_type == "estimate"
    assert response.citations[0].source_uri == "/data/docs/claim/C-1000-estimate.pdf"
    assert response.citations[0].page_number == 1
    assert response.citations[0].file_link == "file:///data/docs/claim/C-1000-estimate.pdf#page=1"
    assert response.citations[0].markdown_link == "[DOC-C-1000-ESTIMATE (p. 1)](file:///data/docs/claim/C-1000-estimate.pdf#page=1)"

    assert response.citations[1].chunk_id == "DOC-C-1000-ADJ#c2"
    assert response.citations[1].document_id == "DOC-C-1000-ADJ"
    assert response.citations[1].file_link == "file:///data/docs/claim/C-1000-adjuster.pdf#page=2"

    # Verify answer with clickable markdown links
    linked_answer = response.answer_with_markdown_links
    assert "[[DOC-C-1000-ESTIMATE#c3]](file:///data/docs/claim/C-1000-estimate.pdf#page=1)" in linked_answer
    assert "[[DOC-C-1000-ADJ#c2]](file:///data/docs/claim/C-1000-adjuster.pdf#page=2)" in linked_answer


def test_qa_flow_empty_retrieval(sample_context):
    mock_retriever = MagicMock(spec=SecureRetriever)
    mock_retriever.retrieve.return_value = []

    mock_llm = MockLLM(answer="Should not be called")
    flow = RAGQuestionAnsweringFlow(retriever=mock_retriever, llm=mock_llm)

    response = flow.answer(
        query="What is policy XYZ?",
        security_context=sample_context,
    )

    # LLM was never called
    assert mock_llm.last_messages is None
    assert "not have enough information" in response.answer.lower()
    assert response.citations == []
    assert response.retrieved_chunks == []


def test_qa_flow_ignores_hallucinated_citations(sample_context, sample_retrieval_chunks):
    mock_retriever = MagicMock(spec=SecureRetriever)
    mock_retriever.retrieve.return_value = sample_retrieval_chunks

    # Model hallucinates a citation [DOC-NONEXISTENT#c1] along with real [DOC-C-1000-ESTIMATE#c3]
    answer_text = (
        "Payable amount is €22,950.00 [DOC-C-1000-ESTIMATE#c3]. "
        "Here is a fake fact [DOC-NONEXISTENT#c1]."
    )
    mock_llm = MockLLM(answer=answer_text)
    flow = RAGQuestionAnsweringFlow(retriever=mock_retriever, llm=mock_llm)

    response = flow.answer(query="Claim details", security_context=sample_context)

    assert len(response.citations) == 1
    assert response.citations[0].chunk_id == "DOC-C-1000-ESTIMATE#c3"


def test_qa_flow_requires_authentication():
    mock_retriever = MagicMock(spec=SecureRetriever)
    mock_llm = MockLLM(answer="hi")
    flow = RAGQuestionAnsweringFlow(retriever=mock_retriever, llm=mock_llm)

    with pytest.raises(AuthenticationError):
        flow.answer(query="Tell me about claim C-1000")


def test_qa_flow_with_jwt_token(sample_retrieval_chunks):
    mock_retriever = MagicMock(spec=SecureRetriever)
    mock_retriever.retrieve.return_value = sample_retrieval_chunks

    secret = "a_very_secure_test_secret_32_bytes!!"
    validator = JWTValidator(secret_key=secret)
    token = validator.create_token(
        user_id="jwt-user-1",
        roles=["adjuster"],
        claim_ids=["C-1000"],
    )

    mock_llm = MockLLM(answer="Answer [DOC-C-1000-ESTIMATE#c3]")
    flow = RAGQuestionAnsweringFlow(
        retriever=mock_retriever,
        llm=mock_llm,
        jwt_validator=validator,
    )

    resp = flow.answer(query="Query", jwt_token=token)
    assert resp.security_user_id == "jwt-user-1"
    assert len(resp.citations) == 1


def test_answer_request_convenience_method(sample_context, sample_retrieval_chunks):
    mock_retriever = MagicMock(spec=SecureRetriever)
    mock_retriever.retrieve.return_value = sample_retrieval_chunks

    mock_llm = MockLLM(answer="Answer [DOC-C-1000-ESTIMATE#c3]")
    flow = RAGQuestionAnsweringFlow(retriever=mock_retriever, llm=mock_llm)

    req = QARequest(
        query="What is the net amount?",
        security_context=sample_context,
        as_of_date=date(2024, 1, 1),
    )
    resp = flow.answer_request(req)
    assert resp.query == "What is the net amount?"
    assert len(resp.citations) == 1
