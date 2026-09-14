"""
Live Demonstration: End-to-End RAG Question-Answering with Qwen3 8B.

Demonstrates:
1. Scenario 1: Financial settlement question on Claim C-1000 (deductibles and net payable).
2. Scenario 2: Factual adjuster assignment and cause of loss question.
3. Scenario 3: Temporal / Point-in-Time policy retrieval (as_of_date).
4. Scenario 4: Security Access Control enforcement (unauthorized user prevented from accessing claim data).
"""
import json
from datetime import date
from pathlib import Path
import sys
import time

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from apps.agent import (
    OllamaLLM,
    PromptBuilder,
    QARequest,
    RAGQuestionAnsweringFlow,
)
from apps.auth import (
    FieldRedactor,
    JWTValidator,
    SecureRetriever,
    SecurityContext,
)
from apps.ingestion.embeddings import OllamaEmbeddingProvider
from apps.ingestion.indexing.opensearch import OpenSearchIndexer
from apps.ingestion.models import AccessLevel
from apps.retrieval import CrossEncoderReranker, HybridRetriever, RetrievalFilters


def build_pipeline() -> RAGQuestionAnsweringFlow:
    """Instantiate the full production-grade RAG pipeline."""
    print("Initializing components...")
    
    # Storage & Embedding layer
    indexer = OpenSearchIndexer(
        endpoint="http://localhost:9200",
        index_name="insurance_documents",
    )
    embedder = OllamaEmbeddingProvider(
        model_name="bge-m3",
        base_url="http://localhost:11434",
    )
    reranker = CrossEncoderReranker(
        model_name="BAAI/bge-reranker-v2-m3",
    )
    
    # Hybrid Retriever (BM25 + Dense k-NN + RRF + CrossEncoder)
    retriever = HybridRetriever(
        vector_store=indexer,
        embedding_provider=embedder,
        reranker=reranker,
    )
    
    # Security Layer (PII Redaction + Mandatory Auth Filters)
    redactor = FieldRedactor()
    secure_retriever = SecureRetriever(
        retriever=retriever,
        redactor=redactor,
    )
    
    # LLM Generation Layer (Qwen3 8B via Ollama)
    llm = OllamaLLM(
        model_name="qwen3:8b",
        base_url="http://localhost:11434",
        timeout=120.0,
    )
    
    # JWT Validator for token-based authentication
    jwt_validator = JWTValidator(secret_key="production_secret_key_insurance_agent_32b")
    
    return RAGQuestionAnsweringFlow(
        retriever=secure_retriever,
        llm=llm,
        prompt_builder=PromptBuilder(),
        jwt_validator=jwt_validator,
    )


output_lines = []

def log(msg: str = ""):
    print(msg)
    output_lines.append(msg)

def print_response(title: str, response):
    log("\n" + "=" * 80)
    log(f"SCENARIO: {title}")
    log("=" * 80)
    log(f"User Query      : {response.query}")
    log(f"Authorized User : {response.security_user_id}")
    log(f"Model           : {response.model}")
    log(f"Total Latency   : {response.latency_seconds:.2f}s")
    log(f"Token Usage     : {response.usage}")
    
    if response.thinking:
        log("\n--- [Qwen3 8B Reasoning / Thinking Trace] ---")
        log(response.thinking[:400] + ("..." if len(response.thinking) > 400 else ""))
    
    log("\n--- [Synthesized Grounded Answer] ---")
    log(response.answer)

    if response.citations and response.answer_with_markdown_links != response.answer:
        log("\n--- [Answer with Clickable File Links] ---")
        log(response.answer_with_markdown_links)
    
    log("\n--- [Authoritative Citations & Source Links] ---")
    if response.citations:
        for idx, cit in enumerate(response.citations, 1):
            log(f"  [{idx}] Chunk ID    : {cit.chunk_id}")
            log(f"      Document ID : {cit.document_id} | Type: {cit.document_type} | Section: {cit.section}")
            if cit.page_number:
                log(f"      Page Number : {cit.page_number}")
            if cit.source_uri:
                log(f"      File Path   : {cit.source_uri}")
            if cit.file_link:
                log(f"      Direct Link : {cit.file_link}")
            if cit.markdown_link:
                log(f"      Markdown    : {cit.markdown_link}")
            if cit.snippet:
                log(f"      Snippet     : {cit.snippet[:100]}...")
    else:
        log("  (No citations extracted)")

    log("\n--- [Retrieved Context Passages] ---")
    for idx, c in enumerate(response.retrieved_chunks, 1):
        sec = c.metadata.get("section", "N/A")
        log(f"  [{idx}] {c.chunk_id} (Score: {c.score:.4f}, Method: {c.retrieval_method.value}, Section: '{sec}')")


def main():
    flow = build_pipeline()
    print("✓ Pipeline successfully initialized.\n")

    # -------------------------------------------------------------
    # Scenario 1: Financial Inquiry on Claim C-1000
    # -------------------------------------------------------------
    adjuster_context = SecurityContext(
        user_id="ADJ-104",
        roles=[AccessLevel.ADJUSTER],
        allowed_claim_ids=["C-1000"],
    )
    
    resp1 = flow.answer(
        query="What is the net claim amount payable after the deductible for claim C-1000?",
        security_context=adjuster_context,
        top_k=3,
    )
    print_response("Scenario 1 - Claim C-1000 Financial Settlement", resp1)

    # -------------------------------------------------------------
    # Scenario 2: Cause of Loss & Assignment for Claim C-1000
    # -------------------------------------------------------------
    resp2 = flow.answer(
        query="What was the cause of loss, the insured party, and who was the assigned adjuster for claim C-1000?",
        security_context=adjuster_context,
        top_k=3,
    )
    print_response("Scenario 2 - Claim C-1000 Assignment & Incident Details", resp2)

    # -------------------------------------------------------------
    # Scenario 3: Point-in-Time Policy Term Inquiry (Version-Aware)
    # -------------------------------------------------------------
    agent_context = SecurityContext(
        user_id="AGT-007",
        roles=[AccessLevel.AGENT],
        allowed_policy_ids=["COM-0000077"],
    )
    
    resp3 = flow.answer(
        query="What policy contract version, coverage, and deductible applied to commercial policy COM-0000077 when the accident occurred in March 2024?",
        security_context=agent_context,
        as_of_date=date(2024, 3, 15),
        top_k=3,
    )
    print_response("Scenario 3 - Version-Aware Historical Policy Inquiry (as_of_date=2024-03-15)", resp3)

    # -------------------------------------------------------------
    # Scenario 4: Security Access Control Enforcement (Unauthorized User)
    # -------------------------------------------------------------
    unauthorized_context = SecurityContext(
        user_id="PH-UNAUTHORIZED-999",
        roles=[AccessLevel.POLICYHOLDER],
        allowed_policy_ids=["COM-OTHER-999"],
        allowed_claim_ids=["C-OTHER-999"],
    )
    
    resp4 = flow.answer(
        query="What is the net payable settlement amount and adjuster name for claim C-1000?",
        security_context=unauthorized_context,
        top_k=3,
    )
    print_response("Scenario 4 - Security Enforcement (Unauthorized User Inquiry on C-1000)", resp4)

    out_file = BASE_DIR / "rag_qa_demonstration.txt"
    out_file.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"\n✓ Saved full demonstration output to {out_file}")


if __name__ == "__main__":
    main()
