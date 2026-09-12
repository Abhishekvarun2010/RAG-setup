"""
Unit and integration tests for document-type-aware chunking.

Tests:
1. Chunker protocol conformance.
2. PolicyChunker: structure awareness, section grouping, table preservation, token budgeting.
3. FAQChunker: Question-Answer pairing and question repetition on split answers.
4. ClaimsChunker: section preservation and claim entity metadata inheritance.
5. ChunkingRouter: DocumentType-based routing and execution.
6. End-to-end pipeline on real policy PDF and customer FAQ.
"""
from pathlib import Path
from typing import List

import pytest

from apps.ingestion import (
    BlockType,
    Chunk,
    Chunker,
    ChunkingRouter,
    ClaimsChunker,
    DocumentType,
    FAQChunker,
    LineOfBusiness,
    ParsedDocument,
    ParsedPage,
    PDFParser,
    PolicyChunker,
    SourceType,
    StructuredBlock,
    StructuredDocument,
    StructureExtractor,
)


# =====================================================================
# 1. Protocol Conformance Tests
# =====================================================================

def test_chunker_protocol_conformance():
    """Verify built-in chunkers satisfy the Chunker runtime protocol."""
    assert isinstance(PolicyChunker(), Chunker)
    assert isinstance(FAQChunker(), Chunker)
    assert isinstance(ClaimsChunker(), Chunker)


def test_custom_duck_typed_chunker_conforms():
    """Verify structural subtyping works for custom chunkers."""
    class CustomChunker:
        def chunk(self, document: StructuredDocument, **kwargs) -> List[Chunk]:
            return []

    assert isinstance(CustomChunker(), Chunker)


# =====================================================================
# 2. PolicyChunker Tests
# =====================================================================

def test_policy_chunker_keeps_heading_paragraph_and_table_together():
    """
    Verify that a policy section heading, explanatory paragraph, and coverage table
    remain unified as one logical chunk.
    """
    blocks = [
        StructuredBlock(
            block_type=BlockType.HEADING,
            content="Collision Coverage",
            page_number=2,
        ),
        StructuredBlock(
            block_type=BlockType.PARAGRAPH,
            content="We will pay for direct, accidental loss or damage to your covered auto.",
            page_number=2,
        ),
        StructuredBlock(
            block_type=BlockType.PARAGRAPH,
            content="Subject to the deductible specified in your coverage schedule.",
            page_number=2,
        ),
        StructuredBlock(
            block_type=BlockType.TABLE,
            content="| Coverage | Limit | Deductible |\n| --- | --- | --- |\n| Collision | $50,000 | $1,000 |",
            page_number=2,
            metadata={
                "headers": ["Coverage", "Limit", "Deductible"],
                "rows": [["Collision", "$50,000", "$1,000"]],
            },
        ),
    ]

    doc = StructuredDocument(
        document_id="DOC-POL-001",
        blocks=blocks,
        metadata={
            "document_type": DocumentType.POLICY_CONTRACT,
            "source_type": SourceType.PDF,
            "policy_id": "MOT-0000001",
            "line_of_business": LineOfBusiness.PERSONAL_AUTO,
        },
    )

    chunker = PolicyChunker(max_tokens=500)
    chunks = chunker.chunk(doc)

    assert len(chunks) == 1
    c = chunks[0]
    assert c.document_id == "DOC-POL-001"
    assert c.document_type == DocumentType.POLICY_CONTRACT
    assert c.policy_id == "MOT-0000001"
    assert c.line_of_business == LineOfBusiness.PERSONAL_AUTO
    assert c.section == "Collision Coverage"
    assert c.page_number == 2
    assert "### Collision Coverage" in c.content
    assert "We will pay for direct" in c.content
    assert "| Collision | $50,000 | $1,000 |" in c.content


def test_policy_chunker_splits_oversized_sections():
    """
    Verify that when a policy section exceeds max_tokens, it is split cleanly
    with section header continuation.
    """
    # Create a long section with multiple paragraphs
    blocks = [
        StructuredBlock(block_type=BlockType.HEADING, content="General Conditions", page_number=1),
        StructuredBlock(block_type=BlockType.PARAGRAPH, content="Paragraph A: " + "Long terms condition text. " * 30, page_number=1),
        StructuredBlock(block_type=BlockType.PARAGRAPH, content="Paragraph B: " + "More policy terms text. " * 30, page_number=1),
    ]

    doc = StructuredDocument(document_id="DOC-LONG", blocks=blocks)
    # Set small max_tokens to force splitting
    chunker = PolicyChunker(max_tokens=50)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 2
    assert chunks[0].section == "General Conditions"
    assert "General Conditions" in chunks[0].content
    assert "General Conditions" in chunks[1].content


# =====================================================================
# 3. FAQChunker Tests
# =====================================================================

def test_faq_chunker_pairs_question_and_answer():
    """Verify FAQ questions and answers are packaged together into standalone chunks."""
    blocks = [
        StructuredBlock(block_type=BlockType.HEADING, content="What is my collision deductible?", page_number=1),
        StructuredBlock(block_type=BlockType.PARAGRAPH, content="Your collision deductible is $1,000 per occurrence.", page_number=1),
        StructuredBlock(block_type=BlockType.HEADING, content="How do I file a claim?", page_number=1),
        StructuredBlock(block_type=BlockType.PARAGRAPH, content="Contact your agent or call our 24/7 hotline.", page_number=1),
    ]

    doc = StructuredDocument(
        document_id="DOC-FAQ-01",
        blocks=blocks,
        metadata={"document_type": DocumentType.CUSTOMER_FAQ, "source_type": SourceType.KB},
    )

    chunker = FAQChunker()
    chunks = chunker.chunk(doc)

    assert len(chunks) == 2

    # Chunk 1
    assert chunks[0].section == "What is my collision deductible?"
    assert "Question: What is my collision deductible?" in chunks[0].content
    assert "Answer:\nYour collision deductible is $1,000 per occurrence." in chunks[0].content

    # Chunk 2
    assert chunks[1].section == "How do I file a claim?"
    assert "Question: How do I file a claim?" in chunks[1].content
    assert "Answer:\nContact your agent or call our 24/7 hotline." in chunks[1].content


def test_faq_chunker_repeats_question_on_split_answers():
    """
    Verify that if an answer exceeds the token budget, the Question is repeated
    in each split chunk so every chunk is self-contained.
    """
    blocks = [
        StructuredBlock(block_type=BlockType.HEADING, content="What is covered under Comprehensive?", page_number=1),
        StructuredBlock(block_type=BlockType.PARAGRAPH, content="Part 1: " + "Covered loss description. " * 25, page_number=1),
        StructuredBlock(block_type=BlockType.PARAGRAPH, content="Part 2: " + "Additional perils and exclusions. " * 25, page_number=1),
    ]

    doc = StructuredDocument(document_id="DOC-FAQ-SPLIT", blocks=blocks)
    # Small max_tokens to force splitting the answer
    chunker = FAQChunker(max_tokens=60)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 2
    # Question must be repeated in every chunk
    for c in chunks:
        assert "Question: What is covered under Comprehensive?" in c.content
        assert c.section == "What is covered under Comprehensive?"


# =====================================================================
# 4. ClaimsChunker Tests
# =====================================================================

def test_claims_chunker_preserves_claim_sections_and_metadata():
    """Verify ClaimsChunker groups logical claim sections and preserves claim_id."""
    blocks = [
        StructuredBlock(block_type=BlockType.HEADING, content="Claim & Assignment", page_number=1),
        StructuredBlock(block_type=BlockType.PARAGRAPH, content="Claim Number: C-1058\nPolicy: COM-0000101", page_number=1),
        StructuredBlock(block_type=BlockType.HEADING, content="Description of Loss", page_number=1),
        StructuredBlock(block_type=BlockType.PARAGRAPH, content="Burglary reported at commercial premises on 21/08/2024.", page_number=1),
        StructuredBlock(block_type=BlockType.HEADING, content="Recommendation", page_number=1),
        StructuredBlock(block_type=BlockType.PARAGRAPH, content="Denial recommended due to lack of forcible entry endorsement.", page_number=1),
    ]

    doc = StructuredDocument(
        document_id="DOC-C-1058-ADJ",
        blocks=blocks,
        metadata={
            "document_type": DocumentType.ADJUSTER_REPORT,
            "source_type": SourceType.PDF,
            "claim_id": "C-1058",
            "policy_id": "COM-0000101",
        },
    )

    chunker = ClaimsChunker()
    chunks = chunker.chunk(doc)

    assert len(chunks) == 3
    assert chunks[0].section == "Claim & Assignment"
    assert chunks[0].claim_id == "C-1058"
    assert chunks[1].section == "Description of Loss"
    assert chunks[1].claim_id == "C-1058"
    assert chunks[2].section == "Recommendation"
    assert chunks[2].claim_id == "C-1058"


# =====================================================================
# 5. ChunkingRouter Tests
# =====================================================================

def test_chunking_router_dispatches_by_document_type():
    """Verify ChunkingRouter dispatches to PolicyChunker, FAQChunker, and ClaimsChunker."""
    router = ChunkingRouter()

    # Policy
    assert isinstance(router.get_chunker(DocumentType.POLICY_CONTRACT), PolicyChunker)
    assert isinstance(router.get_chunker(DocumentType.POLICY_DECLARATIONS), PolicyChunker)
    assert isinstance(router.get_chunker(DocumentType.POLICY_ENDORSEMENTS), PolicyChunker)
    assert isinstance(router.get_chunker(DocumentType.POLICY_SCHEDULE), PolicyChunker)

    # FAQ
    assert isinstance(router.get_chunker(DocumentType.CUSTOMER_FAQ), FAQChunker)

    # Claims
    assert isinstance(router.get_chunker(DocumentType.ADJUSTER_REPORT), ClaimsChunker)
    assert isinstance(router.get_chunker(DocumentType.FNOL), ClaimsChunker)
    assert isinstance(router.get_chunker(DocumentType.SETTLEMENT_LETTER), ClaimsChunker)
    assert isinstance(router.get_chunker(DocumentType.DENIAL_LETTER), ClaimsChunker)


# =====================================================================
# 6. Real Policy PDF End-to-End Pipeline Test
# =====================================================================

def test_real_policy_pdf_end_to_end_pipeline():
    """
    Test the full ingestion pipeline on an actual policy PDF from the corpus:
    PDFParser -> StructureExtractor -> ChunkingRouter -> List[Chunk]
    """
    pdf_path = Path("insurance-support-agent/data/raw/docs/policy/MOT-0000001-declarations.pdf")
    if not pdf_path.exists():
        pdf_path = Path("data/raw/docs/policy/MOT-0000001-declarations.pdf")

    assert pdf_path.exists(), f"Sample policy PDF not found: {pdf_path}"

    # 1. Parse
    parser = PDFParser()
    parsed_doc = parser.parse(pdf_path)

    # 2. Extract Structure
    extractor = StructureExtractor()
    struct_doc = extractor.extract(parsed_doc)

    # 3. Route and Chunk
    router = ChunkingRouter()
    chunks = router.route_and_chunk(struct_doc)

    # Validate output
    assert len(chunks) >= 3

    # Check chunk attributes
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.document_id == "DOC-MOT-0000001-DEC"
        assert c.document_type == DocumentType.POLICY_DECLARATIONS
        assert c.policy_id == "MOT-0000001"
        assert c.line_of_business == LineOfBusiness.PERSONAL_AUTO
        assert c.page_number == 1
        assert c.section is not None
        assert len(c.content.strip()) > 10

    # Verify coverage table is contained in the Limits of Insurance chunk
    limits_chunks = [c for c in chunks if c.section == "Limits of Insurance"]
    assert len(limits_chunks) == 1
    assert "| Coverage | Limit |" in limits_chunks[0].content
    assert "Bodily injury" in limits_chunks[0].content


# =====================================================================
# 7. Customer FAQ End-to-End Pipeline Test
# =====================================================================

def test_customer_faq_end_to_end_pipeline():
    """
    Test ingestion and chunking on the actual Customer FAQ file from the corpus:
    ParsedDocument -> StructureExtractor -> ChunkingRouter -> List[Chunk]
    """
    faq_path = Path("insurance-support-agent/data/raw/docs/kb/customer-faq.md")
    if not faq_path.exists():
        faq_path = Path("data/raw/docs/kb/customer-faq.md")

    assert faq_path.exists(), f"FAQ file not found: {faq_path}"

    content = faq_path.read_text(encoding="utf-8")
    parsed_doc = ParsedDocument(
        document_id="DOC-CUSTOMER-FAQ",
        document_type=DocumentType.CUSTOMER_FAQ,
        source_type=SourceType.MARKDOWN,
        source_uri=str(faq_path),
        pages=[ParsedPage(page_number=1, content=content)],
    )

    extractor = StructureExtractor()
    struct_doc = extractor.extract(parsed_doc)

    router = ChunkingRouter()
    chunks = router.route_and_chunk(struct_doc)

    assert len(chunks) == 4
    for c in chunks:
        assert c.document_type == DocumentType.CUSTOMER_FAQ
        assert c.source_type == SourceType.MARKDOWN
        assert c.content.startswith("Question:")
        assert "Answer:" in c.content
