"""
Unit and integration tests for the Structure Extraction Layer.

Tests:
1. BlockType enumeration values and string behavior.
2. StructuredBlock data model validation, constraints, and metadata.
3. StructuredDocument data model, filtering helpers, and page tracking.
4. StructureExtractor on actual Strata corpus policy PDF (headings, paragraphs, tables, page numbers).
5. Table extraction with headers and rows preserved on policy declaration and schedule PDFs.
6. List extraction (bulleted and numbered) on PDFs and text fallbacks.
7. End-to-end pipeline: PDFParser -> ParsedDocument -> StructureExtractor.
8. Multi-page document handling and page number preservation.
9. Edge cases: blank/scanned PDFs, fallback extraction, invalid files.
"""
from pathlib import Path
from typing import List

import pymupdf
import pytest
from pydantic import ValidationError

from apps.ingestion import (
    BlockType,
    ParsedDocument,
    ParsedPage,
    PDFParser,
    StructuredBlock,
    StructuredDocument,
    StructureExtractor,
)


# =====================================================================
# 1. BlockType Enumeration Tests
# =====================================================================

def test_block_type_enum_members():
    """Verify that BlockType supports heading, paragraph, table, and list."""
    assert BlockType.HEADING == "heading"
    assert BlockType.PARAGRAPH == "paragraph"
    assert BlockType.TABLE == "table"
    assert BlockType.LIST == "list"
    assert len(BlockType) == 4


def test_block_type_string_behavior():
    """Verify BlockType works with string equality and JSON serialization."""
    assert BlockType("heading") == BlockType.HEADING
    assert BlockType("paragraph") == BlockType.PARAGRAPH
    assert BlockType("table") == BlockType.TABLE
    assert BlockType("list") == BlockType.LIST
    with pytest.raises(ValueError):
        BlockType("invalid_type")


# =====================================================================
# 2. StructuredBlock Model Tests
# =====================================================================

def test_valid_structured_block_creation():
    """Verify creating a valid StructuredBlock with required fields."""
    block = StructuredBlock(
        block_type=BlockType.HEADING,
        content="Policy Declarations",
        page_number=1,
        metadata={"font_size": 14.0, "is_bold": True},
    )
    assert block.block_type == BlockType.HEADING
    assert block.content == "Policy Declarations"
    assert block.page_number == 1
    assert block.metadata["font_size"] == 14.0
    assert block.metadata["is_bold"] is True


def test_structured_block_page_number_must_be_ge_1():
    """Verify page_number validation rejects values < 1."""
    with pytest.raises(ValidationError):
        StructuredBlock(
            block_type=BlockType.PARAGRAPH,
            content="Sample text",
            page_number=0,
        )

    with pytest.raises(ValidationError):
        StructuredBlock(
            block_type=BlockType.PARAGRAPH,
            content="Sample text",
            page_number=-1,
        )


def test_structured_block_forbids_empty_content():
    """Verify min_length=1 validation on block content."""
    with pytest.raises(ValidationError):
        StructuredBlock(
            block_type=BlockType.PARAGRAPH,
            content="",
            page_number=1,
        )


def test_structured_block_forbids_extra_fields():
    """Verify extra='forbid' prevents unexpected fields."""
    with pytest.raises(ValidationError):
        StructuredBlock(
            block_type=BlockType.PARAGRAPH,
            content="Valid content",
            page_number=1,
            unrecognized_field="illegal",  # type: ignore
        )


# =====================================================================
# 3. StructuredDocument Model Tests
# =====================================================================

def test_valid_structured_document_creation():
    """Verify creating a StructuredDocument and its helper filtering properties."""
    blocks = [
        StructuredBlock(block_type=BlockType.HEADING, content="Declarations", page_number=1),
        StructuredBlock(block_type=BlockType.PARAGRAPH, content="Named Insured: Robin Hardy", page_number=1),
        StructuredBlock(block_type=BlockType.HEADING, content="Coverage", page_number=2),
        StructuredBlock(block_type=BlockType.PARAGRAPH, content="We will pay covered losses", page_number=2),
        StructuredBlock(block_type=BlockType.TABLE, content="Limit Table", page_number=2),
        StructuredBlock(block_type=BlockType.LIST, content="- Item 1\n- Item 2", page_number=2),
    ]

    doc = StructuredDocument(
        document_id="DOC-TEST-001",
        blocks=blocks,
        metadata={"total_pages": 2},
    )

    assert doc.document_id == "DOC-TEST-001"
    assert len(doc.blocks) == 6
    assert doc.page_count == 2
    assert len(doc.headings) == 2
    assert doc.headings[0].content == "Declarations"
    assert doc.headings[1].content == "Coverage"
    assert len(doc.paragraphs) == 2
    assert len(doc.tables) == 1
    assert len(doc.lists) == 1

    page_1_blocks = doc.get_blocks_by_page(1)
    assert len(page_1_blocks) == 2
    assert all(b.page_number == 1 for b in page_1_blocks)

    page_2_blocks = doc.get_blocks_by_page(2)
    assert len(page_2_blocks) == 4
    assert all(b.page_number == 2 for b in page_2_blocks)


def test_structured_document_page_count_fallback():
    """Verify page_count falls back to max page_number in blocks if total_pages not in metadata."""
    blocks = [
        StructuredBlock(block_type=BlockType.PARAGRAPH, content="Page 3 text", page_number=3),
    ]
    doc = StructuredDocument(document_id="DOC-PAGE-3", blocks=blocks)
    assert doc.page_count == 3


def test_structured_document_empty_blocks_page_count():
    """Verify page_count is 0 when no blocks and no metadata total_pages."""
    doc = StructuredDocument(document_id="DOC-EMPTY")
    assert doc.page_count == 0
    assert doc.blocks == []
    assert doc.headings == []


# =====================================================================
# 4. StructureExtractor on Real Policy PDF (Headings, Paragraphs, Page Numbers)
# =====================================================================

def test_structure_extractor_on_real_policy_pdf():
    """
    Verify StructureExtractor extracts headings, paragraphs, and tables from an actual
    policy PDF in the Strata corpus, retaining page numbers and metadata.
    """
    policy_pdf = Path("insurance-support-agent/data/raw/docs/policy/MOT-0000001-declarations.pdf")
    if not policy_pdf.exists():
        policy_pdf = Path("data/raw/docs/policy/MOT-0000001-declarations.pdf")

    assert policy_pdf.exists(), f"Sample policy PDF not found at {policy_pdf}"

    extractor = StructureExtractor()
    structured_doc = extractor.extract(policy_pdf)

    # 1. Document attributes
    assert isinstance(structured_doc, StructuredDocument)
    assert structured_doc.document_id == f"DOC-{policy_pdf.stem.upper()}"
    assert structured_doc.page_count == 1
    assert len(structured_doc.blocks) > 10

    # 2. Page number preservation: all blocks from page 1 must retain page_number == 1
    for block in structured_doc.blocks:
        assert block.page_number == 1
        assert "bbox" in block.metadata

    # 3. Heading extraction: must identify major bold/prominent section titles
    heading_texts = [b.content for b in structured_doc.headings]
    assert any("Meridian Mutual" in h for h in heading_texts)
    assert any("Policy Declarations" in h for h in heading_texts)
    assert any("Limits of Insurance" in h for h in heading_texts)
    assert any("Endorsements" in h for h in heading_texts)

    # 4. Paragraph extraction: body and field content classified as paragraph
    paragraph_texts = [b.content for b in structured_doc.paragraphs]
    assert any("Robin Hardy" in p for p in paragraph_texts)
    assert any("Peugeot 208" in p for p in paragraph_texts)
    assert any("The full policy contract" in p for p in paragraph_texts)


# =====================================================================
# 5. Table Extraction & Structure Preservation Tests
# =====================================================================

def test_table_extraction_on_policy_declarations_pdf():
    """
    Verify table extraction preserves table structure, headers, rows,
    and formats content into clean markdown without flattening.
    """
    policy_pdf = Path("insurance-support-agent/data/raw/docs/policy/MOT-0000001-declarations.pdf")
    if not policy_pdf.exists():
        policy_pdf = Path("data/raw/docs/policy/MOT-0000001-declarations.pdf")

    extractor = StructureExtractor()
    structured_doc = extractor.extract(policy_pdf)

    assert len(structured_doc.tables) == 1
    table_block = structured_doc.tables[0]

    # Verify table attributes
    assert table_block.block_type == BlockType.TABLE
    assert table_block.page_number == 1
    assert "| Coverage | Limit |" in table_block.content
    assert "| Bodily injury — per person | €100,000.00 |" in table_block.content

    # Verify structured metadata (headers, rows, counts)
    meta = table_block.metadata
    assert meta["headers"] == ["Coverage", "Limit"]
    assert meta["col_count"] == 2
    assert meta["row_count"] == 4
    assert len(meta["rows"]) == 3
    assert meta["rows"][0] == ["Bodily injury — per person", "€100,000.00"]
    assert meta["rows"][1] == ["Bodily injury — per accident", "€300,000.00"]
    assert meta["rows"][2] == ["Property damage", "€100,000.00"]


def test_table_extraction_on_policy_schedule_pdf():
    """
    Verify table extraction on a 3-column coverage schedule PDF.
    Preserves headers: ['Coverage', 'Limit of insurance', 'Deductible'] and all rows.
    """
    schedule_pdf = Path("insurance-support-agent/data/raw/docs/policy/MOT-0000001-schedule.pdf")
    if not schedule_pdf.exists():
        schedule_pdf = Path("data/raw/docs/policy/MOT-0000001-schedule.pdf")

    extractor = StructureExtractor()
    structured_doc = extractor.extract(schedule_pdf)

    assert len(structured_doc.tables) == 1
    table_block = structured_doc.tables[0]

    assert table_block.page_number == 1
    assert table_block.metadata["headers"] == ["Coverage", "Limit of insurance", "Deductible"]
    assert table_block.metadata["col_count"] == 3
    assert len(table_block.metadata["rows"]) == 3

    # Check deductible values preserved
    first_row = table_block.metadata["rows"][0]
    assert "Bodily injury — per person" in first_row[0]
    assert "€100,000.00" in first_row[1]
    assert "€1,000.00" in first_row[2]


# =====================================================================
# 6. List Extraction Tests (Bulleted & Numbered)
# =====================================================================

def test_list_extraction_from_pdf(tmp_path):
    """
    Verify list extraction detects bullet points and numbered lists
    and preserves each list as one logical StructuredBlock of type LIST.
    """
    pdf_path = tmp_path / "policy_with_lists.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)

    page.insert_text((50, 60), "Covered Perils", fontsize=16, fontname="helv")
    # Bulleted list
    page.insert_text((50, 100), "• Fire and lightning\n• Theft and vandalism\n• Windstorm and hail\n• Flood damage", fontsize=10, fontname="helv")

    page.insert_text((50, 200), "Claims Procedures", fontsize=16, fontname="helv")
    # Numbered list
    page.insert_text((50, 240), "1. Notify the insurer immediately\n2. Submit a formal loss estimate\n3. Cooperate with the adjuster inspection", fontsize=10, fontname="helv")

    doc.save(str(pdf_path))
    doc.close()

    extractor = StructureExtractor()
    structured_doc = extractor.extract(pdf_path)

    assert len(structured_doc.lists) == 2
    bullet_list = structured_doc.lists[0]
    numbered_list = structured_doc.lists[1]

    # Verify bullet list block
    assert bullet_list.block_type == BlockType.LIST
    assert bullet_list.page_number == 1
    assert "Fire and lightning" in bullet_list.content
    assert "Flood damage" in bullet_list.content
    assert bullet_list.metadata["item_count"] == 4

    # Verify numbered list block
    assert numbered_list.block_type == BlockType.LIST
    assert numbered_list.page_number == 1
    assert "1. Notify the insurer" in numbered_list.content
    assert "3. Cooperate with the adjuster" in numbered_list.content
    assert numbered_list.metadata["item_count"] == 3


def test_markdown_fallback_table_and_list_extraction():
    """
    Verify fallback structure extraction correctly identifies Markdown tables
    and lists from an in-memory text ParsedDocument.
    """
    content = (
        "# Coverage Overview\n\n"
        "Here are the coverage options:\n\n"
        "| Coverage | Limit | Deductible |\n"
        "| --- | --- | --- |\n"
        "| Collision | $50,000 | $1,000 |\n"
        "| Comprehensive | $25,000 | $500 |\n\n"
        "Exclusions include:\n\n"
        "- Intentional damage\n"
        "- Wear and tear\n"
        "- War and nuclear hazard"
    )

    parsed_doc = ParsedDocument(
        document_id="DOC-KB-MD",
        document_type="customer_faq",
        source_type="kb",
        pages=[ParsedPage(page_number=1, content=content)],
    )

    extractor = StructureExtractor()
    structured_doc = extractor.extract(parsed_doc)

    assert len(structured_doc.headings) == 1
    assert structured_doc.headings[0].content == "Coverage Overview"

    assert len(structured_doc.tables) == 1
    table = structured_doc.tables[0]
    assert table.metadata["headers"] == ["Coverage", "Limit", "Deductible"]
    assert len(table.metadata["rows"]) == 2
    assert table.metadata["rows"][0] == ["Collision", "$50,000", "$1,000"]

    assert len(structured_doc.lists) == 1
    lst = structured_doc.lists[0]
    assert lst.block_type == BlockType.LIST
    assert "Wear and tear" in lst.content
    assert lst.metadata["item_count"] == 3


# =====================================================================
# 7. Pipeline Integration: PDFParser -> ParsedDocument -> StructureExtractor
# =====================================================================

def test_pipeline_parsed_document_to_structure_extractor():
    """
    Verify the complete ingestion pipeline flow:
    PDFParser -> ParsedDocument -> StructureExtractor -> StructuredDocument
    """
    policy_pdf = Path("insurance-support-agent/data/raw/docs/policy/MOT-0000001-declarations.pdf")
    if not policy_pdf.exists():
        policy_pdf = Path("data/raw/docs/policy/MOT-0000001-declarations.pdf")

    # Step 1: Parse with PDFParser
    parser = PDFParser()
    parsed_doc = parser.parse(policy_pdf)
    assert isinstance(parsed_doc, ParsedDocument)
    assert parsed_doc.document_id == "DOC-MOT-0000001-DEC"
    assert parsed_doc.has_text is True

    # Step 2: Extract structure
    extractor = StructureExtractor()
    structured_doc = extractor.extract(parsed_doc)

    assert isinstance(structured_doc, StructuredDocument)
    assert structured_doc.document_id == "DOC-MOT-0000001-DEC"
    assert structured_doc.metadata.get("parser") == "PDFParser"
    assert structured_doc.metadata.get("extractor") == "StructureExtractor"
    assert len(structured_doc.headings) >= 3
    assert len(structured_doc.paragraphs) >= 5
    assert len(structured_doc.tables) == 1


# =====================================================================
# 8. Multi-Page Document Handling & Page Preservation
# =====================================================================

def test_multi_page_pdf_structure_extraction(tmp_path):
    """
    Verify that across multiple pages:
    - Each block strictly retains its correct 1-indexed page_number.
    - Headings, paragraphs, and tables on each page are properly extracted and attributed.
    """
    pdf_path = tmp_path / "multi_page_policy.pdf"
    doc = pymupdf.open()

    # Page 1: Declarations section
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text((50, 60), "Policy Declarations", fontsize=16, fontname="helv")
    page1.insert_text((50, 100), "Named Insured: Robin Hardy\nPolicy Number: MOT-123456", fontsize=10, fontname="helv")

    # Page 2: Coverage & Exclusions section
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((50, 60), "Coverage and Exclusions", fontsize=16, fontname="helv")
    page2.insert_text((50, 100), "We will pay for covered accidental loss or damage to your vehicle.", fontsize=10, fontname="helv")

    doc.save(str(pdf_path))
    doc.close()

    extractor = StructureExtractor()
    structured_doc = extractor.extract(pdf_path)

    assert structured_doc.page_count == 2
    p1_blocks = structured_doc.get_blocks_by_page(1)
    p2_blocks = structured_doc.get_blocks_by_page(2)

    assert len(p1_blocks) >= 2
    assert len(p2_blocks) >= 2

    # Check page 1
    assert all(b.page_number == 1 for b in p1_blocks)
    p1_headings = [b for b in p1_blocks if b.block_type == BlockType.HEADING]
    assert len(p1_headings) >= 1
    assert "Policy Declarations" in p1_headings[0].content

    # Check page 2
    assert all(b.page_number == 2 for b in p2_blocks)
    p2_headings = [b for b in p2_blocks if b.block_type == BlockType.HEADING]
    assert len(p2_headings) >= 1
    assert "Coverage and Exclusions" in p2_headings[0].content


# =====================================================================
# 9. Edge Cases & Fallback Extraction
# =====================================================================

def test_structure_extractor_empty_scanned_pdf(tmp_path):
    """Verify that a blank scanned PDF produces an empty StructuredDocument without crashing."""
    blank_pdf = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page(width=595, height=842)
    doc.save(str(blank_pdf))
    doc.close()

    extractor = StructureExtractor()
    structured_doc = extractor.extract(blank_pdf)

    assert structured_doc.blocks == []
    assert structured_doc.headings == []
    assert structured_doc.paragraphs == []
    assert structured_doc.tables == []
    assert structured_doc.lists == []
    assert structured_doc.metadata["total_pages"] == 1


def test_structure_extractor_nonexistent_file():
    """Verify FileNotFoundError when target PDF does not exist."""
    extractor = StructureExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.extract("non_existent_policy.pdf")


def test_structure_extractor_non_pdf_file(tmp_path):
    """Verify ValueError when passing a non-PDF file to PDF path extractor."""
    text_file = tmp_path / "policy.txt"
    text_file.write_text("Some text")

    extractor = StructureExtractor()
    with pytest.raises(ValueError, match="File is not a PDF"):
        extractor.extract(text_file)
