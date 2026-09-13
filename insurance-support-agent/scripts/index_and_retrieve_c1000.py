"""
End-to-end demonstration script:
1. Parse DOC-C-1000-ESTIMATE (C-1000-estimate.pdf).
2. Extract structure (tables, sections, paragraphs).
3. Chunk using ClaimsChunker (via ChunkingRouter).
4. Generate 1024-dimensional dense vectors using Ollama BGE-M3.
5. Index chunks and embeddings into OpenSearch 'insurance_documents' index.
6. Retrieve them back via Point Retrieval (by ID) and Semantic k-NN Vector Search.
"""
import json
from pathlib import Path
import sys

# Ensure apps is on path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from apps.ingestion import (
    ChunkingRouter,
    DocumentType,
    OllamaEmbeddingProvider,
    OpenSearchIndexer,
    PDFParser,
    SourceType,
    StructureExtractor,
)


def main():
    print("=" * 80)
    print("END-TO-END PIPELINE: DOC-C-1000-ESTIMATE -> BGE-M3 -> OPENSEARCH -> RETRIEVE")
    print("=" * 80)

    pdf_path = BASE_DIR / "data" / "raw" / "docs" / "claim" / "C-1000-estimate.pdf"
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        sys.exit(1)

    print(f"\n[Step 1] Parsing PDF: {pdf_path.name}")
    parser = PDFParser()
    parsed_doc = parser.parse(
        str(pdf_path),
        document_id="DOC-C-1000-ESTIMATE",
        document_type=DocumentType.ESTIMATE,
        source_type=SourceType.PDF,
        claim_id="C-1000",
    )
    print(f"  ✓ Parsed {len(parsed_doc.pages)} page(s)")

    print("\n[Step 2] Extracting Document Structure (tables, headings, paragraphs)...")
    extractor = StructureExtractor()
    struct_doc = extractor.extract(parsed_doc)
    print(f"  ✓ Extracted {len(struct_doc.blocks)} structured blocks")

    print("\n[Step 3] Routing and Chunking...")
    router = ChunkingRouter()
    chunks = router.route_and_chunk(struct_doc)
    print(f"  ✓ Generated {len(chunks)} logical chunks:")
    for idx, chunk in enumerate(chunks, 1):
        print(f"    - Chunk {idx}: id={chunk.chunk_id} | section='{chunk.section}' | length={len(chunk.content)} chars")

    print("\n[Step 4] Generating BGE-M3 Dense Embeddings via Ollama (1024-dim)...")
    provider = OllamaEmbeddingProvider(model_name="bge-m3")
    vectors = provider.embed_chunks(chunks)
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors), 1):
        preview = [round(v, 4) for v in vector[:4]]
        print(f"    - Vector {idx} for {chunk.chunk_id}: len={len(vector)}, first 4 values={preview}...")

    print("\n[Step 5] Indexing into OpenSearch ('insurance_documents')...")
    indexer = OpenSearchIndexer(endpoint="http://localhost:9200", index_name="insurance_documents")
    indexed_count = indexer.index_batch(chunks, vectors, refresh=True)
    total_in_index = indexer.count()
    print(f"  ✓ Successfully bulk-indexed {indexed_count} documents.")
    print(f"  ✓ Total document count in 'insurance_documents' index: {total_in_index}")

    print("\n[Step 6] Retrieving Documents Back from OpenSearch:")

    # 6a. Direct ID Lookup
    print("\n  --- A. Direct ID Retrieval (Point Lookup) ---")
    test_id = chunks[0].chunk_id
    retrieved = indexer.get_by_id(test_id)
    if retrieved:
        source = retrieved.get("_source", {})
        print(f"  ✓ Found document by ID '{test_id}':")
        print(f"    - Document ID   : {source.get('document_id')}")
        print(f"    - Claim ID      : {source.get('claim_id')}")
        print(f"    - Section       : {source.get('section')}")
        print(f"    - Stored Vector : length {len(source.get('embedding', []))} floats")
        print(f"    - Content Snippet:\n      {source.get('content', '').strip()[:100]}...")
    else:
        print(f"  ✗ Failed to find document '{test_id}'")

    # 6b. Semantic k-NN Vector Search
    print("\n  --- B. Semantic k-NN Vector Search ---")
    queries = [
        "What is the estimated net payable amount and labor cost for claim C-1000?",
        "emergency water extraction and dehumidification drying equipment",
    ]

    for q in queries:
        print(f"\n  Query: \"{q}\"")
        q_vector = provider.embed(q)
        hits = indexer.search_knn(q_vector, k=2)
        print(f"  Top {len(hits)} nearest neighbors:")
        for rank, hit in enumerate(hits, 1):
            score = hit.get("_score")
            doc_source = hit.get("_source", {})
            c_id = hit.get("_id")
            sec = doc_source.get("section")
            snippet = doc_source.get("content", "").replace("\n", " ")[:120]
            print(f"    [{rank}] Score: {score:.4f} | ID: {c_id} | Section: '{sec}'")
            print(f"        Preview: {snippet}...")

    print("\n" + "=" * 80)
    print("SUCCESS: End-to-end Parse -> Chunk -> BGE-M3 Embed -> OpenSearch -> k-NN Retrieve complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
