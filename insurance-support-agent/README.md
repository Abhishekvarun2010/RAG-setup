# Insurance Support Agent — Project Documentation

This repository contains the enterprise RAG (Retrieval-Augmented Generation) system for **Meridian Mutual Insurance SE**, built on the multi-format **Strata Insurance Corpus**.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [What Has Been Done So Far](#what-has-been-done-so-far)
   - [Step 1: Dataset & Corpus Integration](#step-1-dataset--corpus-integration)
   - [Step 2: Environment & Dependency Configuration](#step-2-environment--dependency-configuration)
   - [Step 3: Application & Package Architecture](#step-3-application--package-architecture)
   - [Step 4: Ingestion Data Model (Pydantic V2)](#step-4-ingestion-data-model-pydantic-v2)
   - [Step 5: Testing & Quality Assurance](#step-5-testing--quality-assurance)
3. [Repository Structure](#repository-structure)
4. [Chunk Data Model Reference](#chunk-data-model-reference)
5. [How to Run & Verify](#how-to-run--verify)

---

## Project Overview

The goal of this system is to ingest, index, and retrieve multimodal insurance documents (policies, claims, FNOLs, knowledge base manuals, tabular registers, and scanned forms) to support an intelligent insurance support agent with accurate, provenance-grounded answers.

---

## What Has Been Done So Far

### Step 1: Dataset & Corpus Integration
- Downloaded and verified the synthetic **Strata Insurance Corpus** under `insurance-support-agent/data/raw/`:
  - **1,311 documents** across 5 categories: `claim`, `policy`, `kb`, `tabular`, `identity`, plus `evidence` and `faces`.
  - Multi-format coverage: PDF, Word (`.docx`), Markdown (`.md`), Excel (`.xlsx`), CSV, and scanned JPGs.
  - Entity models and ground truth: `model.json`, `manifest.json`, `model.schema.json`, and `golden.jsonl`.

### Step 2: Environment & Dependency Configuration
- Configured Python 3.11 virtual environment using Poetry:
  - Added `huggingface-hub` for corpus fetching.
  - Added `pydantic (>=2.13.5)` for robust data validation.
  - Added `pytest (>=9.1.1)` for automated test execution.

### Step 3: Application & Package Architecture
- Scaffolded modular directory structure under `insurance-support-agent/`:
  - `apps/ingestion/`: Ingestion pipeline, parsers, and chunk models.
  - `apps/agent/`: LLM orchestrator and agent workflows.
  - `apps/api/`: REST API service.
  - `services/`: Downstream microservices (`policy-engine`, `insurance-api`).
  - `infra/`: Docker and OpenSearch cluster infrastructure.
  - `tests/`: Automated unit and integration test suites.
- Created `apps/__init__.py` and `apps/ingestion/__init__.py` with PEP 562 lazy attribute loading to avoid circular `runpy` execution warnings when running modules with `-m`.
- Created a top-level symlink `apps -> insurance-support-agent/apps` ensuring imports resolve smoothly whether executed from the project root or the service folder.

### Step 4: Ingestion Data Model (Pydantic V2)
Created `apps/ingestion/models.py` with the core `Chunk` model and strongly-typed domain Enums:

1. **Domain Enums**:
   - `DocumentType`: All 25+ insurance document types (`policy_contract`, `policy_declarations`, `customer_faq`, `underwriting_guidelines`, `fnol`, `adjuster_report`, `estimate`, etc.).
   - `SourceType`: Source domain categories (`kb`, `policy`, `claim`, `tabular`, `identity`, `evidence`) and file modalities (`pdf`, `docx`, `markdown`, `csv`, `xlsx`, `image`, `scan`).
   - `LineOfBusiness`: `personal_auto`, `homeowners`, `bop` (plus customer-facing aliases: `motor`, `household`, `commercial`).
   - `Status`: Document, policy, and claim statuses (`active`, `expired`, `closed`, `open`, `denied`, `draft`, etc.).
   - `AccessLevel`: Security tiers and roles (`public`, `internal`, `agent`, `adjuster`, `policyholder`, etc.).

2. **`Chunk` Pydantic Model (19 Fields)**:
   - **Core mandatory fields**: `content` (min_length=1), `chunk_id` (auto UUID default), `document_id`, `document_type`, `source_type`.
   - **Optional entity fields**: `policy_id`, `claim_id`, `policyholder_id`, `line_of_business`, `product`, `version`, `status`. *(Designed so that KB documents or policy documents without claims do not require non-applicable IDs).*
   - **Temporal metadata**: `effective_from` (`date`) and `effective_to` (`date`), with validator enforcing `effective_to >= effective_from`.
   - **Structural metadata**: `section` (`str`), `page_number` (`int >= 1`).
   - **Security & Provenance**: `access_control` (`List[str]` with validator normalizing single strings), `source_uri` (`str`), `ingested_at` (`datetime` in UTC).

### Step 5: Testing & Quality Assurance
- Implemented an automated test suite in `insurance-support-agent/tests/test_chunk_model.py`.
- Tested 15 distinct scenarios across minimal/rich chunks, validators, access control, and dates.

### Step 6: Document Parsers & Interface Protocol
- Created `apps/ingestion/parsers/base.py` defining:
  - `ParsedDocument`: Pydantic V2 model representing an ingested document before chunking.
  - `ParsedPage`: Sub-model for page-by-page extractions (1-indexed page numbering).
  - `DocumentParser`: `@runtime_checkable` `typing.Protocol` with `parse(file_path)` and `can_parse(file_path)`.
  - `BaseParser`: Abstract base class (`ABC`) with automatic file extension matching.
- Created `apps/ingestion/parsers/__init__.py` exposing the parser interface.
- Implemented 23 pytest tests in `insurance-support-agent/tests/test_parsers.py` testing document creation, required field validation, `source_type` representation, and parser protocol conformance.

---

## Repository Structure

```
.
├── pyproject.toml                         # Poetry configuration & dependencies
├── poetry.lock
├── .venv/                                 # Python 3.11 virtualenv
├── apps/ -> insurance-support-agent/apps  # Convenience symlink
│
├── insurance-support-agent/
│   ├── .env                               # Environment & OpenSearch config
│   ├── .gitignore                         # Git exclusion rules
│   ├── docker-compose.yml                 # OpenSearch & OpenSearch Dashboards
│   │
│   ├── apps/
│   │   ├── agent/                         # Support agent logic
│   │   ├── api/                           # FastAPI service
│   │   └── ingestion/                     # Ingestion pipeline
│   │       ├── __init__.py                # Package exports (PEP 562 lazy loading)
│   │       └── models.py                  # Chunk Pydantic model & Enums
│   │
│   ├── data/
│   │   ├── raw/                           # Raw Strata corpus (1,311 docs, manifest, schema)
│   │   ├── processed/                     # Chunks & embeddings target
│   │   └── eval/                          # Golden evaluation queries
│   │
│   ├── infra/
│   │   ├── docker/                        # Dockerfiles
│   │   └── opensearch/                    # Index mappings & pipelines
│   │
│   ├── services/
│   │   ├── insurance-api/                 # Core mock backend API
│   │   └── policy-engine/                 # Policy validation engine
│   │
│   └── tests/
│       ├── __init__.py
│       └── test_chunk_model.py            # Automated tests for chunk schemas
```

---

## Chunk Data Model Reference

```python
from apps.ingestion.models import (
    Chunk,
    DocumentType,
    SourceType,
    LineOfBusiness,
    Status,
    AccessLevel
)
from datetime import date

# Example 1: Knowledge Base Chunk (customer-faq.md)
kb_chunk = Chunk(
    content="Contact us to report the loss. We will record a First Notice of Loss...",
    document_id="DOC-KB-FAQ",
    document_type=DocumentType.CUSTOMER_FAQ,
    source_type=SourceType.KB,
    section="How do I file a claim?",
    source_uri="docs/kb/customer-faq.md",
    access_control=["public"],
)

# Example 2: Policy Declarations Chunk (MOT-0000001-declarations.pdf)
policy_chunk = Chunk(
    content="MERIDIAN MUTUAL - Peugeot 208 Reg: DO-2633-QO...",
    document_id="DOC-MOT-0000001-DEC",
    document_type=DocumentType.POLICY_DECLARATIONS,
    source_type=SourceType.POLICY,
    policy_id="MOT-0000001",
    policyholder_id="PH-00053",
    line_of_business=LineOfBusiness.PERSONAL_AUTO,
    effective_from=date(2023, 11, 29),
    effective_to=date(2024, 11, 28),
    page_number=1,
    access_control=[AccessLevel.AGENT, AccessLevel.POLICYHOLDER],
    source_uri="docs/policy/MOT-0000001-declarations.pdf",
)
```

---

## How to Run & Verify

### Run the Chunk Demonstration
```bash
poetry run python insurance-support-agent/apps/ingestion/models.py
```

### Run the Test Suite
```bash
poetry run pytest -v
```
Output:
```
============================== test session starts ==============================
collected 38 items

insurance-support-agent/tests/test_chunk_model.py (15 passed)
insurance-support-agent/tests/test_parsers.py (23 passed)

============================== 38 passed in 0.09s ==============================
```
