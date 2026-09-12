"""
Pydantic data models representing document chunks produced by the ingestion pipeline.

These models define the standardized schema for chunked text and associated metadata
before they are embedded and indexed into OpenSearch or a vector store.
"""
from __future__ import annotations  # Enables postponed evaluation of type annotations (PEP 563)

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =====================================================================
# Domain Enumerations
# =====================================================================

class DocumentType(str, Enum):
    """
    Document classification types corresponding to the Strata Insurance Corpus.
    
    Inheriting from (str, Enum) ensures clean JSON serialization while maintaining
    type safety across Python code.
    """

    # --- Policy Documents ---
    POLICY_CONTRACT = "policy_contract"
    POLICY_DECLARATIONS = "policy_declarations"
    POLICY_ENDORSEMENTS = "policy_endorsements"
    POLICY_SCHEDULE = "policy_schedule"

    # --- Claim Documents ---
    FNOL = "fnol"                                       # First Notice of Loss
    FNOL_SCANNED = "fnol_scanned"                       # Scanned variant (OCR target)
    ADJUSTER_REPORT = "adjuster_report"
    ESTIMATE = "estimate"
    SETTLEMENT_LETTER = "settlement_letter"
    SETTLEMENT_LETTER_SCANNED = "settlement_letter_scanned"
    DENIAL_LETTER = "denial_letter"
    DENIAL_LETTER_SCANNED = "denial_letter_scanned"
    ACCIDENT_STATEMENT = "accident_statement"
    ACCIDENT_STATEMENT_SCANNED = "accident_statement_scanned"
    POLICE_REPORT = "police_report"

    # --- Knowledge Base & Manuals ---
    CUSTOMER_FAQ = "customer_faq"
    UNDERWRITING_GUIDELINES = "underwriting_guidelines"
    CLAIMS_MANUAL = "claims_manual"

    # --- Tabular / Financial Registers ---
    COMMISSION_SUMMARY = "commission_summary"
    LOSS_RUN = "loss_run"
    PREMIUM_REGISTER = "premium_register"
    RESERVE_REGISTER = "reserve_register"

    # --- Identity & Evidence Media ---
    ID_CARD = "id_card"
    ID_CARD_SCANNED = "id_card_scanned"
    ID_PHOTO = "id_photo"
    EVIDENCE_PHOTO = "evidence_photo"

    # --- Fallback ---
    OTHER = "other"


class SourceType(str, Enum):
    """
    Source origin domain category or file format for the document chunk.
    
    Covers both high-level business domains (e.g. 'kb', 'policy') and raw
    file modalities (e.g. 'pdf', 'docx', 'markdown').
    """

    # High-level domain categories (matches data/raw/docs/ folder structure)
    POLICY = "policy"
    CLAIM = "claim"
    KB = "kb"
    TABULAR = "tabular"
    IDENTITY = "identity"
    EVIDENCE = "evidence"

    # File formats and input modalities
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    CSV = "csv"
    XLSX = "xlsx"
    IMAGE = "image"
    SCAN = "scan"


class DocumentFormat(str, Enum):
    """
    File format / modality of the original raw document.
    """

    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "md"
    CSV = "csv"
    XLSX = "xlsx"
    JPG = "jpg"
    IMAGE = "image"


class LineOfBusiness(str, Enum):
    """
    Line of business categorization for insurance policies and claims.
    
    Includes canonical schema identifiers as well as customer-facing aliases.
    """

    PERSONAL_AUTO = "personal_auto"
    HOMEOWNERS = "homeowners"
    BOP = "bop"  # Business Owner's Policy (Commercial small business)

    # General / customer-facing aliases used in guidelines and FAQs
    MOTOR = "motor"
    HOUSEHOLD = "household"
    COMMERCIAL = "commercial"


class Status(str, Enum):
    """
    Lifecycle status for policies, claims, or document validity.
    """

    # Policy / Document lifecycle
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    PENDING = "pending"
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"

    # Claim lifecycle
    OPEN = "open"
    CLOSED = "closed"
    DENIED = "denied"


class AccessLevel(str, Enum):
    """
    Standard access control roles and data classification tiers.
    
    Used to filter retrieval results based on user identity or clearance.
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    POLICYHOLDER = "policyholder"
    AGENT = "agent"
    ADJUSTER = "adjuster"
    UNDERWRITER = "underwriter"
    ADMIN = "admin"


# =====================================================================
# Main Chunk Model
# =====================================================================

class Chunk(BaseModel):
    """
    Represents an ingested, searchable chunk of an insurance document
    ready for embedding, retrieval, and indexing into a vector store.
    """

    # Model configuration:
    # - str_strip_whitespace: automatically strips leading/trailing spaces
    # - validate_assignment: re-runs validation if fields are modified after creation
    # - extra='forbid': prevents accidental typo fields from silently passing
    model_config = ConfigDict(
        use_enum_values=False,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    # -----------------------------------------------------------------
    # Core Chunk Content & Identifiers (Mandatory for every chunk)
    # -----------------------------------------------------------------
    content: str = Field(
        ...,
        min_length=1,
        description="The extracted textual content of the document chunk.",
    )
    chunk_id: str = Field(
        default_factory=lambda: f"chk_{uuid4().hex[:12]}",
        description="Unique identifier for this chunk (e.g. UUID or 'DOC-ID#c1').",
    )
    document_id: str = Field(
        ...,
        min_length=1,
        description="Parent document identifier (e.g. 'DOC-C-1000-ADJ', 'DOC-KB-FAQ').",
    )
    document_type: DocumentType = Field(
        ...,
        description="Classification of the parent document (e.g. policy_contract, customer_faq).",
    )
    source_type: SourceType = Field(
        ...,
        description="Source category or format (e.g. kb, policy, claim, tabular).",
    )

    # -----------------------------------------------------------------
    # Optional Insurance Entity References
    # (These fields only exist for specific document types, so they default to None)
    # -----------------------------------------------------------------
    policy_id: Optional[str] = Field(
        default=None,
        description="Associated policy identifier (e.g. 'MOT-0000001'). None for KB or ID docs.",
    )
    claim_id: Optional[str] = Field(
        default=None,
        description="Associated claim identifier (e.g. 'C-1000'). None for policy or KB docs.",
    )
    policyholder_id: Optional[str] = Field(
        default=None,
        description="Associated policyholder identifier (e.g. 'PH-00001'). None for general KB docs.",
    )
    line_of_business: Optional[LineOfBusiness] = Field(
        default=None,
        description="Line of business (e.g. personal_auto, bop). None for company-wide docs.",
    )
    product: Optional[str] = Field(
        default=None,
        description="Specific insurance product name or coverage tier (e.g. 'Meridian Motor Standard').",
    )
    version: Optional[str] = Field(
        default=None,
        description="Document or policy version string (e.g. '1.0', '2024.1').",
    )
    status: Optional[Status] = Field(
        default=None,
        description="Lifecycle status of document, policy, or claim (e.g. active, closed, open).",
    )

    # -----------------------------------------------------------------
    # Optional Temporal Metadata (Policy/guideline validity window)
    # -----------------------------------------------------------------
    effective_from: Optional[date] = Field(
        default=None,
        description="Effective start date of policy term or document validity.",
    )
    effective_to: Optional[date] = Field(
        default=None,
        description="Effective expiry or renewal date.",
    )

    # -----------------------------------------------------------------
    # Optional Structural Metadata (Location in source document)
    # -----------------------------------------------------------------
    section: Optional[str] = Field(
        default=None,
        description="Section heading, clause, or chapter title within the source document.",
    )
    page_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="1-indexed page number in the original source document (PDF/Word).",
    )

    # -----------------------------------------------------------------
    # Security & Provenance Metadata
    # -----------------------------------------------------------------
    access_control: List[str] = Field(
        default_factory=lambda: ["internal"],
        description="Access control tags or permitted roles (e.g. ['public'], ['agent', 'adjuster']).",
    )
    source_uri: Optional[str] = Field(
        default=None,
        description="URI, relative path, or locator to the original source document.",
    )
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the chunk was processed and ingested.",
    )

    # -----------------------------------------------------------------
    # Custom Validators
    # -----------------------------------------------------------------
    @field_validator("access_control", mode="before")
    @classmethod
    def normalize_access_control(cls, v: Any) -> List[str]:
        """
        Normalizes access control input into a uniform List[str].
        
        Handles:
        - None -> []
        - Single string "public" -> ["public"]
        - List of Enums [AccessLevel.ADMIN] -> ["admin"]
        - Standard List[str] -> unchanged
        """
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, (list, tuple, set)):
            return [item.value if isinstance(item, Enum) else str(item) for item in v]
        return v

    @model_validator(mode="after")
    def validate_dates(self) -> Chunk:
        """
        Ensures chronological integrity between effective_from and effective_to.
        Raises ValueError if effective_to precedes effective_from.
        """
        if self.effective_from and self.effective_to:
            if self.effective_to < self.effective_from:
                raise ValueError(
                    f"effective_to ({self.effective_to}) cannot precede effective_from ({self.effective_from})"
                )
        return self
