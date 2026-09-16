"""
Database connection, SQLAlchemy models, and auto-seeding for Insurance API.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

logger = logging.getLogger(__name__)

Base = declarative_base()

# Default PostgreSQL connection URL (can be overridden via DATABASE_URL env)
DEFAULT_DB_URL = "postgresql://localhost:5432/insurance_db"
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_DB_URL)

LINE_PRODUCT_NAMES: Dict[str, str] = {
    "bop": "Commercial BOP",
    "commercial": "Commercial General Liability",
    "personal_auto": "Personal Auto",
    "homeowners": "Homeowners Comprehensive",
    "strata": "Residential Strata",
}


# =====================================================================
# SQLAlchemy ORM Models
# =====================================================================

class PolicyholderModel(Base):
    __tablename__ = "policyholders"

    id = Column(String(50), primary_key=True, index=True)
    name = Column(String(200), nullable=False, index=True)
    email = Column(String(200), nullable=True)
    phone = Column(String(50), nullable=True)
    street = Column(String(250), nullable=True)
    city = Column(String(100), nullable=True)
    postcode = Column(String(30), nullable=True)
    country = Column(String(10), nullable=True)
    dob = Column(String(20), nullable=True)
    gender = Column(String(20), nullable=True)
    national_id = Column(String(50), nullable=True)

    policies = relationship("PolicyModel", back_populates="policyholder", cascade="all, delete-orphan")


class PolicyModel(Base):
    __tablename__ = "policies"

    id = Column(String(50), primary_key=True, index=True)
    holder_id = Column(String(50), ForeignKey("policyholders.id"), nullable=False, index=True)
    agent_id = Column(String(50), nullable=True)
    line = Column(String(50), nullable=False)
    product = Column(String(100), nullable=False)
    status = Column(String(30), default="active", index=True)
    annual_premium = Column(Float, nullable=False, default=0.0)
    deductible = Column(Float, nullable=False, default=0.0)
    effective_date = Column(String(20), nullable=True)
    expiry_date = Column(String(20), nullable=True)
    limits = Column(JSON, nullable=True)
    endorsements = Column(JSON, nullable=True)

    policyholder = relationship("PolicyholderModel", back_populates="policies")
    claims = relationship("ClaimModel", back_populates="policy", cascade="all, delete-orphan")


class ClaimModel(Base):
    __tablename__ = "claims"

    id = Column(String(50), primary_key=True, index=True)
    policy_id = Column(String(50), ForeignKey("policies.id"), nullable=False, index=True)
    holder_id = Column(String(50), nullable=False, index=True)
    adjuster_id = Column(String(50), nullable=True)
    cause = Column(String(100), nullable=False)
    date_of_loss = Column(String(20), nullable=True)
    reported_date = Column(String(20), nullable=True)
    paid = Column(Float, default=0.0)
    reserve = Column(Float, default=0.0)
    status = Column(String(30), default="open", index=True)

    policy = relationship("PolicyModel", back_populates="claims")


# =====================================================================
# Database Engine & Session Management
# =====================================================================

def get_engine(db_url: Optional[str] = None):
    url = db_url or DATABASE_URL
    try:
        engine = create_engine(url, pool_pre_ping=True)
        # Test connection
        with engine.connect():
            pass
        return engine
    except Exception as e:
        logger.warning(f"Could not connect to PostgreSQL at '{url}': {e}. Falling back to SQLite in-memory.")
        return create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})


_engine = None
_session_factory = None


def init_db(db_url: Optional[str] = None):
    global _engine, _session_factory
    _engine = get_engine(db_url)
    Base.metadata.create_all(bind=_engine)
    _session_factory = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine


def get_db_session() -> Session:
    global _session_factory
    if _session_factory is None:
        init_db()
    return _session_factory()


# =====================================================================
# Seeding from Strata Insurance Corpus model.json
# =====================================================================

def find_model_json_path() -> Optional[Path]:
    """Locate data/raw/model.json relative to repository paths."""
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "model.json",
        Path(__file__).resolve().parent.parent / "data" / "raw" / "model.json",
        Path("insurance-support-agent/data/raw/model.json").resolve(),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def seed_database_from_strata(session: Session, force: bool = False) -> Dict[str, int]:
    """
    Seed policyholders, policies, and claims from Strata model.json into the database.
    """
    policy_count = session.query(PolicyModel).count()
    if policy_count > 0 and not force:
        return {
            "policyholders": session.query(PolicyholderModel).count(),
            "policies": policy_count,
            "claims": session.query(ClaimModel).count(),
        }

    json_path = find_model_json_path()
    if not json_path or not json_path.exists():
        logger.warning(f"Strata model.json not found; skipping seeding.")
        return {"policyholders": 0, "policies": 0, "claims": 0}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Insert Policyholders
    holder_map = {}
    for h in data.get("policyholders", []):
        holder_obj = PolicyholderModel(
            id=h.get("id"),
            name=h.get("name", "Unknown"),
            email=h.get("email"),
            phone=h.get("phone"),
            street=h.get("street"),
            city=h.get("city"),
            postcode=h.get("postcode"),
            country=h.get("country"),
            dob=h.get("dob"),
            gender=h.get("gender"),
            national_id=h.get("national_id"),
        )
        session.merge(holder_obj)
        holder_map[holder_obj.id] = holder_obj

    # 2. Insert Policies
    for p in data.get("policies", []):
        line = p.get("line", "general")
        product_name = LINE_PRODUCT_NAMES.get(line, f"{line.replace('_', ' ').title()} Policy")
        policy_obj = PolicyModel(
            id=p.get("id"),
            holder_id=p.get("holder_id"),
            agent_id=p.get("agent_id"),
            line=line,
            product=product_name,
            status="active",
            annual_premium=float(p.get("annual_premium", 0.0)),
            deductible=float(p.get("deductible", 0.0)),
            effective_date=p.get("effective_date"),
            expiry_date=p.get("expiry_date"),
            limits=p.get("limits", {}),
            endorsements=p.get("endorsements", []),
        )
        session.merge(policy_obj)

    # 3. Insert Claims
    for c in data.get("claims", []):
        claim_obj = ClaimModel(
            id=c.get("id"),
            policy_id=c.get("policy_id"),
            holder_id=c.get("holder_id"),
            adjuster_id=c.get("adjuster_id"),
            cause=c.get("cause", "unspecified"),
            date_of_loss=c.get("date_of_loss"),
            reported_date=c.get("reported_date"),
            paid=float(c.get("paid", 0.0)),
            reserve=float(c.get("reserve", 0.0)),
            status=c.get("status", "open"),
        )
        session.merge(claim_obj)

    session.commit()

    stats = {
        "policyholders": session.query(PolicyholderModel).count(),
        "policies": session.query(PolicyModel).count(),
        "claims": session.query(ClaimModel).count(),
    }
    logger.info(f"Successfully seeded database: {stats}")
    return stats
