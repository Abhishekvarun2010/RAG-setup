"""
FastAPI Insurance API Microservice.

Provides RESTful endpoints for live insurance policy and customer records,
backed by PostgreSQL (or in-memory database) seeded with Strata corpus data.
Used exclusively by PolicyDataTool for Agent Runtime lookups.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import Generator

from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy.orm import Session

from services.insurance_api.database import (
    ClaimModel,
    PolicyholderModel,
    PolicyModel,
    get_db_session,
    init_db,
    seed_database_from_strata,
)
from services.insurance_api.models import (
    ClaimResponse,
    PolicyholderBrief,
    PolicyholderResponse,
    PolicyResponse,
)

logger = logging.getLogger("insurance_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database schema and seed data upon startup."""
    init_db()
    with get_db_session() as session:
        stats = seed_database_from_strata(session)
        logger.info(f"Insurance API initialized with data: {stats}")
    yield


app = FastAPI(
    title="Insurance Support Agent - Core API",
    description="Customer Data & Policy Management Microservice",
    version="1.0.0",
    lifespan=lifespan,
)


def get_db() -> Generator[Session, None, None]:
    """Dependency providing a transactional database session."""
    session = get_db_session()
    try:
        yield session
    finally:
        session.close()


@app.get("/health", summary="Service Health Check")
def health_check(db: Session = Depends(get_db)):
    """Check API and database connection health."""
    policies_count = db.query(PolicyModel).count()
    claims_count = db.query(ClaimModel).count()
    return {
        "status": "healthy",
        "service": "insurance-api",
        "database": "connected",
        "records": {
            "policies": policies_count,
            "claims": claims_count,
        },
    }


@app.get(
    "/policies/{policy_id}",
    response_model=PolicyResponse,
    summary="Get Policy Details",
    description="Retrieve structured policy details, coverage limits, deductible, and policyholder contact.",
)
def get_policy(policy_id: str, db: Session = Depends(get_db)):
    """
    Fetch policy by its unique ID (e.g. 'COM-0000077', 'MOT-0000001').
    """
    policy = db.query(PolicyModel).filter(PolicyModel.id == policy_id.strip()).first()
    if not policy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Policy '{policy_id}' not found.",
        )

    # Attach policyholder brief info if present
    holder_brief = None
    if policy.policyholder:
        h = policy.policyholder
        holder_brief = PolicyholderBrief(
            id=h.id,
            name=h.name,
            email=h.email,
            phone=h.phone,
            city=h.city,
            country=h.country,
        )

    return PolicyResponse(
        policy_id=policy.id,
        policyholder_id=policy.holder_id,
        status=policy.status,
        product=policy.product,
        line_of_business=policy.line,
        deductible=policy.deductible,
        annual_premium=policy.annual_premium,
        effective_date=policy.effective_date,
        expiry_date=policy.expiry_date,
        limits=policy.limits or {},
        endorsements=policy.endorsements or [],
        policyholder=holder_brief,
    )


@app.get(
    "/policyholders/{policyholder_id}",
    response_model=PolicyholderResponse,
    summary="Get Policyholder Details",
)
def get_policyholder(policyholder_id: str, db: Session = Depends(get_db)):
    """Fetch policyholder identity and contact details."""
    holder = db.query(PolicyholderModel).filter(PolicyholderModel.id == policyholder_id.strip()).first()
    if not holder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Policyholder '{policyholder_id}' not found.",
        )
    return PolicyholderResponse(
        id=holder.id,
        name=holder.name,
        email=holder.email,
        phone=holder.phone,
        street=holder.street,
        city=holder.city,
        postcode=holder.postcode,
        country=holder.country,
        dob=holder.dob,
        gender=holder.gender,
        national_id=holder.national_id,
    )


@app.get(
    "/claims/{claim_id}",
    response_model=ClaimResponse,
    summary="Get Claim Details",
)
def get_claim(claim_id: str, db: Session = Depends(get_db)):
    """Fetch claim details by claim ID (e.g. 'C-1000')."""
    claim = db.query(ClaimModel).filter(ClaimModel.id == claim_id.strip()).first()
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim '{claim_id}' not found.",
        )
    return ClaimResponse(
        id=claim.id,
        policy_id=claim.policy_id,
        holder_id=claim.holder_id,
        adjuster_id=claim.adjuster_id,
        cause=claim.cause,
        date_of_loss=claim.date_of_loss,
        reported_date=claim.reported_date,
        paid=claim.paid,
        reserve=claim.reserve,
        status=claim.status,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.insurance_api.main:app", host="0.0.0.0", port=8001, reload=True)
