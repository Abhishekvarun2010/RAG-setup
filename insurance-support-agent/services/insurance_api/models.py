"""
Pydantic data models for the Insurance API service.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PolicyholderBrief(BaseModel):
    """Brief summary of a policyholder attached to policy responses."""
    id: str = Field(..., description="Policyholder ID, e.g. PH-00029")
    name: str = Field(..., description="Full name of policyholder")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    city: Optional[str] = Field(None, description="City")
    country: Optional[str] = Field(None, description="Country code, e.g. ES")


class PolicyResponse(BaseModel):
    """Response container for GET /policies/{policy_id}."""
    policy_id: str = Field(..., description="Unique policy identifier, e.g. COM-0000077")
    policyholder_id: str = Field(..., description="Foreign key to policyholder")
    status: str = Field(default="active", description="Policy lifecycle status (active, expired, cancelled)")
    product: str = Field(..., description="Human-readable product name, e.g. 'Commercial BOP'")
    line_of_business: str = Field(..., description="Line of business, e.g. 'commercial', 'personal_auto'")
    deductible: float = Field(..., description="Per-occurrence deductible in Euros")
    annual_premium: float = Field(..., description="Annual premium amount in Euros")
    effective_date: Optional[str] = Field(None, description="Policy effective date (YYYY-MM-DD)")
    expiry_date: Optional[str] = Field(None, description="Policy expiry date (YYYY-MM-DD)")
    limits: Dict[str, Any] = Field(default_factory=dict, description="Coverage limits dictionary")
    endorsements: List[str] = Field(default_factory=list, description="Attached endorsement codes")
    policyholder: Optional[PolicyholderBrief] = Field(None, description="Attached policyholder contact details")


class PolicyholderResponse(BaseModel):
    """Response container for GET /policyholders/{policyholder_id}."""
    id: str = Field(..., description="Policyholder ID")
    name: str = Field(..., description="Full name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    street: Optional[str] = Field(None, description="Street address")
    city: Optional[str] = Field(None, description="City")
    postcode: Optional[str] = Field(None, description="Postal code")
    country: Optional[str] = Field(None, description="Country code")
    dob: Optional[str] = Field(None, description="Date of birth")
    gender: Optional[str] = Field(None, description="Gender")
    national_id: Optional[str] = Field(None, description="National ID or passport number")


class ClaimResponse(BaseModel):
    """Response container for GET /claims/{claim_id}."""
    id: str = Field(..., description="Claim reference ID, e.g. C-1000")
    policy_id: str = Field(..., description="Associated policy ID")
    holder_id: str = Field(..., description="Associated policyholder ID")
    adjuster_id: Optional[str] = Field(None, description="Assigned adjuster ID")
    cause: str = Field(..., description="Cause of loss, e.g. slip_and_fall")
    date_of_loss: Optional[str] = Field(None, description="Date when loss occurred")
    reported_date: Optional[str] = Field(None, description="Date when claim was reported")
    paid: float = Field(0.0, description="Total amount paid to date in Euros")
    reserve: float = Field(0.0, description="Outstanding reserve in Euros")
    status: str = Field(..., description="Claim lifecycle status (open, closed, denied)")
