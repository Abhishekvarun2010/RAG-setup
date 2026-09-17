"""
Automated unit tests for the Insurance API service endpoints.
"""
from fastapi.testclient import TestClient
import pytest

from services.insurance_api.database import init_db, get_db_session, seed_database_from_strata
from services.insurance_api.main import app


@pytest.fixture(scope="module")
def client():
    # Ensure database is initialized and seeded
    init_db()
    with get_db_session() as session:
        seed_database_from_strata(session)
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert data["records"]["policies"] > 0


def test_get_policy_com_0000077(client):
    response = client.get("/policies/COM-0000077")
    assert response.status_code == 200
    data = response.json()
    assert data["policy_id"] == "COM-0000077"
    assert data["policyholder_id"] == "PH-00029"
    assert data["deductible"] == 5000.0
    assert data["status"] == "active"
    assert data["product"] == "Commercial BOP"
    assert "general_liability" in data["limits"]
    assert "EPLI" in data["endorsements"]
    
    # Check attached policyholder summary
    assert data["policyholder"] is not None
    assert data["policyholder"]["id"] == "PH-00029"
    assert data["policyholder"]["name"] == "Graciano Solé"


def test_get_policy_not_found(client):
    response = client.get("/policies/UNKNOWN-999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_policyholder(client):
    response = client.get("/policyholders/PH-00029")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "PH-00029"
    assert data["name"] == "Graciano Solé"
    assert data["email"] == "carolina01@example.net"
    assert data["city"] == "Huesca"


def test_get_claim_c1000(client):
    response = client.get("/claims/C-1000")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "C-1000"
    assert data["policy_id"] == "COM-0000077"
    assert data["cause"] == "slip_and_fall"
    assert data["paid"] == 22950.0
    assert data["status"] == "closed"
