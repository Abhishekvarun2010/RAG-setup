"""
Unit tests for PolicyDataTool.
"""
import httpx
import pytest

from apps.agent.tools.base import BaseTool, ToolResult
from apps.agent.tools.policy_tool import PolicyDataTool


def test_policy_tool_implements_interface():
    tool = PolicyDataTool()
    assert isinstance(tool, BaseTool)
    assert tool.name == "get_policy_details"
    assert "policy_id" in tool.parameters["properties"]


def test_policy_tool_success():
    mock_payload = {
        "policy_id": "COM-0000077",
        "policyholder_id": "PH-00029",
        "status": "active",
        "product": "Commercial BOP",
        "line_of_business": "commercial",
        "deductible": 5000.0,
        "annual_premium": 2710.0,
        "effective_date": "2023-08-01",
        "expiry_date": "2024-07-31",
        "limits": {"general_liability": 1000000.0},
        "endorsements": ["EPLI"],
        "policyholder": {
            "id": "PH-00029",
            "name": "Graciano Solé",
            "email": "carolina01@example.net",
            "phone": "+34987 451 462",
            "city": "Huesca",
            "country": "ES",
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/policies/COM-0000077"
        return httpx.Response(200, json=mock_payload)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testapi")
    tool = PolicyDataTool(client=client)

    result = tool.execute(policy_id="COM-0000077")
    assert result.success is True
    assert result.data["policy_id"] == "COM-0000077"
    assert "€5,000.00" in result.formatted_output
    assert "Graciano Solé" in result.formatted_output
    assert "Commercial BOP" in result.formatted_output


def test_policy_tool_missing_id():
    tool = PolicyDataTool()
    result = tool.execute(policy_id="")
    assert result.success is False
    assert "must be provided" in result.formatted_output.lower()


def test_policy_tool_not_found_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Policy 'UNKNOWN' not found."})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testapi")
    tool = PolicyDataTool(client=client)

    result = tool.execute(policy_id="UNKNOWN")
    assert result.success is False
    assert "not found" in result.formatted_output.lower()


def test_policy_tool_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testapi")
    tool = PolicyDataTool(client=client)

    result = tool.execute(policy_id="COM-0000077")
    assert result.success is False
    assert "unavailable" in result.formatted_output.lower()
