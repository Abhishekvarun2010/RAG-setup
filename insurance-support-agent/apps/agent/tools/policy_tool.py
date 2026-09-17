"""
PolicyDataTool for querying customer policy records via the Insurance API.

The agent never connects directly to PostgreSQL; it must query the Insurance API
via this tool.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional
import httpx

from apps.agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class PolicyDataTool(BaseTool):
    """
    Tool allowing the Agent Runtime to retrieve structured customer policy details.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: float = 10.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = client

    @property
    def name(self) -> str:
        return "get_policy_details"

    @property
    def description(self) -> str:
        return (
            "Retrieve verified, live customer policy details including lifecycle status, "
            "product line, per-occurrence deductible, coverage limits, attached endorsements, "
            "and policyholder contact information from the authoritative core Insurance API."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "policy_id": {
                    "type": "string",
                    "description": "Unique policy reference number (e.g. 'COM-0000077', 'MOT-0000001').",
                }
            },
            "required": ["policy_id"],
        }

    def _get_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute policy lookup.

        Args:
            policy_id: The ID of the policy to retrieve.

        Returns:
            ToolResult with formatted output and raw data dictionary.
        """
        policy_id = kwargs.get("policy_id")
        if not policy_id or not isinstance(policy_id, str) or not policy_id.strip():
            return ToolResult.fail(
                error="Missing or empty 'policy_id' parameter.",
                formatted_output="Error: A valid policy ID must be provided (e.g. COM-0000077).",
            )

        clean_policy_id = policy_id.strip()
        client = self._get_client()
        should_close = self._client is None

        try:
            url = f"{self.base_url}/policies/{clean_policy_id}"
            response = client.get(url)
        except (httpx.ConnectError, httpx.NetworkError) as e:
            logger.error(f"Cannot connect to Insurance API at {self.base_url}: {e}")
            return ToolResult.fail(
                error=f"Cannot connect to Insurance API at {self.base_url}. Is the service running?",
                formatted_output="System Notice: Customer Insurance API is temporarily unavailable.",
            )
        except httpx.TimeoutException as e:
            return ToolResult.fail(
                error=f"Timeout ({self.timeout}s) waiting for Insurance API response.",
                formatted_output="System Notice: Customer Insurance API timed out.",
            )
        except httpx.RequestError as e:
            return ToolResult.fail(
                error=f"HTTP request failed: {e}",
                formatted_output=f"Error connecting to Insurance API: {e}",
            )
        finally:
            if should_close:
                client.close()

        if response.status_code == 404:
            return ToolResult.fail(
                error=f"Policy '{clean_policy_id}' not found.",
                formatted_output=f"Policy '{clean_policy_id}' was not found in the insurance database.",
            )
        elif response.status_code != 200:
            return ToolResult.fail(
                error=f"Insurance API returned HTTP {response.status_code}: {response.text}",
                formatted_output=f"Insurance API error (HTTP {response.status_code}).",
            )

        try:
            data = response.json()
        except Exception as e:
            return ToolResult.fail(
                error=f"Failed to parse JSON response: {e}",
                formatted_output="Error: Invalid response format from Insurance API.",
            )

        # Format output into concise, factual presentation for LLM and user
        lines = [
            f"--- Policy Record: {data.get('policy_id')} ---",
            f"Status          : {data.get('status', 'active').upper()}",
            f"Product         : {data.get('product')}",
            f"Line of Business: {data.get('line_of_business')}",
            f"Deductible      : €{data.get('deductible', 0.0):,.2f}",
            f"Annual Premium  : €{data.get('annual_premium', 0.0):,.2f}",
            f"Effective Period: {data.get('effective_date')} to {data.get('expiry_date')}",
        ]

        holder = data.get("policyholder")
        if holder:
            lines.append(
                f"Policyholder    : {holder.get('name')} (ID: {holder.get('id')}, Phone: {holder.get('phone')}, Email: {holder.get('email')})"
            )

        limits = data.get("limits")
        if limits and isinstance(limits, dict):
            limits_str = ", ".join(f"{k.replace('_', ' ').title()}: €{v:,.2f}" for k, v in limits.items())
            lines.append(f"Coverage Limits : {limits_str}")

        endorsements = data.get("endorsements")
        if endorsements and isinstance(endorsements, list):
            lines.append(f"Endorsements    : {', '.join(endorsements)}")

        formatted_summary = "\n".join(lines)
        return ToolResult.ok(data=data, formatted_output=formatted_summary)
