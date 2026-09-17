"""
Base interfaces and protocols for Agent Runtime tools.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """
    Standardized result returned by any tool executed within the Agent Runtime.
    """
    success: bool = Field(..., description="Whether tool execution succeeded")
    data: Optional[Any] = Field(None, description="Structured raw data returned by the tool")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    formatted_output: str = Field(..., description="Human- and LLM-readable text summary of tool output")

    @classmethod
    def ok(cls, data: Any, formatted_output: str) -> ToolResult:
        """Convenience constructor for successful tool execution."""
        return cls(success=True, data=data, error=None, formatted_output=formatted_output)

    @classmethod
    def fail(cls, error: str, formatted_output: Optional[str] = None) -> ToolResult:
        """Convenience constructor for failed tool execution."""
        return cls(
            success=False,
            data=None,
            error=error,
            formatted_output=formatted_output or f"Tool execution failed: {error}",
        )


class BaseTool(ABC):
    """
    Abstract Base Class for tools available to the Agent Runtime.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool identifier used by the planner (e.g. 'get_policy_details')."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """High-level summary of tool function and usage instructions."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema dictionary describing accepted arguments."""
        pass

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments and return a ToolResult."""
        pass
