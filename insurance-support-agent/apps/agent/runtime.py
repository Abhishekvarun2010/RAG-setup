"""
Agent Runtime orchestrator.

Coordinates user requests, capability planning, safe tool execution, and final synthesis:
User
 ↓
Agent Runtime
 ↓
Decide what capabilities are needed
 ├── RAG
 ├── Customer Data
 ├── Actions
 └── Direct Answer
LLM proposes the plan; Runtime executes approved tools.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from apps.agent.llm import LLMProvider, OllamaLLM
from apps.agent.models import ChatMessage
from apps.agent.planner import AgentPlanner, CapabilityType, ExecutionPlan
from apps.agent.tools.base import BaseTool, ToolResult
from apps.agent.tools.policy_tool import PolicyDataTool
from apps.auth.models import SecurityContext

logger = logging.getLogger(__name__)


class AgentResponse(BaseModel):
    """Container for the Agent Runtime's final execution outcome."""
    query: str = Field(..., description="User's original query")
    answer: str = Field(..., description="Synthesized grounded answer")
    plan: ExecutionPlan = Field(..., description="Approved execution plan proposed by the planner")
    tool_results: List[ToolResult] = Field(default_factory=list, description="Outputs from executed tools")
    latency_seconds: float = Field(0.0, description="Total execution time in seconds")


SYNTHESIS_SYSTEM_PROMPT = """You are an expert Insurance Support Agent.
Answer the user's inquiry directly, accurately, and factually using the verified information obtained from the tools below.

RULES:
1. Rely strictly on the tool outputs provided.
2. Present policy details clearly (e.g. policyholder name, active status, deductible amount, coverage limits).
3. Do not invent or extrapolate numbers not present in the tool results.
"""


class AgentRuntime:
    """
    Central execution engine for the Insurance Support Agent.
    """

    def __init__(
        self,
        planner: Optional[AgentPlanner] = None,
        tools: Optional[Dict[str, BaseTool]] = None,
        llm: Optional[LLMProvider] = None,
    ) -> None:
        self._llm = llm or OllamaLLM(model_name="qwen3:8b")
        self._planner = planner or AgentPlanner(llm=self._llm)
        self._tools: Dict[str, BaseTool] = tools or {
            "get_policy_details": PolicyDataTool(),
        }

    @property
    def tools(self) -> Dict[str, BaseTool]:
        return self._tools

    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool capability with the runtime."""
        self._tools[tool.name] = tool

    def run(
        self,
        query: str,
        security_context: Optional[SecurityContext] = None,
    ) -> AgentResponse:
        """
        Execute the agent workflow for a given inquiry.

        Args:
            query: User's input question or instruction.
            security_context: Authenticated caller security identity.

        Returns:
            AgentResponse containing the answer, plan, and executed tool results.
        """
        start_time = time.perf_counter()

        # 1. Decide capabilities and obtain execution plan
        plan = self._planner.plan(query)

        # 2. Handle direct answer without tools
        if CapabilityType.DIRECT_ANSWER in plan.capabilities and plan.direct_response:
            elapsed = round(time.perf_counter() - start_time, 4)
            return AgentResponse(
                query=query,
                answer=plan.direct_response,
                plan=plan,
                tool_results=[],
                latency_seconds=elapsed,
            )

        # 3. Execute approved tool steps
        tool_results: List[ToolResult] = []
        for step in plan.steps:
            if not step.tool_name:
                continue

            tool = self._tools.get(step.tool_name)
            if not tool:
                logger.error(f"Plan requested unknown tool '{step.tool_name}'")
                tool_results.append(
                    ToolResult.fail(
                        error=f"Tool '{step.tool_name}' is not registered with Agent Runtime.",
                        formatted_output=f"Error: Requested capability '{step.tool_name}' is unavailable.",
                    )
                )
                continue

            # Execute tool with validated arguments
            logger.info(f"Agent executing step {step.step_number}: {tool.name} with args {step.arguments}")
            res = tool.execute(**step.arguments)
            tool_results.append(res)

        # 4. Synthesize final answer with LLM
        tool_outputs_str = "\n\n".join(
            f"[Tool: {step.tool_name}]\n{res.formatted_output}"
            for step, res in zip(plan.steps, tool_results)
        )

        synthesis_messages = [
            ChatMessage(role="system", content=SYNTHESIS_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=(
                    f"User Query: {query}\n\n"
                    f"Verified Tool Results:\n"
                    f"====================\n"
                    f"{tool_outputs_str}\n"
                    f"====================\n\n"
                    f"Provide a helpful, precise summary addressing the user's question."
                ),
            ),
        ]

        llm_resp = self._llm.chat(messages=synthesis_messages, temperature=0.0)
        elapsed = round(time.perf_counter() - start_time, 4)

        return AgentResponse(
            query=query,
            answer=llm_resp.content,
            plan=plan,
            tool_results=tool_results,
            latency_seconds=elapsed,
        )
