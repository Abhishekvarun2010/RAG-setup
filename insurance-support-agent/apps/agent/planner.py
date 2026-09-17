"""
Agent Runtime Capability Planner.

Enforces deterministic capability selection:
- RAG (unstructured documents, contract clauses, estimates, adjuster notes)
- CUSTOMER_DATA (structured policy/claim lookups via Insurance API)
- ACTIONS (future transaction workflows)
- DIRECT_ANSWER (greetings, conversational replies)

The LLM proposes the plan; it does not execute arbitrary operations.
"""
from __future__ import annotations

from enum import Enum
import json
import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from apps.agent.llm import LLMProvider, OllamaLLM
from apps.agent.models import ChatMessage

logger = logging.getLogger(__name__)


class CapabilityType(str, Enum):
    """Core capabilities available to the Insurance Agent Runtime."""
    RAG = "rag"
    CUSTOMER_DATA = "customer_data"
    ACTIONS = "actions"
    DIRECT_ANSWER = "direct_answer"


class PlanStep(BaseModel):
    """An individual execution step within an approved plan."""
    model_config = ConfigDict(str_strip_whitespace=True)

    step_number: int = Field(..., ge=1, description="Sequential step index")
    capability: CapabilityType = Field(..., description="Target capability for this step")
    tool_name: Optional[str] = Field(None, description="Registered tool to execute (e.g. 'get_policy_details')")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Validated arguments for the tool")
    rationale: str = Field(..., description="Explanation of why this step is required")


class ExecutionPlan(BaseModel):
    """Complete plan proposed by the LLM and validated by the Agent Runtime."""
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., description="Original user inquiry")
    capabilities: List[CapabilityType] = Field(default_factory=list, description="Capabilities chosen for this request")
    steps: List[PlanStep] = Field(default_factory=list, description="Ordered plan steps to execute")
    reasoning: Optional[str] = Field(None, description="High-level planning rationale")
    direct_response: Optional[str] = Field(None, description="Direct text response if capability is direct_answer")


PLANNER_SYSTEM_PROMPT = """You are the Lead Planning Agent for an enterprise Insurance Support Assistant.
Your sole job is to analyze the user's inquiry and propose a structured Execution Plan choosing from 4 capabilities:

CAPABILITIES:
1. "customer_data":
   Use when the user asks for verified structured policy details, deductibles, premiums, active coverage status, or policyholder info.
   Available Tool: "get_policy_details" with argument {"policy_id": "..."}.
   Example triggers: "What is my deductible on COM-0000077?", "Is policy MOT-0000001 active?", "Look up policy details".

2. "rag":
   Use when the user asks about contract clauses, legal wording, exclusions, definitions, damage estimates, adjuster investigation notes, or FAQs.
   Available Tool: "rag_retrieval" with argument {"query": "..."}.
   Example triggers: "Does water damage cover mold?", "What was the adjuster's conclusion?", "Explain standard coverage".

3. "actions":
   Use when the user requests an operational state change (e.g. "cancel my policy", "file a new claim", "update my address").
   (Note: Actions require explicit confirmation).

4. "direct_answer":
   Use for greetings, general insurance principles, polite remarks, or clarifying questions that require no database or document lookups.

RULES:
- You ONLY propose the plan. You CANNOT execute database queries or arbitrary code.
- Return your plan strictly as a JSON object matching this schema:
{
  "capabilities": ["customer_data"],
  "steps": [
    {
      "step_number": 1,
      "capability": "customer_data",
      "tool_name": "get_policy_details",
      "arguments": {"policy_id": "COM-0000077"},
      "rationale": "Retrieve live deductible and active status from Insurance API"
    }
  ],
  "direct_response": null
}
- If no tool is needed (e.g. "Hello!"), set capabilities to ["direct_answer"] and provide "direct_response".
- Never invent tool names. Only use "get_policy_details" or "rag_retrieval".
"""


class AgentPlanner:
    """
    Decides what capabilities are needed and compiles a safe, validated execution plan.
    """

    # Regex heuristic to identify explicit policy numbers (e.g. COM-0000077, MOT-0000001)
    POLICY_ID_REGEX = re.compile(r"\b([A-Z]{3}-\d{7})\b", re.IGNORECASE)

    def __init__(
        self,
        llm: Optional[LLMProvider] = None,
        system_prompt: str = PLANNER_SYSTEM_PROMPT,
    ) -> None:
        self._llm = llm or OllamaLLM(model_name="qwen3:8b")
        self.system_prompt = system_prompt

    def _fallback_heuristic_plan(self, query: str) -> ExecutionPlan:
        """Deterministic rule-based planning fallback if LLM response cannot be parsed."""
        clean_q = query.strip()
        policy_match = self.POLICY_ID_REGEX.search(clean_q)

        # Check for policy lookup
        if policy_match and any(w in clean_q.lower() for w in ["deductible", "policy", "premium", "status", "details", "limit"]):
            policy_id = policy_match.group(1).upper()
            return ExecutionPlan(
                query=clean_q,
                capabilities=[CapabilityType.CUSTOMER_DATA],
                steps=[
                    PlanStep(
                        step_number=1,
                        capability=CapabilityType.CUSTOMER_DATA,
                        tool_name="get_policy_details",
                        arguments={"policy_id": policy_id},
                        rationale=f"Retrieve live policy data for {policy_id} from Insurance API",
                    )
                ],
                reasoning="Heuristic fallback identified policy reference inquiry.",
            )

        # Check for greetings
        if clean_q.lower() in ["hi", "hello", "hey", "good morning", "good afternoon"]:
            return ExecutionPlan(
                query=clean_q,
                capabilities=[CapabilityType.DIRECT_ANSWER],
                steps=[],
                direct_response="Hello! I am your Insurance Support Agent. How can I assist you with your policy or claim today?",
            )

        # Default fallback to RAG for general insurance questions
        return ExecutionPlan(
            query=clean_q,
            capabilities=[CapabilityType.RAG],
            steps=[
                PlanStep(
                    step_number=1,
                    capability=CapabilityType.RAG,
                    tool_name="rag_retrieval",
                    arguments={"query": clean_q},
                    rationale="Search corpus for relevant policy documentation and records",
                )
            ],
            reasoning="Defaulted to knowledge base RAG search.",
        )

    def plan(self, query: str) -> ExecutionPlan:
        """
        Analyze user query and return an approved ExecutionPlan.

        Args:
            query: The user's prompt or question.

        Returns:
            ExecutionPlan instance containing steps and capabilities.
        """
        if not query or not query.strip():
            return ExecutionPlan(
                query="",
                capabilities=[CapabilityType.DIRECT_ANSWER],
                steps=[],
                direct_response="Please provide an insurance inquiry or policy number.",
            )

        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(
                role="user",
                content=f"Plan the execution for this user request:\n\"{query.strip()}\"\n\nReturn JSON only:",
            ),
        ]

        try:
            resp = self._llm.chat(messages=messages, temperature=0.0)
            raw_text = resp.content.strip()

            # Strip markdown code blocks if wrapped in ```json ... ```
            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_text = "\n".join(lines).strip()

            data = json.loads(raw_text)

            # Validate and construct ExecutionPlan
            plan = ExecutionPlan(
                query=query.strip(),
                capabilities=[CapabilityType(c) for c in data.get("capabilities", [])],
                steps=[
                    PlanStep(
                        step_number=s.get("step_number", idx),
                        capability=CapabilityType(s.get("capability", "customer_data")),
                        tool_name=s.get("tool_name"),
                        arguments=s.get("arguments", {}),
                        rationale=s.get("rationale", "Execute planned step"),
                    )
                    for idx, s in enumerate(data.get("steps", []), 1)
                ],
                reasoning=data.get("reasoning"),
                direct_response=data.get("direct_response"),
            )
            return plan

        except Exception as e:
            logger.warning(f"Failed to generate plan via LLM ({e}); falling back to heuristic planner.")
            return self._fallback_heuristic_plan(query)
