"""
Unit tests for AgentPlanner and AgentRuntime.
"""
from typing import Dict, List, Optional
import pytest

from apps.agent.llm import LLMProvider
from apps.agent.models import ChatMessage, LLMResponse
from apps.agent.planner import AgentPlanner, CapabilityType, ExecutionPlan, PlanStep
from apps.agent.runtime import AgentRuntime
from apps.agent.tools.base import BaseTool, ToolResult


class MockPlannerLLM(LLMProvider):
    def __init__(self, response_text: str):
        self.response_text = response_text

    @property
    def model_name(self) -> str:
        return "mock-qwen3"

    def chat(self, messages, temperature=0.0, **kwargs) -> LLMResponse:
        return LLMResponse(content=self.response_text, model="mock-qwen3")

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        return self.chat([ChatMessage(role="user", content=prompt)])


class FakePolicyTool(BaseTool):
    @property
    def name(self) -> str:
        return "get_policy_details"

    @property
    def description(self) -> str:
        return "Fake policy tool"

    @property
    def parameters(self) -> Dict:
        return {"type": "object", "properties": {"policy_id": {"type": "string"}}}

    def execute(self, **kwargs) -> ToolResult:
        p_id = kwargs.get("policy_id")
        if p_id == "COM-0000077":
            return ToolResult.ok(
                data={"policy_id": "COM-0000077", "deductible": 5000.0, "status": "active"},
                formatted_output="Policy COM-0000077: Status active, Deductible €5,000.00.",
            )
        return ToolResult.fail(error="Not found", formatted_output="Policy not found.")


def test_planner_direct_answer_greeting():
    json_plan = '{"capabilities": ["direct_answer"], "steps": [], "direct_response": "Hello! How can I help you?"}'
    planner = AgentPlanner(llm=MockPlannerLLM(json_plan))

    plan = planner.plan("Hello")
    assert CapabilityType.DIRECT_ANSWER in plan.capabilities
    assert plan.direct_response == "Hello! How can I help you?"
    assert len(plan.steps) == 0


def test_planner_customer_data_selection():
    json_plan = '''{
        "capabilities": ["customer_data"],
        "steps": [
            {
                "step_number": 1,
                "capability": "customer_data",
                "tool_name": "get_policy_details",
                "arguments": {"policy_id": "COM-0000077"},
                "rationale": "Retrieve live policy deductible"
            }
        ]
    }'''
    planner = AgentPlanner(llm=MockPlannerLLM(json_plan))

    plan = planner.plan("What is my deductible for COM-0000077?")
    assert CapabilityType.CUSTOMER_DATA in plan.capabilities
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_name == "get_policy_details"
    assert plan.steps[0].arguments["policy_id"] == "COM-0000077"


def test_planner_heuristic_fallback_on_invalid_json():
    planner = AgentPlanner(llm=MockPlannerLLM("Sorry, I am not generating JSON."))

    # Heuristic should catch the policy number COM-0000077 and "deductible"
    plan = planner.plan("What is my deductible on policy COM-0000077?")
    assert CapabilityType.CUSTOMER_DATA in plan.capabilities
    assert len(plan.steps) == 1
    assert plan.steps[0].arguments["policy_id"] == "COM-0000077"


def test_agent_runtime_greeting_flow():
    json_plan = '{"capabilities": ["direct_answer"], "steps": [], "direct_response": "Hello, welcome to insurance support!"}'
    planner = AgentPlanner(llm=MockPlannerLLM(json_plan))
    runtime = AgentRuntime(planner=planner, tools={}, llm=MockPlannerLLM("unused"))

    resp = runtime.run("Hi")
    assert resp.answer == "Hello, welcome to insurance support!"
    assert len(resp.tool_results) == 0


def test_agent_runtime_customer_data_flow():
    json_plan = '''{
        "capabilities": ["customer_data"],
        "steps": [
            {
                "step_number": 1,
                "capability": "customer_data",
                "tool_name": "get_policy_details",
                "arguments": {"policy_id": "COM-0000077"},
                "rationale": "Lookup COM-0000077"
            }
        ]
    }'''
    planner = AgentPlanner(llm=MockPlannerLLM(json_plan))
    synthesis_llm = MockPlannerLLM("Your deductible for policy COM-0000077 is €5,000.00.")

    tool = FakePolicyTool()
    runtime = AgentRuntime(
        planner=planner,
        tools={"get_policy_details": tool},
        llm=synthesis_llm,
    )

    resp = runtime.run("What is my deductible for COM-0000077?")
    assert len(resp.tool_results) == 1
    assert resp.tool_results[0].success is True
    assert "€5,000.00" in resp.tool_results[0].formatted_output
    assert "€5,000.00" in resp.answer
