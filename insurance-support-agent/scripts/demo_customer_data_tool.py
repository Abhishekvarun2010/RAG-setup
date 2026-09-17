"""
Live Demonstration: Customer Data Capability & Agent Runtime.

Demonstrates:
User -> Agent Runtime -> Capability Planner (Qwen3 8B) -> PolicyDataTool -> Insurance API -> Local PostgreSQL

Scenarios:
1. Structured Policy Lookup for COM-0000077 (Deductible, Status, Insured Name).
2. Vehicle Policy Lookup for MOT-0000001 (Premium, Deductible, Limits).
3. Non-Existent Policy Lookup (Graceful error handling).
4. Direct Conversational Greeting (Direct answer capability without API execution).
"""
import asyncio
from pathlib import Path
import sys
import threading
import time
import uvicorn

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from apps.agent import (
    AgentPlanner,
    AgentRuntime,
    OllamaLLM,
    PolicyDataTool,
)
from services.insurance_api.database import (
    get_db_session,
    init_db,
    seed_database_from_strata,
)
from services.insurance_api.main import app as insurance_api_app

output_lines = []

def log(msg: str = ""):
    print(msg)
    output_lines.append(msg)


class UvicornServerThread(threading.Thread):
    """Run FastAPI Insurance API in a background daemon thread on port 8001."""
    def __init__(self, host="127.0.0.1", port=8001):
        super().__init__(daemon=True)
        config = uvicorn.Config(insurance_api_app, host=host, port=port, log_level="warning")
        self.server = uvicorn.Server(config)

    def run(self):
        self.server.run()

    def stop(self):
        self.server.should_exit = True


def print_agent_run(title: str, response):
    log("\n" + "=" * 80)
    log(f"SCENARIO: {title}")
    log("=" * 80)
    log(f"User Query      : {response.query}")
    log(f"Capabilities    : {[c.value for c in response.plan.capabilities]}")
    log(f"Planned Steps   : {len(response.plan.steps)}")
    for s in response.plan.steps:
        log(f"  - Step {s.step_number}: [{s.capability.value}] Tool: '{s.tool_name}' | Args: {s.arguments}")
        log(f"    Rationale: {s.rationale}")
    
    if response.tool_results:
        log("\n--- [Executed Tool Results (via Insurance API -> PostgreSQL)] ---")
        for idx, tr in enumerate(response.tool_results, 1):
            status_str = "SUCCESS" if tr.success else "FAILED"
            log(f"  [{idx}] Status: {status_str}")
            log(f"      Output:\n{tr.formatted_output}")
    
    log("\n--- [Final Synthesized Response] ---")
    log(response.answer)
    log(f"\nExecution Latency: {response.latency_seconds:.2f}s")


def main():
    log("=" * 80)
    log("STARTING AGENT RUNTIME & CUSTOMER DATA CAPABILITY DEMONSTRATION")
    log("=" * 80)

    # 1. Initialize local PostgreSQL and seed from Strata
    log("\n[1] Initializing and verifying local PostgreSQL ('insurance_db')...")
    init_db()
    with get_db_session() as session:
        stats = seed_database_from_strata(session)
        log(f"    ✓ Local PostgreSQL verified & seeded: {stats}")

    # 2. Start Insurance API background server on port 8001
    log("\n[2] Starting Insurance API server on http://localhost:8001...")
    api_server = UvicornServerThread(port=8001)
    api_server.start()
    time.sleep(1.0)
    log("    ✓ Insurance API active on port 8001.")

    # 3. Instantiate Agent Runtime components
    log("\n[3] Initializing Agent Runtime and PolicyDataTool...")
    llm = OllamaLLM(model_name="qwen3:8b", base_url="http://localhost:11434")
    planner = AgentPlanner(llm=llm)
    policy_tool = PolicyDataTool(base_url="http://127.0.0.1:8001")
    
    runtime = AgentRuntime(
        planner=planner,
        tools={"get_policy_details": policy_tool},
        llm=llm,
    )
    log("    ✓ Agent Runtime ready (decides capabilities: RAG, Customer Data, Actions, Direct Answer).")

    # -------------------------------------------------------------
    # Scenario 1: Customer Data Lookup for COM-0000077
    # -------------------------------------------------------------
    resp1 = runtime.run("What is my deductible and who is the insured party on commercial policy COM-0000077?")
    print_agent_run("Scenario 1: Live Policy Lookup (COM-0000077)", resp1)

    # -------------------------------------------------------------
    # Scenario 2: Motor Policy Lookup for MOT-0000001
    # -------------------------------------------------------------
    resp2 = runtime.run("Can you look up policy MOT-0000001 and tell me what the annual premium and deductible are?")
    print_agent_run("Scenario 2: Motor Policy Details (MOT-0000001)", resp2)

    # -------------------------------------------------------------
    # Scenario 3: Non-Existent Policy Error Handling
    # -------------------------------------------------------------
    resp3 = runtime.run("What is the status of policy COM-9999999?")
    print_agent_run("Scenario 3: Non-Existent Policy Handling (COM-9999999)", resp3)

    # -------------------------------------------------------------
    # Scenario 4: Direct Conversational Greeting
    # -------------------------------------------------------------
    resp4 = runtime.run("Hello! Can you help me today?")
    print_agent_run("Scenario 4: Direct Conversational Greeting", resp4)

    # 4. Save output to artifact file
    out_file = BASE_DIR / "customer_data_tool_demonstration.txt"
    out_file.write_text("\n".join(output_lines), encoding="utf-8")
    log(f"\n✓ Saved full demonstration output to {out_file}")

    api_server.stop()


if __name__ == "__main__":
    main()
