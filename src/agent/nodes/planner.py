"""Research Planner node (§4.3, §9.4). Turns the interpreted task into a concrete
research plan, without baking in Lakshmi-Devi assumptions unless the prompt has them.
"""

from __future__ import annotations

from src.agent.llm import call_json, render

_SYSTEM = (
    "You plan precedent research over a fixed corpus of Indian judgments. "
    "Return JSON only. Do not assume facts not present in the request."
)

_USER = """Create a research plan for this task.

Client side: {client_side}
Legal area: {legal_area}
Issues: {issues}
Facts: {facts}
Needs adverse search: {requires_adverse}
Needs compensation analysis: {requires_compensation}

Return JSON with keys:
- "research_plan": ordered list of 4-7 concrete research steps. MUST include a supporting-precedent
  step, an adverse-precedent step (search as opposing counsel) when adverse is required, a
  compensation step when relevant, and a factual/statutory similarity step.
- "issue_map": deduplicated list of the distinct legal issues to cover.
"""


def plan(state: dict) -> dict:
    data = call_json(_SYSTEM, render(_USER,
        client_side=state.get("client_side"),
        legal_area=state.get("legal_area"),
        issues=state.get("issue_map"),
        facts=state.get("extracted_facts"),
        requires_adverse=state.get("requires_adverse"),
        requires_compensation=state.get("requires_compensation"),
    )) or {}
    return {
        "research_plan": data.get("research_plan", []) or [],
        "issue_map": data.get("issue_map") or state.get("issue_map", []),
    }
