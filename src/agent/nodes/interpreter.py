"""Input Interpreter node (§4.1, §9.3). Classifies the request and extracts the
legal frame. Flexible — works on any prompt, not just the Lakshmi Devi brief.
"""

from __future__ import annotations

from src.agent.llm import call_json, render

_SYSTEM = (
    "You classify legal research requests over a FIXED corpus of Indian court judgments "
    "(mostly motor-accident insurance cases, plus criminal, trademark, banking, civil "
    "procedure, etc.). You do NOT answer the legal question; you only analyze the request. "
    "Return JSON only."
)

_USER = """Analyze this user request and return JSON with EXACTLY these keys:
- "intent": short label, one of: "simple_lookup", "deep_precedent_research", "comparison", "no_answer_check", "other"
- "complexity": "simple" or "deep"
    * "deep" = needs supporting + adverse precedents and/or litigation strategy, multi-issue research, or "find precedents that support/oppose ...".
    * "simple" = factual/listing/metadata lookups ("which judgments involve commercial vehicles?", "which cases discuss fake licence?").
- "client_side": "claimant" | "insurer" | "owner_driver" | "unknown" (whose side the user is on, if inferable)
- "legal_area": best guess like "motor_accident", "criminal", "trademark_ip", "civil_procedure", etc., or "unknown"
- "issues": list of concise legal issues raised (e.g., ["driving licence validity","insurer liability","compensation"])
- "facts": object of any concrete facts present (e.g., {"death": true, "commercial_vehicle": true, "driver_unlicensed": true, "income": 35000, "age": 42, "dependents": 3})
- "requires_adverse": boolean (true for deep research / strategy)
- "requires_strategy": boolean
- "requires_compensation": boolean (true if compensation/quantum is relevant)
- "requires_clarification": boolean (true ONLY if the request is too vague to act on at all)

USER REQUEST:
\"\"\"{query}\"\"\"
"""


def interpret(state: dict) -> dict:
    query = state["user_query"]
    data = call_json(_SYSTEM, render(_USER, query=query)) or {}

    complexity = data.get("complexity", "deep")
    if data.get("requires_clarification"):
        complexity = "clarify"
    if complexity not in ("simple", "deep", "clarify"):
        complexity = "deep"

    return {
        "detected_intent": data.get("intent", "deep_precedent_research"),
        "query_complexity": complexity,
        "client_side": data.get("client_side", "unknown"),
        "legal_area": data.get("legal_area", "unknown"),
        "issue_map": data.get("issues", []) or [],
        "extracted_facts": data.get("facts", {}) or {},
        "requires_adverse": bool(data.get("requires_adverse", complexity == "deep")),
        "requires_strategy": bool(data.get("requires_strategy", complexity == "deep")),
        "requires_compensation": bool(data.get("requires_compensation", False)),
        "router_reason": f"intent={data.get('intent')} complexity={complexity}",
    }


def route(state: dict) -> str:
    """Intent Router (§4.2): semantic routing, not a hard-coded Lakshmi branch."""
    return state.get("query_complexity", "deep")
