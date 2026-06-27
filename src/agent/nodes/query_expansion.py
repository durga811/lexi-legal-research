"""Query Expansion node (§4.4, §9). The LLM generates several purpose-tagged
search queries (support / adverse / compensation / factual / statutory) and
suggests metadata boost tags. Boost tags are validated against the controlled
vocabulary so a hallucinated tag can't reach the retriever.
"""

from __future__ import annotations

from src.agent.llm import call_json, render
from src.ingestion.vocab import ISSUE_TAGS, VEHICLE_TAGS

_SYSTEM = (
    "You generate retrieval queries for a hybrid (dense + lexical) search over Indian "
    "court judgments. Return JSON only. Use exact legal phrasing where it helps lexical recall."
)

_VALID_PURPOSES = {"support", "adverse", "compensation", "factual", "statutory"}

_USER = """Generate search queries for this research task.

Client side: {client_side}
Issues: {issues}
Facts: {facts}
Research plan: {plan}
Needs adverse: {requires_adverse}
Needs compensation: {requires_compensation}

Return JSON with key "queries": a list of 5-9 objects, each:
  {"purpose": one of ["support","adverse","compensation","factual","statutory"],
    "query": "a focused natural-language search query"}
Rules:
- Include at least 2 "support" queries from the CLIENT's perspective.
- If adverse is needed, include at least 2 "adverse" queries written from the OPPOSING side
  (e.g., for a claimant client, search for insurer-exonerating / fake-licence-proved / contributory-negligence cases).
- If compensation is needed, include 1-2 "compensation" queries (multiplier, future prospects, loss of dependency).
- Include 1 "factual"/"statutory" query for exact terms (statute sections, case-specific phrases).

Also return "boost_issue_tags" (subset of these only): {issue_tags}
and "boost_vehicle_tags" (subset of these only): {vehicle_tags}
"""


def expand_queries(state: dict) -> dict:
    data = call_json(_SYSTEM, render(_USER,
        client_side=state.get("client_side"),
        issues=state.get("issue_map"),
        facts=state.get("extracted_facts"),
        plan=state.get("research_plan"),
        requires_adverse=state.get("requires_adverse"),
        requires_compensation=state.get("requires_compensation"),
        issue_tags=sorted(ISSUE_TAGS),
        vehicle_tags=sorted(VEHICLE_TAGS),
    )) or {}

    queries = []
    for q in data.get("queries", []) or []:
        purpose = q.get("purpose", "support")
        if purpose not in _VALID_PURPOSES:
            purpose = "support"
        text = (q.get("query") or "").strip()
        if text:
            queries.append({"purpose": purpose, "query": text})

    # Fallback: guarantee at least a support + adverse query from the user request.
    if not any(q["purpose"] == "support" for q in queries):
        queries.append({"purpose": "support", "query": state["user_query"]})
    if state.get("requires_adverse") and not any(q["purpose"] == "adverse" for q in queries):
        queries.append({"purpose": "adverse",
                        "query": "cases where the opposing party succeeded against this position"})

    boosts = {
        "issue_tags": [t for t in (data.get("boost_issue_tags") or []) if t in ISSUE_TAGS],
        "vehicle_tags": [t for t in (data.get("boost_vehicle_tags") or []) if t in VEHICLE_TAGS],
    }
    boosts = {k: v for k, v in boosts.items() if v}

    return {"generated_queries": queries, "retrieval_filters": {"boosts": boosts}}
