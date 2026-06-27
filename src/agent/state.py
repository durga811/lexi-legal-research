"""Typed LangGraph state (§3.3). Nodes return partial dicts that LangGraph merges.

The UI reads these fields directly to render the visible evidence trace (§9.2):
generated_queries, retrieved_candidates (with dense/lexical/RRF/boost scores),
reranked_candidates (with rerank scores), selected_evidence, case_analyses,
validation_report, and final_answer.
"""

from __future__ import annotations

from typing import Any, TypedDict


class LegalResearchState(TypedDict, total=False):
    user_query: str

    # interpreter
    detected_intent: str
    query_complexity: str          # "simple" | "deep" | "clarify"
    client_side: str | None        # claimant | insurer | owner_driver | unknown
    legal_area: str | None
    extracted_facts: dict[str, Any]
    issue_map: list[str]
    requires_adverse: bool
    requires_strategy: bool
    requires_compensation: bool
    router_reason: str

    # planning + query generation
    research_plan: list[str]
    generated_queries: list[dict]  # {purpose, query, boosts}
    retrieval_filters: dict

    # retrieval
    retrieved_candidates: list[dict]   # merged fused candidates across passes
    reranked_candidates: list[dict]
    selected_evidence: dict            # {supporting, adverse, neutral, parents}

    # analysis + answer
    case_analyses: list[dict]
    compensation_findings: dict | None
    final_answer: str
    validation_report: dict

    error: str | None
