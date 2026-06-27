"""Agent-behavior metrics (§12.3) + helpers to extract the agent's doc lists
from the final state for the retrieval/answer metrics.
"""

from __future__ import annotations


def agent_ranked_docs(state: dict) -> list[str]:
    """Doc-level ranked list the agent retrieved (for precision/recall@k)."""
    docs: list[str] = []
    for c in state.get("reranked_candidates") or []:
        if c["doc_id"] not in docs:
            docs.append(c["doc_id"])
    dm = (state.get("selected_evidence") or {}).get("doc_matches")
    if dm:
        for d in dm:
            if d["doc_id"] not in docs:
                docs.append(d["doc_id"])
    for c in sorted(state.get("retrieved_candidates") or [],
                    key=lambda c: c.get("fused_score", 0), reverse=True):
        if c["doc_id"] not in docs:
            docs.append(c["doc_id"])
    return docs


def agent_presented_docs(state: dict) -> set[str]:
    """Docs the agent presents as relevant (for answer-level precision)."""
    sel = state.get("selected_evidence") or {}
    if sel.get("doc_matches") is not None:  # simple path
        return {d["doc_id"] for d in sel["doc_matches"]}
    docs = set()
    for bucket in ("supporting", "adverse", "neutral"):
        docs |= {c["doc_id"] for c in sel.get(bucket, [])}
    for a in state.get("case_analyses") or []:
        if a.get("doc_id"):
            docs.add(a["doc_id"])
    return docs


def agent_adverse_docs(state: dict) -> set[str]:
    """Docs the agent flags as adverse (eval dimension 4)."""
    sel = state.get("selected_evidence") or {}
    docs = {c["doc_id"] for c in sel.get("adverse", [])}
    for a in state.get("case_analyses") or []:
        if a.get("client_impact") == "adverse" and a.get("doc_id"):
            docs.add(a["doc_id"])
    return docs


def agent_supporting_docs(state: dict) -> set[str]:
    sel = state.get("selected_evidence") or {}
    docs = {c["doc_id"] for c in sel.get("supporting", [])}
    for a in state.get("case_analyses") or []:
        if a.get("client_impact") in ("supporting", "mixed") and a.get("doc_id"):
            docs.add(a["doc_id"])
    return docs


def router_correct(q, state: dict) -> bool:
    """Did the agent pick simple vs deep correctly per the golden workflow?"""
    expected_deep = q.expects_deep()
    actual_deep = state.get("query_complexity") == "deep"
    return expected_deep == actual_deep


def adverse_search_executed(state: dict) -> bool:
    """Did the agent actually run an adverse retrieval pass?"""
    queries = state.get("generated_queries") or []
    if any(x.get("purpose") == "adverse" for x in queries):
        return True
    return bool((state.get("selected_evidence") or {}).get("adverse"))
