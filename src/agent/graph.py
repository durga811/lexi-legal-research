"""LangGraph wiring (§2.2, §3, §18.1). A stateful, inspectable graph that routes
between a simple-lookup path and a deep precedent-research path — semantically,
not via hard-coded branches.

Node-file map (guide §4 node design):
  interpreter.interpret/route   -> Input Interpreter + Intent Router
  planner.plan                  -> Research Planner
  query_expansion.expand_queries-> Query Expansion
  retrieval.{support,adverse,compensation}_retrieval, select_evidence_node -> §4.5–4.8
  analysis.{case_analyzer, strategy_synthesizer, validator}               -> §4.9–4.11
  simple.{simple_retrieval, simple_answer, clarify}                       -> §3.1 / clarify
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from src.agent.nodes.analysis import case_analyzer, strategy_synthesizer, validator
from src.agent.nodes.interpreter import interpret, route
from src.agent.nodes.planner import plan
from src.agent.nodes.query_expansion import expand_queries
from src.agent.nodes.retrieval import (
    adverse_retrieval,
    compensation_retrieval,
    select_evidence_node,
    support_retrieval,
)
from src.agent.nodes.simple import clarify, simple_answer, simple_retrieval
from src.agent.state import LegalResearchState


def build_graph():
    g = StateGraph(LegalResearchState)

    g.add_node("interpret", interpret)
    # deep path
    g.add_node("plan", plan)
    g.add_node("expand", expand_queries)
    g.add_node("support", support_retrieval)
    g.add_node("adverse", adverse_retrieval)
    g.add_node("compensation", compensation_retrieval)
    g.add_node("select", select_evidence_node)
    g.add_node("analyze", case_analyzer)
    g.add_node("synthesize", strategy_synthesizer)
    g.add_node("validate", validator)
    # simple + clarify paths
    g.add_node("simple_retrieve", simple_retrieval)
    g.add_node("simple_answer", simple_answer)
    g.add_node("clarify", clarify)

    g.add_edge(START, "interpret")
    g.add_conditional_edges("interpret", route, {
        "deep": "plan",
        "simple": "simple_retrieve",
        "clarify": "clarify",
    })

    # deep chain
    g.add_edge("plan", "expand")
    g.add_edge("expand", "support")
    g.add_edge("support", "adverse")
    g.add_edge("adverse", "compensation")
    g.add_edge("compensation", "select")
    g.add_edge("select", "analyze")
    g.add_edge("analyze", "synthesize")
    g.add_edge("synthesize", "validate")
    g.add_edge("validate", END)

    # simple chain
    g.add_edge("simple_retrieve", "simple_answer")
    g.add_edge("simple_answer", END)
    g.add_edge("clarify", END)

    return g.compile()


@lru_cache(maxsize=1)
def get_app():
    return build_graph()


def run_agent(query: str, recursion_limit: int = 30) -> dict:
    """Run the agent end-to-end; returns the full final state for the UI/eval."""
    app = get_app()
    return app.invoke({"user_query": query}, config={"recursion_limit": recursion_limit})
