"""Offline tests for the agent graph (no network): compilation + routing + the
deterministic interpreter post-processing. Full end-to-end runs (which call Gemini
and Pinecone) live in src/agent/agent_smoke.py.
"""

from src.agent.graph import build_graph
from src.agent.nodes.interpreter import route


def test_graph_compiles_with_expected_nodes():
    app = build_graph()
    nodes = set(app.get_graph().nodes)
    for n in ("interpret", "plan", "expand", "support", "adverse", "compensation",
              "select", "analyze", "synthesize", "validate", "simple_retrieve",
              "simple_answer", "clarify"):
        assert n in nodes


def test_route_maps_complexity():
    assert route({"query_complexity": "deep"}) == "deep"
    assert route({"query_complexity": "simple"}) == "simple"
    assert route({"query_complexity": "clarify"}) == "clarify"
    assert route({}) == "deep"  # safe default
