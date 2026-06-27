"""UI render tests. We can't drive Streamlit's browser here, so we replace the
`st` object with a mock and run every render function against representative agent
states — this exercises all the rendering logic and catches KeyError/format bugs.
"""

from unittest.mock import MagicMock

import app


def _candidate(doc="DOC_001", rid="DOC_001_child_0007", rerank=0.9):
    return {
        "doc_id": doc, "record_id": rid, "parent_id": f"{doc}_parent_0001",
        "case_title": "United India Insurance vs Neelam Devi", "court": "P&H HC", "year": 2023,
        "chunk_type": "driving_license_finding", "legal_area": "motor_accident",
        "issue_tags": ["driving_license_validity"], "stance_tags": ["adverse_to_insurer"],
        "vehicle_tags": ["truck"], "page_start": 5, "page_end": 6, "text": "The driving licence ...",
        "dense_score": 0.87, "dense_rank": 2, "lexical_score": 27.5, "lexical_rank": 5,
        "rrf_score": 0.031, "metadata_boost": 0.02, "fused_score": 0.051,
        "rerank_score": rerank, "passes": ["support"], "client_stance": "supporting",
    }


def _deep_state():
    c1, c2 = _candidate(), _candidate("DOC_031", "DOC_031_child_0003", 0.8)
    c2["client_stance"] = "adverse"
    return {
        "user_query": "q", "detected_intent": "deep_precedent_research", "query_complexity": "deep",
        "client_side": "claimant", "legal_area": "motor_accident",
        "extracted_facts": {"death": True, "commercial_vehicle": True},
        "issue_map": ["insurer liability", "driving licence validity"],
        "research_plan": ["step 1", "step 2"],
        "generated_queries": [{"purpose": "support", "query": "pay and recover"},
                              {"purpose": "adverse", "query": "insurer exonerated"}],
        "retrieval_filters": {"boosts": {"issue_tags": ["driving_license_validity"]}},
        "retrieved_candidates": [c1, c2],
        "reranked_candidates": [c1, c2],
        "selected_evidence": {"supporting": [c1], "adverse": [c2], "neutral": [],
                              "selected": [c1, c2], "parents": {"DOC_001_parent_0001": "parent text"}},
        "case_analyses": [{"doc_id": "DOC_001", "case_title": "x", "client_impact": "supporting",
                           "confidence": "high", "holding": "h", "why": "w", "evidence_ids": ["DOC_001_child_0007"]}],
        "validation_report": {"passes": True, "adverse_section_present": True,
                              "uncited_or_invented_docs": [], "cited_docs": ["DOC_001", "DOC_031"],
                              "issues": [], "required_fixes": []},
        "final_answer": "## Answer\nSupporting: DOC_001 ...",
    }


def _simple_state():
    return {
        "user_query": "Which judgments involve commercial vehicles?",
        "detected_intent": "simple_lookup", "query_complexity": "simple",
        "client_side": "unknown", "legal_area": "motor_accident",
        "extracted_facts": {}, "issue_map": [],
        "retrieved_candidates": [_candidate()],
        "selected_evidence": {"doc_matches": [
            {"doc_id": "DOC_027", "case_title": "The Manager vs Chennamma", "legal_area": "motor_accident",
             "issue_tags": ["commercial_vehicle_use"], "vehicle_tags": ["truck"], "score": 0.82}]},
        "final_answer": "DOC_027 involves a commercial vehicle.",
    }


def _run_all(state):
    app.st = MagicMock()  # absorb all Streamlit calls; exercise pure render logic
    app.render_routing(state)
    app.render_plan_queries(state)
    app.render_retrieved(state)
    app.render_reranked(state)
    app.render_selected(state)
    app.render_analyses(state)
    app.render_validation(state)
    assert app.st.markdown.called or app.st.dataframe.called or app.st.caption.called


def test_render_deep_state():
    _run_all(_deep_state())


def test_render_simple_state():
    _run_all(_simple_state())


def test_render_handles_empty_state():
    _run_all({"final_answer": "", "selected_evidence": {}})
