"""Offline tests for the evaluation metric functions (no network)."""

from src.evaluation.agent_metrics import agent_adverse_docs, agent_ranked_docs
from src.evaluation.answer_metrics import extract_cited_docs
from src.evaluation.retrieval_metrics import (
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    set_recall,
)


def test_precision_recall():
    ranked = ["DOC_001", "DOC_999", "DOC_002", "DOC_003"]
    gold = {"DOC_001", "DOC_002"}
    assert precision_at_k(ranked, gold, 2) == 0.5
    assert recall_at_k(ranked, gold, 3) == 1.0
    assert recall_at_k(ranked, gold, 1) == 0.5
    assert recall_at_k(ranked, set(), 5) == 1.0  # nothing to recall


def test_mrr_and_ndcg():
    ranked = ["DOC_999", "DOC_001", "DOC_002"]
    gold = {"DOC_001"}
    assert mrr(ranked, gold) == 0.5
    # gold at rank 0 gives perfect nDCG
    assert ndcg_at_k(["DOC_001", "DOC_999"], {"DOC_001"}, 5) == 1.0
    assert ndcg_at_k(["DOC_999", "DOC_001"], {"DOC_001"}, 5) < 1.0


def test_set_recall():
    assert set_recall({"DOC_001"}, {"DOC_001", "DOC_002"}) == 0.5
    assert set_recall(set(), set()) == 1.0


def test_extract_cited_docs():
    assert extract_cited_docs("see DOC_001 and DOC_031, also doc_x") == {"DOC_001", "DOC_031"}


def test_agent_doc_extractors():
    state = {
        "reranked_candidates": [{"doc_id": "DOC_001"}, {"doc_id": "DOC_001"}, {"doc_id": "DOC_005"}],
        "retrieved_candidates": [{"doc_id": "DOC_009", "fused_score": 0.1}],
        "selected_evidence": {"adverse": [{"doc_id": "DOC_031"}], "supporting": [], "neutral": []},
        "case_analyses": [{"doc_id": "DOC_002", "client_impact": "adverse"}],
    }
    assert agent_ranked_docs(state) == ["DOC_001", "DOC_005", "DOC_009"]
    assert agent_adverse_docs(state) == {"DOC_031", "DOC_002"}
