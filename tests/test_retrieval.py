"""Offline tests for the retrieval layer (no Pinecone / no network).

Dense search and reranking are exercised by src/retrieval/retrieval_smoke.py
(they require the live index). Here we test the deterministic pieces: RRF, the
filter matcher, query expansion, stance classification, and the BM25 lexical
signal (which only needs the local JSONL artifacts).
"""

from src.ingestion.config import CHILD_CHUNKS_PATH
from src.retrieval.hybrid_retriever import _match, lexical_search
from src.retrieval.evidence_selector import classify_stance
from src.retrieval.query_expansion import expand_query, expansion_terms
from src.retrieval.rrf import rrf_fuse


def test_rrf_rewards_agreement_and_top_ranks():
    dense = ["a", "b", "c"]
    lexical = ["b", "a", "d"]
    scores = rrf_fuse([dense, lexical])
    # 'a' (ranks 0,1) and 'b' (ranks 1,0) beat items in only one list
    assert scores["a"] > scores["c"]
    assert scores["b"] > scores["d"]


def test_filter_matcher_eq_in_and():
    cand = {"legal_area": "motor_accident", "vehicle_tags": ["truck", "commercial_vehicle"]}
    assert _match(cand, {"legal_area": {"$eq": "motor_accident"}})
    assert not _match(cand, {"legal_area": {"$eq": "criminal"}})
    assert _match(cand, {"vehicle_tags": {"$in": ["truck"]}})
    assert not _match(cand, {"vehicle_tags": {"$in": ["bus"]}})
    assert _match(cand, {"$and": [{"legal_area": {"$eq": "motor_accident"}},
                                  {"vehicle_tags": {"$in": ["truck"]}}]})


def test_query_expansion_triggers_synonyms():
    assert "driving license" in expansion_terms("valid driving licence dispute")
    out = expand_query("commercial vehicle accident")
    assert "truck" in out and out.startswith("commercial vehicle accident")


def test_classify_stance_for_claimant():
    supporting = {"doc_id": "DOC_001", "stance_tags": ["adverse_to_insurer"]}
    adverse = {"doc_id": "DOC_001", "stance_tags": ["insurer_supporting"]}
    mixed = {"doc_id": "DOC_001", "stance_tags": ["adverse_to_insurer", "adverse_to_claimant"]}
    assert classify_stance(supporting, "claimant") == "supporting"
    assert classify_stance(adverse, "claimant") == "adverse"
    assert classify_stance(mixed, "claimant") == "mixed"


def test_bm25_lexical_finds_distinctive_terms():
    if not CHILD_CHUNKS_PATH.exists():
        return
    hits = lexical_search("trademark infringement passing off", top_k=8)
    assert hits, "BM25 returned no hits for a distinctive query"
    # the top lexical hit for 'trademark' should come from a trademark-IP doc
    from src.retrieval.corpus_store import load_corpus
    cb = load_corpus()["child_by_id"]
    top_areas = [cb[rid]["legal_area"] for rid, _ in hits[:5]]
    assert "trademark_ip" in top_areas
