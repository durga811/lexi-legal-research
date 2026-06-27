"""Hybrid retrieval: dense (Pinecone e5) + lexical (BM25) + metadata, fused by RRF.

Each returned candidate carries every sub-score (dense, lexical, RRF, metadata
boost, fused) so the UI can render the full evidence trace (§9.2). Reranking is a
separate stage (reranker.py).
"""

from __future__ import annotations

from src.retrieval.corpus_store import get_bm25, load_corpus, tokenize
from src.retrieval.pinecone_client import NAMESPACE, get_index
from src.retrieval.rrf import rrf_fuse

DENSE_K = 40
LEXICAL_K = 40
FUSED_K = 40
BOOST_WEIGHT = 0.005  # per matching tag; small relative to top RRF (~0.016)


# --------------------------------------------------------------------------- #
# Filter helpers (Pinecone-style dict, reused for post-filtering BM25 results)
# --------------------------------------------------------------------------- #
def _match(cand: dict, flt: dict) -> bool:
    if "$and" in flt:
        return all(_match(cand, sub) for sub in flt["$and"])
    if "$or" in flt:
        return any(_match(cand, sub) for sub in flt["$or"])
    for field, cond in flt.items():
        val = cand.get(field)
        if isinstance(cond, dict):
            if "$eq" in cond and val != cond["$eq"]:
                return False
            if "$in" in cond:
                want = cond["$in"]
                if isinstance(val, list):
                    if not set(val) & set(want):
                        return False
                elif val not in want:
                    return False
        elif val != cond:
            return False
    return True


def _child_filter(meta_filter: dict | None) -> dict:
    base = {"record_type": {"$eq": "child_chunk"}}
    return {"$and": [base, meta_filter]} if meta_filter else base


# --------------------------------------------------------------------------- #
# Single-signal searches
# --------------------------------------------------------------------------- #
def dense_search(query: str, top_k: int = DENSE_K, meta_filter: dict | None = None) -> list[tuple[str, float]]:
    res = get_index().search(
        namespace=NAMESPACE,
        query={"inputs": {"text": query}, "top_k": top_k, "filter": _child_filter(meta_filter)},
        fields=["doc_id"],
    )
    return [(h.id, float(h.score)) for h in res["result"]["hits"]]


def lexical_search(query: str, top_k: int = LEXICAL_K, meta_filter: dict | None = None) -> list[tuple[str, float]]:
    bm25, ids = get_bm25()
    child_by_id = load_corpus()["child_by_id"]
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
    out: list[tuple[str, float]] = []
    for i in order:
        if scores[i] <= 0:
            break
        rid = ids[i]
        if meta_filter and not _match(child_by_id[rid], meta_filter):
            continue
        out.append((rid, float(scores[i])))
        if len(out) >= top_k:
            break
    return out


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def _boost(cand: dict, boosts: dict | None) -> float:
    if not boosts:
        return 0.0
    total = 0.0
    for field, wanted in boosts.items():
        val = cand.get(field) or []
        if isinstance(val, list):
            total += BOOST_WEIGHT * len(set(val) & set(wanted))
    return total


def _candidate(record_id: str) -> dict | None:
    c = load_corpus()["child_by_id"].get(record_id)
    if not c:
        return None
    cand = dict(c)
    cand.update(dense_score=None, dense_rank=None, lexical_score=None, lexical_rank=None,
                metadata_boost=0.0, rrf_score=0.0, fused_score=0.0, rerank_score=None)
    return cand


def hybrid_search(
    query: str,
    top_k: int = FUSED_K,
    meta_filter: dict | None = None,
    boosts: dict | None = None,
    dense_k: int = DENSE_K,
    lexical_k: int = LEXICAL_K,
) -> list[dict]:
    dense = dense_search(query, dense_k, meta_filter)
    lexical = lexical_search(query, lexical_k, meta_filter)

    dense_rank = {rid: r for r, (rid, _) in enumerate(dense)}
    dense_sc = {rid: s for rid, s in dense}
    lex_rank = {rid: r for r, (rid, _) in enumerate(lexical)}
    lex_sc = {rid: s for rid, s in lexical}

    rrf = rrf_fuse([[rid for rid, _ in dense], [rid for rid, _ in lexical]])

    cands: list[dict] = []
    for rid in rrf:
        c = _candidate(rid)
        if c is None:
            continue
        c["dense_score"] = dense_sc.get(rid)
        c["dense_rank"] = dense_rank.get(rid)
        c["lexical_score"] = lex_sc.get(rid)
        c["lexical_rank"] = lex_rank.get(rid)
        c["rrf_score"] = rrf[rid]
        c["metadata_boost"] = _boost(c, boosts)
        c["fused_score"] = c["rrf_score"] + c["metadata_boost"]
        cands.append(c)

    cands.sort(key=lambda c: c["fused_score"], reverse=True)
    return cands[:top_k]


def case_card_search(query: str, top_k: int = 10, meta_filter: dict | None = None) -> list[dict]:
    """Doc-level retrieval over case cards (simple-query path / broad lookups)."""
    base = {"record_type": {"$eq": "case_card"}}
    flt = {"$and": [base, meta_filter]} if meta_filter else base
    res = get_index().search(
        namespace=NAMESPACE,
        query={"inputs": {"text": query}, "top_k": top_k, "filter": flt},
        fields=["doc_id", "case_title", "legal_area", "issue_tags", "stance_tags", "vehicle_tags", "year"],
    )
    return [
        {
            "doc_id": h.fields["doc_id"],
            "case_title": h.fields.get("case_title"),
            "legal_area": h.fields.get("legal_area"),
            "year": h.fields.get("year"),
            "issue_tags": h.fields.get("issue_tags", []),
            "stance_tags": h.fields.get("stance_tags", []),
            "vehicle_tags": h.fields.get("vehicle_tags", []),
            "score": float(h.score),
        }
        for h in res["result"]["hits"]
    ]
