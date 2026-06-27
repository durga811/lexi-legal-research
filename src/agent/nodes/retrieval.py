"""Retrieval nodes (§4.5–4.8): support / adverse / compensation passes that each
run the generated queries through the hybrid retriever and accumulate candidates,
then an evidence-selection node that reranks the merged set and picks a diverse,
support/adverse-balanced evidence set with parent context.
"""

from __future__ import annotations

from src.retrieval.corpus_store import load_corpus
from src.retrieval.evidence_selector import select_evidence
from src.retrieval.hybrid_retriever import hybrid_search
from src.retrieval.query_expansion import expand_query
from src.retrieval.reranker import rerank

PER_QUERY_K = 12
MERGE_TOP_FOR_RERANK = 28
RERANK_N = 16

# Doc-level stance that is adverse FOR the given client.
_ADVERSE_STANCE = {
    "claimant": {"adverse_to_claimant", "insurer_supporting"},
    "insurer": {"adverse_to_insurer", "claimant_supporting"},
    "owner_driver": {"adverse_to_claimant"},
}


def _client_adverse_docs(client_side: str, legal_area: str | None) -> list[str]:
    """Docs whose document-level stance/adverse_value make them genuinely adverse
    to the client (Fix #2 — guarantees the adverse pass draws from real adverse docs)."""
    adverse_stance = _ADVERSE_STANCE.get(client_side, _ADVERSE_STANCE["claimant"])
    restrict = legal_area if legal_area and legal_area not in ("unknown", "mixed") else None
    out = []
    for m in load_corpus()["docmeta"].values():
        if restrict and m["legal_area"] != restrict:
            continue
        # require an explicit adverse stance tag (drop adverse_value-only noise)
        if set(m["stance_tags"]) & adverse_stance:
            out.append(m["doc_id"])
    return out

_SCORE_KEYS = ("dense_score", "dense_rank", "lexical_score", "lexical_rank",
               "rrf_score", "metadata_boost", "fused_score")


def _run_pass(state: dict, purposes: set[str], pass_name: str) -> dict:
    boosts = (state.get("retrieval_filters") or {}).get("boosts") or None
    queries = [q for q in state.get("generated_queries", []) if q["purpose"] in purposes]
    by_id = {c["record_id"]: c for c in (state.get("retrieved_candidates") or [])}

    for q in queries:
        for cand in hybrid_search(expand_query(q["query"]), top_k=PER_QUERY_K, boosts=boosts):
            rid = cand["record_id"]
            if rid in by_id:
                existing = by_id[rid]
                if cand["fused_score"] > existing["fused_score"]:
                    existing.update({k: cand[k] for k in _SCORE_KEYS})
                existing["passes"] = sorted(set(existing.get("passes", [])) | {pass_name})
            else:
                cand["passes"] = [pass_name]
                by_id[rid] = cand

    return {"retrieved_candidates": list(by_id.values())}


def support_retrieval(state: dict) -> dict:
    return _run_pass(state, {"support", "factual", "statutory"}, "support")


def adverse_retrieval(state: dict) -> dict:
    res = _run_pass(state, {"adverse"}, "adverse")
    if not state.get("requires_adverse"):
        return res

    # Fix #2: force a retrieval pass restricted to genuinely-adverse documents so
    # adverse precedents aren't crowded out by topically-similar supporting ones.
    client = state.get("client_side") or "claimant"
    adv_docs = _client_adverse_docs(client, state.get("legal_area"))
    if not adv_docs:
        return res

    by_id = {c["record_id"]: c for c in res["retrieved_candidates"]}
    flt = {"doc_id": {"$in": adv_docs}}
    boosts = {"stance_tags": list(_ADVERSE_STANCE.get(client, _ADVERSE_STANCE["claimant"]))}
    adverse_qs = [q["query"] for q in state.get("generated_queries", []) if q["purpose"] == "adverse"]
    for q in (adverse_qs or [state["user_query"]]):
        for cand in hybrid_search(expand_query(q), top_k=10, meta_filter=flt, boosts=boosts):
            rid = cand["record_id"]
            if rid in by_id:
                by_id[rid]["passes"] = sorted(set(by_id[rid].get("passes", [])) | {"adverse"})
            else:
                cand["passes"] = ["adverse"]
                by_id[rid] = cand
    return {"retrieved_candidates": list(by_id.values())}


def compensation_retrieval(state: dict) -> dict:
    if not state.get("requires_compensation"):
        return {}
    return _run_pass(state, {"compensation"}, "compensation")


def _adverse_bucket(state: dict, client: str, n: int) -> list[dict]:
    """Build the adverse precedents from a DEDICATED retrieval over genuinely-adverse
    documents (best chunk per doc, reranked among themselves) so they are not crowded
    out by topically-similar supporting docs (Fix #2)."""
    adv_docs = set(_client_adverse_docs(client, state.get("legal_area")))
    if not adv_docs:
        return []
    flt = {"doc_id": {"$in": list(adv_docs)}}
    boosts = {"stance_tags": list(_ADVERSE_STANCE.get(client, _ADVERSE_STANCE["claimant"]))}
    pool: dict[str, dict] = {}
    for c in state.get("retrieved_candidates") or []:
        if c["doc_id"] in adv_docs and (c["doc_id"] not in pool or c["fused_score"] > pool[c["doc_id"]]["fused_score"]):
            pool[c["doc_id"]] = c
    adverse_qs = [q["query"] for q in state.get("generated_queries", []) if q["purpose"] == "adverse"]
    for q in (adverse_qs or [state["user_query"]]):
        for c in hybrid_search(expand_query(q), top_k=40, meta_filter=flt, boosts=boosts):
            d = c["doc_id"]
            if d not in pool or c["fused_score"] > pool[d]["fused_score"]:
                pool[d] = c
    if not pool:
        return []
    reranked = rerank(state["user_query"], list(pool.values()), top_n=min(len(pool), n * 2))
    out, seen = [], set()
    for c in reranked:
        if c["doc_id"] in seen:
            continue
        item = dict(c)
        item["client_stance"] = "adverse"
        out.append(item)
        seen.add(c["doc_id"])
        if len(out) >= n:
            break
    return out


def select_evidence_node(state: dict) -> dict:
    from src.retrieval.corpus_store import get_parent

    cands = state.get("retrieved_candidates") or []
    client = state.get("client_side") or "claimant"
    if client not in ("claimant", "insurer", "owner_driver"):
        client = "claimant"

    top = sorted(cands, key=lambda c: c["fused_score"], reverse=True)[:MERGE_TOP_FOR_RERANK]
    reranked = rerank(state["user_query"], top, top_n=RERANK_N)

    adverse = _adverse_bucket(state, client, n=6) if state.get("requires_adverse") else []
    adv_ids = {c["doc_id"] for c in adverse}

    # supporting / neutral from the main reranked set, excluding adverse docs
    sel = select_evidence([c for c in reranked if c["doc_id"] not in adv_ids],
                          client_side=client, max_per_doc=2, n_support=5, n_adverse=0, n_neutral=3)
    sel["adverse"] = adverse

    chosen = {c["record_id"] for c in sel["supporting"] + adverse + sel["neutral"]}
    if len(sel["supporting"]) + len(adverse) + len(sel["neutral"]) < 6:
        for c in reranked:
            if c["record_id"] not in chosen and c["doc_id"] not in adv_ids:
                item = dict(c)
                item["client_stance"] = "neutral"
                sel["neutral"].append(item)
                chosen.add(c["record_id"])
                if len(sel["supporting"]) + len(adverse) + len(sel["neutral"]) >= 8:
                    break

    sel["selected"] = sel["supporting"] + adverse + sel["neutral"]
    parents = sel.get("parents") or {}
    for it in adverse:
        pid = it.get("parent_id")
        if pid and pid not in parents:
            p = get_parent(pid)
            if p:
                parents[pid] = p["text"]
    sel["parents"] = parents

    return {"reranked_candidates": reranked, "selected_evidence": sel}
