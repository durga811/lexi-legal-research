"""One-call retrieval entry point used by the agent's retrieval nodes (Step 5).

Chains hybrid search -> rerank -> evidence selection and returns every stage so
the UI can render the full evidence trace (§9.2): fused candidates with their
dense/lexical/RRF/boost scores, the reranked order with relevance scores, and the
support/adverse/neutral selection with parent context.
"""

from __future__ import annotations

from src.retrieval.evidence_selector import select_evidence
from src.retrieval.hybrid_retriever import hybrid_search
from src.retrieval.reranker import rerank


def retrieve_evidence(
    query: str,
    client_side: str = "claimant",
    meta_filter: dict | None = None,
    boosts: dict | None = None,
    fused_k: int = 30,
    rerank_n: int = 12,
    max_per_doc: int = 2,
    n_support: int = 5,
    n_adverse: int = 4,
    n_neutral: int = 2,
) -> dict:
    fused = hybrid_search(query, top_k=fused_k, meta_filter=meta_filter, boosts=boosts)
    reranked = rerank(query, fused, top_n=rerank_n)
    selection = select_evidence(
        reranked,
        client_side=client_side,
        max_per_doc=max_per_doc,
        n_support=n_support,
        n_adverse=n_adverse,
        n_neutral=n_neutral,
    )
    return {"query": query, "fused": fused, "reranked": reranked, "selection": selection}
