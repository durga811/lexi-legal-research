"""Reranking stage (§4.7, §8.5).

Primary: Pinecone-hosted ``bge-reranker-v2-m3`` over the fused candidate set.
Fallback: if the hosted reranker errors (rate limit / unavailable), keep the
fused order and copy the fused score into ``rerank_score`` so the pipeline still
returns a ranked list. Each candidate keeps all earlier sub-scores for the trace.
"""

from __future__ import annotations

from src.retrieval.pinecone_client import get_pc

RERANK_MODEL = "bge-reranker-v2-m3"
MAX_DOC_CHARS = 2000  # bge handles ~512 tokens; our chunks are smaller, cap defensively


def rerank(query: str, candidates: list[dict], top_n: int = 12, model: str = RERANK_MODEL) -> list[dict]:
    if not candidates:
        return []
    docs = [{"id": c["record_id"], "text": c["text"][:MAX_DOC_CHARS]} for c in candidates]
    by_id = {c["record_id"]: c for c in candidates}
    try:
        result = get_pc().inference.rerank(
            model=model,
            query=query,
            documents=docs,
            rank_fields=["text"],
            top_n=min(top_n, len(docs)),
            return_documents=True,
        )
        out: list[dict] = []
        for item in result.data:
            rid = item.document["id"]
            cand = dict(by_id[rid])
            cand["rerank_score"] = float(item.score)
            out.append(cand)
        return out
    except Exception as exc:  # noqa: BLE001 — graceful fallback is the point
        return _fallback(candidates, top_n, reason=str(exc))


def _fallback(candidates: list[dict], top_n: int, reason: str) -> list[dict]:
    ranked = sorted(candidates, key=lambda c: c.get("fused_score", 0.0), reverse=True)[:top_n]
    out = []
    for c in ranked:
        cc = dict(c)
        cc["rerank_score"] = cc.get("fused_score", 0.0)
        cc["rerank_fallback"] = reason[:120]
        out.append(cc)
    return out
