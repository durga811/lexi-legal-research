"""Reproducible live verification of the retrieval pipeline (Step 4).

Requires the populated Pinecone index. Confirms the hybrid -> rerank -> select
pipeline returns relevant, scored, support/adverse-balanced evidence.

Run:  uv run python -m src.retrieval.retrieval_smoke
"""

from __future__ import annotations

from src.retrieval.hybrid_retriever import case_card_search
from src.retrieval.pipeline import retrieve_evidence

QUERY = ("insurer denies liability because the commercial truck driver had no valid "
         "driving licence; death compensation claim")
BOOSTS = {
    "issue_tags": ["driving_license_validity", "insurance_liability", "policy_breach", "pay_and_recover"],
    "vehicle_tags": ["truck", "commercial_vehicle"],
}


def main() -> int:
    res = retrieve_evidence(QUERY, client_side="claimant", boosts=BOOSTS)
    fused, reranked, sel = res["fused"], res["reranked"], res["selection"]
    failures = []

    print(f"fused candidates: {len(fused)} | reranked: {len(reranked)}")
    print("top-5 fused (dense/lexical/rrf/boost):")
    for c in fused[:5]:
        print(f"  {c['record_id']} d={(c['dense_score'] or 0):.3f}(r{c['dense_rank']}) "
              f"l={(c['lexical_score'] or 0):.1f}(r{c['lexical_rank']}) rrf={c['rrf_score']:.4f} +{c['metadata_boost']:.3f}")

    print("top-5 reranked (bge relevance):")
    for c in reranked[:5]:
        print(f"  {c['doc_id']} {c['record_id']} rerank={c['rerank_score']:.3f} {c['chunk_type']}")

    print("selection:")
    for bucket in ("supporting", "adverse", "neutral"):
        print(f"  {bucket}: {[c['doc_id'] for c in sel[bucket]]}")

    # sanity checks
    if not fused:
        failures.append("no fused candidates")
    if not any(c.get("lexical_rank") is not None for c in fused):
        failures.append("lexical signal absent from fusion")
    if not reranked or reranked[0].get("rerank_score") is None:
        failures.append("reranking produced no scores")
    if not sel["adverse"]:
        failures.append("no adverse precedents surfaced (eval dim 4)")
    if len({c["doc_id"] for c in sel["selected"]}) < 3:
        failures.append("evidence not diverse across documents")

    cards = case_card_search("which judgments involve commercial vehicles", top_k=6)
    print("case_card simple path docs:", [c["doc_id"] for c in cards])
    if not cards:
        failures.append("case_card search returned nothing")

    if failures:
        print("\nRETRIEVAL SMOKE FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nRETRIEVAL SMOKE PASSED — hybrid + rerank + select + trace verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
