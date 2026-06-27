"""Evidence selection (§4.8): from a reranked candidate list, pick a diverse,
balanced set of supporting / adverse / neutral evidence and attach parent context.

Stance is judged relative to the client's side. A chunk's own stance tags win;
the document-level stance is a fallback. Doc-diversity is capped so one judgment
can't dominate the evidence set. Parent-expansion (§7.5) attaches the larger
parent chunk for each selected child so the analyzer has full legal context.
"""

from __future__ import annotations

from src.retrieval.corpus_store import get_doc_meta, get_parent

SUPPORT_FOR = {
    "claimant": {"claimant_supporting", "adverse_to_insurer"},
    "insurer": {"insurer_supporting", "adverse_to_claimant"},
    "owner_driver": {"owner_driver_supporting"},
}
ADVERSE_FOR = {
    "claimant": {"insurer_supporting", "adverse_to_claimant"},
    "insurer": {"claimant_supporting", "adverse_to_insurer"},
    "owner_driver": {"adverse_to_claimant"},
}


def classify_stance(cand: dict, client_side: str) -> str:
    """'supporting' | 'adverse' | 'mixed' | 'neutral' for the given client.

    Whether a *precedent* helps or hurts the client is a document-level property
    (a claimant-supporting judgment is supporting authority even where a chunk
    quotes the insurer's argument), so the document-level stance is primary; the
    chunk-level stance only disambiguates documents that are themselves mixed.
    """
    sup = SUPPORT_FOR.get(client_side, set())
    adv = ADVERSE_FOR.get(client_side, set())

    meta = get_doc_meta(cand["doc_id"]) or {}
    dtags = set(meta.get("stance_tags") or [])
    doc_sup, doc_adv = bool(dtags & sup), bool(dtags & adv)
    if doc_adv and not doc_sup:
        return "adverse"
    if doc_sup and not doc_adv:
        return "supporting"

    # document is itself mixed (or unlabelled) -> use the chunk's own stance
    tags = set(cand.get("stance_tags") or [])
    chunk_sup, chunk_adv = bool(tags & sup), bool(tags & adv)
    if chunk_adv and not chunk_sup:
        return "adverse"
    if chunk_sup and not chunk_adv:
        return "supporting"
    if doc_sup and doc_adv:
        return "mixed"
    return "neutral"


def select_evidence(
    reranked: list[dict],
    client_side: str = "claimant",
    max_per_doc: int = 2,
    n_support: int = 5,
    n_adverse: int = 4,
    n_neutral: int = 2,
    expand_parents: bool = True,
) -> dict:
    supporting: list[dict] = []
    adverse: list[dict] = []
    neutral: list[dict] = []
    per_doc: dict[str, int] = {}

    for cand in reranked:  # already ordered by rerank_score
        doc = cand["doc_id"]
        if per_doc.get(doc, 0) >= max_per_doc:
            continue
        stance = classify_stance(cand, client_side)
        item = dict(cand)
        item["client_stance"] = stance
        if stance in ("supporting", "mixed") and len(supporting) < n_support:
            supporting.append(item)
        elif stance == "adverse" and len(adverse) < n_adverse:
            adverse.append(item)
        elif stance == "neutral" and len(neutral) < n_neutral:
            neutral.append(item)
        else:
            continue
        per_doc[doc] = per_doc.get(doc, 0) + 1

    selected = supporting + adverse + neutral
    parents: dict[str, str] = {}
    if expand_parents:
        for it in selected:
            pid = it.get("parent_id")
            if pid and pid not in parents:
                p = get_parent(pid)
                if p:
                    parents[pid] = p["text"]

    return {
        "client_side": client_side,
        "supporting": supporting,
        "adverse": adverse,
        "neutral": neutral,
        "selected": selected,
        "parents": parents,
    }
