"""Simple-query path (§3.1) + clarify path (§4.2).

Simple lookups ("which judgments involve commercial vehicles?") skip the full
support/adverse/strategy workflow and answer concisely from case-card + chunk
retrieval, still exposing the retrieved evidence.
"""

from __future__ import annotations

from src.agent.llm import call_json, call_text, render
from src.ingestion.vocab import ISSUE_TAGS, LEGAL_AREAS, PROCEDURAL_STAGES, VEHICLE_TAGS
from src.retrieval.corpus_store import load_corpus
from src.retrieval.hybrid_retriever import case_card_search, hybrid_search

# ----------------------------------------------------------------------------- #
# Fix #1: metadata enumeration for listing / factual / negation queries.
# These under-recall with semantic top-k even though the tags are indexed, so we
# map the query to a metadata predicate and enumerate the FULL matching doc set.
# ----------------------------------------------------------------------------- #
_META_SYSTEM = ("You map a corpus listing/factual question to a metadata filter over a fixed set "
                "of Indian court judgments. Return JSON only.")

_META_USER = """Question: {query}

Decide whether this is best answered by ENUMERATING documents that match a metadata filter
(e.g. 'which involve commercial vehicles', 'which are NOT motor accident cases', 'which are
trademark cases', 'which cases discuss fake licence', 'which are appeals from the tribunal'),
versus needing semantic reasoning.

Return JSON:
- "enumerable": boolean — true only if a metadata filter cleanly captures the intent.
- "legal_area": list from {legal_areas} ([] if N/A)
- "is_motor": true | false | null  (false for "NOT motor accident" questions)
- "issue_tags": list from the issue vocabulary ([] if N/A)
- "vehicle_tags": list from {vehicle_tags} — include ALL relevant types (for "commercial vehicle"
  include truck, lorry, bus, tempo, tanker, dumper, goods_carriage, commercial_vehicle)
- "procedural_stage": list from {stages} ([] if N/A)
"""


def _validate_spec(spec: dict) -> dict:
    spec["legal_area"] = [x for x in (spec.get("legal_area") or []) if x in LEGAL_AREAS]
    spec["issue_tags"] = [x for x in (spec.get("issue_tags") or []) if x in ISSUE_TAGS]
    spec["vehicle_tags"] = [x for x in (spec.get("vehicle_tags") or []) if x in VEHICLE_TAGS]
    spec["procedural_stage"] = [x for x in (spec.get("procedural_stage") or []) if x in PROCEDURAL_STAGES]
    return spec


def _enumerate_docs(spec: dict) -> list[dict]:
    out = []
    for m in load_corpus()["docmeta"].values():
        if spec["legal_area"] and m["legal_area"] not in spec["legal_area"]:
            continue
        if spec.get("is_motor") is not None and bool(m["is_motor_accident"]) != bool(spec["is_motor"]):
            continue
        if spec["issue_tags"] and not (set(m["issue_tags"]) & set(spec["issue_tags"])):
            continue
        if spec["vehicle_tags"] and not (set(m["vehicle_tags"]) & set(spec["vehicle_tags"])):
            continue
        if spec["procedural_stage"] and m.get("procedural_stage") not in spec["procedural_stage"]:
            continue
        out.append(m)
    return out


def _meta_match(m: dict) -> dict:
    return {"doc_id": m["doc_id"], "case_title": m["case_title"], "legal_area": m["legal_area"],
            "year": m["year"], "issue_tags": m["issue_tags"], "stance_tags": m["stance_tags"],
            "vehicle_tags": m["vehicle_tags"], "score": 1.0, "source": "metadata_enumeration"}

_SIMPLE_SYSTEM = (
    "You answer factual questions about a fixed corpus of Indian court judgments. "
    "Be concise. Cite doc_ids. Only use the supplied matches; if the corpus does not "
    "clearly contain something, say so rather than inventing."
)

_SIMPLE_USER = """Question: {query}

Document-level matches (case cards):
{cards}

Supporting passages:
{chunks}

Answer concisely: list the matching judgments (doc_id + case title) with a one-line reason each,
grouped sensibly. If none clearly match, say the corpus does not appear to contain such a case and
name the closest ones.
"""


def simple_retrieval(state: dict) -> dict:
    query = state["user_query"]

    # Fix #1: try metadata enumeration first (complete recall for listing queries).
    spec = call_json(_META_SYSTEM, render(_META_USER, query=query,
        legal_areas=sorted(LEGAL_AREAS), vehicle_tags=sorted(VEHICLE_TAGS),
        stages=sorted(PROCEDURAL_STAGES))) or {}
    doc_matches: list[dict] = []
    if spec.get("enumerable"):
        spec = _validate_spec(spec)
        # only enumerate if at least one predicate is set (avoid returning the whole corpus)
        if any(spec.get(k) for k in ("legal_area", "issue_tags", "vehicle_tags", "procedural_stage")) \
                or spec.get("is_motor") is not None:
            doc_matches = [_meta_match(m) for m in _enumerate_docs(spec)]

    cards = case_card_search(query, top_k=12)
    chunks = hybrid_search(query, top_k=12)

    seen = {d["doc_id"] for d in doc_matches}
    for c in cards:  # supplement enumerated set with semantic case-card hits
        if c["doc_id"] not in seen:
            doc_matches.append(c)
            seen.add(c["doc_id"])

    return {"retrieved_candidates": chunks,
            "selected_evidence": {"doc_matches": doc_matches},
            "retrieval_filters": {"metadata_spec": spec if spec.get("enumerable") else None}}


def simple_answer(state: dict) -> dict:
    cards = (state.get("selected_evidence") or {}).get("doc_matches", [])
    chunks = state.get("retrieved_candidates") or []
    card_lines = "\n".join(
        f"- {c['doc_id']} | {c.get('case_title','')} | area={c.get('legal_area')} "
        f"| issues={c.get('issue_tags')} | vehicles={c.get('vehicle_tags')} (score {c['score']:.2f})"
        for c in cards
    ) or "(none)"
    chunk_lines = "\n".join(
        f"- {c['doc_id']} ({c['record_id']}): \"{c['text'][:200]}\"" for c in chunks[:8]
    ) or "(none)"
    answer = call_text(_SIMPLE_SYSTEM, render(_SIMPLE_USER,
        query=state["user_query"], cards=card_lines, chunks=chunk_lines))
    return {"final_answer": answer}


_CLARIFY_SYSTEM = "You help scope an ambiguous legal research request over a fixed judgment corpus."

_CLARIFY_USER = """The user's request is ambiguous: "{query}"

Possibly-relevant judgments:
{cards}

Briefly: (1) state the most reasonable interpretation you will assume, (2) ask 1-2 specific
clarifying questions, and (3) list a few possibly-relevant doc_ids the user might mean.
"""


def clarify(state: dict) -> dict:
    cards = case_card_search(state["user_query"], top_k=6)
    card_lines = "\n".join(f"- {c['doc_id']} | {c.get('case_title','')} | {c.get('legal_area')}"
                           for c in cards) or "(none)"
    answer = call_text(_CLARIFY_SYSTEM, render(_CLARIFY_USER,
        query=state["user_query"], cards=card_lines))
    return {"final_answer": answer, "retrieved_candidates": [],
            "selected_evidence": {"doc_matches": cards}}
