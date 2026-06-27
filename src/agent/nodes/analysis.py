"""Answer-generation nodes (§4.9–4.11, §9.5–9.7): case analyzer, strategy
synthesizer, and grounding validator. All grounded ONLY in retrieved evidence.
"""

from __future__ import annotations

from src.agent.llm import call_json, call_text, render
from src.retrieval.corpus_store import get_doc_meta

_CHUNK_CHARS = 650


def _evidence_context(state: dict) -> tuple[str, list[str]]:
    """Build a compact, grouped evidence context + the list of doc_ids present."""
    sel = state.get("selected_evidence") or {}
    items = sel.get("selected") or []
    by_doc: dict[str, list[dict]] = {}
    for it in items:
        by_doc.setdefault(it["doc_id"], []).append(it)

    blocks = []
    for doc_id, chunks in by_doc.items():
        meta = get_doc_meta(doc_id) or {}
        stance = chunks[0].get("client_stance", "neutral")
        head = (f"[{doc_id} | {meta.get('case_title', chunks[0].get('case_title',''))} | "
                f"{meta.get('court','')} {meta.get('year','')} | stance_for_client={stance}]")
        summary = (meta.get("outcome", "") + " " + " ".join(meta.get("key_holdings", [])[:2])).strip()
        lines = [head]
        if summary:
            lines.append(f"  case summary: {summary[:400]}")
        for c in chunks[:3]:
            lines.append(f"  evidence ({c['record_id']}, p.{c.get('page_start')}, "
                         f"{c.get('chunk_type')}): \"{c['text'][:_CHUNK_CHARS]}\"")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks), list(by_doc.keys())


_ANALYZER_SYSTEM = (
    "You are a precedent analyst. Analyze ONLY the supplied judgment evidence. "
    "Do not use outside knowledge or invent facts. Return JSON only."
)

_ANALYZER_USER = """Client side: {client_side}
Issues: {issues}

For EACH judgment below, produce an analysis object. Return JSON with key "analyses": a list of:
{"doc_id","case_title","relevant_facts","legal_issue","holding","client_impact": one of
["supporting","adverse","mixed","neutral"],"why","how_to_use","how_to_distinguish","confidence":
["high","medium","low"],"evidence_ids": list of the cited evidence chunk ids}
Be honest about adverse impact. Ground every statement in the evidence text.

EVIDENCE:
{context}
"""


def case_analyzer(state: dict) -> dict:
    context, _ = _evidence_context(state)
    if not context.strip():
        return {"case_analyses": []}
    data = call_json(_ANALYZER_SYSTEM, render(_ANALYZER_USER,
        client_side=state.get("client_side"),
        issues=state.get("issue_map"),
        context=context,
    ), temperature=0.1) or {}
    return {"case_analyses": data.get("analyses", []) or []}


_SYNTH_SYSTEM = (
    "You are a legal research assistant preparing precedent analysis for a lawyer. "
    "Use ONLY the analyzed precedents. Frame output as research assistance, not definitive "
    "legal advice. Cite doc_ids. Be honest about adverse precedents and gaps."
)

_SYNTH_USER = """User request: {query}
Client side: {client_side}
Issues: {issues}
Facts: {facts}

Analyzed precedents (JSON): {analyses}

Write a structured precedent-research answer in Markdown with these sections:
1. **Issue Map** — the legal issues in play.
2. **Supporting Precedents** — cite doc_ids; why the facts align and what principle each establishes.
3. **Adverse Precedents** — cite doc_ids the opposing side could use; honest risk assessment and how to distinguish/counter each.
4. **Distinguishing Arguments** — how to neutralise the adverse cases.
5. **Strategy Recommendation** — prioritized arguments, realistic risk level.
6. **Compensation Discussion** — only if relevant to this request.
7. **Evidence Gaps / Risks** — what is missing or weak in the corpus for this request.
If there are no adverse precedents in the evidence, say so explicitly. Do not invent cases or holdings.
"""


def strategy_synthesizer(state: dict) -> dict:
    analyses = state.get("case_analyses") or []
    answer = call_text(_SYNTH_SYSTEM, render(_SYNTH_USER,
        query=state["user_query"],
        client_side=state.get("client_side"),
        issues=state.get("issue_map"),
        facts=state.get("extracted_facts"),
        analyses=analyses,
    ), temperature=0.2)
    return {"final_answer": answer}


_VALIDATOR_SYSTEM = (
    "You validate a legal research answer for grounding and honesty. Return JSON only."
)

_VALIDATOR_USER = """Retrieved doc_ids available as evidence: {retrieved_docs}
Deep research requires an adverse-precedents discussion.

ANSWER TO VALIDATE:
\"\"\"{answer}\"\"\"

Return JSON with keys:
- "passes": boolean
- "cited_docs": list of doc_ids cited in the answer
- "uncited_or_invented_docs": doc_ids cited that are NOT in the retrieved list
- "adverse_section_present": boolean
- "issues": list of problems (overstated holdings, invented facts, party confusion, missing gaps)
- "required_fixes": list of concrete fixes
"""


def validator(state: dict) -> dict:
    _, doc_ids = _evidence_context(state)
    data = call_json(_VALIDATOR_SYSTEM, render(_VALIDATOR_USER,
        retrieved_docs=sorted(set(doc_ids)),
        answer=state.get("final_answer", ""),
    )) or {}
    return {"validation_report": data}
