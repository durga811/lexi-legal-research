"""Lexi Legal Precedent Research Agent — hosted Streamlit UI.

Accepts ANY prompt and shows the full visible reasoning + evidence trace (§9.2):
detected intent, extracted facts/issues, research plan, generated queries, the
exact retrieved chunks with dense/lexical/RRF/boost scores, the reranker scores
and order, the selected support/adverse evidence, per-case analysis, validation
status, and the final answer. Intermediate steps are shown — not just the answer.
"""

from __future__ import annotations

import streamlit as st

from src.agent.graph import get_app

st.set_page_config(page_title="Lexi Legal Precedent Research Agent", layout="wide")

# Live progress messages, emitted as each graph node populates state (so the user
# sees the reasoning happen and the connection stays alive across the ~100s deep run).
STAGE_SEQUENCE = [
    ("detected_intent", lambda s: f"Interpreted request → {s.get('query_complexity')} "
                                  f"(client: {s.get('client_side')}, area: {s.get('legal_area')})"),
    ("research_plan", lambda s: f"Built research plan ({len(s.get('research_plan') or [])} steps)"),
    ("generated_queries", lambda s: f"Generated {len(s.get('generated_queries') or [])} search queries "
                                    f"(support / adverse / compensation)"),
    ("retrieved_candidates", lambda s: f"Retrieved {len(s.get('retrieved_candidates') or [])} candidate "
                                       f"chunks (dense + BM25 + RRF)"),
    ("reranked_candidates", lambda s: f"Reranked to top {len(s.get('reranked_candidates') or [])} (bge-reranker)"),
    ("selected_evidence", lambda s: "Selected evidence: "
        f"{len((s.get('selected_evidence') or {}).get('supporting', []))} supporting, "
        f"{len((s.get('selected_evidence') or {}).get('adverse', []))} adverse"
        if (s.get('selected_evidence') or {}).get('supporting') is not None
        else f"Matched {len((s.get('selected_evidence') or {}).get('doc_matches', []))} documents"),
    ("case_analyses", lambda s: f"Analyzed {len(s.get('case_analyses') or [])} precedents"),
    ("final_answer", lambda s: "Synthesized answer"),
    ("validation_report", lambda s: "Validated grounding (citations, adverse coverage, no hallucination)"),
]

EXAMPLES = [
    "Find supporting and adverse precedents and a strategy for a claimant where the insurer denies "
    "liability because the commercial truck driver had no valid driving licence.",
    "Which judgments involve commercial vehicles?",
    "Which documents are not motor accident cases?",
    "Find precedents on contributory negligence and the strongest adverse cases.",
    "Find cases interpreting Section 167 of the Motor Vehicles Act.",
]


def _fmt(x, n: int = 3) -> str:
    return "" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


# --------------------------------------------------------------------------- #
# Trace renderers
# --------------------------------------------------------------------------- #
def render_routing(s: dict) -> None:
    c = st.columns(4)
    c[0].metric("Intent", s.get("detected_intent", "-"))
    c[1].metric("Complexity", s.get("query_complexity", "-"))
    c[2].metric("Client side", s.get("client_side", "-"))
    c[3].metric("Legal area", s.get("legal_area", "-"))
    facts = s.get("extracted_facts") or {}
    issues = s.get("issue_map") or []
    if facts:
        st.markdown("**Extracted facts**")
        st.json(facts, expanded=False)
    if issues:
        st.markdown("**Issue map**")
        for i in issues:
            st.markdown(f"- {i}")


def render_plan_queries(s: dict) -> None:
    plan = s.get("research_plan") or []
    if plan:
        st.markdown("**Research plan**")
        for i, step in enumerate(plan, 1):
            st.markdown(f"{i}. {step}")
    queries = s.get("generated_queries") or []
    if queries:
        st.markdown("**Generated search queries**")
        st.dataframe(
            [{"purpose": q.get("purpose"), "query": q.get("query")} for q in queries],
            use_container_width=True, hide_index=True,
        )
    boosts = (s.get("retrieval_filters") or {}).get("boosts")
    if boosts:
        st.caption(f"Metadata boosts: {boosts}")


def render_retrieved(s: dict) -> None:
    cands = s.get("retrieved_candidates") or []
    if not cands:
        st.caption("No chunk candidates (e.g. clarify path).")
        return
    rows = sorted(cands, key=lambda c: c.get("fused_score", 0), reverse=True)[:30]
    st.caption(f"{len(cands)} fused candidates (showing top {len(rows)}). "
               "Dense = e5 cosine, Lexical = BM25, RRF = reciprocal-rank fusion, Boost = metadata.")
    st.dataframe(
        [{
            "doc_id": c["doc_id"], "record_id": c["record_id"], "type": c.get("chunk_type"),
            "dense": _fmt(c.get("dense_score")), "d_rank": c.get("dense_rank"),
            "lexical": _fmt(c.get("lexical_score"), 2), "l_rank": c.get("lexical_rank"),
            "rrf": _fmt(c.get("rrf_score"), 4), "boost": _fmt(c.get("metadata_boost")),
            "fused": _fmt(c.get("fused_score"), 4), "passes": ",".join(c.get("passes", [])),
        } for c in rows],
        use_container_width=True, hide_index=True,
    )


def render_reranked(s: dict) -> None:
    rr = s.get("reranked_candidates") or []
    if not rr:
        st.caption("No reranked candidates.")
        return
    st.caption(f"Reranked by Pinecone bge-reranker-v2-m3 — {len(rr)} candidates in final relevance order.")
    for i, c in enumerate(rr, 1):
        with st.expander(f"#{i} · {c['doc_id']} · {c['record_id']} · rerank={_fmt(c.get('rerank_score'))} "
                         f"· {c.get('chunk_type')} · stance={c.get('stance_tags')}"):
            st.caption(f"dense={_fmt(c.get('dense_score'))} (rank {c.get('dense_rank')}) · "
                       f"lexical={_fmt(c.get('lexical_score'),2)} (rank {c.get('lexical_rank')}) · "
                       f"rrf={_fmt(c.get('rrf_score'),4)} · fused={_fmt(c.get('fused_score'),4)} · "
                       f"pages {c.get('page_start')}-{c.get('page_end')}")
            st.write(c.get("text", ""))


def _render_evidence_items(items: list[dict]) -> None:
    for c in items:
        st.markdown(f"**{c['doc_id']} — {c.get('case_title','')}**  "
                    f"·  stance: `{c.get('client_stance','-')}`  ·  rerank: {_fmt(c.get('rerank_score'))}")
        st.caption(f"{c['record_id']} · {c.get('chunk_type')} · pages {c.get('page_start')}-{c.get('page_end')}")
        st.write(c.get("text", ""))
        st.divider()


def render_selected(s: dict) -> None:
    sel = s.get("selected_evidence") or {}
    if sel.get("doc_matches") is not None:  # simple path
        st.markdown("**Document matches (case cards)**")
        st.dataframe(
            [{"doc_id": d["doc_id"], "case_title": d.get("case_title"), "legal_area": d.get("legal_area"),
              "issue_tags": ",".join(d.get("issue_tags", [])), "score": _fmt(d.get("score"))}
             for d in sel["doc_matches"]],
            use_container_width=True, hide_index=True,
        )
        return
    cols = st.columns(3)
    with cols[0]:
        st.markdown(f"#### 🟢 Supporting ({len(sel.get('supporting', []))})")
        _render_evidence_items(sel.get("supporting", []))
    with cols[1]:
        st.markdown(f"#### 🔴 Adverse ({len(sel.get('adverse', []))})")
        _render_evidence_items(sel.get("adverse", []))
    with cols[2]:
        st.markdown(f"#### ⚪ Neutral ({len(sel.get('neutral', []))})")
        _render_evidence_items(sel.get("neutral", []))
    parents = sel.get("parents") or {}
    if parents:
        with st.expander(f"Parent-chunk context ({len(parents)} parents expanded)"):
            for pid, text in parents.items():
                st.markdown(f"**{pid}**")
                st.write(text)
                st.divider()


def render_analyses(s: dict) -> None:
    analyses = s.get("case_analyses") or []
    if not analyses:
        st.caption("No per-case analysis (simple path).")
        return
    for a in analyses:
        impact = a.get("client_impact", "-")
        emoji = {"supporting": "🟢", "adverse": "🔴", "mixed": "🟡", "neutral": "⚪"}.get(impact, "•")
        with st.expander(f"{emoji} {a.get('doc_id')} — {a.get('case_title','')} · impact={impact} "
                         f"· confidence={a.get('confidence')}"):
            for key in ("relevant_facts", "legal_issue", "holding", "why", "how_to_use", "how_to_distinguish"):
                if a.get(key):
                    st.markdown(f"**{key.replace('_', ' ').title()}:** {a[key]}")
            if a.get("evidence_ids"):
                st.caption(f"Cited evidence: {a['evidence_ids']}")


def render_validation(s: dict) -> None:
    vr = s.get("validation_report") or {}
    if not vr:
        st.caption("No validation report (simple path).")
        return
    if vr.get("passes"):
        st.success("Validation passed")
    else:
        st.warning("Validation flagged issues")
    c = st.columns(2)
    c[0].markdown(f"**Adverse section present:** {vr.get('adverse_section_present')}")
    c[1].markdown(f"**Invented/uncited docs:** {vr.get('uncited_or_invented_docs') or 'none'}")
    if vr.get("cited_docs"):
        st.caption(f"Cited docs: {vr['cited_docs']}")
    for key in ("issues", "required_fixes"):
        if vr.get(key):
            st.markdown(f"**{key.replace('_', ' ').title()}:**")
            for x in vr[key]:
                st.markdown(f"- {x}")


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("⚖️ Lexi Legal Precedent Research Agent")
    st.caption("Research precedents across 56 Indian court judgments. Ask any question, from a quick "
               "lookup to a full precedent-research task. The agent's full reasoning and evidence trace "
               "(retrieved documents, ranking, and scores) is shown below the final answer.")

    busy = st.session_state.get("busy", False)

    with st.sidebar:
        st.subheader("Try an example")
        for ex in EXAMPLES:
            if st.button(ex, use_container_width=True, disabled=busy):
                st.session_state["prompt"] = ex
    prompt = st.text_area("Research prompt", value=st.session_state.get("prompt", ""), height=120,
                          disabled=busy,
                          placeholder="e.g. Find supporting and adverse precedents for a claimant where the "
                                      "insurer denies liability due to an unlicensed commercial truck driver.")
    run = st.button("Run research" if not busy else "Researching…", type="primary", disabled=busy)

    # Click → mark busy and rerun so the button renders disabled during the (~100s) run.
    if run and prompt.strip() and not busy:
        st.session_state.update(prompt=prompt, pending=prompt.strip(), busy=True, state=None, error=None)
        st.rerun()

    # Streamed execution: render each reasoning step as the graph node completes.
    if st.session_state.get("busy") and st.session_state.get("pending"):
        q = st.session_state["pending"]
        status = st.status(f"Researching: {q[:70]}…", expanded=True)
        final: dict = {}
        emitted: set[str] = set()
        try:
            for state in get_app().stream({"user_query": q}, stream_mode="values",
                                          config={"recursion_limit": 30}):
                final = state
                for key, msg in STAGE_SEQUENCE:
                    if key not in emitted and state.get(key):
                        emitted.add(key)
                        status.write(f"✓ {msg(state)}")
            status.update(label="Research complete", state="complete", expanded=False)
            st.session_state["state"] = final
        except Exception as exc:  # noqa: BLE001 — surface failures instead of a blank page
            status.update(label="Research failed", state="error")
            st.session_state["state"] = final or None  # keep partial trace if any
            st.session_state["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            st.session_state.update(busy=False, pending=None)
            st.rerun()

    if st.session_state.get("error"):
        st.error(st.session_state["error"])

    state = st.session_state.get("state")
    if state:
        st.markdown("## Answer")
        st.markdown(state.get("final_answer") or "_No answer produced._")

        st.markdown("## Reasoning & evidence trace")
        with st.expander("1 · Routing — intent, facts, issues", expanded=True):
            render_routing(state)
        if state.get("query_complexity") == "deep":
            with st.expander("2 · Research plan & generated queries", expanded=False):
                render_plan_queries(state)
        with st.expander("3 · Retrieved candidates — dense / lexical / RRF / boost scores", expanded=False):
            render_retrieved(state)
        if state.get("reranked_candidates"):
            with st.expander("4 · Reranked candidates — bge relevance scores & order", expanded=False):
                render_reranked(state)
        with st.expander("5 · Selected evidence — supporting / adverse / neutral (+ parents)", expanded=True):
            render_selected(state)
        if state.get("case_analyses"):
            with st.expander("6 · Per-case analysis", expanded=False):
                render_analyses(state)
        if state.get("validation_report"):
            with st.expander("7 · Grounding validation", expanded=False):
                render_validation(state)


if __name__ == "__main__":
    main()
