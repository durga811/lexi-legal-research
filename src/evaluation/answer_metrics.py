"""Answer-quality metrics (§12.2): objective deterministic checks + an
LLM-as-judge rubric. The deterministic checks are the trustworthy backbone; the
LLM judge (Gemini) adds qualitative scores and is intentionally paired with the
objective checks to offset self-judging bias.
"""

from __future__ import annotations

import re

from src.agent.llm import call_json, render

_DOC_RE = re.compile(r"DOC_\d{3}")
_NO_ANSWER_CUES = ["no judgment", "no case", "does not", "did not find", "no precedent",
                   "not appear", "no such", "none of", "could not find", "no direct",
                   "not contain", "no document", "no matching"]


def extract_cited_docs(text: str) -> set[str]:
    return set(_DOC_RE.findall(text or ""))


def _issue_covered(issue: str, answer_l: str) -> bool:
    words = [w for w in re.split(r"[^a-z0-9]+", issue.lower()) if len(w) > 4]
    return any(w in answer_l for w in words) if words else True


def deterministic_answer_checks(q, state: dict, retrieved_docs: set[str]) -> dict:
    answer = state.get("final_answer", "") or ""
    answer_l = answer.lower()
    cited = extract_cited_docs(answer)
    valid = {f"DOC_{i:03d}" for i in range(1, 57)}

    invented_nonexistent = sorted(cited - valid)
    uncited_unretrieved = sorted(cited - retrieved_docs - set(q.expected_relevant_doc_ids))

    issues = q.must_include_issues or []
    issue_cov = (sum(_issue_covered(i, answer_l) for i in issues) / len(issues)) if issues else None

    deep = q.expects_deep()
    adverse_present = ("adverse" in answer_l) or bool(state.get("validation_report", {}).get("adverse_section_present"))

    no_answer_ok = None
    if q.no_answer:
        # Correct = explicitly states the corpus lacks such a case. Listing the
        # closest related docs is allowed (§8.5), so we don't penalise citations;
        # invented non-existent docs are caught separately by invented_nonexistent_docs.
        no_answer_ok = any(c in answer_l for c in _NO_ANSWER_CUES) and not invented_nonexistent

    return {
        "cited_docs": sorted(cited),
        "citations_present": bool(cited) if not q.no_answer else True,
        "invented_nonexistent_docs": invented_nonexistent,
        "cited_not_retrieved": uncited_unretrieved,
        "issue_coverage": None if issue_cov is None else round(issue_cov, 2),
        "adverse_section_present": adverse_present if deep else None,
        "no_answer_handled": no_answer_ok,
        "answer_chars": len(answer),
    }


_JUDGE_SYSTEM = (
    "You are a strict legal-research answer judge. Score 1-5 (5 best). Be critical. "
    "Return JSON only."
)

_JUDGE_USER = """Research request: {query}
Ideal answer should: {facets}
The answer must NOT claim: {must_not}
Documents that are actually relevant (ground truth): {gold}

ANSWER TO JUDGE:
\"\"\"{answer}\"\"\"

Return JSON with integer scores 1-5 and booleans:
- "faithfulness": are all claims grounded in cited precedents (no invented facts/holdings)?
- "legal_reasoning": does it correctly identify issues and explain why precedents apply?
- "adverse_reasoning": does it honestly surface and assess adverse precedents/risks? (5 if not applicable and it says so)
- "completeness": did it address all parts of the request?
- "must_not_claim_respected": boolean — true if the answer avoids every forbidden claim
- "justification": one sentence.
"""


def judge_answer(q, state: dict) -> dict:
    data = call_json(_JUDGE_SYSTEM, render(_JUDGE_USER,
        query=q.query,
        facets="; ".join(q.ideal_answer_facets) or "n/a",
        must_not="; ".join(q.must_not_claim) or "n/a",
        gold=", ".join(q.expected_relevant_doc_ids) or "(none — no-answer)",
        answer=(state.get("final_answer", "") or "")[:6000],
    )) or {}
    out = {}
    for k in ("faithfulness", "legal_reasoning", "adverse_reasoning", "completeness"):
        v = data.get(k)
        out[k] = int(v) if isinstance(v, (int, float)) else None
    out["must_not_claim_respected"] = bool(data.get("must_not_claim_respected", True))
    out["justification"] = data.get("justification", "")
    return out
