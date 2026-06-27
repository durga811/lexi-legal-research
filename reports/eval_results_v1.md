# Lexi Agent — Evaluation Results (v1)

Golden set: 25 queries (0 errored). Agent: LangGraph + Pinecone hybrid (dense e5 + BM25 + bge rerank) + Gemini. Metrics are document-level; recall is measured against the grounded golden set (see data/golden/golden_set_readme.md).

## Dimension 1 — Precision
- Precision@5: **0.53** · Precision@10: 0.431
- Answer precision (presented docs that are relevant): **0.474**

## Dimension 2 — Recall
- Recall@10: **0.651** · Recall@20: 0.73
- MRR: 0.765 · nDCG@10: 0.67 · Support recall: 0.402
- Recall@10 by difficulty: {'easy': 0.588, 'medium': 0.615, 'hard': 0.706}

## Dimension 3 — Reasoning Quality (1–5 judge + objective checks)
- Faithfulness: **3.84** · Legal reasoning: 3.32 · Completeness: 2.68
- Issue coverage: 0.969 · must-not-claim respected: 0.96 · no invented docs: 1.0

## Dimension 4 — Adverse Identification
- Adverse recall: **0.342** · Adverse-reasoning score: 3.48
- Adverse search executed (deep): 0.889 · Adverse section present: 0.889

## Agent behavior
- Router accuracy (simple vs deep): **0.92** · No-answer handled: 1.0

## Per-query results

| query_id | type | route ok | P@5 | R@10 | adv_recall | faith | preview |
|---|---|---|---|---|---|---|---|
| g_det_01 | factual_metadata | ✓ | 1.0 | 0.389 | None | 5 | Based on the provided documents, the following jud |
| g_det_02 | factual_metadata | ✓ | 0.0 | 0.0 | None | 2 | Based on the provided document-level matches, the  |
| g_det_03 | single_issue | ✓ | 0.4 | 1.0 | None | 4 | Based on the provided documents, the following cas |
| g_det_04 | single_issue | ✓ | 0.2 | 0.75 | None | 5 | Based on the provided documents, the following cas |
| g_det_05 | factual_metadata | ✓ | 0.8 | 1.0 | None | 5 | Based on the provided documents, the following jud |
| g_det_06 | statutory | ✓ | 0.2 | 1.0 | None | 5 | Based on the provided documents, the case that int |
| g_det_07 | procedural | ✓ | 0.2 | 1.0 | None | 5 | Based on the provided documents, the case that inv |
| g_det_08 | no_answer | ✓ | None | None | None | 5 | Based on the provided matches, the corpus **does n |
| g_det_09 | no_answer | ✓ | None | None | None | 5 | Based on the provided documents, the corpus **does |
| DSQ_01 | procedural | ✓ | 0.4 | 1.0 | None | 5 | Based on the provided documents, the following cas |
| DSQ_02 | procedural | ✓ | 0.2 | 0.25 | None | 3 | Based on the provided documents, the corpus does n |
| DSQ_03 | single_issue | ✓ | 0.4 | 0.375 | None | 3 | Based on the provided documents, the corpus does n |
| DSQ_04 | single_issue | ✓ | 0.6 | 0.444 | None | 4 | Based on the provided documents, the following cas |
| DSQ_05 | multi_hop | ✓ | 0.6 | 0.538 | 0.2 | 0 | Here is a structured precedent-research analysis p |
| DSQ_06 | multi_hop | ✓ | 0.8 | 0.857 | None | 5 | Based on the provided documents, the following cas |
| DSQ_07 | multi_hop | ✓ | 0.8 | 0.833 | 0.5 | 1 | ### 1. Issue Map  This research analysis addresses |
| DSQ_08 | comparison | ✓ | 0.4 | 0.625 | 0.5 | 2 | ### 1. Issue Map  The legal issues surrounding the |
| DSQ_09 | comparison | ✓ | 0.6 | 0.5 | None | 4 | Here is a structured precedent-research analysis c |
| DSQ_10 | ambiguous | ✗ | 0.4 | 0.333 | 0.0 | 5 | To help narrow down your request, here is how we c |
| DSQ_11 | ambiguous | ✓ | 0.2 | 0.8 | None | 4 | Based on the provided matches, the cases where ins |
| DSQ_12 | statutory | ✗ | 0.4 | 0.571 | None | 5 | ### 1. Issue Map  To determine "just compensation" |
| DSQ_13 | cross_document_strategy | ✓ | 0.8 | 0.615 | 0.2 | 3 | ### 1. Issue Map  The legal issues in play for thi |
| DSQ_14 | cross_document_strategy | ✓ | 1.0 | 0.778 | 0.5 | 5 | ### 1. Issue Map  In driving-licence disputes invo |
| DSQ_15 | cross_document_strategy | ✓ | 1.0 | 0.615 | 0.333 | 5 | Here is a structured precedent analysis and strate |
| DSQ_16 | cross_document_strategy | ✓ | 0.8 | 0.7 | 0.5 | 1 | ### Precedent Analysis: Fake Driving Licence Alleg |

---

## Failure Analysis — where the agent fails and what I'd fix first

### What works (the safety properties hold)
- **Grounding is strong:** faithfulness 3.84/5, **no invented documents (1.0)**, must-not-claim respected 0.96, issue coverage 0.97 — these last three are objective (not the LLM judge). The agent does not fabricate cases or holdings.
- **Routing 0.92** (23/25 correct simple-vs-deep) and **no-answer 1.0** — both no-answer queries correctly stated the corpus lacks the case (and only listed *closest* real docs, never invented one).
- **MRR 0.765** — the first relevant precedent is usually in the top ~1.3 results; recall@20 0.73.
- The **adverse mechanism runs** (adverse search executed 0.89, adverse section present 0.89) and reliably surfaces the headline adverse precedent (DOC_031, *National Insurance v. Laxmi Narain Dhut*).

### #1 weakness — metadata / listing recall (fix first)
Factual / negation / procedural queries with large gold sets under-recall because the simple path
uses **semantic top-k instead of metadata enumeration**, even though the needed tags already exist in
the index: `g_det_02` "which are NOT motor cases" **R@10 = 0.0** (a negation that pure similarity
cannot answer), `DSQ_02` MACT appeals 0.25, `g_det_01` commercial vehicles 0.39 (yet **P@5 = 1.0** —
precise but incomplete). These pull the precision/recall and completeness averages down the most.
**Fix:** add a metadata-enumeration branch to the simple path — when a query maps to a known
`legal_area` / `vehicle_tags` / `issue_tags` / `procedural_stage` (or is a negation/"which …"
listing), run a Pinecone **metadata filter to enumerate the full matching set** rather than top-k
semantic. This single change should sharply raise recall on ~6 queries and, with it, completeness.

### #2 weakness — adverse recall 0.34 (eval dimension 4)
The agent runs the adverse pass and includes an adverse section, but recovers only ~1/3 of the
*specific* gold adverse docs (per-query 0.0–0.5). The adverse pass is crowded out by topical
relevance: selection caps adverse at 4 and dedups per doc, and the reranker scores topical similarity,
not stance. **Fix:** in the adverse pass, hard-boost `stance_tags ∈ {adverse_to_claimant,
insurer_supporting}` and `adverse_value`-bearing docs; generate more opposing-side query variants;
raise the adverse selection cap; add stance-aware scoring so adverse precedents aren't displaced by
supporting ones.

### #3 weakness — completeness 2.68 (downstream of recall)
The lowest reasoning sub-score, and mostly a consequence of #1: when docs are missed the answer is
judged incomplete (`g_det_02`, `DSQ_02`, `DSQ_03` all completeness 1). A few deep multi-issue answers
(`DSQ_05`, `DSQ_16`) also scored low on faithfulness/completeness from the judge, suggesting the
synthesizer drifts on complex multi-part prompts. **Fix:** improve recall (#1), and have the
synthesizer explicitly enumerate and answer each requested facet (a per-facet checklist).

### #4 — router boundary cases (2/25)
`DSQ_10` ("find cases that help us on negligence") routed to *clarify* — defensible since it is
genuinely ambiguous, but the golden label expected a best-effort deep answer; `DSQ_12` (a statutory
lookup) over-escalated to deep. **Fix:** bias the interpreter away from *clarify* toward best-effort
deep-with-stated-assumptions, and treat "which/find cases interpreting Section X" as `simple_lookup`.

### Honest caveats
- Recall is measured against **my** grounded golden set (the assessment asks for my own methodology);
  labels are grounded in validated metadata + pooling + spot-review.
- The reasoning judge is **Gemini judging Gemini** — a self-judge bias, deliberately offset by the
  objective deterministic checks (no-invented 1.0, issue-coverage 0.97).
- **support_recall 0.40** partly reflects that the designer's labelled "supporting" subset is narrower
  than the (still relevant) set the agent surfaces — i.e. it is stricter than the precision picture suggests.

### Fix order
1. Metadata-enumeration for listing/factual queries → biggest recall + completeness gain.
2. Stance-aware adverse retrieval → dimension 4.
3. Router threshold tuning (clarify vs deep; statutory → simple).
4. Synthesizer facet checklist for multi-part deep prompts.