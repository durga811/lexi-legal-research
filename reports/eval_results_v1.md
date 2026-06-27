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
| g_det_01 | factual_metadata | yes | 1.0 | 0.389 | None | 5 | Based on the provided documents, the following jud |
| g_det_02 | factual_metadata | yes | 0.0 | 0.0 | None | 2 | Based on the provided document-level matches, the  |
| g_det_03 | single_issue | yes | 0.4 | 1.0 | None | 4 | Based on the provided documents, the following cas |
| g_det_04 | single_issue | yes | 0.2 | 0.75 | None | 5 | Based on the provided documents, the following cas |
| g_det_05 | factual_metadata | yes | 0.8 | 1.0 | None | 5 | Based on the provided documents, the following jud |
| g_det_06 | statutory | yes | 0.2 | 1.0 | None | 5 | Based on the provided documents, the case that int |
| g_det_07 | procedural | yes | 0.2 | 1.0 | None | 5 | Based on the provided documents, the case that inv |
| g_det_08 | no_answer | yes | None | None | None | 5 | Based on the provided matches, the corpus **does n |
| g_det_09 | no_answer | yes | None | None | None | 5 | Based on the provided documents, the corpus **does |
| DSQ_01 | procedural | yes | 0.4 | 1.0 | None | 5 | Based on the provided documents, the following cas |
| DSQ_02 | procedural | yes | 0.2 | 0.25 | None | 3 | Based on the provided documents, the corpus does n |
| DSQ_03 | single_issue | yes | 0.4 | 0.375 | None | 3 | Based on the provided documents, the corpus does n |
| DSQ_04 | single_issue | yes | 0.6 | 0.444 | None | 4 | Based on the provided documents, the following cas |
| DSQ_05 | multi_hop | yes | 0.6 | 0.538 | 0.2 | 0 | Here is a structured precedent-research analysis p |
| DSQ_06 | multi_hop | yes | 0.8 | 0.857 | None | 5 | Based on the provided documents, the following cas |
| DSQ_07 | multi_hop | yes | 0.8 | 0.833 | 0.5 | 1 | ### 1. Issue Map  This research analysis addresses |
| DSQ_08 | comparison | yes | 0.4 | 0.625 | 0.5 | 2 | ### 1. Issue Map  The legal issues surrounding the |
| DSQ_09 | comparison | yes | 0.6 | 0.5 | None | 4 | Here is a structured precedent-research analysis c |
| DSQ_10 | ambiguous | no | 0.4 | 0.333 | 0.0 | 5 | To help narrow down your request, here is how we c |
| DSQ_11 | ambiguous | yes | 0.2 | 0.8 | None | 4 | Based on the provided matches, the cases where ins |
| DSQ_12 | statutory | no | 0.4 | 0.571 | None | 5 | ### 1. Issue Map  To determine "just compensation" |
| DSQ_13 | cross_document_strategy | yes | 0.8 | 0.615 | 0.2 | 3 | ### 1. Issue Map  The legal issues in play for thi |
| DSQ_14 | cross_document_strategy | yes | 1.0 | 0.778 | 0.5 | 5 | ### 1. Issue Map  In driving-licence disputes invo |
| DSQ_15 | cross_document_strategy | yes | 1.0 | 0.615 | 0.333 | 5 | Here is a structured precedent analysis and strate |
| DSQ_16 | cross_document_strategy | yes | 0.8 | 0.7 | 0.5 | 1 | ### Precedent Analysis: Fake Driving Licence Alleg |

---

## Failure analysis — where it fails, what to fix first

**What works.** Grounding is strong: faithfulness 3.84/5, **0 invented documents**, must-not-claim 0.96,
issue-coverage 0.97 (the last three are objective, not the judge). Routing 0.92, no-answer 1.0, MRR 0.77.

**Weaknesses, ranked:**

1. **Metadata / listing recall (fix first).** Listing & negation queries under-recall because the simple
   path uses semantic top-k, not metadata filters — even though the tags exist: `g_det_02` ("NOT motor")
   **R@10 = 0.0**, `g_det_01` 0.39 (yet P@5 = 1.0 — precise but incomplete), `DSQ_02` 0.25.
   → *Fix:* enumerate matching docs by metadata filter.
2. **Adverse recall 0.34 (dimension 4).** The adverse pass runs (0.89) but specific adverse docs get
   crowded out by topical relevance. → *Fix:* retrieve adverse by document **stance label**, not ranking.
3. **Completeness 2.68.** Mostly downstream of #1 (missing docs → incomplete answers).
   → *Fix:* better recall + a per-facet synthesis checklist.
4. **Router 2/25 misses.** `DSQ_10` (ambiguous) over-clarified; `DSQ_12` (statutory) over-escalated to deep.

**Caveats.** Recall is vs my grounded golden set (my own methodology, as the brief asks). The judge is
Gemini-on-Gemini — offset by the objective checks (no-invented 1.0). `support_recall 0.40` is partly
because the labelled "supporting" subset is narrower than what the agent reasonably surfaces.

**Fix order:** (1) metadata enumeration → (2) stance-aware adverse → (3) router tuning → (4) facet
checklist. **Fixes #1 and #2 were applied in v2** ([`eval_results_v2.md`](eval_results_v2.md)).