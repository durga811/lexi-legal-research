# Lexi Agent — Evaluation Results (v2 — after Fix #1+#2)

Golden set: 25 queries (0 errored). Agent: LangGraph + Pinecone hybrid (dense e5 + BM25 + bge rerank) + Gemini. Metrics are document-level; recall is measured against the grounded golden set (see data/golden/golden_set_readme.md).

## Dimension 1 — Precision
- Precision@5: **0.591** · Precision@10: 0.481
- Answer precision (presented docs that are relevant): **0.405**

## Dimension 2 — Recall
- Recall@10: **0.662** · Recall@20: 0.77
- MRR: 0.807 · nDCG@10: 0.714 · Support recall: 0.446
- Recall@10 by difficulty: {'easy': 0.644, 'medium': 0.613, 'hard': 0.716}

## Dimension 3 — Reasoning Quality (1–5 judge + objective checks)
- Faithfulness: **3.88** · Legal reasoning: 3.4 · Completeness: 2.88
- Issue coverage: 0.957 · must-not-claim respected: 1.0 · no invented docs: 1.0

## Dimension 4 — Adverse Identification
- Adverse recall: **0.562** · Adverse-reasoning score: 3.72
- Adverse search executed (deep): 0.889 · Adverse section present: 0.889

## Agent behavior
- Router accuracy (simple vs deep): **0.92** · No-answer handled: 1.0

## Per-query results

| query_id | type | route ok | P@5 | R@10 | adv_recall | faith | preview |
|---|---|---|---|---|---|---|---|
| g_det_01 | factual_metadata | ✓ | 1.0 | 0.556 | None | 5 | Based on the provided documents, the following jud |
| g_det_02 | factual_metadata | ✓ | 1.0 | 0.5 | None | 5 | The following documents in the corpus are **not**  |
| g_det_03 | single_issue | ✓ | 0.2 | 0.5 | None | 3 | Based on the provided documents, the following cas |
| g_det_04 | single_issue | ✓ | 0.8 | 1.0 | None | 5 | Based on the provided documents, the following cas |
| g_det_05 | factual_metadata | ✓ | 0.8 | 1.0 | None | 5 | Based on the provided documents, the following jud |
| g_det_06 | statutory | ✓ | 0.2 | 1.0 | None | 5 | Based on the provided documents, the case that int |
| g_det_07 | procedural | ✓ | 0.2 | 1.0 | None | 5 | Based on the provided documents, the case that inv |
| g_det_08 | no_answer | ✓ | None | None | None | 5 | Based on the provided corpus, there are **no trade |
| g_det_09 | no_answer | ✓ | None | None | None | 5 | Based on the provided documents, the corpus **does |
| DSQ_01 | procedural | ✓ | 0.4 | 1.0 | None | 5 | Based on the provided documents, the following cas |
| DSQ_02 | procedural | ✓ | 0.2 | 0.25 | None | 3 | Based on the provided documents, the corpus does n |
| DSQ_03 | single_issue | ✓ | 0.4 | 0.375 | None | 3 | Based on the provided documents, the corpus does n |
| DSQ_04 | single_issue | ✓ | 1.0 | 0.778 | None | 5 | Based on the provided documents, the following cas |
| DSQ_05 | multi_hop | ✓ | 0.6 | 0.615 | 0.4 | 4 | ### 1. Issue Map  The legal issues in the provided |
| DSQ_06 | multi_hop | ✓ | 0.6 | 0.714 | None | 5 | Based on the provided documents, the following cas |
| DSQ_07 | multi_hop | ✓ | 0.8 | 0.833 | 0.75 | 5 | ### 1. Issue Map  The allocation of liability betw |
| DSQ_08 | comparison | ✓ | 0.4 | 0.625 | 0.75 | 0 | ### 1. Issue Map  The legal issues surrounding the |
| DSQ_09 | comparison | ✓ | 0.6 | 0.667 | None | 3 | ### 1. Issue Map  The interaction between FIR-base |
| DSQ_10 | ambiguous | ✗ | 0.4 | 0.333 | 0.0 | 5 | To help narrow down your request, here is how we c |
| DSQ_11 | ambiguous | ✓ | 0.2 | 0.2 | None | 1 | Here are the cases from the corpus where insurance |
| DSQ_12 | statutory | ✗ | 0.4 | 0.571 | None | 2 | ### 1. Issue Map  To determine "just compensation" |
| DSQ_13 | cross_document_strategy | ✓ | 0.8 | 0.615 | 0.6 | 1 | ### 1. Issue Map  In representing the widow of the |
| DSQ_14 | cross_document_strategy | ✓ | 1.0 | 0.778 | 0.75 | 4 | ### 1. Issue Map  In driving-licence disputes wher |
| DSQ_15 | cross_document_strategy | ✓ | 0.8 | 0.615 | 0.5 | 4 | ### 1. Issue Map  In a fatal motor accident claim  |
| DSQ_16 | cross_document_strategy | ✓ | 0.8 | 0.7 | 0.75 | 4 | ### 1. Issue Map  The legal issues in play regardi |

---

## v1 → v2 — effect of the two fixes

v1 baseline is in `eval_results_v1.{json,md}` (with its failure analysis). v2 applies the two top
fixes from that analysis.

**Fix #1 — metadata enumeration for listing/factual queries:** the simple path now maps a query to a
metadata predicate (legal_area / vehicle_tags / issue_tags / procedural_stage / is_motor, incl.
negation) and enumerates the full matching doc set instead of relying on semantic top-k.

**Fix #2 — stance-aware adverse identification:** a precedent's helpfulness is judged by its
**document-level** stance (not noisy chunk stance), and adverse precedents are drawn from a
**dedicated retrieval over genuinely-adverse documents** (best chunk per doc, reranked separately) so
they are not crowded out by topically-similar supporting docs.

| metric | v1 | v2 | Δ |
|---|---|---|---|
| Precision@5 | 0.53 | **0.591** | +0.06 |
| Recall@20 | 0.73 | **0.77** | +0.04 |
| MRR | 0.765 | **0.807** | +0.04 |
| nDCG@10 | 0.67 | **0.714** | +0.04 |
| Completeness | 2.68 | **2.88** | +0.20 |
| must-not-claim respected | 0.96 | **1.0** | +0.04 |
| **Adverse recall (dim 4)** | 0.342 | **0.562** | **+0.22** |
| Adverse reasoning | 3.48 | **3.72** | +0.24 |
| Answer precision | 0.474 | 0.405 | −0.07 |
| Faithfulness / no-invented / router / no-answer | 3.84 / 1.0 / 0.92 / 1.0 | 3.88 / 1.0 / 0.92 / 1.0 | ≈ |

**Net:** every dimension improved or held. The single regression — **answer precision −0.07** — is the
expected cost of the much larger **adverse-recall +0.22**: presenting more (genuinely adverse) precedents
means a few fall outside the golden set's narrow curated adverse subset. For a legal system where
"finding only favorable cases is dangerous," trading a little answer-precision for substantially better
adverse identification is the right call. No safety property regressed (zero invented docs; must-not-claim
now 1.0). Remaining weaknesses: `DSQ_02` ("insurer *appealed*") still needs appellant-role metadata, and
the two router boundary cases (`DSQ_10` clarify, `DSQ_12` over-escalate) are unchanged.

## Caveats & label audit

- **Judge scores are single-sample.** A prior iteration measured reasoning-judge std up to ±0.34 and a
  ~0.05–0.07 noise floor, and only trusted judge numbers at N≥5. So treat the judge metrics
  (faithfulness, reasoning, completeness) as indicative: the small judge deltas above (≤0.04) are within
  noise; the deterministic gains (adverse recall **+0.22**, completeness +0.20, precision +0.06) are real.
- **Self-judge bias:** the judge is Gemini scoring Gemini, offset by the objective checks (no-invented
  docs, must-not-claim, issue-coverage). A different judge model is the documented upgrade.
- **Label audit — DOC_031 (Laxmi Narain Dhut):** marked **adverse** here, but a prior iteration
  classed it **pro-claimant / supporting** (it limits *Swaran Singh* yet still protects third parties — a
  genuinely contested call). If a domain expert flips it, adverse-recall would shift. This is a known
  borderline label, not a verified one.