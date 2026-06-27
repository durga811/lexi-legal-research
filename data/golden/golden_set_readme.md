# Golden Evaluation Set — v1

`golden_set_v1.json` is the frozen evaluation set for the Lexi agent. It is
deliberately **lean (25 queries) but broad** — enough to test core behaviour and
coverage without becoming unmanageable. Labels prioritise **accuracy**: every
expected document is grounded in the corpus and verified.

## How it was built (§11)

Two label sources, both grounded in the **validated** `document_metadata.jsonl`
(case cards whose tags were checked against verbatim source quotes in Step 2):

1. **Deterministic (9 queries, `label_source: deterministic`).** Factual / metadata /
   statutory / no-answer queries whose `expected_relevant_doc_ids` are computed
   directly from metadata predicates (e.g. `is_motor_accident == false`,
   `issue_tags ∋ fake_license`). These are exact set-membership labels — the most
   accurate possible. Built by `src/evaluation/golden_deterministic.py`.

2. **Sub-agent designed (16 queries, `label_source: subagent`).** An Evaluation
   Designer (Claude Opus) read the 56-doc corpus index and authored the
   procedural / single-issue / multi-hop / comparison / ambiguous / statutory /
   cross-document-strategy queries, labelling expected / supporting / adverse docs
   from each doc's `issue_tags`, `stance_tags`, `outcome` and `adverse_value`.

## How labels were verified (§11.8)

`src/evaluation/build_golden.py` runs:
- **Schema + vocab + existence:** pydantic, controlled query-type/workflow/difficulty,
  every referenced `doc_id` exists, no-answer queries have empty expected sets.
- **Retrieval pooling:** for each query the real hybrid (dense + BM25) and case-card
  retrieval is run; `pool_recall` measures whether labelled docs are retrievable and
  `pooling_suggestions` lists retrieved-but-unlabelled docs. Mean pool_recall = **0.94**.
  Pooling suggestions were reviewed but **not auto-added** (that would make the gold
  set circular — gold = whatever retrieval finds). The metadata-grounded labels are
  the authority. Report: `golden_verification.json`.
- **Developer spot-review:** the supporting/adverse splits of the deep strategy
  queries were checked against `stance_tags` — every supporting doc is
  claimant-supporting / adverse-to-insurer, every adverse doc is
  adverse-to-claimant / insurer-supporting (or carries an `adverse_value`).

### Note on `g_det_02` (low pool_recall, intentionally hard)
"Which documents are NOT motor accident cases?" has exact labels (the 20 non-motor
docs) but low pool_recall, because a negation query is not answerable by dense/lexical
similarity — it must be answered from case-card **metadata** (`legal_area`). This is a
deliberate test of whether the agent uses metadata rather than pure semantic search.

## Composition

- 25 queries · 9 deep (support_adverse_strategy / comparison) · 55/56 docs covered.
- Types: factual_metadata, procedural, single_issue, multi_hop, comparison, ambiguous,
  statutory, no_answer, cross_document_strategy.
- Difficulty: 3 easy / 10 medium / 12 hard.
- 8 queries carry explicit `adverse_doc_ids` (drive eval dimension 4: adverse identification).

## Schema
See `src/evaluation/golden_schema.py`. Per query: `query`, `query_type`, `legal_area`,
`required_workflow`, `expected_relevant_doc_ids`, `supporting_doc_ids`,
`adverse_doc_ids`, `neutral_doc_ids`, `must_include_issues`, `must_not_claim`,
`ideal_answer_facets`, `difficulty`, `no_answer`, `label_source`.

## Regenerate
```
uv run python -m src.evaluation.build_golden
```
