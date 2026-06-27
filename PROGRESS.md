# Lexi Legal Precedent Research Agent — Build Progress Log

Living log of the staged build. Each step is planned → executed (deterministic code and/or
Claude sub-agents) → **verified against the requirements** before the next step starts.
Authoritative requirements: `goal/Lexi BE EG 2026 Research Assessment.pdf`.
Implementation plan: `goal/goal-guide.md`.

## Execution plan (10 steps)

| # | Step | Sub-agents? | Guide § | Status |
|---|------|-------------|---------|--------|
| 1 | Corpus ingestion (parse/clean/chunk, deterministic) | no | §5, §16-P1 | ✅ Done |
| 2 | Build-time enrichment (case cards, metadata, chunk labels) | yes | §5.6–5.7, §18.5 | ✅ Done |
| 3 | Pinecone index + upsert | no | §7, §16-P2 | ✅ Done |
| 4 | Retrieval layer (hybrid + RRF + rerank) | no | §8, §18.2 | ✅ Done |
| 5 | LangGraph agent (runtime Gemini nodes) | no | §3–4, §18.1 | ✅ Done |
| 6 | Streamlit UI with visible trace | no | §9.2 | ✅ Done |
| 7 | Golden set (25 queries, accuracy-verified) | yes | §11 | ✅ Done |
| 8 | Evaluation framework + results | no | §12 | ✅ Done |
| 9 | Railway deployment | no | §14, §16-P7 | ✅ Live |
| 10 | README + ADR + submission | no | §15.2, §18 | ✅ Done (URL in README; user sends email) |

Dependency chain: 1→2→3→4→5→6; 7 needs 2; 8 needs 5+6+7; 9 needs 6; 10 needs 8+9.

---

## ✅ Step 1 — Corpus ingestion (deterministic code)

**Built** (`src/ingestion/`, `src/utils/`; pure deterministic code, no LLM):
- `parse_pdfs.py` — PyMuPDF primary + pdfplumber per-page fallback.
- `clean_text.py` — strips Indian Kanoon URL, recurring case-title footer, page numbers; de-hyphenates; preserves paragraphs. Captures a `source_hint` (footer title/date) for Step 2 grounding.
- `chunker.py` — paragraph-aware hierarchical chunking: parents (~1200 / ≤1500 tok), children (≤470 est tok, word-overlap ~60 tok). Oversized paragraphs window-split; intact form preserved in the parent.
- `run_ingest.py` — normalize → parse → clean → chunk → **validate** → report.
- `config.py`, `utils/tokens.py` (conservative e5 token estimator), `utils/ids.py`, `utils/schemas.py` (pydantic).

**Results (verified):**
- 56/56 PDFs parsed, 0 empty, no OCR needed.
- 717 parent chunks, 3,647 child chunks.
- Child token estimate max **469 < 507** (e5 limit) → no silent truncation; 3 clamped, 5 tiny fragments (0.14%).
- 0 footer/URL leaks; page-1 case caption preserved (good for grounding).
- All child→parent→doc links valid; no orphans, duplicates, or dangling IDs.
- Per-doc cleaned text → `data/processed/extracted/DOC_xxx.{txt,json}` (Step 2 input).

**Verification:** `run_ingest` validation PASSED all §13.2 guarantees; `tests/test_ingestion.py` 5/5 pass.

**Bug caught & fixed:** first run failed validation (17 children over the token clamp) because the
clamp truncated by word-count while the estimate is char-aware. Fixed the clamp to trim until the
real estimate is under bound; tightened budgets; dropped heading-fragment children. Re-ran clean.

**Artifacts:** `data/processed/{parent_chunks.jsonl, child_chunks.jsonl, ingest_report.json}`, `extracted/`.

**Design note:** parents are NOT embedded in Pinecone (they exceed 507 tokens); they live in
`parent_chunks.jsonl` and are fetched by `record_id` for parent-expansion (§7.5). Only child chunks
and case cards get embedded.

---

## ✅ Step 2 — Build-time enrichment (Claude sub-agents + deterministic labeling)

Two planes per §5.7: **Claude sub-agents** for the LLM-grade document reading; **deterministic
code** for assembly, validation, and chunk-grounded tagging. No runtime LLM.

**Contract + tooling (deterministic, written first):**
- `src/ingestion/enrichment_spec.md` — schema + controlled vocabularies + grounding rules the sub-agents follow.
- `src/ingestion/vocab.py` — controlled vocab (legal_area, stage, issue/stance/vehicle tags) + grounding lexicons + chunk-type cues.
- `src/utils/schemas.py::DocEnrichment` — pydantic schema.
- `src/ingestion/assemble_enrichment.py` — validates sub-agent output (pydantic + vocab + **grounding** + coverage + motor-consistency) → freezes `case_cards.jsonl` + `document_metadata.jsonl`.
- `src/ingestion/chunk_labeler.py` — deterministic, chunk-grounded `chunk_type` + issue/vehicle/stance tags → `chunk_labels.jsonl`.

**Sub-agent fan-out:** 8 Document Analyst sub-agents (Sonnet), 7 disjoint docs each, read full
`extracted/*.txt` and wrote one `data/processed/_enrich/DOC_xxx.json` per doc with verbatim
`key_passages` as the anti-hallucination anchor.

**Validation (verified):**
- Coverage: 56/56 docs enriched exactly once; all DOC_001..056 present.
- Grounding: verbatim `key_passages` checked as substrings of source text — clean (only 1 minor non-verbatim quote in DOC_048, doc still ≥50% verbatim). Title/statute/motor cross-checks pass.
- Vocabulary: all tags in controlled vocab; motor-consistency (non-motor ⇒ `vehicle_tags == []`) holds.
- Legal-area mix: 33 motor_accident (+3 criminal-rash-driving with is_motor=true) + criminal 5, trademark_ip 4, consumer 3, specific_performance 3, banking_finance 2, service 2, tax 1, civil_procedure 1, other 2 — matches the heterogeneous corpus the guide describes.
- **Adverse coverage (eval dim 4):** 9 `adverse_to_claimant`, 13 insurer-supporting/adverse docs, 25 with explicit `adverse_value`; developer spot-review confirmed they are genuinely adverse (insurer-exoneration, license-breach defence, comp reduction, *Laxmi Narain Dhut*).
- Chunk labels: 3,647/3,647 labeled; 64% carry ≥1 grounded issue tag; chunk-type distribution sensible after tightening the compensation-calc cue.

**Verification:** `assemble_enrichment` PASSED; `chunk_labeler` OK; `tests/test_enrichment.py` 4/4 (+ 5 ingestion) = **9/9 pass**.

**Bug caught & fixed:** DOC_027 used an out-of-vocab `vehicle_tag` (`heavy_goods_vehicle`) → re-processed
by the same sub-agent via SendMessage (faithful to "re-process the offending doc"), not hand-edited.
Also fixed a false-positive in the statute grounding check (it matched the act *year* instead of the
section number) → warnings dropped 82 → 41 (all remaining are coarse issue-lexicon notes, non-blocking).

**Artifacts:** `data/processed/{case_cards.jsonl, document_metadata.jsonl, chunk_labels.jsonl,
enrichment_report.json, chunk_labels_report.json}`, raw `_enrich/DOC_xxx.json` (kept for reproducibility/diff).

**Design note:** chunk-level issue/vehicle tags = doc-level tag ∩ chunk-text lexicon hit (grounded to
both case and chunk); chunk stance derived directly from chunk text so a chunk can carry adverse
reasoning inside a claimant-supporting case (§10.4). These feed metadata boosting in Step 4 retrieval.

---

## ✅ Step 3 — Pinecone index + upsert

**Built:**
- `src/retrieval/pinecone_client.py` — shared client + `ensure_index()` (creates the integrated-inference index, waits until ready) + `get_index()`. Config from `.env`. Used by both ingestion and the runtime app.
- `src/ingestion/pinecone_upsert.py` — merges `child_chunks.jsonl` (text+structure) with `chunk_labels.jsonl` (tags) into flat Pinecone records, plus the 56 case cards; throttled token-rate upsert with 429 retry + skip-existing.
- `src/ingestion/pinecone_smoke.py` — reproducible retrieval/filter verification.

**Index:** `lexi-legal-precedents-v1`, namespace `corpus-v1`, **integrated inference** with hosted
`multilingual-e5-large` (1024-dim) — so no embedding model runs in our app (Railway stays light).
Field map `{"text": "text"}`. Parents are NOT embedded (kept in JSONL for parent-expansion).

**Upserted:** 3,703 records = 3,647 child_chunk + 56 case_card. `total_vector_count` = **3,703** (verified).
Flat metadata per record: doc_id, record_type, parent_id, legal_area, case_title, court, year,
chunk_type, issue_tags, stance_tags, vehicle_tags, page_start/end (null/empty fields dropped).

**Verification (`pinecone_smoke` PASSED):**
- T1 license/compensation scenario → top-8 child chunks all `motor_accident`, `driving_license_finding`/`compensation` (0.86–0.88).
- T2 `legal_area=trademark_ip` hard filter → only DOC_046–049.
- T3 case-card "fake license" → DOC_031 (*Laxmi Narain Dhut*), 034, 003, 005, 024, 032.
- T4 `vehicle_tags $in [truck]` → distinct truck docs (001, 019, 021, 027, 010).
- T5 adverse query → surfaces insurer-liability/adverse cases.

**Issues caught & fixed:** (1) hosted e5 rate limit (250k tok/min) → added a token-rate throttle
(220k/min) + 429 backoff + skip-existing so re-runs are cheap and idempotent. (2) Pinecone MCP key
is stale/empty → used the Python SDK with the `.env` key directly (correct for the deliverable anyway).
(3) v9 SDK hit objects expose `.id`/`.score`/`.fields` (not `_score`); stats expose `total_vector_count`.

---

## ✅ Step 4 — Retrieval layer (hybrid + RRF + rerank + evidence selection)

**Built (`src/retrieval/`):**
- `corpus_store.py` — in-memory loader (cached) for merged child records, parents, doc metadata, case cards; builds a **BM25** index over child text (lexical signal — our dense e5 index can't do BM25 itself). Added `rank-bm25` dep.
- `rrf.py` — reciprocal rank fusion.
- `query_expansion.py` — deterministic synonym broadening (license↔licence/DL, commercial vehicle↔truck/tempo…); the LLM-driven expansion is a Step 5 agent node.
- `hybrid_retriever.py` — `dense_search` (Pinecone e5) + `lexical_search` (BM25) + metadata soft-boost → RRF fuse; plus `case_card_search` for the simple/doc-level path. Pinecone-style filter matcher reused to post-filter BM25.
- `reranker.py` — Pinecone-hosted **`bge-reranker-v2-m3`**; deterministic fused-order fallback on error.
- `evidence_selector.py` — stance classification relative to client side, doc-diversity cap, support/adverse/neutral buckets, parent-expansion (§7.5).
- `pipeline.py::retrieve_evidence` — one-call entry point (hybrid→rerank→select) returning every stage for the trace.
- `retrieval_smoke.py` — reproducible live verification.

**Each candidate carries every sub-score** (dense score+rank, lexical score+rank, RRF, metadata boost, fused, rerank) → the §9.2 visible-trace data is fully available for Step 6.

**Verification:** `tests/test_retrieval.py` (RRF, filter matcher, query expansion, stance classification, BM25) — offline suite now **14/14 pass**; live `retrieval_smoke` PASSED:
- Hybrid value demonstrated: DOC_006_child_0037 was dense-rank 23 but lexical-rank 2 → fused into the top (a chunk dense alone would have missed).
- bge reranking active (no fallback); top reranked chunks are `driving_license_finding` at 0.96–0.98.
- Selection surfaces **both supporting and adverse** precedents (DOC_032/027/033 adverse) with the 2-per-doc diversity cap respected and 8 parents expanded.
- Simple `case_card_search` returns the right commercial-vehicle docs.

**Design note:** lexical retrieval is local BM25 (3,647 chunks, in-memory, ms-fast) rather than a second
Pinecone sparse index — keeps infra/cost/rate-limit surface minimal and is fully reproducible. Metadata
filters are soft boosts by default (§8.3); hard filters available via `meta_filter`.

---

## ✅ Step 5 — LangGraph agent (runtime Gemini nodes)

**Built (`src/agent/`):** all runtime nodes are programmatic **Gemini `gemini-3.5-flash`** calls (NOT Claude sub-agents — they run on every query).
- `llm.py` — Gemini wrapper; tolerant `call_json` (handles Gemini's list-of-blocks content + prose/fences) + `render()` (token substitution; prompts contain literal JSON braces) + LangSmith env wiring.
- `state.py` — `LegalResearchState` (§3.3); UI reads its fields for the trace.
- `nodes/interpreter.py` — Input Interpreter + Intent Router (semantic routing).
- `nodes/planner.py` — Research Planner. `nodes/query_expansion.py` — purpose-tagged query generation + vocab-validated boost tags.
- `nodes/retrieval.py` — support / adverse / compensation passes (merge + dedup) + evidence-selection node (rerank merged set → support/adverse/neutral, parent-expand, top-up).
- `nodes/analysis.py` — case analyzer + strategy synthesizer + grounding validator.
- `nodes/simple.py` — simple-lookup path + clarify path.
- `graph.py` — `StateGraph` wiring (interpret → route → deep chain | simple chain | clarify) + `run_agent()`.

**Graph:** `interpret →[route]→` deep: `plan→expand→support→adverse→compensation→select→analyze→synthesize→validate→END`; simple: `simple_retrieve→simple_answer→END`; clarify→END.

**Verification:** offline `tests/test_agent.py` (graph compiles, routing) — suite now **16/16 pass**; live `agent_smoke` PASSED on 3 query types:
- simple "which judgments involve commercial vehicles?" → routes simple, concise grounded list.
- deep licence/insurer dispute → routes deep; support DOC_032/027/006/034/003, adverse DOC_031 (*Laxmi Narain Dhut*)/035; validation passes, **adverse section present, zero invented citations**; structured Issue Map/Supporting/Adverse/Strategy answer.
- deep **contributory negligence** (different topic) → routes deep with its own support/adverse docs — **proves the agent is not a hard-coded Lakshmi pipeline** (Critical Requirement: Flexibility).

**Bug caught & fixed:** `str.format` treated literal JSON braces in prompts as placeholders (`KeyError: '"death"'`) → added `render()` token-replacement and switched all nodes; also handled Gemini returning content as a list of blocks.

**Design note:** node files are grouped by concern (interpreter/planner/expansion/retrieval/analysis/simple)
rather than one-file-per-node, but each maps 1:1 to a guide §4 node (documented in `graph.py`). The trace
data (§9.2) is the live state — every query, candidate score, rerank score, selection and validation is captured.

---

## ✅ Step 6 — Streamlit UI with visible trace

**Built:** `app.py` — single-page Streamlit app. Prompt box accepts ANY prompt; sidebar has example
prompts (simple + deep + no-answer). Renders the **final answer** AND the full §9.2 trace in ordered expanders:
1. Routing — intent / complexity / client side / legal area + extracted facts + issue map.
2. Research plan & generated queries (purpose-tagged) + metadata boosts (deep only).
3. Retrieved candidates — table of **exact** doc_id/record_id/chunk_type with **dense, lexical, RRF, boost, fused** scores and which passes hit each.
4. Reranked candidates — bge relevance score + new order; per-chunk expander shows all sub-scores, pages, and the **exact chunk text**.
5. Selected evidence — 🟢 supporting / 🔴 adverse / ⚪ neutral columns with exact passages + stance + parent-chunk context (simple path shows the case-card matches table instead).
6. Per-case analysis — facts/issue/holding/why/use/distinguish/confidence/cited evidence ids.
7. Grounding validation — pass/fail, adverse-section-present, invented/uncited docs, issues, fixes.

Directly satisfies the task: *"Intermediate reasoning steps must be visible — which documents retrieved,
how ranked, how it concluded. Do not show only the final output."*

**Verification:**
- App boots headless and serves (`/_stcore/health` → ok) — confirms the whole agent+retrieval stack imports.
- `tests/test_ui.py` (render functions run against representative deep / simple / empty states via a mock `st`) — full suite now **19/19 pass**.
- `app.py` page body wrapped in `main()` under `if __name__ == "__main__"` so it stays importable for testing while Streamlit still runs it.

**Note:** the Claude-in-Chrome extension wasn't connected in this session, so the browser-driven click-through
was substituted with headless-boot + render-unit verification; the live hosted click-through is covered by the
Step 9 Railway smoke test.

---

## ✅ Step 7 — Golden set (lean, accuracy-verified)

Scoped to **25 queries** (per the "minimal but good coverage" instruction) with **label accuracy** as the priority.

**Built (`src/evaluation/`):**
- `golden_schema.py` — pydantic `GoldenQuery` + controlled vocabularies.
- `golden_deterministic.py` — 9 factual/single-issue/statutory/procedural/no-answer queries whose `expected_relevant_doc_ids` are computed from **validated metadata predicates** (e.g. `is_motor_accident==false`, `issue_tags ∋ fake_license`) — exact set-membership, the most accurate labels possible.
- Evaluation Designer **sub-agent (Opus)** read the 56-doc corpus index and authored 16 procedural/single-issue/multi-hop/comparison/ambiguous/statutory/cross-document-strategy queries, labelling expected/supporting/adverse from each doc's tags/outcome/`adverse_value`.
- `build_golden.py` — assembles, validates (pydantic + vocab + doc existence + no-answer emptiness), runs **retrieval pooling** verification, and freezes the set.

**Verification (the accuracy gate):**
- Schema/vocab/existence: all pass; no-answer queries verified empty.
- **Retrieval pooling** (§11.8): real hybrid + case-card retrieval per query; mean **pool_recall = 0.94** (labels are retrievable). Pooling suggestions reviewed but NOT auto-added (avoids gold↔retrieval circularity). The one low-recall query (`g_det_02`, "which are NOT motor cases") is an intentional metadata/negation test — labels are exact.
- **Developer spot-review** of the deep strategy queries' supporting/adverse splits vs `stance_tags`: every supporting doc is claimant-supporting/adverse-to-insurer; every adverse doc is adverse-to-claimant/insurer-supporting (DOC_031 *Laxmi Narain Dhut* consistently adverse).

**Composition:** 25 queries · 9 deep · 55/56 docs covered · 3 easy/10 medium/12 hard · 8 with explicit adverse labels (eval dim 4) · all 9 query types represented.

**Verification:** `tests/test_golden.py` (schema, label validity, coverage, adverse presence) — suite now **21/21 pass**.

**Artifacts:** `data/golden/{golden_set_v1.json, golden_verification.json, golden_set_readme.md, _corpus_index.json, _designer_queries.json}`.

---

## ✅ Step 8 — Evaluation framework + results

**Built (`src/evaluation/`):** `retrieval_metrics.py` (P@k, R@k, MRR, nDCG, set recall/precision),
`answer_metrics.py` (objective deterministic checks + Gemini LLM-as-judge rubric),
`agent_metrics.py` (router accuracy, adverse-search execution, doc-list extractors),
`run_eval.py` (runs the agent over all 25 golden queries → aggregates the 4 dimensions → writes
`reports/eval_results_v1.{json,md}`), `langsmith_dataset.py` (uploaded golden set as LangSmith
dataset `lexi-golden-v1`, 25 examples).

**Ran the full baseline (25/25 queries, 0 errors):**
- **Precision:** P@5 0.53 · P@10 0.43 · answer-precision 0.47
- **Recall:** R@10 0.65 · R@20 0.73 · MRR 0.77 · nDCG@10 0.67 (recall@10 by difficulty: easy 0.59 / med 0.62 / hard 0.71)
- **Reasoning:** faithfulness 3.84/5 · legal 3.32 · completeness 2.68 · **no invented docs 1.0** · issue-coverage 0.97 · must-not-claim 0.96
- **Adverse ID (dim 4):** adverse-recall 0.34 · adverse-reasoning 3.48 · adverse-search-executed 0.89 · adverse-section-present 0.89
- **Behavior:** router accuracy 0.92 · no-answer handled 1.0

**Failure analysis written** (`reports/eval_results_v1.md`) — the PDF's required "where it fails / what
to fix first": (1) metadata/listing recall is the #1 gap (negation & large-gold factual queries use
semantic top-k not metadata filters — `g_det_02` R@10=0.0); (2) adverse recall 0.34 (mechanism runs but
specific adverse docs get crowded out — fix with stance-aware adverse retrieval); (3) completeness is
downstream of recall; (4) 2 router boundary misses. Honest caveats noted (recall vs my golden set;
Gemini self-judge offset by objective checks).

**Verification:** `tests/test_eval_metrics.py` (metric math + extractors) — suite now **26/26 pass**.
**Metric bug caught & fixed:** the no-answer check wrongly penalised listing *closest* docs (§8.5 allows
this) → relaxed to "states no such case + invents nothing"; no-answer rate corrected 0.5 → **1.0**.
