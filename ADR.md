# Architecture Decision Record — Lexi Legal Precedent Research Agent

## Context

Build a flexible agent over 56 Indian court judgments that handles both simple corpus questions
("which judgments involve commercial vehicles?") and deep precedent research (supporting precedents,
adverse precedents, litigation strategy) — without hard-coding the provided case brief, and with all
intermediate retrieval/ranking steps visible. The corpus is heterogeneous: mostly motor-accident
insurance cases, plus criminal, trademark/IP, banking, civil-procedure, and consumer matters.

## Decisions and rationale

### 1. Agent framework — LangGraph (not a single RAG chain)
The task needs *flexible* behavior: the same system must answer a one-line lookup and run a multi-step
research workflow. I modelled this as a **stateful, inspectable state machine**: `interpret → route →`
either a simple-lookup path or a deep path (`plan → query-expansion → support/adverse/compensation
retrieval → rerank → select → analyze → synthesize → validate`). LangGraph gives typed state, explicit
nodes, and a natural place to expose every intermediate step to the UI and to LangSmith.
**Tradeoff:** more moving parts than a single chain, but the inspectability and the clean simple-vs-deep
split are exactly what the assessment grades. **Not allowed / rejected:** CrewAI-style drag-and-drop.

### 2. Retrieval — hybrid (dense + lexical + metadata), RRF, reranked
Legal text needs both semantics *and* exact-term recall (statute sections, "pay and recover", case
names). I combined **dense** (Pinecone hosted `multilingual-e5-large`) + **lexical BM25** (local, over
3,647 child chunks) + **metadata** boosts, fused with **Reciprocal Rank Fusion**, then reordered by
Pinecone's hosted **`bge-reranker-v2-m3`**. RRF avoids either signal dominating; the eval shows the
value directly (a dense-rank-23 chunk rescued to the top by lexical-rank-2).
**Tradeoff:** BM25 in-process rather than a second Pinecone sparse index — minimal infra/cost/rate-limit
surface and fully reproducible, at the price of holding the corpus in memory (trivial at this scale).

### 3. Chunking — hierarchical parent/child + document case cards
Judgments are inconsistently structured, so a section-aware splitter alone fails. I used **child chunks**
(~300 tokens, retrieval units, kept under the e5 507-token limit with a conservative estimator) inside
**parent chunks** (~1,200 tokens, context for reasoning, fetched via parent-expansion), plus a
**document-level case card** for broad corpus queries. Parents are *not* embedded (they exceed 507
tokens) — they live in JSONL and are fetched by id. **Tradeoff:** more artifacts to manage, but it gives
both retrieval precision (small children) and reasoning context (parents) without truncation.

### 4. Build-time enrichment by Claude sub-agents (not a runtime LLM loop)
Case cards, document metadata, and chunk issue/stance/vehicle tags are produced **once, offline, by
Claude Code sub-agents** reading each document in full, then frozen into JSONL. This keeps the deployed
app light (no per-query enrichment cost) and yields higher-quality, grounded labels. Reliability is
enforced in code: a strict schema, **verbatim-quote grounding** (a hallucinated quote fails validation),
controlled vocabularies, coverage and motor-consistency checks; offending docs are re-processed.
The **runtime** nodes (interpreter, planner, analyzer, synthesizer, validator) stay programmatic Gemini
calls because they run on every query. **Tradeoff:** enrichment isn't regenerated live, so corpus changes
require a rebuild — acceptable for a fixed corpus.

### 5. Models — open-source embedding + Gemini Flash
`multilingual-e5-large` (open-source, Pinecone-hosted integrated inference) satisfies the open-source
embedding allowance while avoiding running a model server on the host. Runtime reasoning uses
**Gemini `gemini-3.5-flash`** for cost/latency; the synthesis/validation nodes are the place to bump to a
Pro tier if quality demands.

### 6. Stance-aware adverse identification
"Adverse" is a **document-level** property of a precedent (a claimant-winning judgment that nonetheless
reserves the insurer's recovery rights is adverse authority), so adverse precedents are drawn from a
dedicated retrieval over genuinely-adverse documents and classified by document-level stance, not noisy
chunk stance. This is what made adverse recall jump (see Evaluation).

## How the agent decides simple vs deep

The Input Interpreter (an LLM call returning JSON) classifies each prompt's `complexity` as
`simple`, `deep`, or `clarify`, based on intent — *deep* when the prompt needs supporting + adverse
precedents or strategy or multi-issue research; *simple* for factual/listing/metadata lookups. A
conditional edge routes accordingly. This is **semantic**, not an `if "Lakshmi Devi"` branch — verified
by the eval (router accuracy 0.92; a contributory-negligence prompt and the licence prompt both route
deep with their own evidence). Simple lookups additionally use **metadata enumeration** (filter by
legal_area / tags) so "which documents are NOT motor cases?" is answered from metadata, not similarity.

## Evaluation (summary)

A 25-query grounded golden set (deterministic metadata-derived labels + a Claude designer sub-agent,
pooling-verified) drives automated metrics across the four required dimensions. After applying the top
two fixes from the v1 failure analysis (metadata-enumeration recall; stance-aware adverse), **v2** vs v1:
adverse recall **0.34 → 0.56**, completeness 2.68 → 2.88, precision@5 0.53 → 0.59, MRR/nDCG/recall@20 up,
**zero invented documents** throughout, router 0.92, no-answer 1.0. The one regression — answer precision
−0.07 — is the deliberate cost of surfacing more genuinely-adverse precedents. Full numbers + per-query
failure analysis in [`reports/`](reports/).

## What I'd change for 5,000 documents (instead of 50)

- **Enrichment:** the same Claude sub-agent methodology, but batched/queued with a coordinator and
  incremental re-runs; spot-review a sample rather than every adverse doc.
- **Lexical signal:** move BM25 off in-process memory to a **Pinecone sparse index** (`pinecone-sparse-
  english-v0`) or OpenSearch, so retrieval doesn't depend on loading the corpus into RAM.
- **Retrieval:** rely more on **hard metadata pre-filters** (legal_area, year, court) to shrink the
  candidate space before dense/rerank; cache embeddings; paginate enumeration queries.
- **Eval:** stratified sampling of the golden set, retrieval pooling across more systems, and human
  adjudication of a labelled subset rather than exhaustive labels.
- **Cost/latency:** batch the per-case analysis, add a cheaper first-pass filter model, and cache
  query-expansion/interpreter outputs for repeated query shapes.

## What I'd change with another week

1. **Close the recall gaps the eval exposed:** add appellant-role metadata (so "cases where the *insurer
   appealed*" works) and tune the router's clarify-vs-deep and statutory-vs-simple boundaries (the 2
   router misses).
2. **Per-facet synthesis checklist** so multi-part strategy answers explicitly address every requested
   facet (lifts completeness further).
3. **Stronger answer-quality eval:** a second, different judge model (reduce self-judge bias) and
   chunk-level recall labels in the golden set.
4. **Compensation calculator** as a deterministic tool the agent calls, instead of free-form arithmetic.
5. **Caching + streaming** in the UI for lower latency, and a feedback loop to capture corrections.

## Key tradeoffs, summarized

| Decision | Gained | Gave up |
|---|---|---|
| LangGraph state machine | inspectability, flexible routing | more components than one chain |
| In-process BM25 | reproducible, low infra | corpus held in memory (fine at 56 docs) |
| Parent/child + case cards | precision + context, no truncation | more artifacts |
| Build-time Claude enrichment | light app, grounded labels | rebuild needed on corpus change |
| Gemini Flash | cost/latency | occasional depth (bump to Pro if needed) |
| Doc-level adverse selection | +0.22 adverse recall | −0.07 answer precision |
