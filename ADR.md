# ADR — Lexi Legal Precedent Research Agent

## What this is

An agent over 56 Indian court judgments that answers both **simple lookups** ("which judgments involve
commercial vehicles?") and **deep precedent research** (supporting precedents, adverse precedents,
strategy) — without hard-coding the case brief, and showing every retrieval/ranking step.

## Stack — and why

| Component | Choice | Why |
|---|---|---|
| Agent | **LangGraph** state machine | flexible simple-vs-deep routing; the graph *is* the visible reasoning trace |
| Retrieval | **hybrid**: dense + BM25 + metadata, **RRF** fused | dense finds paraphrase, BM25 finds exact legal terms; RRF avoids score-scaling issues |
| Embeddings | **Pinecone-hosted `multilingual-e5-large`** (open-source) | no embedding model in the app → light deploy; managed inference |
| Reranker | **Pinecone `bge-reranker-v2-m3`** | reorders the fused pool by true relevance |
| Chunking | **child (~300 tok) + parent (~1200 tok) + case cards** | small chunks = precision; parents = context; cards = doc-level lookups |
| Enrichment | **Claude sub-agents, build-time** | grounded case cards / metadata / stance labels, frozen once — no per-query cost |
| LLM (runtime) | **Gemini `gemini-3.5-flash`** | cost/latency; bump to Pro for synthesis if needed |
| Deploy / eval | **Railway · LangSmith** | hosted URL, no local infra to evaluate |

## Key decisions (and the tradeoff)

- **State machine, not one chain or one ReAct loop.** Explicit nodes (interpret → route → plan →
  retrieve(support/adverse/comp) → rerank → select → analyze → synthesize → validate) make the
  workflow inspectable and the routing semantic. *Tradeoff:* more parts than a single chain.
- **Hybrid retrieval + RRF + rerank.** Legal text needs both meaning and exact terms. *Tradeoff:*
  BM25 runs in-process (corpus in memory) instead of a second Pinecone index — trivial at 56 docs,
  minimal infra.
- **Hierarchical chunks + case cards.** Retrieve on small children, reason on parents (fetched by id,
  not embedded since they exceed the 507-token limit). *Tradeoff:* more artifacts to manage.
- **Enrichment at build time by Claude sub-agents**, validated by **verbatim-quote grounding** (a
  hallucinated quote fails the build). Runtime nodes stay Gemini calls (they run every query).
  *Tradeoff:* a corpus change needs a rebuild.
- **Stance is a document-level property.** Whether a precedent helps or hurts is decided by the
  document's stance label, not noisy chunk text — this is what makes adverse identification work.

## How the agent decides simple vs deep

The interpreter (one Gemini call → JSON) classifies each prompt as **simple**, **deep**, or **clarify**
from intent — *deep* when it needs supporting + adverse precedents or strategy; *simple* for
factual/listing lookups. A conditional edge routes accordingly. This is **semantic**, not an
`if "Lakshmi Devi"` branch (verified: a contributory-negligence prompt routes deep with its own
evidence; router accuracy 0.92). Simple lookups also use **metadata enumeration** (filter by
legal_area / tags), so "which documents are NOT motor cases?" is answered from metadata, not similarity.

## Evaluation (summary)

A 25-query grounded golden set (metadata-derived labels + a Claude designer sub-agent, pooling-verified)
drives automated metrics on the four required dimensions. After the two top fixes from the v1 failure
analysis (metadata-enumeration recall; stance-aware adverse), **v2**: adverse recall **0.34 → 0.56**,
completeness 2.68 → 2.88, precision@5 0.53 → 0.59, **zero invented documents**, router 0.92,
no-answer 1.0. Full numbers + failure analysis: [`reports/`](reports/).

## If the corpus were 5,000 docs (not 50)

- Move BM25 off in-process memory → a **Pinecone sparse index** (no loading the corpus into RAM).
- Lean on **hard metadata pre-filters** (area, year, court) to shrink the candidate set before rerank.
- Batch the Claude enrichment with a coordinator + incremental re-runs; spot-review a sample, not every doc.
- Stratified golden-set sampling + human adjudication of a subset, instead of exhaustive labels.
- Cache embeddings + interpreter/query-expansion outputs; add a cheap first-pass filter model.

## If I had another week

1. Add **appellant-role metadata** so "cases where the *insurer appealed*" works; tune the two router
   boundary cases (clarify-vs-deep, statutory-vs-simple).
2. **Per-facet synthesis checklist** so multi-part answers address every requested part (lifts completeness).
3. **Second, different judge model** (cut self-judge bias) and **multi-sample** judge scores.
4. **Compensation calculator** as a deterministic tool the agent calls.

## Learnings from a prior iteration

An earlier version (local Chroma + `bge-small` embeddings + cross-encoder + a **single-agent ReAct loop
driven by one big prompt**) was built and evaluated first. Its numbers ran on a *different* gold set, so
they are **not directly comparable** to this build — they are used below only to show *direction*.

- **A single-prompt agent limited flexibility.** One monolithic system prompt had to cover lookups,
  deep research, and no-answer cases at once, so behavior couldn't adapt cleanly to different prompt
  shapes. This build splits the work into **typed nodes with an explicit router and per-node prompts**,
  so behavior varies by design — not by overloading one prompt. (The prior build's own note: "a 22-step
  ReAct log is a worse reasoning artifact than an up-front plan.")
- **Adverse identification is a classification problem, not a ranking problem.** In the prior build,
  counter-query *prompting* didn't move adverse recall at all, and regex outcome-tagging failed
  (adverse docs *discuss* "pay and recover" while *rejecting* it — the signal is in the holding). The
  fix it deferred — an ingest-time stance classifier, retrieve adverse by label — is what this build
  implements.
- **Lookup recall is an enumeration problem**, not a ranking one → solved here by the metadata path.
- **Eval lessons inherited:** score retrieval-level vs answer-level separately; gate metrics by query
  kind; use nDCG; and use a **custom grounding rubric** (a stock faithfulness metric scored a
  *fabricated* claim 1.0). **Not yet fixed:** LLM judges are noisy (the prior build needed N≥5); our
  judge scores are single-sample, so small judge deltas are within noise.
- **Label audit (carry-forward):** the prior build found that mislabeling pro-claimant Supreme Court
  cases as adverse inflated adverse-recall. This build marks **DOC_031 (Laxmi Narain Dhut)** as adverse
  — a contested call flagged for domain audit (see the eval report).

## Tradeoffs at a glance

| Decision | Gained | Gave up |
|---|---|---|
| LangGraph state machine | inspectability, flexible routing | more parts than one chain |
| In-process BM25 | reproducible, low infra | corpus in memory (fine at 56) |
| Parent/child + case cards | precision + context | more artifacts |
| Build-time Claude enrichment | light app, grounded labels | rebuild on corpus change |
| Pinecone-hosted embeddings | no model in app, scale path | external dep + rate limits |
| Doc-level adverse selection | adverse recall up | a little answer precision |
