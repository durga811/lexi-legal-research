# Lexi — Legal Precedent Research Agent

A flexible legal precedent research agent over a corpus of 56 Indian court judgments. Given any
prompt — a simple corpus lookup or a deep precedent-research task — it retrieves relevant judgments,
separates **supporting** vs **adverse** precedents, explains the legal reasoning, and recommends a
litigation strategy, while exposing every intermediate retrieval/ranking step.

> Status: work in progress. Build log: [`PROGRESS.md`](PROGRESS.md).

## Architecture (high level)

```
56 PDFs ─▶ ingestion (parse → clean → hierarchical chunk, deterministic)
        ─▶ build-time enrichment (Claude sub-agents → case cards, metadata, chunk labels)
        ─▶ Pinecone (hosted multilingual-e5-large, integrated inference) + local BM25
        ─▶ LangGraph agent (Gemini nodes): interpret → route → plan → query-expansion
              → support/adverse/compensation retrieval → rerank → select → analyze
              → synthesize → validate
        ─▶ Streamlit UI (full visible evidence trace)
        ─▶ LangSmith (tracing) + automated evaluation over a grounded golden set
```

- **Agent orchestration:** LangGraph (stateful, inspectable; simple vs deep routing is semantic, not hard-coded)
- **Retrieval:** hybrid — dense (Pinecone `multilingual-e5-large`) + lexical (BM25) + metadata, fused by RRF, reranked by Pinecone `bge-reranker-v2-m3`
- **LLM (runtime):** Google Gemini (`gemini-3.5-flash`)
- **Build-time enrichment + golden set:** Claude Code sub-agents (offline, one-time)
- **Deploy:** Streamlit on Railway · **Eval/observability:** LangSmith

## Setup

```bash
# Python 3.12, uv
uv sync
cp .env.example .env   # fill in PINECONE_API_KEY, GOOGLE_API_KEY, LANGSMITH_API_KEY

# (offline, one-time) build the corpus + index — needs the source PDFs in raw-docs/
uv run python -m src.ingestion.run_ingest          # parse + chunk (deterministic)
# build-time enrichment is produced by Claude sub-agents into data/processed/_enrich/
uv run python -m src.ingestion.assemble_enrichment # validate + freeze case cards/metadata
uv run python -m src.ingestion.chunk_labeler       # grounded chunk labels
uv run python -m src.ingestion.pinecone_upsert     # create index + upsert

# run the app
uv run streamlit run app.py
```

The hosted app only needs `data/processed/` (tracked) + a populated Pinecone index; the raw PDFs are
required only for offline re-ingestion.

## Evaluation

```bash
uv run python -m src.evaluation.build_golden   # assemble + verify the golden set
uv run python -m src.evaluation.run_eval       # run the agent over the golden set → reports/
```

Measures the four required dimensions — Precision, Recall, Reasoning Quality, Adverse Identification —
plus agent behavior (routing, no-answer). Results + failure analysis in [`reports/`](reports/).

## Layout

```
app.py                      Streamlit UI (visible trace)
src/ingestion/              parse, clean, chunk, enrichment assembly, Pinecone upsert
src/retrieval/              corpus store, hybrid retriever, RRF, reranker, evidence selector
src/agent/                  LangGraph state, graph, Gemini nodes
src/evaluation/             metrics, golden-set build, eval runner
data/processed/             frozen JSONL artifacts (chunks, case cards, metadata, labels)
data/golden/                golden_set_v1.json + verification
tests/                      offline unit/artifact tests
```

## Hosted app

URL: _to be added (Railway)._
