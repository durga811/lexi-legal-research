# ADR — Lexi Legal Precedent Research Agent

## Scope

The system is a Python legal precedent research agent over **56 Indian court judgments**. It supports both simple corpus questions and deep legal research. Deep research outputs supporting precedents, adverse precedents, and strategy recommendations. The app also exposes intermediate retrieval/ranking steps so the answer is not a black box.

---

## 1. Architecture choices

### Agent framework

I used a **LangGraph state-machine agent** instead of a single ReAct loop.

Reason: the task needs two different behaviours:

- **Simple lookup:** factual/listing questions such as “which cases involve commercial vehicles?”
- **Deep legal research:** questions requiring supporting precedents, adverse precedents, legal risk, and strategy.

A single ReAct loop tends to send every query through the same heavy path, causing slow and unnecessary retrieval for simple questions. LangGraph gives explicit routing, observability, and debuggability.

Runtime graph:

```text
interpret -> route
  simple: simple_retrieve -> simple_answer
  clarify: clarify
  deep: plan -> expand -> support -> adverse -> compensation
        -> select -> analyze -> synthesize -> validate
```

### Retrieval strategy

I used **hybrid retrieval: dense + BM25 + metadata + reranking**.

- **Dense retrieval** with Pinecone `multilingual-e5-large` finds semantic matches and paraphrases.
- **BM25** catches exact legal words and phrases such as “fake licence”, “pay and recover”, “Section 167”, and vehicle terms.
- **Metadata filters/tags** are applied where they help rather than uniformly: issue and vehicle tags boost the main hybrid path, stance and legal area drive the dedicated adverse retrieval, and procedural stage / legal area filter the simple enumeration path.
- **RRF fusion** combines dense and BM25 rankings without trying to normalize incompatible scores.
- **Pinecone `bge-reranker-v2-m3`** reranks the fused candidate pool for final relevance.

This was chosen because legal research needs both exact legal phrase matching and semantic recall.

### Chunking strategy

I used **parent-child chunking**:

- **Child chunks:** ~300-token target (max 360, with a 60-token overlap; observed mean ~313), used for retrieval.
- **Parent chunks:** ~1200-token paragraph-aware chunks, used for legal reasoning context.
- **Case cards:** one document-level summary record per judgment, used for broad lookup and metadata answers.

Final artifacts:

- **56 documents**
- **717 parent chunks**
- **3,647 child chunks**
- **56 case cards**

Reason: small child chunks improve precision, parent chunks preserve judgment reasoning, and case cards make document-level questions reliable.

---

## 2. Trade-offs made

| Decision | Benefit | Trade-off |
|---|---|---|
| LangGraph instead of single ReAct chain | Control, routing, visible trace, easier debugging | More moving parts |
| Hybrid retrieval instead of only vector search | Better legal recall for exact terms and paraphrases | More retrieval complexity |
| In-memory BM25 | Simple and fast for 56 docs | Not suitable for 5,000+ docs |
| Build-time enrichment | Faster runtime and stable metadata | Corpus changes require rebuild |
| Parent-child chunking | Precision + legal context | More artifacts to manage |
| Doc-level adverse stance | Better adverse recall | Slight answer-precision drop because more adverse docs are surfaced |
| Gemini Flash runtime model | Lower latency/cost | Weaker than a larger reasoning model for hard synthesis |

---

## 3. How the agent decides simple vs deep

The **interpreter node** calls the LLM and returns structured JSON:

```json
{
  "intent": "simple_lookup | deep_precedent_research | comparison | no_answer_check | other",
  "complexity": "simple | deep",
  "client_side": "claimant | insurer | owner_driver | unknown",
  "legal_area": "motor_accident | criminal | trademark_ip | ...",
  "issues": [],
  "facts": {},
  "requires_adverse": true,
  "requires_strategy": true,
  "requires_compensation": false,
  "requires_clarification": false
}
```

(These are the raw LLM output keys. The interpreter node renames several before writing them to graph state — e.g. `intent` → `detected_intent`, `complexity` → `query_complexity`, `issues` → `issue_map`, `facts` → `extracted_facts`.)

Routing rule:

- **Simple**: factual/listing/metadata queries.
- **Deep**: supporting/adverse precedents, strategy, multi-issue comparison, litigation advice, or compensation analysis.
- **Clarify**: query is too vague to act on.

This is semantic routing, not a hard-coded branch for the Lakshmi Devi brief.

---

## 4. What changes for 5,000 documents

I would keep the same logical architecture, but change the scale layer:

1. Move BM25 from in-memory to **Pinecone sparse index / Elasticsearch / OpenSearch**.
2. Add stronger metadata pre-filters: court, year, legal area, issue, party role, appellant/respondent role, outcome, vehicle type.
3. Build a proper **DAG ingestion pipeline** with incremental updates, retries, validation, and versioned enrichment.
4. Use a legal-domain taxonomy designed/reviewed by a lawyer so metadata remains consistent across a larger corpus.
5. Rerank fewer but better candidates after hard filters.
6. Cache embeddings, query expansions, router outputs, and reranker results.
7. Build a lawyer-reviewed, stratified, diverse golden set instead of relying mainly on small-corpus pooling.

---

## 5. What I would change with another week

1. Add a **facet checklist before final synthesis** so multi-part questions are answered more completely.
2. Improve **support recall** through better query expansion and issue-specific retrieval passes.
3. Improve adverse recall further using better stance labels and document-level adverse pools.
4. Add **appellant-role metadata**, because queries like “cases where the insurer appealed” are currently weak.
5. Try a faster/stronger reranker such as a **late-interaction ColBERT-style reranker** to rerank more chunks within the same latency.
6. Use **multi-sample LLM-as-judge** and/or a different judge model to reduce evaluation noise.
7. Test a stronger reasoning model for the final legal synthesis step.
8. Add a deterministic compensation calculator for multiplier/future-prospects calculations.

---

## 6. Evaluation report

The system was evaluated on a **25-query golden set** covering factual metadata, single-issue retrieval, procedural queries, statutory queries, no-answer cases, comparisons, ambiguous questions, multi-hop questions, and cross-document strategy questions.

### Results

| Dimension | Metric | Result |
|---|---:|---:|
| Precision | Precision@5 | **0.591** |
| Precision | Precision@10 | **0.481** |
| Precision | Answer precision | **0.405** |
| Recall | Recall@10 | **0.662** |
| Recall | Recall@20 | **0.770** |
| Recall | MRR | **0.807** |
| Recall | nDCG@10 | **0.714** |
| Recall | Support recall | **0.446** |
| Reasoning | Faithfulness | **3.88 / 5** |
| Reasoning | Legal reasoning | **3.40 / 5** |
| Reasoning | Completeness | **2.88 / 5** |
| Reasoning | Issue coverage | **0.957** |
| Safety | No invented docs | **1.0** |
| Safety | Must-not-claim respected | **1.0** |
| Adverse | Adverse recall | **0.562** |
| Adverse | Adverse reasoning | **3.72 / 5** |
| Adverse | Adverse search executed for deep queries | **0.889** |
| Adverse | Adverse section present | **0.889** |
| Agent behaviour | Router accuracy | **0.92** |
| Agent behaviour | No-answer handled | **1.0** |

Important v2 improvement: **adverse recall improved from 0.342 to 0.562** after adding stance-aware adverse retrieval.

---

## 7. Agent flow

### Offline ingestion

1. Parse PDFs.
2. Clean extracted text.
3. Split text into paragraphs.
4. Build parent chunks.
5. Build child retrieval chunks.
6. Generate case cards and document metadata.
7. Label chunks with issue, vehicle, stance, and chunk-type tags.
8. Upsert child chunks and case cards to Pinecone.
9. Store parent chunks locally for context expansion.

### Runtime

1. User enters query.
2. Interpreter extracts intent, facts, issues, legal area, client side, and complexity.
3. Router chooses simple, deep, or clarify.
4. Simple path uses metadata enumeration, case cards, and supporting chunks.
5. Deep path creates a research plan.
6. Query expansion creates support, adverse, factual, statutory, and compensation queries.
7. Hybrid retrieval runs for support and adverse evidence.
8. Dedicated adverse retrieval searches document-level adverse pools.
9. Candidates are fused, reranked, and selected.
10. Parent context is attached.
11. Case analyzer summarizes each selected judgment.
12. Strategy synthesizer writes the final answer.
13. Validator checks grounding and avoids invented citations/documents.

---

## 8. How chunks are scored/ranked

Each child chunk receives:

1. **Dense score** from Pinecone vector retrieval.
2. **BM25 score** from lexical retrieval.
3. **Dense rank** and **lexical rank**.
4. **RRF score**:

```text
RRF = sum(1 / (60 + rank + 1))
```

5. **Metadata boost** for matching issue/vehicle/stance tags.
6. **Fused score**:

```text
fused_score = rrf_score + metadata_boost
```

Then the top fused candidates are sent to the **bge reranker**, which assigns the final rerank score. Evidence selection then buckets chunks into supporting, adverse, and neutral evidence, with limits per document and parent-context expansion.

---

## 9. Evaluation system

The evaluation system has three layers:

1. **Golden set:** 25 labeled queries with expected relevant document IDs and adverse/supporting expectations.
2. **Retrieval metrics:** Precision@K, Recall@K, MRR, nDCG@10, support recall, adverse recall.
3. **Answer-quality checks:** LLM judge scores for faithfulness/legal reasoning/completeness, plus deterministic checks for issue coverage, no invented documents, no-answer handling, adverse section presence, and must-not-claim compliance.

I separated retrieval metrics from answer metrics because a system can retrieve the right documents but synthesize poorly, or write a plausible answer while missing key cases.

---

## 10. Current failures and first fixes

Main failures:

- **Completeness is low** at 2.88/5, so multi-part answers sometimes miss requested facets.
- **Support recall is low** at 0.446.
- **Adverse recall improved but is still incomplete** at 0.562.
- **Router missed 2 of 25 cases**.
- **Appellant-role queries fail** because the system does not yet track who appealed.
- **LLM judge noise** exists because the judge is single-sample and similar to the runtime model.
- Some stance labels are borderline and need domain expert audit.

First fixes:

1. Add a final-answer **facet checklist** to improve completeness.
2. Add appellant/respondent role metadata.
3. Improve support recall with better query expansion and more diverse retrieval passes.
4. Add multi-sample / second-model judging.

---

## 11. Alternatives considered

| Decision | Alternatives considered | Final choice and reason |
|---|---|---|
| Agent orchestration | Single chain, ReAct loop, CrewAI | LangGraph: explicit routing, traceability, no drag-and-drop builder |
| Routing | Keyword rules | LLM JSON interpreter: handles semantic intent and unseen prompts |
| Retrieval | Pure dense search | Rejected because legal keywords and statutes need exact matching |
| Retrieval | Pure BM25 | Rejected because legal issues are often phrased semantically |
| Fusion | Weighted score merge | RRF chosen because dense and BM25 scores are not comparable |
| Lexical index | External sparse index | In-memory BM25 chosen because corpus has only 56 docs |
| Reranking | No reranker | bge reranker chosen to improve final candidate ordering |
| Chunking | Whole-document chunks | Rejected because too noisy and too large for embeddings |
| Chunking | Small chunks only | Rejected because legal reasoning context gets lost |
| Chunking | Parent chunks only | Rejected because retrieval precision drops |
| Final chunking | Child + parent + case card | Best balance of precision, context, and document-level lookup |
| Enrichment | Runtime extraction | Build-time enrichment chosen for speed and consistency |
| Adverse search | Prompt-only “find adverse cases” | Dedicated stance-aware adverse retrieval chosen because adverse cases were crowded out |
| Eval | Generic RAG eval only | Custom legal eval chosen to measure adverse identification, reasoning, and no-answer behaviour |
| Judge | Single deterministic checks only | Mixed deterministic + LLM judge chosen because reasoning quality needs qualitative scoring |

---

## Conclusion

The architecture is optimized for legal precedent research, not generic Q&A. The main design principle is: use deterministic structure where possible, use hybrid retrieval for legal recall, use LLMs for interpretation and synthesis, and expose the retrieval/ranking trace so the system can be evaluated and debugged.
