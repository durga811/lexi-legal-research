"""Assemble + verify + freeze the golden set (§11.4, §11.8).

Combines deterministic queries (exact metadata-derived labels) with the
sub-agent-designed semantic/strategy queries, then VERIFIES every label:
  - schema (pydantic) + controlled vocab,
  - every referenced doc_id exists; no-answer queries have empty expected,
  - retrieval pooling: run the real hybrid + case-card retrieval per query and
    report pool_recall (are labeled docs retrievable?) and pooling_suggestions
    (docs the retriever surfaces that are NOT labeled — candidates to add).

Run:  uv run python -m src.evaluation.build_golden
Outputs: data/golden/golden_set_v1.json, data/golden/golden_verification.json
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from src.evaluation.golden_deterministic import deterministic_queries
from src.evaluation.golden_schema import (
    DIFFICULTIES,
    QUERY_TYPES,
    WORKFLOWS,
    GoldenQuery,
)
from src.ingestion.config import N_DOCS, PROCESSED
from src.retrieval.hybrid_retriever import case_card_search, hybrid_search
from src.utils.ids import doc_id

GOLDEN_DIR = PROCESSED.parent / "golden"
DESIGNER_PATH = GOLDEN_DIR / "_designer_queries.json"
GOLDEN_PATH = GOLDEN_DIR / "golden_set_v1.json"
VERIFY_PATH = GOLDEN_DIR / "golden_verification.json"

VALID_DOCS = {doc_id(n) for n in range(1, N_DOCS + 1)}
POOL_CHILD_K = 40
POOL_CARD_K = 25


def _load_designer() -> list[GoldenQuery]:
    raw = json.loads(DESIGNER_PATH.read_text(encoding="utf-8"))
    out = []
    for i, q in enumerate(raw["queries"], 1):
        q.setdefault("query_id", f"g_sub_{i:02d}")
        q["label_source"] = "subagent"
        out.append(GoldenQuery(**q))
    return out


def _static_checks(q: GoldenQuery) -> list[str]:
    errs = []
    if q.query_type not in QUERY_TYPES:
        errs.append(f"bad query_type {q.query_type}")
    if q.required_workflow not in WORKFLOWS:
        errs.append(f"bad required_workflow {q.required_workflow}")
    if q.difficulty not in DIFFICULTIES:
        errs.append(f"bad difficulty {q.difficulty}")
    for field in ("expected_relevant_doc_ids", "supporting_doc_ids", "adverse_doc_ids", "neutral_doc_ids"):
        for d in getattr(q, field):
            if d not in VALID_DOCS:
                errs.append(f"{field} references non-existent {d}")
    if q.no_answer and q.expected_relevant_doc_ids:
        errs.append("no_answer query has non-empty expected docs")
    if not q.no_answer and not q.expected_relevant_doc_ids:
        errs.append("non-no_answer query has empty expected docs")
    return errs


def _pool(query: str) -> list[str]:
    """Ranked distinct doc_ids from hybrid child + case-card retrieval."""
    seen: list[str] = []
    for c in hybrid_search(query, top_k=POOL_CHILD_K):
        if c["doc_id"] not in seen:
            seen.append(c["doc_id"])
    for c in case_card_search(query, top_k=POOL_CARD_K):
        if c["doc_id"] not in seen:
            seen.append(c["doc_id"])
    return seen


def main() -> int:
    queries = deterministic_queries()
    try:
        queries += _load_designer()
    except (ValidationError, FileNotFoundError, KeyError) as e:
        print(f"FATAL loading designer queries: {e}")
        return 1

    # uniqueness of ids
    ids = [q.query_id for q in queries]
    if len(set(ids)) != len(ids):
        print("FATAL: duplicate query_ids")
        return 1

    hard_failures: list[str] = []
    per_query = []
    for q in queries:
        errs = _static_checks(q)
        pooled = _pool(q.query)
        exp = set(q.expected_relevant_doc_ids)
        pool_recall = round(len(exp & set(pooled)) / len(exp), 2) if exp else None
        suggestions = [d for d in pooled if d not in exp][:6]
        per_query.append({
            "query_id": q.query_id, "query_type": q.query_type,
            "required_workflow": q.required_workflow, "difficulty": q.difficulty,
            "n_expected": len(exp), "n_adverse": len(q.adverse_doc_ids),
            "pool_recall": pool_recall,
            "missing_from_pool": sorted(exp - set(pooled)),
            "pooling_suggestions": suggestions if not q.no_answer else [],
            "closest_for_no_answer": pooled[:5] if q.no_answer else [],
            "errors": errs,
        })
        if errs:
            hard_failures.extend(f"{q.query_id}: {e}" for e in errs)

    # coverage / distribution stats
    covered = sorted({d for q in queries for d in q.expected_relevant_doc_ids})
    from collections import Counter
    verification = {
        "n_queries": len(queries),
        "by_type": dict(Counter(q.query_type for q in queries)),
        "by_workflow": dict(Counter(q.required_workflow for q in queries)),
        "by_difficulty": dict(Counter(q.difficulty for q in queries)),
        "deep_queries": sum(1 for q in queries if q.expects_deep()),
        "n_with_adverse_labels": sum(1 for q in queries if q.adverse_doc_ids),
        "distinct_docs_covered": len(covered),
        "mean_pool_recall": round(
            sum(p["pool_recall"] for p in per_query if p["pool_recall"] is not None)
            / max(1, sum(1 for p in per_query if p["pool_recall"] is not None)), 3),
        "low_pool_recall_queries": [p["query_id"] for p in per_query
                                    if p["pool_recall"] is not None and p["pool_recall"] < 0.6],
        "hard_failures": hard_failures,
        "per_query": per_query,
    }
    VERIFY_PATH.write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")

    if hard_failures:
        print(f"VALIDATION FAILED ({len(hard_failures)}):")
        for f in hard_failures[:20]:
            print("  -", f)
        return 1

    GOLDEN_PATH.write_text(
        json.dumps({"version": "v1", "queries": [q.model_dump() for q in queries]},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Golden set: {verification['n_queries']} queries "
          f"({verification['deep_queries']} deep), {verification['distinct_docs_covered']}/{N_DOCS} docs covered.")
    print("by_type:", verification["by_type"])
    print("by_workflow:", verification["by_workflow"])
    print("by_difficulty:", verification["by_difficulty"])
    print("queries with adverse labels:", verification["n_with_adverse_labels"])
    print("mean pool_recall:", verification["mean_pool_recall"],
          "| low-recall queries:", verification["low_pool_recall_queries"])
    print(f"Frozen: {GOLDEN_PATH.name} + {VERIFY_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
