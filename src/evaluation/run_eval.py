"""Run the agent over the golden set and report the 4 required dimensions
(Precision, Recall, Reasoning Quality, Adverse Identification) + agent behavior.

Run:  uv run python -m src.evaluation.run_eval [limit]
  optional `limit` runs only the first N queries (quick smoke).
Outputs: reports/eval_results_v1.json, reports/eval_results_v1.md
"""

from __future__ import annotations

import json
import sys
import traceback
from collections import Counter

from src.agent.graph import run_agent
from src.evaluation.agent_metrics import (
    adverse_search_executed,
    agent_adverse_docs,
    agent_presented_docs,
    agent_ranked_docs,
    agent_supporting_docs,
    router_correct,
)
from src.evaluation.answer_metrics import deterministic_answer_checks, judge_answer
from src.evaluation.golden_schema import GoldenQuery
from src.evaluation.retrieval_metrics import (
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    set_precision,
    set_recall,
)
from src.ingestion.config import ROOT

import os

GOLDEN_PATH = ROOT / "data" / "golden" / "golden_set_v1.json"
REPORTS = ROOT / "reports"
# EVAL_VERSION lets re-runs write a fresh report without clobbering an earlier one.
VERSION = os.environ.get("EVAL_VERSION", "v2")
JSON_OUT = REPORTS / f"eval_results_{VERSION}.json"
MD_OUT = REPORTS / f"eval_results_{VERSION}.md"


def _mean(vals):
    v = [x for x in vals if x is not None]
    return round(sum(v) / len(v), 3) if v else None


def eval_query(q: GoldenQuery) -> dict:
    state = run_agent(q.query)
    gold = set(q.expected_relevant_doc_ids)
    ranked = agent_ranked_docs(state)
    presented = agent_presented_docs(state)
    judge = judge_answer(q, state)
    det = deterministic_answer_checks(q, state, set(ranked))

    return {
        "query_id": q.query_id, "query_type": q.query_type,
        "required_workflow": q.required_workflow, "difficulty": q.difficulty,
        "no_answer": q.no_answer,
        "expected_deep": q.expects_deep(), "actual_complexity": state.get("query_complexity"),
        "router_correct": router_correct(q, state),
        "adverse_search_executed": adverse_search_executed(state),
        "n_gold": len(gold), "n_retrieved": len(ranked), "n_presented": len(presented),
        # retrieval (skip rank metrics for no-answer: gold is empty)
        "precision_at_5": None if q.no_answer else round(precision_at_k(ranked, gold, 5), 3),
        "precision_at_10": None if q.no_answer else round(precision_at_k(ranked, gold, 10), 3),
        "recall_at_10": None if q.no_answer else round(recall_at_k(ranked, gold, 10), 3),
        "recall_at_20": None if q.no_answer else round(recall_at_k(ranked, gold, 20), 3),
        "mrr": None if q.no_answer else round(mrr(ranked, gold), 3),
        "ndcg_at_10": None if q.no_answer else round(ndcg_at_k(ranked, gold, 10), 3),
        "answer_precision": None if q.no_answer else round(set_precision(presented, gold), 3),
        # adverse / support (eval dim 4)
        "adverse_recall": (round(set_recall(agent_adverse_docs(state), set(q.adverse_doc_ids)), 3)
                           if q.adverse_doc_ids else None),
        "support_recall": (round(set_recall(agent_supporting_docs(state), set(q.supporting_doc_ids)), 3)
                           if q.supporting_doc_ids else None),
        "deterministic": det,
        "judge": judge,
        "final_answer_preview": (state.get("final_answer", "") or "")[:300],
    }


def aggregate(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    deep = [r for r in ok if r["expected_deep"]]
    judged = [r["judge"] for r in ok]
    det = [r["deterministic"] for r in ok]
    na = [r for r in ok if r["no_answer"]]

    return {
        "n_queries": len(rows), "n_errors": len(rows) - len(ok),
        "precision": {
            "precision_at_5": _mean([r["precision_at_5"] for r in ok]),
            "precision_at_10": _mean([r["precision_at_10"] for r in ok]),
            "answer_precision": _mean([r["answer_precision"] for r in ok]),
        },
        "recall": {
            "recall_at_10": _mean([r["recall_at_10"] for r in ok]),
            "recall_at_20": _mean([r["recall_at_20"] for r in ok]),
            "mrr": _mean([r["mrr"] for r in ok]),
            "ndcg_at_10": _mean([r["ndcg_at_10"] for r in ok]),
            "support_recall": _mean([r["support_recall"] for r in ok]),
        },
        "reasoning_quality": {
            "faithfulness": _mean([j.get("faithfulness") for j in judged]),
            "legal_reasoning": _mean([j.get("legal_reasoning") for j in judged]),
            "completeness": _mean([j.get("completeness") for j in judged]),
            "must_not_claim_respected_rate": _mean([1.0 if j.get("must_not_claim_respected") else 0.0 for j in judged]),
            "issue_coverage": _mean([d.get("issue_coverage") for d in det]),
            "no_invented_docs_rate": _mean([0.0 if d["invented_nonexistent_docs"] else 1.0 for d in det]),
        },
        "adverse_identification": {
            "adverse_recall": _mean([r["adverse_recall"] for r in ok]),
            "adverse_reasoning_score": _mean([j.get("adverse_reasoning") for j in judged]),
            "adverse_search_executed_rate": _mean([1.0 if r["adverse_search_executed"] else 0.0 for r in deep]),
            "adverse_section_present_rate": _mean([1.0 if d.get("adverse_section_present") else 0.0
                                                   for d in det if d.get("adverse_section_present") is not None]),
        },
        "agent_behavior": {
            "router_accuracy": _mean([1.0 if r["router_correct"] else 0.0 for r in ok]),
            "no_answer_handled_rate": _mean([1.0 if r["deterministic"].get("no_answer_handled") else 0.0 for r in na]),
        },
        "by_difficulty_recall10": {
            d: _mean([r["recall_at_10"] for r in ok if r["difficulty"] == d])
            for d in ("easy", "medium", "hard")
        },
    }


def _md(agg: dict, rows: list[dict]) -> str:
    p, r, rq, ai, ab = (agg["precision"], agg["recall"], agg["reasoning_quality"],
                        agg["adverse_identification"], agg["agent_behavior"])
    lines = [
        "# Lexi Agent — Evaluation Results (v1)",
        "",
        f"Golden set: {agg['n_queries']} queries ({agg['n_errors']} errored). Agent: LangGraph + "
        "Pinecone hybrid (dense e5 + BM25 + bge rerank) + Gemini. Metrics are document-level; "
        "recall is measured against the grounded golden set (see data/golden/golden_set_readme.md).",
        "",
        "## Dimension 1 — Precision",
        f"- Precision@5: **{p['precision_at_5']}** · Precision@10: {p['precision_at_10']}",
        f"- Answer precision (presented docs that are relevant): **{p['answer_precision']}**",
        "",
        "## Dimension 2 — Recall",
        f"- Recall@10: **{r['recall_at_10']}** · Recall@20: {r['recall_at_20']}",
        f"- MRR: {r['mrr']} · nDCG@10: {r['ndcg_at_10']} · Support recall: {r['support_recall']}",
        f"- Recall@10 by difficulty: {agg['by_difficulty_recall10']}",
        "",
        "## Dimension 3 — Reasoning Quality (1–5 judge + objective checks)",
        f"- Faithfulness: **{rq['faithfulness']}** · Legal reasoning: {rq['legal_reasoning']} · Completeness: {rq['completeness']}",
        f"- Issue coverage: {rq['issue_coverage']} · must-not-claim respected: {rq['must_not_claim_respected_rate']} · no invented docs: {rq['no_invented_docs_rate']}",
        "",
        "## Dimension 4 — Adverse Identification",
        f"- Adverse recall: **{ai['adverse_recall']}** · Adverse-reasoning score: {ai['adverse_reasoning_score']}",
        f"- Adverse search executed (deep): {ai['adverse_search_executed_rate']} · Adverse section present: {ai['adverse_section_present_rate']}",
        "",
        "## Agent behavior",
        f"- Router accuracy (simple vs deep): **{ab['router_accuracy']}** · No-answer handled: {ab['no_answer_handled_rate']}",
        "",
        "## Per-query results",
        "",
        "| query_id | type | route ok | P@5 | R@10 | adv_recall | faith | preview |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if "error" in row:
            lines.append(f"| {row['query_id']} | {row.get('query_type','?')} | ERROR | | | | | {row['error'][:40]} |")
            continue
        j = row["judge"]
        lines.append(
            f"| {row['query_id']} | {row['query_type']} | {'yes' if row['router_correct'] else 'no'} "
            f"| {row['precision_at_5']} | {row['recall_at_10']} | {row['adverse_recall']} "
            f"| {j.get('faithfulness')} | {row['final_answer_preview'][:50].replace(chr(10),' ')} |")
    return "\n".join(lines)


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    queries = [GoldenQuery(**q) for q in json.loads(GOLDEN_PATH.read_text())["queries"]]
    if limit:
        queries = queries[:limit]

    REPORTS.mkdir(exist_ok=True)
    rows: list[dict] = []
    for i, q in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {q.query_id} ({q.query_type}) …", flush=True)
        try:
            rows.append(eval_query(q))
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            rows.append({"query_id": q.query_id, "query_type": q.query_type, "error": f"{type(exc).__name__}: {exc}"})

    agg = aggregate(rows)
    JSON_OUT.write_text(json.dumps({"aggregate": agg, "per_query": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    MD_OUT.write_text(_md(agg, rows), encoding="utf-8")

    print("\n=== AGGREGATE ===")
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    print(f"\nWrote {JSON_OUT.name} + {MD_OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
