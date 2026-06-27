"""Retrieval metrics (§12.1). Pure functions over a ranked list of doc_ids and a
gold set. Document-level (precedents = documents)."""

from __future__ import annotations

import math


def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if k <= 0 or not ranked:
        return 0.0
    topk = ranked[:k]
    return sum(1 for d in topk if d in gold) / len(topk)


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 1.0  # nothing to recall (e.g. no-answer)
    topk = set(ranked[:k])
    return len(topk & gold) / len(gold)


def mrr(ranked: list[str], gold: set[str]) -> float:
    for i, d in enumerate(ranked):
        if d in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 1.0
    dcg = sum((1.0 / math.log2(i + 2)) for i, d in enumerate(ranked[:k]) if d in gold)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def set_recall(found: set[str], gold: set[str]) -> float:
    """Recall of a labelled subset (e.g. adverse / supporting docs)."""
    if not gold:
        return 1.0
    return len(found & gold) / len(gold)


def set_precision(found: set[str], gold: set[str]) -> float:
    if not found:
        return 1.0 if not gold else 0.0
    return len(found & gold) / len(found)
