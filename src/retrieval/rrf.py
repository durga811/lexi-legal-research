"""Reciprocal Rank Fusion (§8.4).

RRF score for an item = Σ_lists 1 / (k + rank), rank 0-indexed. Combining the
dense and lexical rankings this way stops either signal from dominating and is
robust to the two scores being on different scales.
"""

from __future__ import annotations

from collections import defaultdict


def rrf_fuse(ranked_id_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """Map each id to its fused RRF score across the provided ranked lists."""
    scores: dict[str, float] = defaultdict(float)
    for ids in ranked_id_lists:
        for rank, rid in enumerate(ids):
            scores[rid] += 1.0 / (k + rank + 1)
    return dict(scores)
