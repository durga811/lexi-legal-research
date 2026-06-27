"""Offline regression tests for the frozen golden set."""

import json

from src.evaluation.golden_schema import (
    DIFFICULTIES,
    QUERY_TYPES,
    WORKFLOWS,
    GoldenQuery,
)
from src.ingestion.config import N_DOCS, PROCESSED
from src.utils.ids import doc_id

GOLDEN_PATH = PROCESSED.parent / "golden" / "golden_set_v1.json"
VALID_DOCS = {doc_id(n) for n in range(1, N_DOCS + 1)}


def _load():
    if not GOLDEN_PATH.exists():
        return None
    return json.loads(GOLDEN_PATH.read_text())["queries"]


def test_golden_schema_and_labels_valid():
    queries = _load()
    if queries is None:
        return
    ids = set()
    for raw in queries:
        q = GoldenQuery(**raw)  # pydantic-validates
        assert q.query_id not in ids, "duplicate query_id"
        ids.add(q.query_id)
        assert q.query_type in QUERY_TYPES
        assert q.required_workflow in WORKFLOWS
        assert q.difficulty in DIFFICULTIES
        for field in ("expected_relevant_doc_ids", "supporting_doc_ids",
                      "adverse_doc_ids", "neutral_doc_ids"):
            assert set(getattr(q, field)) <= VALID_DOCS, f"{q.query_id} {field} invalid doc"
        if q.no_answer:
            assert q.expected_relevant_doc_ids == []
        else:
            assert q.expected_relevant_doc_ids


def test_golden_has_coverage_and_adverse():
    queries = _load()
    if queries is None:
        return
    qs = [GoldenQuery(**r) for r in queries]
    assert len(qs) >= 20
    assert sum(1 for q in qs if q.expects_deep()) >= 5
    assert sum(1 for q in qs if q.adverse_doc_ids) >= 4   # adverse identification coverage
    assert sum(1 for q in qs if q.no_answer) >= 1          # no-answer behaviour
    covered = {d for q in qs for d in q.expected_relevant_doc_ids}
    assert len(covered) >= 40                              # broad corpus coverage
