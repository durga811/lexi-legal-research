"""Artifact tests for Step 2 (build-time enrichment + chunk labeling)."""

import json
from pathlib import Path

from src.ingestion.config import N_DOCS, PROCESSED
from src.ingestion.vocab import (
    ISSUE_TAGS,
    LEGAL_AREAS,
    STANCE_TAGS,
    VEHICLE_TAGS,
)
from src.utils.ids import doc_id

VALID_IDS = {doc_id(n) for n in range(1, N_DOCS + 1)}


def _read_jsonl(path: Path):
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def test_enrichment_report_passed():
    rp = PROCESSED / "enrichment_report.json"
    if not rp.exists():
        return
    report = json.loads(rp.read_text())
    assert report["coverage_ok"] is True
    assert report["failed_docs"] == []
    assert report["n_enriched"] == N_DOCS


def test_case_cards_cover_corpus():
    path = PROCESSED / "case_cards.jsonl"
    if not path.exists():
        return
    cards = _read_jsonl(path)
    ids = {c["doc_id"] for c in cards}
    assert ids == VALID_IDS
    for c in cards:
        assert c["record_type"] == "case_card"
        assert c["record_id"] == f"{c['doc_id']}_case_card"
        assert c["text"].strip()
        assert c["metadata"]["legal_area"] in LEGAL_AREAS


def test_document_metadata_vocab_and_consistency():
    path = PROCESSED / "document_metadata.jsonl"
    if not path.exists():
        return
    meta = _read_jsonl(path)
    assert {m["doc_id"] for m in meta} == VALID_IDS
    for m in meta:
        assert m["legal_area"] in LEGAL_AREAS
        assert set(m["issue_tags"]) <= ISSUE_TAGS
        assert set(m["stance_tags"]) <= STANCE_TAGS
        assert set(m["vehicle_tags"]) <= VEHICLE_TAGS
        # motor consistency: non-motor => no vehicle tags
        if not m["is_motor_accident"]:
            assert m["vehicle_tags"] == []


def test_chunk_labels_align_with_children():
    labels_path = PROCESSED / "chunk_labels.jsonl"
    children_path = PROCESSED / "child_chunks.jsonl"
    if not (labels_path.exists() and children_path.exists()):
        return
    labels = _read_jsonl(labels_path)
    child_ids = {json.loads(l)["record_id"] for l in children_path.open(encoding="utf-8")}
    label_ids = {x["record_id"] for x in labels}
    assert label_ids == child_ids  # exactly one label per child
    for x in labels:
        assert set(x["issue_tags"]) <= ISSUE_TAGS
        assert set(x["vehicle_tags"]) <= VEHICLE_TAGS
        assert set(x["stance_tags"]) <= STANCE_TAGS
        assert x["doc_id"] in VALID_IDS
