"""In-memory corpus store for retrieval (lightweight; ~13MB of JSONL).

Loaded once and cached. Provides:
  - merged child-chunk records (structure + grounded labels) for candidate building,
  - a BM25 index over child-chunk text for the lexical signal,
  - parent lookup for parent-expansion (§7.5),
  - document metadata + case cards for doc-level context and the simple-query path.

The runtime app builds this at startup; no embedding model is loaded here.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from src.ingestion.config import CHILD_CHUNKS_PATH, PROCESSED

CASE_CARDS_PATH = PROCESSED / "case_cards.jsonl"
DOC_METADATA_PATH = PROCESSED / "document_metadata.jsonl"
PARENT_CHUNKS_PATH = PROCESSED / "parent_chunks.jsonl"
CHUNK_LABELS_PATH = PROCESSED / "chunk_labels.jsonl"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _read(path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@lru_cache(maxsize=1)
def load_corpus() -> dict:
    children = _read(CHILD_CHUNKS_PATH)
    labels = {r["record_id"]: r for r in _read(CHUNK_LABELS_PATH)}
    parents = {r["record_id"]: r for r in _read(PARENT_CHUNKS_PATH)}
    docmeta = {r["doc_id"]: r for r in _read(DOC_METADATA_PATH)}
    cards = {r["doc_id"]: r for r in _read(CASE_CARDS_PATH)}

    child_by_id: dict[str, dict] = {}
    for c in children:
        lab = labels.get(c["record_id"], {})
        child_by_id[c["record_id"]] = {
            "record_id": c["record_id"],
            "doc_id": c["doc_id"],
            "parent_id": c["parent_id"],
            "text": c["text"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "legal_area": lab.get("legal_area"),
            "case_title": lab.get("case_title"),
            "court": lab.get("court"),
            "year": lab.get("year"),
            "chunk_type": lab.get("chunk_type"),
            "issue_tags": lab.get("issue_tags", []),
            "stance_tags": lab.get("stance_tags", []),
            "vehicle_tags": lab.get("vehicle_tags", []),
        }
    return {
        "children": children,
        "child_ids": [c["record_id"] for c in children],
        "child_by_id": child_by_id,
        "parents": parents,
        "docmeta": docmeta,
        "cards": cards,
    }


@lru_cache(maxsize=1)
def get_bm25():
    """BM25 index over child-chunk text. Returns (bm25, ids_in_order)."""
    from rank_bm25 import BM25Okapi

    corpus = load_corpus()
    ids = corpus["child_ids"]
    tokenized = [tokenize(corpus["child_by_id"][i]["text"]) for i in ids]
    return BM25Okapi(tokenized), ids


def get_child(record_id: str) -> dict | None:
    return load_corpus()["child_by_id"].get(record_id)


def get_parent(parent_id: str) -> dict | None:
    return load_corpus()["parents"].get(parent_id)


def get_doc_meta(doc_id: str) -> dict | None:
    return load_corpus()["docmeta"].get(doc_id)
