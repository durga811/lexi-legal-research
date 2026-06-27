"""Deterministic, chunk-grounded labeling (Step 2, second half). No LLM.

Assigns each child chunk a chunk_type plus issue / vehicle / stance tags, where
every tag is justified by the chunk's OWN text (§5.7 "each chunk label must be
justified by that chunk's own text"):

  - chunk_type   : first matching rule from CHUNK_TYPE_CUES (else "general").
  - issue_tags   : doc-level issue_tags whose lexicon terms appear in the chunk
                   (intersection keeps it grounded to both the case and the chunk).
  - vehicle_tags : doc-level vehicle_tags that appear in the chunk.
  - stance_tags  : derived directly from chunk text via STANCE_CUES (a chunk can
                   carry adverse reasoning even inside a claimant-supporting case).

Output: data/processed/chunk_labels.jsonl, keyed by child record_id, merged into
Pinecone records at upsert time (Step 3).

Run:  uv run python -m src.ingestion.chunk_labeler
"""

from __future__ import annotations

import json
import re
from collections import Counter

from src.ingestion.config import CHILD_CHUNKS_PATH, PROCESSED
from src.ingestion.vocab import (
    CHUNK_TYPE_CUES,
    ISSUE_LEXICON,
    STANCE_CUES,
    VEHICLE_LEXICON,
)

DOC_METADATA_PATH = PROCESSED / "document_metadata.jsonl"
CHUNK_LABELS_PATH = PROCESSED / "chunk_labels.jsonl"
CHUNK_LABELS_REPORT = PROCESSED / "chunk_labels_report.json"


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower())


def _classify_type(text: str) -> str:
    for ctype, cues in CHUNK_TYPE_CUES:
        if any(cue in text for cue in cues):
            return ctype
    return "general"


def _grounded_issue_tags(text: str, doc_issue_tags: list[str]) -> list[str]:
    out = []
    for tag in doc_issue_tags:
        terms = ISSUE_LEXICON.get(tag)
        if terms and any(term in text for term in terms):
            out.append(tag)
    return out


def _grounded_vehicle_tags(text: str, doc_vehicle_tags: list[str]) -> list[str]:
    out = []
    for tag in doc_vehicle_tags:
        terms = VEHICLE_LEXICON.get(tag, [tag])
        if any(term in text for term in terms):
            out.append(tag)
    return out


def _chunk_stance(text: str) -> list[str]:
    return [stance for stance, cues in STANCE_CUES.items() if any(c in text for c in cues)]


def main() -> int:
    meta = {}
    with DOC_METADATA_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            meta[r["doc_id"]] = r

    children = [json.loads(l) for l in CHILD_CHUNKS_PATH.open(encoding="utf-8")]

    type_counts: Counter[str] = Counter()
    n_issue = n_vehicle = n_stance = 0
    missing_meta = []

    with CHUNK_LABELS_PATH.open("w", encoding="utf-8") as out:
        for c in children:
            m = meta.get(c["doc_id"])
            if m is None:
                missing_meta.append(c["record_id"])
                continue
            text = _norm(c["text"])
            ctype = _classify_type(text)
            itags = _grounded_issue_tags(text, m.get("issue_tags", []))
            vtags = _grounded_vehicle_tags(text, m.get("vehicle_tags", []))
            stags = _chunk_stance(text)

            type_counts[ctype] += 1
            n_issue += bool(itags)
            n_vehicle += bool(vtags)
            n_stance += bool(stags)

            out.write(json.dumps({
                "record_id": c["record_id"],
                "doc_id": c["doc_id"],
                "parent_id": c["parent_id"],
                "legal_area": m.get("legal_area"),
                "case_title": m.get("case_title"),
                "court": m.get("court"),
                "year": m.get("year"),
                "chunk_type": ctype,
                "issue_tags": itags,
                "vehicle_tags": vtags,
                "stance_tags": stags,
            }, ensure_ascii=False) + "\n")

    n = len(children)
    report = {
        "n_children": n,
        "n_labeled": n - len(missing_meta),
        "missing_meta": missing_meta,
        "chunk_type_dist": dict(type_counts.most_common()),
        "pct_with_issue_tag": round(100 * n_issue / n, 1) if n else 0,
        "pct_with_vehicle_tag": round(100 * n_vehicle / n, 1) if n else 0,
        "pct_with_stance_tag": round(100 * n_stance / n, 1) if n else 0,
    }
    CHUNK_LABELS_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Labeled {report['n_labeled']}/{n} child chunks")
    print(f"chunk_type dist : {report['chunk_type_dist']}")
    print(f"with issue_tag  : {report['pct_with_issue_tag']}%  |  vehicle_tag: {report['pct_with_vehicle_tag']}%  |  stance: {report['pct_with_stance_tag']}%")
    if missing_meta:
        print(f"WARNING: {len(missing_meta)} chunks missing doc metadata")
        return 1
    print("chunk_labels.jsonl written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
