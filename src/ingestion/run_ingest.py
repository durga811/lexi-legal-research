"""Step 1 orchestrator: normalize -> parse -> clean -> chunk -> validate -> report.

Run with:  uv run python -m src.ingestion.run_ingest

Produces (all deterministic, no LLM):
  data/raw/DOC_001.pdf .. DOC_056.pdf            (normalized copies)
  data/processed/extracted/DOC_xxx.json|.txt     (cleaned per-doc text, for Step 2)
  data/processed/parent_chunks.jsonl
  data/processed/child_chunks.jsonl
  data/processed/ingest_report.json

Exits non-zero if any §13.2 ingestion guarantee fails.
"""

from __future__ import annotations

import json
import shutil
import statistics
import sys

from src.ingestion.chunker import chunk_document
from src.ingestion.clean_text import clean_document
from src.ingestion.config import (
    CHILD_CHUNKS_PATH,
    CHILD_HARD_CLAMP_TOKENS,
    EXTRACTED,
    INGEST_REPORT_PATH,
    MIN_DOC_CHARS,
    N_DOCS,
    PARENT_CHUNKS_PATH,
    RAW,
    RAW_SRC,
)
from src.ingestion.parse_pdfs import extract_doc
from src.utils.ids import doc_id


def _ensure_dirs() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    EXTRACTED.mkdir(parents=True, exist_ok=True)


def _normalize_pdfs() -> list[str]:
    """Copy raw-docs/doc_NNN.pdf -> data/raw/DOC_NNN.pdf. Returns missing sources."""
    missing: list[str] = []
    for n in range(1, N_DOCS + 1):
        src = RAW_SRC / f"doc_{n:03d}.pdf"
        dst = RAW / f"{doc_id(n)}.pdf"
        if not src.exists():
            missing.append(src.name)
            continue
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copyfile(src, dst)
    return missing


def _write_jsonl(path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _dist(values: list[int]) -> dict:
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "p95": 0}
    s = sorted(values)
    p95 = s[min(len(s) - 1, int(0.95 * len(s)))]
    return {"min": s[0], "max": s[-1], "mean": round(statistics.mean(s), 1), "p95": p95}


def main() -> int:
    _ensure_dirs()

    missing = _normalize_pdfs()
    if missing:
        print(f"FATAL: missing source PDFs: {missing}", file=sys.stderr)
        return 1

    all_parents: list[dict] = []
    all_children: list[dict] = []
    per_doc: list[dict] = []
    total_clamped = 0

    for n in range(1, N_DOCS + 1):
        doc = doc_id(n)
        pdf = RAW / f"{doc}.pdf"
        pages, parser_used = extract_doc(pdf)
        cleaned_pages, source_hint = clean_document(pages)
        full_text = "\n\n".join(p["text"] for p in cleaned_pages).strip()

        # Persist cleaned text for the Step 2 enrichment sub-agents.
        (EXTRACTED / f"{doc}.json").write_text(
            json.dumps(
                {
                    "doc_id": doc,
                    "file_name": pdf.name,
                    "n_pages": len(pages),
                    "char_len": len(full_text),
                    "parser_used": parser_used,
                    "source_hint": source_hint,
                    "pages": cleaned_pages,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (EXTRACTED / f"{doc}.txt").write_text(full_text, encoding="utf-8")

        parents, children, n_clamped = chunk_document(doc, cleaned_pages)
        total_clamped += n_clamped
        all_parents.extend(parents)
        all_children.extend(children)

        per_doc.append(
            {
                "doc_id": doc,
                "pages": len(pages),
                "char_len": len(full_text),
                "parser_used": parser_used,
                "n_parents": len(parents),
                "n_children": len(children),
                "max_child_tokens": max((c["token_estimate"] for c in children), default=0),
                "source_hint": source_hint,
            }
        )

    _write_jsonl(PARENT_CHUNKS_PATH, all_parents)
    _write_jsonl(CHILD_CHUNKS_PATH, all_children)

    # ---------------- Validation (§13.2 ingestion guarantees) ---------------- #
    failures: list[str] = []
    parent_ids = {p["record_id"] for p in all_parents}
    child_ids = {c["record_id"] for c in all_children}
    valid_doc_ids = {doc_id(n) for n in range(1, N_DOCS + 1)}

    if len(per_doc) != N_DOCS:
        failures.append(f"processed {len(per_doc)} docs, expected {N_DOCS}")
    if len(parent_ids) != len(all_parents):
        failures.append("duplicate parent record_ids found")
    if len(child_ids) != len(all_children):
        failures.append("duplicate child record_ids found")

    for d in per_doc:
        if d["char_len"] < MIN_DOC_CHARS:
            failures.append(f"{d['doc_id']} extracted text too short ({d['char_len']} chars)")
        if d["n_parents"] == 0:
            failures.append(f"{d['doc_id']} has no parent chunks")
        if d["n_children"] == 0:
            failures.append(f"{d['doc_id']} has no child chunks")

    parent_by_id = {p["record_id"]: p for p in all_parents}
    for c in all_children:
        if c["doc_id"] not in valid_doc_ids:
            failures.append(f"{c['record_id']} has invalid doc_id {c['doc_id']}")
        parent = parent_by_id.get(c["parent_id"])
        if parent is None:
            failures.append(f"{c['record_id']} references missing parent {c['parent_id']}")
        elif parent["doc_id"] != c["doc_id"]:
            failures.append(f"{c['record_id']} parent doc_id mismatch")
        if c["token_estimate"] > CHILD_HARD_CLAMP_TOKENS:
            failures.append(f"{c['record_id']} exceeds token clamp ({c['token_estimate']})")

    # Every parent.child_ids resolves, and the union covers all children exactly.
    linked: set[str] = set()
    for p in all_parents:
        if p["doc_id"] not in valid_doc_ids:
            failures.append(f"{p['record_id']} has invalid doc_id {p['doc_id']}")
        for cid in p["child_ids"]:
            if cid not in child_ids:
                failures.append(f"{p['record_id']} lists missing child {cid}")
            linked.add(cid)
    orphans = child_ids - linked
    if orphans:
        failures.append(f"{len(orphans)} orphan child chunks not linked to any parent")

    child_tokens = [c["token_estimate"] for c in all_children]
    parent_tokens = [p["token_estimate"] for p in all_parents]
    report = {
        "n_docs": len(per_doc),
        "n_parents": len(all_parents),
        "n_children": len(all_children),
        "n_children_clamped": total_clamped,
        "child_token_dist": _dist(child_tokens),
        "parent_token_dist": _dist(parent_tokens),
        "child_overlap_count": sum(1 for c in all_children if c["has_overlap"]),
        "fallback_docs": [d["doc_id"] for d in per_doc if d["parser_used"] != "pymupdf"],
        "validation_passed": not failures,
        "failures": failures,
        "per_doc": per_doc,
    }
    INGEST_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- Summary ---------------- #
    print(f"Docs processed     : {report['n_docs']}/{N_DOCS}")
    print(f"Parent chunks      : {report['n_parents']}")
    print(f"Child chunks       : {report['n_children']}  (overlap: {report['child_overlap_count']}, clamped: {total_clamped})")
    print(f"Child tokens (est) : {report['child_token_dist']}")
    print(f"Parent tokens (est): {report['parent_token_dist']}")
    print(f"pdfplumber fallback: {report['fallback_docs'] or 'none'}")
    if failures:
        print(f"\nVALIDATION FAILED ({len(failures)}):")
        for f in failures[:25]:
            print(f"  - {f}")
        return 1
    print("\nVALIDATION PASSED — all §13.2 ingestion guarantees hold.")
    print(f"Artifacts: {PARENT_CHUNKS_PATH.name}, {CHILD_CHUNKS_PATH.name}, {INGEST_REPORT_PATH.name}, extracted/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
