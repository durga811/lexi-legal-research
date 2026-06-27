"""Assemble + validate Step 2 sub-agent enrichment into frozen artifacts.

Reads data/processed/_enrich/DOC_xxx.json (one per doc, written by the analyst
sub-agents) and:
  1. pydantic-validates each record,
  2. checks controlled-vocabulary membership,
  3. checks GROUNDING against the cleaned source text (verbatim key_passages,
     case-title tokens, statutes, motor-consistency, issue-tag lexicon support),
  4. checks COVERAGE (every DOC_001..056 present exactly once),
and writes case_cards.jsonl + document_metadata.jsonl for the records that pass.

Hard failures (schema/vocab/coverage/hallucinated-quotes/motor-inconsistency) list
the offending docs for re-processing by a fresh sub-agent. Softer issues are emitted
as warnings for developer spot-review (§5.7 correctness guarantees).

Run:  uv run python -m src.ingestion.assemble_enrichment
"""

from __future__ import annotations

import json
import re
import sys

from pydantic import ValidationError

from src.ingestion.config import EXTRACTED, N_DOCS, PROCESSED
from src.ingestion.vocab import (
    DOCUMENT_TYPES,
    ISSUE_LEXICON,
    ISSUE_TAGS,
    LEGAL_AREAS,
    PROCEDURAL_STAGES,
    STANCE_TAGS,
    VEHICLE_TAGS,
)
from src.utils.ids import doc_id
from src.utils.schemas import DocEnrichment

ENRICH_DIR = PROCESSED / "_enrich"
CASE_CARDS_PATH = PROCESSED / "case_cards.jsonl"
DOC_METADATA_PATH = PROCESSED / "document_metadata.jsonl"
ENRICH_REPORT_PATH = PROCESSED / "enrichment_report.json"

_TITLE_STOP = {
    "vs", "v", "and", "others", "ors", "anr", "etc", "the", "of", "ltd", "co",
    "company", "smt", "sh", "mr", "mrs", "ms", "m/s", "limited", "thr", "through",
    "s/o", "w/o", "d/o", "alias",
}
_MOTOR_TERMS = ["motor vehicle", "motor accident", "motor vehicular", "m.v. act",
                "motor vehicles act", "mact", "macp", "claim petition", "fao", "accident"]


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()


def _alnum(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def _quote_found(quote: str, norm_text: str, alnum_text: str) -> bool:
    q = _norm(quote)
    if len(q) < 8:
        return True  # too short to meaningfully ground
    if q in norm_text:
        return True
    return _alnum(quote)[:120] in alnum_text  # relaxed: ignore punctuation/spacing


def _validate_doc(rec: dict, text: str) -> tuple[list[str], list[str]]:
    """Return (hard_failures, warnings) for one enrichment record."""
    hard: list[str] = []
    warn: list[str] = []

    # Vocabulary
    if rec["legal_area"] not in LEGAL_AREAS:
        hard.append(f"legal_area '{rec['legal_area']}' not in vocab")
    if rec["procedural_stage"] not in PROCEDURAL_STAGES:
        hard.append(f"procedural_stage '{rec['procedural_stage']}' not in vocab")
    if rec["document_type"] not in DOCUMENT_TYPES:
        hard.append(f"document_type '{rec['document_type']}' not in vocab")
    for t in rec["issue_tags"]:
        if t not in ISSUE_TAGS:
            hard.append(f"issue_tag '{t}' not in vocab")
    for t in rec["stance_tags"]:
        if t not in STANCE_TAGS:
            hard.append(f"stance_tag '{t}' not in vocab")
    for t in rec["vehicle_tags"]:
        if t not in VEHICLE_TAGS:
            hard.append(f"vehicle_tag '{t}' not in vocab")

    norm_text, alnum_text = _norm(text), _alnum(text)

    # Grounding: verbatim key_passages (the hallucination detector)
    passages = rec.get("key_passages", [])
    if passages:
        found = sum(1 for p in passages if _quote_found(p["quote"], norm_text, alnum_text))
        if found / len(passages) < 0.5:
            hard.append(f"only {found}/{len(passages)} key_passages quotes found in source text")
        elif found < len(passages):
            warn.append(f"{len(passages) - found} key_passage quote(s) not verbatim")
    else:
        warn.append("no key_passages provided")

    # Grounding: motor consistency
    motor_in_text = any(term in norm_text for term in _MOTOR_TERMS)
    if rec["is_motor_accident"] and not motor_in_text:
        hard.append("is_motor_accident=true but no motor terms found in text")
    if not rec["is_motor_accident"] and rec["vehicle_tags"]:
        hard.append("is_motor_accident=false but vehicle_tags is non-empty")

    # Grounding: case-title tokens present
    toks = [w for w in re.split(r"[^a-z0-9]+", rec["case_title"].lower())
            if len(w) >= 3 and w not in _TITLE_STOP]
    if toks:
        present = sum(1 for w in toks if w in norm_text)
        if present / len(toks) < 0.4:
            warn.append(f"case_title weakly grounded ({present}/{len(toks)} tokens in text)")

    # Grounding: statutes + issue-tag lexicon support (warn-level).
    # A statute is grounded if a section number (not the year) OR a distinctive
    # word of the act name appears in the text.
    for s in rec["statutes"]:
        sl = _norm(s)
        nums = [n for n in re.findall(r"\d{1,4}[a-z]?", sl)
                if not (len(n) == 4 and n[:2] in {"18", "19", "20"})]
        act_words = [w for w in re.split(r"[^a-z]+", sl)
                     if len(w) >= 4 and w not in {"section", "article", "rule", "read", "with", "act"}]
        grounded = any(n in norm_text for n in nums) or any(w in norm_text for w in act_words[:2])
        if not grounded:
            warn.append(f"statute '{s}' core token not found in text")
    for t in rec["issue_tags"]:
        terms = ISSUE_LEXICON.get(t)
        if terms and not any(term in norm_text for term in terms):
            warn.append(f"issue_tag '{t}' has no lexicon evidence in text")

    return hard, warn


def _case_card_text(rec: dict) -> str:
    parts = [
        f"Case: {rec['case_title']}",
        f"Court: {rec['court']} ({rec.get('year') or 'n.d.'})",
        f"Legal area: {rec['legal_area']}; stage: {rec['procedural_stage']}",
        f"Issues: {', '.join(rec['issue_tags']) or 'n/a'}",
    ]
    if rec["vehicle_tags"]:
        parts.append(f"Vehicles: {', '.join(rec['vehicle_tags'])}")
    parts.append(f"Stance: {', '.join(rec['stance_tags']) or 'n/a'}")
    if rec["core_facts"]:
        parts.append("Facts: " + " ".join(rec["core_facts"]))
    if rec["key_holdings"]:
        parts.append("Holdings: " + " ".join(rec["key_holdings"]))
    parts.append(f"Outcome: {rec['outcome']}")
    parts.append(rec["summary"])
    return "\n".join(parts)


def main() -> int:
    valid_ids = [doc_id(n) for n in range(1, N_DOCS + 1)]
    records: dict[str, dict] = {}
    per_doc_report: list[dict] = []
    failed_docs: list[str] = []

    for did in valid_ids:
        path = ENRICH_DIR / f"{did}.json"
        if not path.exists():
            failed_docs.append(did)
            per_doc_report.append({"doc_id": did, "status": "missing", "hard": ["file not found"]})
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            failed_docs.append(did)
            per_doc_report.append({"doc_id": did, "status": "bad_json", "hard": [str(e)]})
            continue

        if raw.get("doc_id") != did:
            raw["doc_id"] = did  # tolerate; enforce the assigned id

        try:
            model = DocEnrichment(**raw)
        except ValidationError as e:
            failed_docs.append(did)
            per_doc_report.append({"doc_id": did, "status": "schema_invalid",
                                   "hard": [str(err) for err in e.errors()][:5]})
            continue

        rec = model.model_dump()
        text = (EXTRACTED / f"{did}.txt").read_text(encoding="utf-8")
        hard, warn = _validate_doc(rec, text)
        status = "ok" if not hard else "grounding_failed"
        per_doc_report.append({"doc_id": did, "status": status, "hard": hard, "warn": warn,
                               "confidence": rec.get("confidence")})
        if hard:
            failed_docs.append(did)
        else:
            records[did] = rec

    # Coverage
    coverage_ok = len(records) == N_DOCS

    # Write artifacts for passing docs (full freeze only when coverage_ok)
    if records:
        with CASE_CARDS_PATH.open("w", encoding="utf-8") as cc, \
             DOC_METADATA_PATH.open("w", encoding="utf-8") as dm:
            for did in valid_ids:
                rec = records.get(did)
                if not rec:
                    continue
                cc.write(json.dumps({
                    "doc_id": did,
                    "record_id": f"{did}_case_card",
                    "record_type": "case_card",
                    "text": _case_card_text(rec),
                    "metadata": {
                        "doc_id": did,
                        "record_type": "case_card",
                        "case_title": rec["case_title"],
                        "court": rec["court"],
                        "year": rec["year"],
                        "legal_area": rec["legal_area"],
                        "issue_tags": rec["issue_tags"],
                        "stance_tags": rec["stance_tags"],
                        "vehicle_tags": rec["vehicle_tags"],
                    },
                }, ensure_ascii=False) + "\n")
                dm.write(json.dumps(rec, ensure_ascii=False) + "\n")

    warn_total = sum(len(d.get("warn", [])) for d in per_doc_report)
    report = {
        "n_enriched": len(records),
        "n_expected": N_DOCS,
        "coverage_ok": coverage_ok,
        "failed_docs": failed_docs,
        "n_warnings": warn_total,
        "per_doc": per_doc_report,
    }
    ENRICH_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Enriched OK : {len(records)}/{N_DOCS}")
    print(f"Warnings    : {warn_total} (developer spot-review)")
    if failed_docs:
        print(f"\nHARD FAILURES — re-process these docs ({len(failed_docs)}):")
        for d in per_doc_report:
            if d["doc_id"] in failed_docs:
                print(f"  - {d['doc_id']} [{d['status']}]: {d.get('hard')}")
        return 1
    if not coverage_ok:
        print("\nCOVERAGE INCOMPLETE — not all 56 docs enriched.")
        return 1
    print("\nVALIDATION PASSED — case_cards.jsonl + document_metadata.jsonl frozen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
