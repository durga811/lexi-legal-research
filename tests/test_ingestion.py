"""Unit + artifact tests for Step 1 (deterministic ingestion)."""

import json
from pathlib import Path

from src.ingestion.chunker import build_paragraphs, chunk_document
from src.ingestion.clean_text import clean_document
from src.ingestion.config import CHILD_HARD_CLAMP_TOKENS, N_DOCS, PROCESSED
from src.utils.tokens import estimate_tokens

ROOT = Path(__file__).resolve().parents[1]


# --------------------------- unit: token estimate --------------------------- #
def test_estimate_is_conservative_and_monotonic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("word " * 100) > estimate_tokens("word " * 50)
    # char-heavy text is counted via the char-based branch
    long_word = "supercalifragilistic " * 30
    assert estimate_tokens(long_word) >= len(long_word) / 3.5 - 1


# --------------------------- unit: cleaning --------------------------------- #
def test_clean_removes_kanoon_scaffolding():
    pages = [
        {"page": 1, "text": "Real legal sentence one.\nABC vs XYZ on 6 November, 2023\n"
                            "Indian Kanoon - http://indiankanoon.org/doc/123/\n1"},
        {"page": 2, "text": "Real legal sentence two.\nABC vs XYZ on 6 November, 2023\n"
                            "Indian Kanoon - http://indiankanoon.org/doc/123/\n2"},
        {"page": 3, "text": "Real legal sentence three.\nABC vs XYZ on 6 November, 2023\n"
                            "Indian Kanoon - http://indiankanoon.org/doc/123/\n3"},
    ]
    cleaned, hint = clean_document(pages)
    body = "\n".join(p["text"] for p in cleaned)
    assert "indiankanoon" not in body.lower()
    assert "ABC vs XYZ on 6 November, 2023" not in body  # repeating footer stripped
    assert "Real legal sentence one." in body
    assert hint.get("footer_title") == "ABC vs XYZ"


# --------------------------- unit: chunking invariants ---------------------- #
def _synthetic_pages():
    para = ("This is paragraph number {n}. " + "lorem ipsum dolor sit amet " * 20)
    text = "\n\n".join(para.format(n=i) for i in range(1, 25))
    return [{"page": 1, "text": text}, {"page": 2, "text": text}]


def test_chunk_links_and_sizes_hold():
    parents, children, _ = chunk_document("DOC_999", _synthetic_pages())
    assert parents and children
    pids = {p["record_id"] for p in parents}
    # every child links to a real parent of the same doc
    for c in children:
        assert c["parent_id"] in pids
        assert c["doc_id"] == "DOC_999"
        assert c["token_estimate"] <= CHILD_HARD_CLAMP_TOKENS
    # parent.child_ids cover children exactly (no orphans / dangling)
    linked = {cid for p in parents for cid in p["child_ids"]}
    assert linked == {c["record_id"] for c in children}
    # paragraphs are globally indexed and contiguous
    paras = build_paragraphs(_synthetic_pages())
    assert [p["idx"] for p in paras] == list(range(len(paras)))


def test_overlap_present_for_multi_child_parent():
    _, children, _ = chunk_document("DOC_998", _synthetic_pages())
    # at least some children carry overlap from the previous child
    assert any(c["has_overlap"] for c in children)


# --------------------------- artifact: real corpus -------------------------- #
def test_real_ingest_report_passed():
    rp = PROCESSED / "ingest_report.json"
    if not rp.exists():
        return  # ingestion not run yet in this environment
    report = json.loads(rp.read_text())
    assert report["validation_passed"] is True
    assert report["n_docs"] == N_DOCS
    assert report["child_token_dist"]["max"] <= CHILD_HARD_CLAMP_TOKENS
