"""Hierarchical, paragraph-aware chunking. No LLM.

Three structural levels are produced (semantic case cards are Step 2):
  - Paragraphs: cleaned page text split on blank lines and the judgment's own
    numbered-paragraph markers ("2. Brief facts ...").
  - Parent chunks: paragraphs packed to ~1200 (<=1500) tokens. Whole legal units
    (compensation calculations, issue lists, orders) stay intact here because
    parents only ever break on paragraph boundaries. Parents are the context unit
    fetched for parent-expansion (§7.5); they are NOT embedded in Pinecone.
  - Child chunks: retrieval units. New content is packed to ~320 (<=380) tokens
    and ~80 tokens of the previous child are prepended as overlap, so the total
    estimate stays < 470 < 507 (the e5 input limit). A single paragraph larger
    than the child cap is window-split by words; its intact form still lives in
    the parent.
"""

from __future__ import annotations

import re

from src.ingestion.config import (
    CHILD_HARD_CLAMP_TOKENS,
    CHILD_MAX_TOKENS,
    CHILD_OVERLAP_TOKENS,
    CHILD_TARGET_TOKENS,
    MIN_CHILD_CHARS,
    PARENT_MAX_TOKENS,
    PARENT_TARGET_TOKENS,
)
from src.utils.ids import child_record_id, parent_record_id, text_hash
from src.utils.schemas import ChildChunk, ParentChunk
from src.utils.tokens import estimate_tokens, words_for_tokens

_NUM_PARA_RE = re.compile(r"^\s*\(?\d{1,3}[.)]\s+\S")


# --------------------------------------------------------------------------- #
# Paragraphs
# --------------------------------------------------------------------------- #
def _emit(paras: list[dict], lines: list[str], page: int) -> None:
    joined = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if joined:
        paras.append({"page": page, "text": joined})


def build_paragraphs(cleaned_pages: list[dict]) -> list[dict]:
    """Flatten cleaned pages into globally-indexed paragraph units."""
    paras: list[dict] = []
    for p in cleaned_pages:
        page = p["page"]
        for block in re.split(r"\n\s*\n", p["text"]):
            block = block.strip()
            if not block:
                continue
            cur: list[str] = []
            for ln in block.split("\n"):
                if _NUM_PARA_RE.match(ln) and cur:
                    _emit(paras, cur, page)
                    cur = []
                cur.append(ln)
            if cur:
                _emit(paras, cur, page)
    for i, pa in enumerate(paras):
        pa["idx"] = i
    return paras


# --------------------------------------------------------------------------- #
# Parents
# --------------------------------------------------------------------------- #
def _finalize_parent(paras: list[dict]) -> dict:
    text = "\n\n".join(p["text"] for p in paras)
    pages = [p["page"] for p in paras]
    idxs = [p["idx"] for p in paras]
    return {
        "paras": paras,
        "text": text,
        "page_start": min(pages),
        "page_end": max(pages),
        "para_start": min(idxs),
        "para_end": max(idxs),
    }


def build_parents(paras: list[dict]) -> list[dict]:
    parents: list[dict] = []
    cur: list[dict] = []
    cur_tokens = 0
    for pa in paras:
        t = estimate_tokens(pa["text"])
        if cur and cur_tokens + t > PARENT_MAX_TOKENS:
            parents.append(_finalize_parent(cur))
            cur, cur_tokens = [], 0
        cur.append(pa)
        cur_tokens += t
        if cur_tokens >= PARENT_TARGET_TOKENS:
            parents.append(_finalize_parent(cur))
            cur, cur_tokens = [], 0
    if cur:
        parents.append(_finalize_parent(cur))
    return parents


# --------------------------------------------------------------------------- #
# Children
# --------------------------------------------------------------------------- #
def _window_split_paragraph(pa: dict) -> list[dict]:
    """Split an oversized paragraph into overlapping word windows."""
    words = pa["text"].split()
    win = words_for_tokens(CHILD_TARGET_TOKENS)
    step = max(1, words_for_tokens(CHILD_TARGET_TOKENS - CHILD_OVERLAP_TOKENS))
    segs: list[dict] = []
    i = 0
    while i < len(words):
        seg_text = " ".join(words[i : i + win])
        segs.append({"paras": [{"page": pa["page"], "idx": pa["idx"], "text": seg_text}],
                     "text": seg_text})
        if i + win >= len(words):
            break
        i += step
    return segs


def _mk_raw_child(paras: list[dict]) -> dict:
    text = re.sub(r"\s+", " ", " ".join(p["text"] for p in paras)).strip()
    return {"paras": list(paras), "text": text}


def build_children_for_parent(parent_paras: list[dict]) -> list[dict]:
    """Paragraph-pack children, window-splitting oversized paragraphs, then
    prepend word-overlap from the previous child."""
    raw: list[dict] = []
    cur: list[dict] = []
    cur_tokens = 0
    for pa in parent_paras:
        t = estimate_tokens(pa["text"])
        if t > CHILD_MAX_TOKENS:
            if cur:
                raw.append(_mk_raw_child(cur))
                cur, cur_tokens = [], 0
            raw.extend(_window_split_paragraph(pa))
            continue
        if cur and cur_tokens + t > CHILD_MAX_TOKENS:
            raw.append(_mk_raw_child(cur))
            cur, cur_tokens = [], 0
        cur.append(pa)
        cur_tokens += t
        if cur_tokens >= CHILD_TARGET_TOKENS:
            raw.append(_mk_raw_child(cur))
            cur, cur_tokens = [], 0
    if cur:
        raw.append(_mk_raw_child(cur))

    overlap_words = words_for_tokens(CHILD_OVERLAP_TOKENS)
    out: list[dict] = []
    for i, ch in enumerate(raw):
        text = ch["text"]
        has_overlap = False
        if i > 0:
            prev_words = raw[i - 1]["text"].split()
            if prev_words:
                text = " ".join(prev_words[-overlap_words:]) + " " + text
                has_overlap = True
        out.append({"paras": ch["paras"], "text": text.strip(), "has_overlap": has_overlap})
    return out


def _clamp(text: str) -> str:
    """Final safety net: trim trailing words until the *estimate* (which counts
    both words and chars) is under the clamp, guaranteeing the real e5 token
    count stays below 507 regardless of long legal words / numbers."""
    if estimate_tokens(text) <= CHILD_HARD_CLAMP_TOKENS:
        return text
    words = text.split()
    while words and estimate_tokens(" ".join(words)) > CHILD_HARD_CLAMP_TOKENS:
        drop = max(1, len(words) // 20)
        words = words[:-drop]
    return " ".join(words)


# --------------------------------------------------------------------------- #
# Document-level assembly
# --------------------------------------------------------------------------- #
def chunk_document(doc: str, cleaned_pages: list[dict]) -> tuple[list[dict], list[dict], int]:
    """Return ``(parent_records, child_records, n_clamped)`` for one document."""
    paras = build_paragraphs(cleaned_pages)
    parent_units = build_parents(paras)

    parent_records: list[dict] = []
    child_records: list[dict] = []
    cidx = 0
    n_clamped = 0

    for pidx, punit in enumerate(parent_units):
        pid = parent_record_id(doc, pidx)
        child_ids: list[str] = []
        children = build_children_for_parent(punit["paras"])
        # Drop heading-fragment children ("ORDER", "2.") but never leave a parent
        # with zero children; if all are tiny, keep the largest.
        kept = [c for c in children if len(c["text"]) >= MIN_CHILD_CHARS]
        if not kept and children:
            kept = [max(children, key=lambda c: len(c["text"]))]
        for ch in kept:
            clamped = _clamp(ch["text"])
            if clamped != ch["text"]:
                n_clamped += 1
            crid = child_record_id(doc, cidx)
            cidx += 1
            pages = [p["page"] for p in ch["paras"]]
            idxs = [p["idx"] for p in ch["paras"]]
            child_records.append(
                ChildChunk(
                    doc_id=doc,
                    record_id=crid,
                    parent_id=pid,
                    text=clamped,
                    token_estimate=estimate_tokens(clamped),
                    char_len=len(clamped),
                    page_start=min(pages),
                    page_end=max(pages),
                    paragraph_start=min(idxs),
                    paragraph_end=max(idxs),
                    has_overlap=ch["has_overlap"],
                    source_text_hash=text_hash(clamped),
                ).model_dump()
            )
            child_ids.append(crid)

        ptext = punit["text"]
        parent_records.append(
            ParentChunk(
                doc_id=doc,
                record_id=pid,
                text=ptext,
                token_estimate=estimate_tokens(ptext),
                char_len=len(ptext),
                page_start=punit["page_start"],
                page_end=punit["page_end"],
                paragraph_start=punit["para_start"],
                paragraph_end=punit["para_end"],
                child_ids=child_ids,
                source_text_hash=text_hash(ptext),
            ).model_dump()
        )

    return parent_records, child_records, n_clamped
