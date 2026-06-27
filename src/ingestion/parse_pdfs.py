"""Deterministic PDF text extraction (PyMuPDF primary, pdfplumber fallback).

No LLM. No OCR is wired in by default; if a page yields almost no text from both
parsers it is left as-is and surfaced in the ingest report so a developer can
decide whether OCR is warranted. For this corpus every page is digital text.
"""

from __future__ import annotations

try:  # PyMuPDF >= 1.24 exposes the ``pymupdf`` module name; ``fitz`` is the alias.
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

import pdfplumber

from src.ingestion.config import MIN_PAGE_CHARS


def extract_doc(pdf_path) -> tuple[list[dict], str]:
    """Extract per-page text from a PDF.

    Returns ``(pages, parser_used)`` where ``pages`` is ``[{"page": int,
    "text": str}, ...]`` (1-indexed) and ``parser_used`` records whether the
    pdfplumber fallback was needed for any page.
    """
    pages: list[dict] = []
    parser_used = "pymupdf"
    plumber = None
    try:
        with fitz.open(pdf_path) as doc:
            n = doc.page_count
            for i in range(n):
                text = doc[i].get_text("text") or ""
                if len(text.strip()) < MIN_PAGE_CHARS:
                    # Retry just this page with pdfplumber.
                    try:
                        if plumber is None:
                            plumber = pdfplumber.open(pdf_path)
                        alt = plumber.pages[i].extract_text() or ""
                        if len(alt.strip()) > len(text.strip()):
                            text = alt
                            parser_used = "pymupdf+pdfplumber"
                    except Exception:
                        pass
                pages.append({"page": i + 1, "text": text})
    finally:
        if plumber is not None:
            plumber.close()
    return pages, parser_used
