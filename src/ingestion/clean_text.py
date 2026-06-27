"""Deterministic text cleanup. No LLM.

Removes Indian Kanoon scaffolding that repeats on every page:
  - the "Indian Kanoon - http://indiankanoon.org/doc/NNN/" footer line
  - the recurring "<case title> on <date>" running footer
  - standalone page numbers ("3") and "N of M" page markers
  - any other line that repeats on a large fraction of the document's pages
De-hyphenates line-wrapped words and normalizes whitespace, while preserving
paragraph structure (blank lines / numbered-paragraph boundaries) for chunking.
"""

from __future__ import annotations

import re
from collections import Counter

_URL_RE = re.compile(r"^\s*indian\s+kanoon\s*-\s*http\S+\s*$", re.IGNORECASE)
_PAGENUM_RE = re.compile(r"^\s*\d{1,4}\s*$")
_PAGE_OF_RE = re.compile(r"^\s*\d{1,4}\s+of\s+\d{1,4}\s*$", re.IGNORECASE)
_FOOTER_TITLE_RE = re.compile(
    r"^(?P<title>.+?)\s+on\s+\d{1,2}\s+[A-Za-z]+,\s+\d{4}\s*$"
)


def _norm(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def find_repeating_lines(pages: list[dict], min_pages: int = 2, frac: float = 0.45) -> set[str]:
    """Normalized lines that appear on >= max(min_pages, frac*n_pages) pages."""
    n = len(pages)
    counter: Counter[str] = Counter()
    for p in pages:
        seen: set[str] = set()
        for raw in p["text"].split("\n"):
            nl = _norm(raw)
            if len(nl) < 8:  # very short lines handled by the regex strippers
                continue
            if nl not in seen:
                seen.add(nl)
                counter[nl] += 1
    threshold = max(min_pages, int(n * frac))
    return {line for line, c in counter.items() if c >= threshold}


def _extract_source_hint(repeating: set[str]) -> dict:
    """Best-effort mechanical hint (title/date) from the running footer.

    This is only a hint to help Step 2 grounding; the enrichment sub-agents must
    still confirm the case title and date against the document body.
    """
    for line in repeating:
        m = _FOOTER_TITLE_RE.match(line)
        if m:
            return {"footer_title": m.group("title").strip(), "footer_line": line}
    return {}


def _clean_page(text: str, repeating: set[str]) -> str:
    kept: list[str] = []
    for raw in text.split("\n"):
        nl = _norm(raw)
        if not nl:
            kept.append("")  # preserve blank line => paragraph separator
            continue
        if nl in repeating:
            continue
        if _URL_RE.match(raw) or _PAGENUM_RE.match(raw) or _PAGE_OF_RE.match(raw):
            continue
        kept.append(raw.rstrip())
    out = "\n".join(kept)
    # Join words split across a line break: "compensa-\ntion" -> "compensation".
    out = re.sub(r"(\w)-\n(\w)", r"\1\2", out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def clean_document(pages: list[dict]) -> tuple[list[dict], dict]:
    """Return ``(cleaned_pages, source_hint)``."""
    repeating = find_repeating_lines(pages)
    source_hint = _extract_source_hint(repeating)
    cleaned = [{"page": p["page"], "text": _clean_page(p["text"], repeating)} for p in pages]
    return cleaned, source_hint
