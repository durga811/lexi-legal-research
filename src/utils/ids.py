"""Deterministic ID and hashing helpers shared across the pipeline."""

import hashlib


def doc_id(n: int) -> str:
    """1 -> 'DOC_001'."""
    return f"DOC_{n:03d}"


def parent_record_id(doc: str, idx: int) -> str:
    return f"{doc}_parent_{idx:04d}"


def child_record_id(doc: str, idx: int) -> str:
    return f"{doc}_child_{idx:04d}"


def text_hash(text: str) -> str:
    """Stable 16-hex-char content hash, used to detect changed source text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
