"""Build + upsert Pinecone records (Step 3).

Embedded record types (namespace corpus-v1):
  - case_card  : one per doc, text = structured case-card summary.
  - child_chunk: retrieval unit, text = chunk text, with grounded chunk metadata.

Parents are NOT upserted (they exceed the 507-token embedding limit); they stay in
parent_chunks.jsonl for parent-expansion at retrieval time.

Each record merges child_chunks.jsonl (text + structure) with chunk_labels.jsonl
(tags), flat-keyed for Pinecone metadata filtering. Null / empty fields are dropped.

Run:  uv run python -m src.ingestion.pinecone_upsert
"""

from __future__ import annotations

import json
import sys

from src.ingestion.config import CHILD_CHUNKS_PATH, PROCESSED
from src.retrieval.pinecone_client import NAMESPACE, ensure_index, get_index
from src.utils.tokens import estimate_tokens

CASE_CARDS_PATH = PROCESSED / "case_cards.jsonl"
CHUNK_LABELS_PATH = PROCESSED / "chunk_labels.jsonl"
BATCH = 90  # integrated-inference upsert batch limit is 96
EMBED_CLAMP_TOKENS = 480  # keep case-card text under the 507 model limit
# Hosted e5 inference is capped at 250k tokens/min on this plan; throttle under it.
TPM_LIMIT = 220_000
WINDOW_S = 60.0


def _clean(rec: dict) -> dict:
    """Drop keys whose value is None, "" or [] (Pinecone metadata can't be null)."""
    return {k: v for k, v in rec.items() if v is not None and v != "" and v != []}


def _clamp(text: str) -> str:
    if estimate_tokens(text) <= EMBED_CLAMP_TOKENS:
        return text
    words = text.split()
    while words and estimate_tokens(" ".join(words)) > EMBED_CLAMP_TOKENS:
        words = words[: -max(1, len(words) // 20)]
    return " ".join(words)


def build_records() -> list[dict]:
    labels = {}
    with CHUNK_LABELS_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            labels[r["record_id"]] = r

    records: list[dict] = []

    # child chunks
    with CHILD_CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            lab = labels.get(c["record_id"], {})
            records.append(_clean({
                "_id": c["record_id"],
                "text": c["text"],
                "doc_id": c["doc_id"],
                "record_type": "child_chunk",
                "parent_id": c["parent_id"],
                "legal_area": lab.get("legal_area"),
                "case_title": lab.get("case_title"),
                "court": lab.get("court"),
                "year": lab.get("year"),
                "chunk_type": lab.get("chunk_type"),
                "issue_tags": lab.get("issue_tags", []),
                "stance_tags": lab.get("stance_tags", []),
                "vehicle_tags": lab.get("vehicle_tags", []),
                "page_start": c["page_start"],
                "page_end": c["page_end"],
            }))

    # case cards
    with CASE_CARDS_PATH.open(encoding="utf-8") as f:
        for line in f:
            cc = json.loads(line)
            m = cc["metadata"]
            records.append(_clean({
                "_id": cc["record_id"],
                "text": _clamp(cc["text"]),
                "doc_id": cc["doc_id"],
                "record_type": "case_card",
                "case_title": m.get("case_title"),
                "court": m.get("court"),
                "year": m.get("year"),
                "legal_area": m.get("legal_area"),
                "issue_tags": m.get("issue_tags", []),
                "stance_tags": m.get("stance_tags", []),
                "vehicle_tags": m.get("vehicle_tags", []),
            }))

    return records


def _existing_ids(index) -> set[str]:
    """IDs already in the namespace, so re-runs skip re-embedding them."""
    ids: set[str] = set()
    try:
        for page in index.list(namespace=NAMESPACE):
            ids.update(page if isinstance(page, list) else [page])
    except Exception:
        pass
    return ids


def _upsert_with_retry(index, batch, max_retries: int = 6) -> None:
    import time

    from pinecone.errors.exceptions import RateLimitError

    for attempt in range(max_retries):
        try:
            index.upsert_records(namespace=NAMESPACE, records=batch)
            return
        except RateLimitError:
            wait = 20 * (attempt + 1)
            print(f"\n  rate-limited; backing off {wait}s...", end="")
            time.sleep(wait)
    index.upsert_records(namespace=NAMESPACE, records=batch)  # final attempt raises


def main() -> int:
    import time

    print("Ensuring index is ready (hosted multilingual-e5-large)...")
    ensure_index()
    index = get_index()

    records = build_records()
    n_cards = sum(1 for r in records if r["record_type"] == "case_card")
    n_child = len(records) - n_cards
    print(f"Built {len(records)} records ({n_child} child_chunk, {n_cards} case_card).")

    existing = _existing_ids(index)
    todo = [r for r in records if r["_id"] not in existing]
    print(f"{len(existing)} already present; upserting {len(todo)} new records "
          f"(throttled to {TPM_LIMIT:,} tok/min)...")

    win_start = time.monotonic()
    win_tokens = 0
    done = 0
    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        btok = sum(estimate_tokens(r["text"]) for r in batch)
        now = time.monotonic()
        if now - win_start >= WINDOW_S:
            win_start, win_tokens = now, 0
        if win_tokens + btok > TPM_LIMIT:
            sleep_for = max(0.0, WINDOW_S - (now - win_start))
            if sleep_for:
                print(f"\n  throttling {sleep_for:.0f}s to respect token/min limit...", end="")
                time.sleep(sleep_for)
            win_start, win_tokens = time.monotonic(), 0
        _upsert_with_retry(index, batch)
        win_tokens += btok
        done += len(batch)
        print(f"  upserted {done}/{len(todo)} new", end="\r")
    print()

    # Validate counts (allow a few seconds for the index to converge)
    import time

    expected = len(records)
    got = 0
    for _ in range(15):
        stats = index.describe_index_stats()
        ns = stats.get("namespaces", {}).get(NAMESPACE, {})
        got = ns.get("record_count") or ns.get("vector_count") or 0
        if got >= expected:
            break
        time.sleep(2)

    print(f"Namespace '{NAMESPACE}' record_count: {got} (expected {expected})")
    if got < expected:
        print("WARNING: fewer records than expected — index may still be converging.")
        return 1
    print("Upsert validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
