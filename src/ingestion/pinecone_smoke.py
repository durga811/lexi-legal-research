"""Reproducible Pinecone smoke test (Step 3 verification).

Confirms the index is populated and that dense retrieval + metadata filtering
behave sensibly. Run after pinecone_upsert.

Run:  uv run python -m src.ingestion.pinecone_smoke
"""

from __future__ import annotations

from collections import Counter

from src.retrieval.pinecone_client import NAMESPACE, get_index


def _search(idx, q, k=8, flt=None, fields=("doc_id", "record_type", "legal_area", "chunk_type", "case_title")):
    query = {"inputs": {"text": q}, "top_k": k}
    if flt:
        query["filter"] = flt
    return idx.search(namespace=NAMESPACE, query=query, fields=list(fields))["result"]["hits"]


def main() -> int:
    idx = get_index()
    stats = idx.describe_index_stats()
    ns = stats.get("namespaces", {}).get(NAMESPACE, {})
    count = ns.get("record_count") or ns.get("vector_count") or stats.get("total_vector_count", 0)
    print(f"Index total_vector_count: {count}")

    failures = []

    t1 = _search(idx, "commercial truck driver had no valid driving licence, insurer denies "
                      "liability for death compensation", 8, {"record_type": {"$eq": "child_chunk"}})
    areas = [h.fields.get("legal_area") for h in t1]
    print("\nT1 license/compensation scenario -> areas:", Counter(areas))
    if areas[:5].count("motor_accident") < 4:
        failures.append("T1: top results not predominantly motor_accident")

    t2 = _search(idx, "trademark infringement and passing off", 6, {"legal_area": {"$eq": "trademark_ip"}})
    print("T2 trademark filter -> docs:", [h.fields["doc_id"] for h in t2])
    if any(h.fields.get("legal_area") != "trademark_ip" for h in t2):
        failures.append("T2: legal_area filter leaked non-trademark docs")

    t3 = _search(idx, "fake or invalid driving license defence by insurer", 6,
                 {"record_type": {"$eq": "case_card"}})
    print("T3 case_card fake-license -> docs:", [h.fields["doc_id"] for h in t3])
    if any(h.fields.get("record_type") != "case_card" for h in t3):
        failures.append("T3: record_type filter leaked non-case_card records")

    t4 = _search(idx, "truck accident", 20, {"vehicle_tags": {"$in": ["truck"]}})
    print("T4 vehicle_tags $in [truck] -> docs:", dict(Counter(h.fields["doc_id"] for h in t4)))
    if not t4:
        failures.append("T4: vehicle_tags $in filter returned nothing")

    if failures:
        print("\nSMOKE FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nSMOKE PASSED — retrieval + metadata filtering verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
