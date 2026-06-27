"""Upload the golden set to LangSmith as a dataset (§12).

The agent already traces every run to the LangSmith project (LANGSMITH_TRACING).
This makes the golden set inspectable as a dataset there too. Idempotent: skips
examples already present (matched by query text).

Run:  uv run python -m src.evaluation.langsmith_dataset
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

from src.ingestion.config import ROOT

load_dotenv(dotenv_path=ROOT / ".env")

GOLDEN_PATH = ROOT / "data" / "golden" / "golden_set_v1.json"
DATASET_NAME = "lexi-golden-v1"


def main() -> int:
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY not set — skipping dataset upload.")
        return 0
    from langsmith import Client

    client = Client()
    queries = json.loads(GOLDEN_PATH.read_text())["queries"]

    if client.has_dataset(dataset_name=DATASET_NAME):
        ds = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        ds = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Lexi precedent-research golden set v1 (grounded, pooling-verified).",
        )

    existing = {e.inputs.get("query") for e in client.list_examples(dataset_id=ds.id)}
    added = 0
    for q in queries:
        if q["query"] in existing:
            continue
        client.create_example(
            dataset_id=ds.id,
            inputs={"query": q["query"]},
            outputs={
                "expected_relevant_doc_ids": q["expected_relevant_doc_ids"],
                "supporting_doc_ids": q["supporting_doc_ids"],
                "adverse_doc_ids": q["adverse_doc_ids"],
                "required_workflow": q["required_workflow"],
                "no_answer": q["no_answer"],
            },
            metadata={"query_id": q["query_id"], "query_type": q["query_type"],
                      "difficulty": q["difficulty"]},
        )
        added += 1

    print(f"Dataset '{DATASET_NAME}': {added} examples added ({len(queries)} total).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
