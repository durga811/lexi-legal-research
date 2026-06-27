"""Shared Pinecone client + index config. Used by ingestion (upsert) and the
runtime retrieval layer. Uses Pinecone integrated inference (hosted
multilingual-e5-large) so no embedding model runs in our app.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache

from dotenv import load_dotenv

from src.ingestion.config import ROOT

load_dotenv(dotenv_path=ROOT / ".env")

INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "lexi-legal-precedents-v1")
NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "corpus-v1")
EMBED_MODEL = "multilingual-e5-large"
EMBED_FIELD = "text"  # record field that Pinecone embeds (index field_map)
CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
REGION = os.environ.get("PINECONE_REGION", "us-east-1")


@lru_cache(maxsize=1)
def get_pc():
    key = os.environ.get("PINECONE_API_KEY")
    if not key:
        raise RuntimeError("PINECONE_API_KEY not set (check .env)")
    from pinecone import Pinecone

    return Pinecone(api_key=key)


def ensure_index(wait: bool = True, timeout_s: int = 180):
    """Create the integrated-inference index if absent; wait until ready."""
    pc = get_pc()
    if not pc.has_index(INDEX_NAME):
        pc.create_index_for_model(
            name=INDEX_NAME,
            cloud=CLOUD,
            region=REGION,
            embed={"model": EMBED_MODEL, "field_map": {"text": EMBED_FIELD}},
        )
    if wait:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            desc = pc.describe_index(INDEX_NAME)
            if getattr(desc, "status", {}).get("ready"):
                return desc
            time.sleep(2)
        raise TimeoutError(f"Index {INDEX_NAME} not ready after {timeout_s}s")
    return pc.describe_index(INDEX_NAME)


@lru_cache(maxsize=1)
def get_index():
    return get_pc().Index(INDEX_NAME)
