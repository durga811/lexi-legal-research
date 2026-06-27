"""Static configuration for the deterministic ingestion pipeline (Step 1).

All values here are mechanical/build-time only. Nothing in this module calls an
LLM. Chunk-size budgets are deliberately conservative so that child chunks stay
safely under the multilingual-e5-large 507-token input limit even after overlap
is prepended (see src/utils/tokens.py for the estimator rationale).
"""

from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parents[2]
RAW_SRC = ROOT / "raw-docs"               # original lowercase doc_001.pdf ..
DATA = ROOT / "data"
RAW = DATA / "raw"                         # normalized DOC_001.pdf ..
PROCESSED = DATA / "processed"
EXTRACTED = PROCESSED / "extracted"        # per-doc cleaned text for Step 2 sub-agents

PARENT_CHUNKS_PATH = PROCESSED / "parent_chunks.jsonl"
CHILD_CHUNKS_PATH = PROCESSED / "child_chunks.jsonl"
INGEST_REPORT_PATH = PROCESSED / "ingest_report.json"

N_DOCS = 56

# --- Token budgets (estimated tokens, see tokens.estimate_tokens) ---
# Child = retrieval unit. Budget NEW content so that NEW(<=380) + OVERLAP(80)
# stays < 470 < 507 (the e5 model limit). The estimator over-counts vs. the real
# XLM-RoBERTa tokenizer, so the real token count is always below these numbers.
CHILD_TARGET_TOKENS = 300          # close a child once new content reaches this
CHILD_MAX_TOKENS = 360             # hard cap on new content before forcing a close
CHILD_OVERLAP_TOKENS = 60          # context carried (prepended) from previous child
CHILD_HARD_CLAMP_TOKENS = 470      # final safety clamp on a child's total estimate
MIN_CHILD_CHARS = 40               # drop heading-fragment children below this

# Parent = context unit (preserves whole legal reasoning blocks). Parents are NOT
# embedded in Pinecone (they exceed 507 tokens); they live in parent_chunks.jsonl
# and are fetched by record_id for parent-expansion (§7.5).
PARENT_TARGET_TOKENS = 1200
PARENT_MAX_TOKENS = 1500

# Conversion factor used for word-window splitting and overlap seeding.
TOKENS_PER_WORD = 1.5

# A page with fewer than this many non-space chars triggers a pdfplumber retry.
MIN_PAGE_CHARS = 20
# A document whose total cleaned text is shorter than this fails the build.
MIN_DOC_CHARS = 50
