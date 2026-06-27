"""Pydantic schemas for ingestion artifacts (Step 1).

Only the *structural* chunk schemas live here. Semantic enrichment fields
(case_title, legal_area, issue/stance/vehicle tags, chunk_type) are produced at
build-time by Claude sub-agents in Step 2 and stored in a separate labels file,
then merged at Pinecone-upsert time. Keeping ingestion structural-only means the
deterministic stage never depends on an LLM.
"""

from pydantic import BaseModel, Field


class ChildChunk(BaseModel):
    doc_id: str
    record_id: str
    record_type: str = "child_chunk"
    parent_id: str
    text: str
    token_estimate: int
    char_len: int
    page_start: int
    page_end: int
    paragraph_start: int
    paragraph_end: int
    has_overlap: bool = False
    source_text_hash: str


class ParentChunk(BaseModel):
    doc_id: str
    record_id: str
    record_type: str = "parent_chunk"
    text: str
    token_estimate: int
    char_len: int
    page_start: int
    page_end: int
    paragraph_start: int
    paragraph_end: int
    child_ids: list[str] = Field(default_factory=list)
    source_text_hash: str


class ExtractedDoc(BaseModel):
    doc_id: str
    file_name: str
    n_pages: int
    char_len: int
    parser_used: str
    source_hint: dict
    pages: list[dict]  # [{"page": int, "text": str}, ...] (cleaned)


class KeyPassage(BaseModel):
    label: str
    quote: str
    page: int | None = None


class DocEnrichment(BaseModel):
    """Per-document build-time enrichment produced by a Claude sub-agent.

    Types only here; controlled-vocabulary membership and grounding (quotes/title/
    statutes present in the source text) are checked in assemble_enrichment.py so
    failures can be collected and the offending doc re-processed.
    """

    doc_id: str
    case_title: str
    court: str
    judge_or_bench: str = ""
    date: str = ""
    year: int | None = None
    legal_area: str
    is_motor_accident: bool
    procedural_stage: str
    document_type: str
    parties: dict = Field(default_factory=dict)
    statutes: list[str] = Field(default_factory=list)
    core_facts: list[str] = Field(default_factory=list)
    legal_issues: list[str] = Field(default_factory=list)
    issue_tags: list[str] = Field(default_factory=list)
    vehicle_tags: list[str] = Field(default_factory=list)
    stance_tags: list[str] = Field(default_factory=list)
    outcome: str = ""
    key_holdings: list[str] = Field(default_factory=list)
    key_passages: list[KeyPassage] = Field(default_factory=list)
    adverse_value: str = ""
    summary: str
    confidence: str = "medium"
