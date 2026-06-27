"""Golden-set schema (§11.7). One record per evaluation query."""

from __future__ import annotations

from pydantic import BaseModel, Field

QUERY_TYPES = {
    "factual_metadata", "procedural", "single_issue", "multi_hop", "comparison",
    "ambiguous", "no_answer", "statutory", "cross_document_strategy",
}
WORKFLOWS = {"simple_lookup", "support_adverse_strategy", "comparison", "no_answer_check"}
DIFFICULTIES = {"easy", "medium", "hard"}

# Query types whose EXPECTED agent behaviour is the deep precedent-research path.
DEEP_WORKFLOWS = {"support_adverse_strategy", "comparison"}


class GoldenQuery(BaseModel):
    query_id: str
    query: str
    query_type: str
    legal_area: str
    required_workflow: str
    expected_relevant_doc_ids: list[str] = Field(default_factory=list)
    supporting_doc_ids: list[str] = Field(default_factory=list)
    adverse_doc_ids: list[str] = Field(default_factory=list)
    neutral_doc_ids: list[str] = Field(default_factory=list)
    must_include_issues: list[str] = Field(default_factory=list)
    must_not_claim: list[str] = Field(default_factory=list)
    ideal_answer_facets: list[str] = Field(default_factory=list)
    difficulty: str = "medium"
    no_answer: bool = False
    label_source: str = "subagent"  # "deterministic" | "subagent"

    def expects_deep(self) -> bool:
        return self.required_workflow in DEEP_WORKFLOWS
