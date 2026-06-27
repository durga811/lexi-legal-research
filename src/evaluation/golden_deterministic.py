"""Deterministic golden queries whose expected_relevant_doc_ids are computed from
the validated metadata predicates (Step 2). These are the most accurate possible
labels — exact set membership — and anchor the golden set for factual / metadata /
statutory / no-answer behaviour.
"""

from __future__ import annotations

from src.evaluation.golden_schema import GoldenQuery
from src.retrieval.corpus_store import load_corpus

COMMERCIAL_VEHICLE_TAGS = {"commercial_vehicle", "truck", "lorry", "bus", "tempo",
                           "tanker", "dumper", "goods_carriage"}


def _meta() -> list[dict]:
    return list(load_corpus()["docmeta"].values())


def by_issue(tag: str) -> list[str]:
    return sorted(m["doc_id"] for m in _meta() if tag in m["issue_tags"])


def by_vehicle_any(tags: set[str]) -> list[str]:
    return sorted(m["doc_id"] for m in _meta() if set(m["vehicle_tags"]) & tags)


def by_area(area: str) -> list[str]:
    return sorted(m["doc_id"] for m in _meta() if m["legal_area"] == area)


def non_motor() -> list[str]:
    return sorted(m["doc_id"] for m in _meta() if not m["is_motor_accident"])


def deterministic_queries() -> list[GoldenQuery]:
    q: list[GoldenQuery] = []

    q.append(GoldenQuery(
        query_id="g_det_01",
        query="Which judgments involve a commercial or goods vehicle such as a truck, lorry, bus, tempo, tanker or goods carriage?",
        query_type="factual_metadata", legal_area="motor_accident", required_workflow="simple_lookup",
        expected_relevant_doc_ids=by_vehicle_any(COMMERCIAL_VEHICLE_TAGS),
        must_include_issues=["commercial vehicle"],
        must_not_claim=["every motor accident case involves a commercial vehicle"],
        ideal_answer_facets=["lists matching doc_ids", "names the vehicle type per case"],
        difficulty="easy", label_source="deterministic"))

    q.append(GoldenQuery(
        query_id="g_det_02",
        query="Which documents in this corpus are NOT motor accident cases?",
        query_type="factual_metadata", legal_area="mixed", required_workflow="simple_lookup",
        expected_relevant_doc_ids=non_motor(),
        must_not_claim=["all documents are motor accident cases"],
        ideal_answer_facets=["identifies non-motor docs", "states their legal area"],
        difficulty="medium", label_source="deterministic"))

    q.append(GoldenQuery(
        query_id="g_det_03",
        query="Which cases specifically discuss a fake driving licence?",
        query_type="single_issue", legal_area="motor_accident", required_workflow="simple_lookup",
        expected_relevant_doc_ids=by_issue("fake_license"),
        must_include_issues=["fake licence"],
        must_not_claim=["a fake licence always exonerates the insurer"],
        ideal_answer_facets=["identifies fake-licence cases", "distinguishes from mere expiry"],
        difficulty="medium", label_source="deterministic"))

    q.append(GoldenQuery(
        query_id="g_det_04",
        query="Which cases involve contributory negligence?",
        query_type="single_issue", legal_area="motor_accident", required_workflow="simple_lookup",
        expected_relevant_doc_ids=by_issue("contributory_negligence"),
        must_include_issues=["contributory negligence"],
        ideal_answer_facets=["lists contributory-negligence cases"],
        difficulty="medium", label_source="deterministic"))

    q.append(GoldenQuery(
        query_id="g_det_05",
        query="Which judgments are trademark or intellectual-property cases?",
        query_type="factual_metadata", legal_area="trademark_ip", required_workflow="simple_lookup",
        expected_relevant_doc_ids=by_area("trademark_ip"),
        must_not_claim=["the corpus is only about motor accidents"],
        ideal_answer_facets=["identifies the trademark/IP docs"],
        difficulty="easy", label_source="deterministic"))

    q.append(GoldenQuery(
        query_id="g_det_06",
        query="Which case interprets Section 167 of the Motor Vehicles Act (election between the Motor Vehicles Act and the Workmen's Compensation Act)?",
        query_type="statutory", legal_area="motor_accident", required_workflow="simple_lookup",
        expected_relevant_doc_ids=by_issue("section_167_mv_act"),
        must_include_issues=["Section 167", "Workmen's Compensation"],
        ideal_answer_facets=["identifies the Section 167 case", "explains the election bar"],
        difficulty="hard", label_source="deterministic"))

    q.append(GoldenQuery(
        query_id="g_det_07",
        query="Which case involves amendment of pleadings or amendment of a written statement?",
        query_type="procedural", legal_area="civil_procedure", required_workflow="simple_lookup",
        expected_relevant_doc_ids=by_issue("amendment_of_pleadings"),
        must_include_issues=["amendment of pleadings"],
        ideal_answer_facets=["identifies the amendment case"],
        difficulty="medium", label_source="deterministic"))

    # No-answer queries — verified empty against the metadata.
    q.append(GoldenQuery(
        query_id="g_det_08",
        query="Find trademark dilution cases that also involve a commercial motor vehicle accident.",
        query_type="no_answer", legal_area="mixed", required_workflow="no_answer_check",
        expected_relevant_doc_ids=[], no_answer=True,
        must_not_claim=["a corpus case combines trademark dilution with a motor accident"],
        ideal_answer_facets=["states no such case exists", "does not hallucinate"],
        difficulty="hard", label_source="deterministic"))

    q.append(GoldenQuery(
        query_id="g_det_09",
        query="Find Supreme Court cyber-fraud sentencing cases arising from a motor accident.",
        query_type="no_answer", legal_area="mixed", required_workflow="no_answer_check",
        expected_relevant_doc_ids=[], no_answer=True,
        must_not_claim=["the corpus contains a cyber-fraud motor-accident case"],
        ideal_answer_facets=["states the corpus lacks such a case", "may point to closest material"],
        difficulty="hard", label_source="deterministic"))

    return q
