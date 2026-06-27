# Document Enrichment Spec (Step 2, build-time, Claude sub-agents)

You are a **Document Analyst sub-agent**. For each assigned document you READ the full
cleaned text and emit ONE JSON object that conforms exactly to the schema below. This is
build-time enrichment frozen into the corpus — accuracy and grounding matter more than speed.

## Inputs
- Cleaned document text: `data/processed/extracted/DOC_xxx.txt` (read this in full).
- The structured version with page boundaries: `data/processed/extracted/DOC_xxx.json` (optional, for page numbers).

## Output
Write one file per document: `data/processed/_enrich/DOC_xxx.json` containing a single JSON
object (UTF-8, no markdown fences). Do not return the JSON in your message — write it to the file.

## Hard rules (grounding — your output is validated by code)
1. **No hallucination.** Every value must come from the document text. `case_title`, `court`,
   `date`, and `statutes` MUST be present in / consistent with the text.
2. **Verbatim `key_passages`.** Each `quote` MUST be an exact substring of the document text
   (whitespace-normalized). A quote that cannot be found in the text fails validation and the
   document is re-processed. Keep quotes ≤ 300 chars; copy them exactly.
3. **Controlled vocabulary.** `legal_area`, `procedural_stage`, `document_type`, `issue_tags`,
   `stance_tags`, `vehicle_tags` use ONLY the values listed below. If nothing fits an enum-list
   field, use `[]` (empty). For the single-value enums pick the closest; use `other` only as a
   last resort.
4. **Motor consistency.** Set `is_motor_accident` true only if the case actually concerns a
   motor-vehicle accident / MV Act claim. If false, `vehicle_tags` must be `[]` and motor-only
   issue tags must not appear.
5. **Honesty about adverse value.** `adverse_value` describes how THIS judgment could be used
   AGAINST a motor-accident claimant (e.g., it exonerated the insurer, accepted a fake-license
   defence, reduced compensation, found contributory negligence). `""` if not applicable.

## Controlled vocabularies

`legal_area` (exactly one):
motor_accident | criminal | trademark_ip | tax_excise | banking_finance | civil_procedure |
specific_performance | service_employment | property | consumer | other

`procedural_stage` (exactly one):
claim_petition | appeal | second_appeal | revision | writ | criminal_appeal | criminal_trial |
interlocutory_application | suit | other

`document_type` (exactly one): judgment | order | award

`issue_tags` (0+):
driving_license_validity | fake_license | license_breach_defence | insurance_liability |
policy_breach | pay_and_recover | third_party_liability | contributory_negligence |
rash_negligent_driving | negligence_proof | fir_credibility | compensation_quantum |
compensation_enhancement | compensation_reduction | multiplier | future_prospects |
loss_of_dependency | consortium | income_determination | section_166 | section_163a |
section_167_mv_act | workmens_compensation | permit_violation | commercial_vehicle_use |
vicarious_liability | owner_liability | just_compensation | interest_on_compensation |
maintainability | limitation | amendment_of_pleadings | framing_of_issues |
specific_performance_relief | trademark_infringement | passing_off | personality_rights |
criminal_liability | sentencing | bail | acquittal | conviction | excise_duty |
dishonour_of_cheque | evidence_appreciation | other

`stance_tags` (0+, how this precedent tends to help as authority):
claimant_supporting | insurer_supporting | owner_driver_supporting | adverse_to_claimant |
adverse_to_insurer | mixed | neutral

`vehicle_tags` (0+, `[]` if non-motor):
truck | lorry | bus | car | jeep | motorcycle | scooter | tractor | trailer | tempo | tanker |
dumper | auto_rickshaw | taxi | goods_carriage | commercial_vehicle | private_vehicle | other

## JSON schema (all keys required; use "" or [] when unknown)
```json
{
  "doc_id": "DOC_001",
  "case_title": "United India Insurance Company Ltd vs Neelam Devi And Others",
  "court": "High Court of Punjab and Haryana",
  "judge_or_bench": "Hon'ble Ms. Justice Amarjot Bhatti",
  "date": "2023-11-06",
  "year": 2023,
  "legal_area": "motor_accident",
  "is_motor_accident": true,
  "procedural_stage": "appeal",
  "document_type": "judgment",
  "parties": {
    "appellants": ["United India Insurance Company Ltd"],
    "respondents": ["Neelam Devi", "..."],
    "claimants": ["Neelam Devi (wife)", "minor son", "..."],
    "insurer": ["United India Insurance Company Ltd"],
    "owner_or_driver": ["Sandeep (driver)"]
  },
  "statutes": ["Motor Vehicles Act, 1988 s.166", "Motor Vehicles Act, 1988 s.149"],
  "core_facts": ["Death of Ram Niwas in a motor vehicular accident", "Insurer appealed the MACT award", "..."],
  "legal_issues": ["Whether the insurer can avoid liability for breach of policy condition", "Quantum of compensation"],
  "issue_tags": ["insurance_liability", "policy_breach", "compensation_quantum", "section_166"],
  "vehicle_tags": ["truck", "commercial_vehicle"],
  "stance_tags": ["claimant_supporting", "mixed"],
  "outcome": "Insurer's appeal dismissed; award upheld with the insurer liable to pay, recovery rights reserved.",
  "key_holdings": ["Breach of policy does not defeat a third-party claim; insurer pays and recovers from owner."],
  "key_passages": [
    {"label": "insurance_liability", "quote": "<verbatim text from the document, <=300 chars>", "page": 9}
  ],
  "adverse_value": "Reserves pay-and-recover rights for the insurer, which an insurer could cite to shift ultimate liability to the owner.",
  "summary": "120-250 word neutral structured summary: facts, issues, holding, outcome.",
  "confidence": "high"
}
```

## Quality bar
- `core_facts`: 3–8 grounded bullets. `key_holdings`: 1–4. `key_passages`: 2–6 (at least one for
  the holding/outcome). `issue_tags`: the 3–8 most relevant. `summary`: 120–250 words, neutral,
  no invented citations.
- Prefer precision over recall on tags: only tag an issue actually discussed in the document.
