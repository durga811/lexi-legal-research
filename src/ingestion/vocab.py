"""Controlled vocabularies for build-time enrichment + the grounding lexicons used
by the deterministic chunk labeler. Single source of truth for both the sub-agent
validation (Step 2 assembly) and chunk tagging."""

LEGAL_AREAS = frozenset({
    "motor_accident", "criminal", "trademark_ip", "tax_excise", "banking_finance",
    "civil_procedure", "specific_performance", "service_employment", "property",
    "consumer", "other",
})

PROCEDURAL_STAGES = frozenset({
    "claim_petition", "appeal", "second_appeal", "revision", "writ", "criminal_appeal",
    "criminal_trial", "interlocutory_application", "suit", "other",
})

DOCUMENT_TYPES = frozenset({"judgment", "order", "award"})

ISSUE_TAGS = frozenset({
    "driving_license_validity", "fake_license", "license_breach_defence", "insurance_liability",
    "policy_breach", "pay_and_recover", "third_party_liability", "contributory_negligence",
    "rash_negligent_driving", "negligence_proof", "fir_credibility", "compensation_quantum",
    "compensation_enhancement", "compensation_reduction", "multiplier", "future_prospects",
    "loss_of_dependency", "consortium", "income_determination", "section_166", "section_163a",
    "section_167_mv_act", "workmens_compensation", "permit_violation", "commercial_vehicle_use",
    "vicarious_liability", "owner_liability", "just_compensation", "interest_on_compensation",
    "maintainability", "limitation", "amendment_of_pleadings", "framing_of_issues",
    "specific_performance_relief", "trademark_infringement", "passing_off", "personality_rights",
    "criminal_liability", "sentencing", "bail", "acquittal", "conviction", "excise_duty",
    "dishonour_of_cheque", "evidence_appreciation", "other",
})

STANCE_TAGS = frozenset({
    "claimant_supporting", "insurer_supporting", "owner_driver_supporting", "adverse_to_claimant",
    "adverse_to_insurer", "mixed", "neutral",
})

VEHICLE_TAGS = frozenset({
    "truck", "lorry", "bus", "car", "jeep", "motorcycle", "scooter", "tractor", "trailer",
    "tempo", "tanker", "dumper", "auto_rickshaw", "taxi", "goods_carriage", "commercial_vehicle",
    "private_vehicle", "other",
})

CONFIDENCE = frozenset({"high", "medium", "low"})

# --------------------------------------------------------------------------- #
# Grounding lexicons: tag -> trigger terms (lowercased substrings). Used to
# (a) verify a doc-level tag is supported somewhere in the text, and (b) attach a
# tag to a child chunk only when the chunk's own text contains a trigger term.
# --------------------------------------------------------------------------- #
ISSUE_LEXICON: dict[str, list[str]] = {
    "driving_license_validity": ["driving licence", "driving license", "valid licence", "valid license", "dl ", "licence to drive"],
    "fake_license": ["fake licence", "fake license", "forged licence", "forged license", "bogus licence", "invalid licence"],
    "license_breach_defence": ["breach of policy", "licence condition", "without a valid", "not holding a valid"],
    "insurance_liability": ["insurer", "insurance company", "indemnify", "liability of the insurance", "liable to pay", "policy of insurance"],
    "policy_breach": ["breach of policy", "policy condition", "violation of the policy", "terms of the policy"],
    "pay_and_recover": ["pay and recover", "recover the amount", "right of recovery", "recovery from the owner"],
    "third_party_liability": ["third party", "third-party", "statutory liability"],
    "contributory_negligence": ["contributory negligence", "contributed to the accident", "negligence on the part of the deceased"],
    "rash_negligent_driving": ["rash and negligent", "rashly and negligently", "rash driving", "negligent driving"],
    "negligence_proof": ["burden to prove", "burden of proof", "prove negligence", "preponderance of probabilities"],
    "fir_credibility": ["fir", "first information report", "one-sided", "delay in lodging"],
    "compensation_quantum": ["compensation", "quantum of compensation", "amount of compensation", "awarded a sum"],
    "compensation_enhancement": ["enhance the compensation", "enhanced", "enhancement of compensation"],
    "compensation_reduction": ["reduce the compensation", "reduced", "excessive compensation"],
    "multiplier": ["multiplier"],
    "future_prospects": ["future prospects"],
    "loss_of_dependency": ["loss of dependency", "dependency", "loss of dependence"],
    "consortium": ["consortium", "loss of consortium"],
    "income_determination": ["monthly income", "income of the deceased", "salary", "earning"],
    "section_166": ["section 166", "u/s 166", "under section 166"],
    "section_163a": ["section 163a", "163-a", "163a"],
    "section_167_mv_act": ["section 167", "u/s 167"],
    "workmens_compensation": ["workmen's compensation", "workmen compensation", "employees compensation", "workmen's compensation act"],
    "permit_violation": ["permit", "route permit", "without permit", "violation of permit"],
    "commercial_vehicle_use": ["commercial vehicle", "goods carriage", "transport vehicle", "goods vehicle"],
    "vicarious_liability": ["vicarious", "vicariously liable"],
    "owner_liability": ["owner of the vehicle", "liability of the owner", "owner is liable"],
    "just_compensation": ["just compensation", "just and reasonable"],
    "interest_on_compensation": ["interest", "per annum", "rate of interest"],
    "maintainability": ["maintainability", "not maintainable", "maintainable"],
    "limitation": ["limitation", "barred by time", "condonation of delay"],
    "amendment_of_pleadings": ["amendment", "amend the", "order vi rule 17", "written statement"],
    "framing_of_issues": ["framing of issues", "issues framed", "framed the issue"],
    "specific_performance_relief": ["specific performance", "agreement to sell", "sale agreement"],
    "trademark_infringement": ["trade mark", "trademark", "infringement of trademark", "infringement"],
    "passing_off": ["passing off", "passing-off"],
    "personality_rights": ["personality right", "publicity right", "celebrity"],
    "criminal_liability": ["accused", "offence", "convicted", "guilty"],
    "sentencing": ["sentence", "rigorous imprisonment", "sentenced to"],
    "bail": ["bail"],
    "acquittal": ["acquit", "acquitted", "acquittal"],
    "conviction": ["convict", "convicted", "conviction"],
    "excise_duty": ["excise", "cenvat", "duty"],
    "dishonour_of_cheque": ["dishonour", "cheque", "section 138", "negotiable instruments"],
    "evidence_appreciation": ["appreciation of evidence", "examined", "deposed", "testimony"],
}

VEHICLE_LEXICON: dict[str, list[str]] = {
    "truck": ["truck"],
    "lorry": ["lorry"],
    "bus": ["bus "],
    "car": ["car ", "motor car"],
    "jeep": ["jeep"],
    "motorcycle": ["motorcycle", "motor cycle", "two wheeler", "two-wheeler", "bike"],
    "scooter": ["scooter"],
    "tractor": ["tractor"],
    "trailer": ["trailer"],
    "tempo": ["tempo"],
    "tanker": ["tanker"],
    "dumper": ["dumper"],
    "auto_rickshaw": ["auto rickshaw", "auto-rickshaw", "autorickshaw", "three wheeler"],
    "taxi": ["taxi", "cab "],
    "goods_carriage": ["goods carriage", "goods vehicle", "goods carrier"],
    "commercial_vehicle": ["commercial vehicle", "transport vehicle"],
    "private_vehicle": ["private vehicle", "private car"],
}

# Chunk-type classification cues (checked roughly top-to-bottom; first/strongest wins).
CHUNK_TYPE_CUES: list[tuple[str, list[str]]] = [
    ("compensation_calculation", ["multiplier", "future prospects", "loss of dependency", "deduction towards", "personal expenses", "loss of estate", "funeral expenses", "loss of consortium"]),
    ("final_order", ["it is ordered", "appeal is allowed", "appeal is dismissed", "petition is allowed", "petition is dismissed", "in the result", "disposed of", "we direct", "hereby directed"]),
    ("holding", ["we are of the view", "we are of the opinion", "held that", "in our opinion", "court held", "we hold"]),
    ("driving_license_finding", ["driving licence", "driving license", "fake licence", "valid licence"]),
    ("negligence_finding", ["rash and negligent", "contributory negligence", "negligence of"]),
    ("issues_framed", ["framing of issues", "issues for determination", "the following issues", "point for consideration"]),
    ("legal_principle", ["supreme court", "apex court", "laid down", "it is settled", "ratio", "precedent"]),
    ("insurer_argument", ["learned counsel for the insurer", "counsel for the insurance", "insurer contended", "insurance company submitted"]),
    ("claimant_argument", ["learned counsel for the claimant", "claimant contended", "counsel for the appellant", "petitioner submitted"]),
    ("evidence", ["pw-", "examined", "deposed", "testimony", "exhibit", "marked as ex"]),
    ("procedural_history", ["tribunal", "trial court", "lower court", "impugned", "appeal arises"]),
    ("facts", ["brief facts", "facts of the case", "case of the claimant", "on the fateful day"]),
]

# Stance cue lexicon (chunk-grounded).
STANCE_CUES: dict[str, list[str]] = {
    "adverse_to_claimant": ["insurer is exonerated", "insurance company is exonerated", "not liable to pay",
                             "contributory negligence", "fake licence", "compensation is reduced",
                             "policy is void", "claim is dismissed", "not maintainable"],
    "adverse_to_insurer": ["insurer is liable", "insurance company is liable", "liable to indemnify",
                            "breach of policy does not", "directed to pay the compensation",
                            "insurer cannot avoid"],
    "claimant_supporting": ["compensation is enhanced", "entitled to compensation", "just compensation",
                             "award is upheld", "claimants are entitled"],
    "insurer_supporting": ["pay and recover", "right of recovery", "recover the amount from the owner"],
}
