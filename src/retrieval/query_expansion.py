"""Deterministic synonym expansion / broadening for retrieval (§8.4).

This is the mechanical fallback that widens recall (license -> licence/DL,
commercial vehicle -> truck/tempo/...). The richer, task-aware query generation
(support/adverse/compensation variants) is an LLM node in the agent (Step 5);
both feed the same hybrid retriever.
"""

from __future__ import annotations

import re

# canonical phrase -> extra terms that should also be searched
SYNONYMS: dict[str, list[str]] = {
    "driving licence": ["driving license", "valid licence", "dl", "licence to drive", "effective licence"],
    "fake licence": ["fake license", "forged licence", "invalid licence", "bogus licence"],
    "commercial vehicle": ["truck", "lorry", "goods carriage", "transport vehicle", "tempo", "tanker", "dumper"],
    "insurer liability": ["insurance company", "policy breach", "indemnify", "pay and recover", "liable to pay"],
    "compensation": ["just compensation", "loss of dependency", "multiplier", "future prospects", "quantum"],
    "contributory negligence": ["rash and negligent", "negligence of the deceased", "fir credibility", "burden of proof"],
    "pay and recover": ["right of recovery", "recover from the owner", "insurer recovery rights"],
    "death": ["deceased", "loss of dependency", "legal heirs", "dependants"],
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def expansion_terms(query: str) -> list[str]:
    """Extra terms triggered by phrases present in the query."""
    q = _norm(query)
    terms: list[str] = []
    for phrase, extra in SYNONYMS.items():
        if phrase in q or any(e in q for e in extra):
            for t in extra:
                if t not in q and t not in terms:
                    terms.append(t)
    return terms


def expand_query(query: str) -> str:
    """Original query enriched with triggered synonym terms (for one search)."""
    terms = expansion_terms(query)
    return query if not terms else f"{query} {' '.join(terms)}"
