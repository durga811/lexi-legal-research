"""Token estimation for chunk sizing.

We do not load the real XLM-RoBERTa tokenizer (heavy dependency, build-time only)
and instead use a conservative heuristic. Both sub-estimates intentionally
*over*-count relative to the real multilingual-e5-large tokenizer on English /
Indian-legal text, so a chunk whose estimate is <= N has a real token count
< N < 507 (the model's input limit). This guarantees Pinecone never silently
truncates a child chunk during embedding.
"""

from src.ingestion.config import TOKENS_PER_WORD


def estimate_tokens(text: str) -> int:
    """Return a conservative (over-)estimate of e5 tokens for ``text``."""
    words = len(text.split())
    word_based = words * TOKENS_PER_WORD
    char_based = len(text) / 3.5
    return int(max(word_based, char_based))


def words_for_tokens(n_tokens: int) -> int:
    """Inverse of the word-based estimate: words that fit in ``n_tokens``."""
    return max(1, int(n_tokens / TOKENS_PER_WORD))
