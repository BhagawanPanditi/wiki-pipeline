"""Shared head-noun normalization and title key generation.

Normalizes only the head noun (last word of noun phrase) deterministically
using spaCy (en_core_web_sm) with parser and NER disabled for high throughput.
No domain-specific stoplists or hardcoded noun lists.
"""

from functools import lru_cache

_nlp_model = None


def _nlp():
    """Load and cache the spaCy pipeline. Fails loud if missing."""
    global _nlp_model
    if _nlp_model is None:
        try:
            import spacy
            _nlp_model = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except Exception as e:
            raise RuntimeError(
                f"spaCy or the 'en_core_web_sm' model is missing: {e}. "
                "Install via: pip install spacy && python -m spacy download en_core_web_sm"
            ) from e
    return _nlp_model


@lru_cache(maxsize=100000)
def normalize_head_noun(term: str) -> str:
    """Normalize the head noun (last word) of a concept title or entity.

    Leaves leading modifiers untouched, lowercases, and lemmatizes the final token.
    Deterministic, domain-agnostic, and idempotent.
    """
    if not term:
        return ""

    words = " ".join(term.strip().split()).casefold().split()
    if not words:
        return ""

    nlp_pipeline = _nlp()
    # Prefix with 'the ' so the tagger reliably treats the final token as a noun
    doc = nlp_pipeline(f"the {words[-1]}")
    if len(doc) >= 2 and doc[1].lemma_:
        words[-1] = doc[1].lemma_.casefold()

    return " ".join(words)


@lru_cache(maxsize=100000)
def canonical_title_key(title: str) -> str:
    """Produce a canonical key for candidate grouping across titles."""
    return normalize_head_noun(title)

