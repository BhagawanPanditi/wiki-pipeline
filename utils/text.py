"""Small text helpers shared by the stages.

These were copied between stages before. They now live here once, so a change to
how a description is shortened applies everywhere.
"""

import re


def snippet_description(text: str, max_words: int = 30) -> str:
    """Shorten a description to its first words, marking any cut with an ellipsis.

    Args:
        text: The description to shorten.
        max_words: Number of whitespace-separated words to keep.

    Returns:
        The stripped text unchanged when it has at most ``max_words`` words;
        otherwise the first ``max_words`` words followed by ``"..."``.
    """
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]) + "..."


def first_sentences(text: str, max_sentences: int = 2) -> str:
    """Return the opening sentences of a text, used when the LLM gives no brief.

    Args:
        text: The text to take sentences from.
        max_sentences: How many sentences to keep.

    Returns:
        The stripped text when it already has ``max_sentences`` sentences or
        fewer; otherwise the first ``max_sentences`` sentences joined by spaces.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if len(sentences) <= max_sentences:
        return text.strip()
    return " ".join(sentences[:max_sentences])
