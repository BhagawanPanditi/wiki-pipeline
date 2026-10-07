"""Small helpers for keeping provenance out of briefs and readable in exports."""

import re


# Briefs are summaries, not source maps. Strip internal document/chunk references
# whether they came from the pipeline or were repeated by the language model.
_BRACKETED_CANON_CITATION = re.compile(
    r"\[\s*canon_[A-Za-z0-9_-]+(?:::[A-Za-z0-9_-]+)?\s*\]",
    re.IGNORECASE,
)
_BRACKETED_CHUNK_CITATION = re.compile(
    r"\[\s*[A-Za-z0-9_-]+::[A-Za-z0-9_-]+\s*\]"
)
_BRACKETED_CHUNK_ID = re.compile(r"\[\s*chunk_[A-Za-z0-9_-]+\s*\]", re.IGNORECASE)
_BARE_CANON_ID = re.compile(
    r"(?<![A-Za-z0-9_-])canon_[A-Za-z0-9_-]+(?:::[A-Za-z0-9_-]+)?(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)

# Any suffix after :: is chunk/source-level detail. The exported wiki cites the
# canonical document only, e.g. [canon_abc::chunk_001] -> [canon_abc].
_CHUNK_CITATION = re.compile(r"\[\s*([A-Za-z0-9_-]+)::[A-Za-z0-9_-]+\s*\]")


def clean_brief_text(text: str) -> str:
    """Remove internal provenance IDs and return a single-line brief."""
    if not text:
        return ""

    cleaned = _BRACKETED_CANON_CITATION.sub("", text)
    cleaned = _BRACKETED_CHUNK_CITATION.sub("", cleaned)
    cleaned = _BRACKETED_CHUNK_ID.sub("", cleaned)
    cleaned = _BARE_CANON_ID.sub("", cleaned)
    # A brief is emitted as a quoted, single-line YAML scalar. Collapse all
    # whitespace (including a single newline) so embedded paragraphs cannot
    # break the frontmatter layout.
    return re.sub(r"\s+", " ", cleaned).strip()


def clean_chunk_citations(text: str) -> str:
    """Convert chunk-level bracket citations to document-level citations."""
    if not text:
        return ""
    return _CHUNK_CITATION.sub(r"[\1]", text)
