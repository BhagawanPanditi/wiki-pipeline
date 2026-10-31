"""Citation & Provenance Sanitization.
-------------------------------------
Utilities for stripping internal document/chunk citation tags from briefs
and standardizing document-level citations in markdown exports.
"""

import re


# Briefs are summaries, not source maps. Strip internal document/chunk references
# whether they came from the pipeline or were repeated by the language model.
_BRACKETED_DOC_CITATION = re.compile(
    r"\[\s*(?:doc_|canon_)[A-Za-z0-9_-]+(?:::[A-Za-z0-9_-]+)?\s*\]",
    re.IGNORECASE,
)
_BRACKETED_CHUNK_CITATION = re.compile(
    r"\[\s*[A-Za-z0-9_-]+::[A-Za-z0-9_-]+\s*\]"
)
_BRACKETED_CHUNK_ID = re.compile(r"\[\s*chunk_[A-Za-z0-9_-]+\s*\]", re.IGNORECASE)
_BARE_DOC_ID = re.compile(
    r"(?<![A-Za-z0-9_-])(?:doc_|canon_)[A-Za-z0-9_-]+(?:::[A-Za-z0-9_-]+)?(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)

# Any suffix after :: is chunk/source-level detail. The exported wiki cites the
# document only, e.g. [doc_00001::chunk_001] -> [doc_00001].
_CHUNK_CITATION = re.compile(r"\[\s*([A-Za-z0-9_-]+)::[A-Za-z0-9_-]+\s*\]")


def clean_brief_text(text: str) -> str:
    """Remove internal provenance IDs and return a single-line brief."""
    if not text:
        return ""

    cleaned = _BRACKETED_DOC_CITATION.sub("", text)
    cleaned = _BRACKETED_CHUNK_CITATION.sub("", cleaned)
    cleaned = _BRACKETED_CHUNK_ID.sub("", cleaned)
    cleaned = _BARE_DOC_ID.sub("", cleaned)
    # A brief is emitted as a quoted, single-line YAML scalar. Collapse all
    # whitespace (including a single newline) so embedded paragraphs cannot
    # break the frontmatter layout.
    return re.sub(r"\s+", " ", cleaned).strip()


def clean_chunk_citations(text: str) -> str:
    """Convert chunk-level bracket citations to document-level citations and deduplicate adjacent ones.
    
    Example:
        [canon_013435abc85f::chunk_000][canon_013435abc85f::chunk_001]
        -> [canon_013435abc85f]
    """
    if not text:
        return ""

    def replace_citation_groups(match):
        group = match.group(0)
        canon_ids = re.findall(r"\[\s*([A-Za-z0-9_-]+)(?:::[A-Za-z0-9_-]+)?\s*\]", group)
        unique = []
        for cid in canon_ids:
            if cid not in unique:
                unique.append(cid)
        return " " + " ".join(f"[{cid}]" for cid in unique) + " "

    pattern = r"(?:\[\s*[A-Za-z0-9_-]+(?:::[A-Za-z0-9_-]+)?\s*\]\s*)+"
    cleaned = re.sub(pattern, replace_citation_groups, text)
    cleaned = re.sub(r" +", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return cleaned.strip()
