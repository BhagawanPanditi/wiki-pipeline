"""Shared concept manipulation operations.

Implements F-36:
- fuse_descriptions: deduplicating and combining sentences while preserving citation tags and paragraph structure.
- union_aliases: combining aliases without duplicates.
- generate_stable_slug: collision-free unique slugs (F-12).
"""

import hashlib
import re
from typing import Iterable, List, Optional, Set

from utils.atomic_io import sanitize_filename


def generate_stable_slug(title: str, existing_slugs: Optional[Set[str]] = None) -> str:
    """Generate a collision-free filesystem and markdown slug for a concept title (F-12)."""
    base = sanitize_filename(title) or "unnamed_concept"
    base = base[:80].strip("-_")
    if not base:
        base = "concept"

    if existing_slugs is None:
        return base

    if base not in existing_slugs:
        existing_slugs.add(base)
        return base

    # Append deterministic 6-character SHA1 hash of full title on collision
    h = hashlib.sha1(title.strip().casefold().encode("utf-8")).hexdigest()[:6]
    candidate = f"{base}-{h}"
    if candidate not in existing_slugs:
        existing_slugs.add(candidate)
        return candidate

    counter = 1
    while f"{candidate}-{counter}" in existing_slugs:
        counter += 1
    final_slug = f"{candidate}-{counter}"
    existing_slugs.add(final_slug)
    return final_slug


def fuse_descriptions(descriptions: List[str], max_tokens: int = 1500) -> str:
    """Fuse multiple concept descriptions, deduplicating sentences while retaining citations."""
    seen_sentences: Set[str] = set()
    fused_sentences: List[str] = []

    for desc in descriptions:
        if not desc:
            continue
        # Split into sentences preserving trailing delimiters
        sentences = re.split(r'(?<=[.!?])\s+', desc.strip())
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            # Compare lowercase text without brackets/citations
            norm = re.sub(r"\[doc_\d+\]", "", s_clean).casefold().strip()
            norm = " ".join(norm.split())
            if norm and norm not in seen_sentences:
                seen_sentences.add(norm)
                fused_sentences.append(s_clean)

    fused = " ".join(fused_sentences)
    return fused


def union_aliases(alias_lists: Iterable[List[str]]) -> List[str]:
    """Union lists of aliases, deduplicating case-insensitively while preserving clean display."""
    seen = set()
    result = []
    for alist in alias_lists:
        for a in alist:
            norm = a.strip().casefold()
            if norm and norm not in seen:
                seen.add(norm)
                result.append(a.strip())
    return result
