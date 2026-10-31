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
            norm = re.sub(r"\[(?:doc_|canon_)[A-Za-z0-9_-]+(?:::[A-Za-z0-9_-]+)?\]", "", s_clean).casefold().strip()
            norm = " ".join(norm.split())
            if norm and norm not in seen_sentences:
                seen_sentences.add(norm)
                fused_sentences.append(s_clean)

    fused = " ".join(fused_sentences)
    return fused


def fuse_descriptions_by_document(concept_list: List[Dict[str, Any]]) -> str:
    """Fuse concept descriptions keeping extractions from each document as separate paragraphs,
    with document citations positioned at the end of each respective paragraph.
    """
    doc_to_descs: Dict[str, List[str]] = defaultdict(list)
    doc_order: List[str] = []

    for c in concept_list:
        doc_id = None
        for sc in c.get("source_chunks", []):
            if sc:
                doc_id = sc.split("::")[0] if "::" in sc else sc
                break
        if not doc_id:
            # Fallback check for in-text citation
            m = re.search(r"\[\s*([A-Za-z0-9_-]+)(?:::[^\]]+)?\s*\]", c.get("description", ""))
            if m:
                doc_id = m.group(1)

        key = doc_id or "unknown"
        if key not in doc_to_descs:
            doc_order.append(key)
        desc = c.get("description", "").strip()
        if desc:
            doc_to_descs[key].append(desc)

    paragraphs = []
    for doc_id in doc_order:
        descs = doc_to_descs[doc_id]
        seen_sents: Set[str] = set()
        sents: List[str] = []

        for d in descs:
            for s in re.split(r'(?<=[.!?])\s+', d.strip()):
                s_clean = s.strip()
                if not s_clean:
                    continue
                # Strip raw chunk citations like [doc_...::chunk_...]
                s_clean = re.sub(r"\[\s*(?:doc_|canon_)[A-Za-z0-9_-]+::[A-Za-z0-9_-]+\s*\]", "", s_clean).strip()
                norm = re.sub(r"\[\s*(?:doc_|canon_)[A-Za-z0-9_-]+\s*\]", "", s_clean).casefold().strip()
                norm = " ".join(norm.split())
                if norm and norm not in seen_sents:
                    seen_sents.add(norm)
                    sents.append(s_clean)

        if not sents:
            continue

        para = " ".join(sents).strip()
        para = re.sub(r"\s+([.,;:!?])", r"\1", para)

        # Place the document citation cleanly at the end of this document's paragraph
        if doc_id != "unknown" and f"[{doc_id}]" not in para:
            para = f"{para} [{doc_id}]"

        paragraphs.append(para)

    return "\n\n".join(paragraphs)


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
