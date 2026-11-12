"""Shared concept manipulation operations.

Implements F-36:
- fuse_descriptions: deduplicating and combining sentences while preserving citation tags and paragraph structure.
- union_aliases: combining aliases without duplicates.
- generate_stable_slug: collision-free unique slugs (F-12).
"""

import hashlib
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set

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


def extract_concept_paragraphs(concept: Dict[str, Any], default_doc_id: Optional[str] = None) -> List[Dict[str, str]]:
    """Extract clean per-document paragraphs from a concept dict.
    
    Returns a list of dicts: [{"doc_id": "doc_XXXXX", "text": "..."}] with ZERO inline citations.
    """
    # 1. If concept already has a structured 'paragraphs' field, return it cleanly
    if "paragraphs" in concept and isinstance(concept["paragraphs"], list) and concept["paragraphs"]:
        valid = []
        for p in concept["paragraphs"]:
            if isinstance(p, dict) and p.get("text"):
                did = p.get("doc_id") or default_doc_id or "unknown"
                clean_t = re.sub(r"\[\s*(?:doc_|canon_)[A-Za-z0-9_-]+(?:::[-A-Za-z0-9_]+)?\s*\]", "", p["text"]).strip()
                clean_t = " ".join(clean_t.split())
                if clean_t:
                    valid.append({"doc_id": did, "text": clean_t})
        if valid:
            return valid

    # 2. Extract doc_id from source_chunks
    expected_doc_ids: List[str] = []
    if default_doc_id:
        expected_doc_ids.append(default_doc_id)
    for sc in concept.get("source_chunks", []):
        did = sc.split("::")[0] if "::" in sc else sc
        if did and did not in expected_doc_ids:
            expected_doc_ids.append(did)

    desc = concept.get("description", "").strip()
    if not desc:
        return []

    # If description contains distinct double newlines, inspect each block
    raw_blocks = [b.strip() for b in re.split(r"\n\s*\n", desc) if b.strip()]
    if len(raw_blocks) > 1 and len(expected_doc_ids) >= len(raw_blocks):
        paras = []
        for i, b in enumerate(raw_blocks):
            did = expected_doc_ids[i] if i < len(expected_doc_ids) else expected_doc_ids[0]
            clean_t = re.sub(r"\[\s*(?:doc_|canon_)[A-Za-z0-9_-]+(?:::[-A-Za-z0-9_]+)?\s*\]", "", b).strip()
            clean_t = " ".join(clean_t.split())
            if clean_t:
                paras.append({"doc_id": did, "text": clean_t})
        return paras

    # Otherwise parse sentences and bracket tags
    sentences = re.split(r'(?<=[.!?])\s+', desc)
    doc_paragraphs: Dict[str, List[str]] = defaultdict(list)
    doc_order: List[str] = []
    current_doc = expected_doc_ids[0] if len(expected_doc_ids) == 1 else None

    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        found_docs = re.findall(r'\[\s*((?:doc_|canon_)[A-Za-z0-9_-]+)(?:::[^\]]+)?\s*\]', s_clean)
        if found_docs:
            current_doc = found_docs[0]
        elif not current_doc and expected_doc_ids:
            current_doc = expected_doc_ids[0]

        key = current_doc or (expected_doc_ids[0] if expected_doc_ids else "unknown")
        if key not in doc_paragraphs:
            doc_order.append(key)

        clean_sentence = re.sub(r'\[\s*(?:doc_|canon_)[A-Za-z0-9_-]+(?:::[^\]]+)?\s*\]', '', s_clean).strip()
        clean_sentence = re.sub(r'\.\s*\.', '.', clean_sentence)
        clean_sentence = re.sub(r'\s+([.,;:!?])', r'\1', clean_sentence)
        clean_sentence = " ".join(clean_sentence.split())
        if clean_sentence and len(clean_sentence) > 3:
            doc_paragraphs[key].append(clean_sentence)

    result = []
    for did in doc_order:
        sents = doc_paragraphs[did]
        if sents:
            txt = " ".join(sents).strip()
            if not txt.endswith((".", "!", "?")):
                txt += "."
            result.append({"doc_id": did, "text": txt})
    return result


def fuse_paragraphs_by_document(concept_list: List[Dict[str, Any]], default_doc_id: Optional[str] = None) -> List[Dict[str, str]]:
    """Merge paragraphs from multiple concepts, combining text belonging to the same doc_id
    and keeping distinct doc_ids separate. Returns list of {"doc_id": ..., "text": ...}.
    """
    doc_to_sents: Dict[str, List[str]] = defaultdict(list)
    doc_order: List[str] = []
    seen_sents: Set[str] = set()

    for c in concept_list:
        paras = extract_concept_paragraphs(c, default_doc_id=default_doc_id)
        for p in paras:
            did = p["doc_id"]
            if did not in doc_to_sents:
                doc_order.append(did)
            for s in re.split(r'(?<=[.!?])\s+', p["text"].strip()):
                s_clean = s.strip()
                if not s_clean:
                    continue
                norm = s_clean.casefold()
                norm = " ".join(norm.split())
                if norm and norm not in seen_sents:
                    seen_sents.add(norm)
                    doc_to_sents[did].append(s_clean)

    merged = []
    for did in doc_order:
        sents = doc_to_sents[did]
        if sents:
            txt = " ".join(sents).strip()
            if not txt.endswith((".", "!", "?")):
                txt += "."
            merged.append({"doc_id": did, "text": txt})
    return merged


def render_paragraphs_to_markdown(paragraphs: List[Dict[str, str]]) -> str:
    """Render a structured paragraphs list into Markdown with [doc_XXXXX] citations at the end."""
    if not paragraphs:
        return ""
    lines = []
    for p in paragraphs:
        t = p.get("text", "").strip()
        did = p.get("doc_id", "").strip()
        if not t:
            continue
        if did and did != "unknown":
            lines.append(f"{t} [{did}]")
        else:
            lines.append(t)
    return "\n\n".join(lines)


def get_concept_full_text(concept: Dict[str, Any]) -> str:
    """Get clean plain prose text of a concept without inline doc citations.
    Prefers concept['paragraphs'], falls back to concept['description'].
    """
    if "paragraphs" in concept and concept["paragraphs"]:
        return " ".join(p.get("text", "").strip() for p in concept["paragraphs"] if p.get("text", "").strip())
    desc = concept.get("description", "")
    # Clean any accidental inline citations
    clean = re.sub(r"\[\s*(?:doc_|canon_)[A-Za-z0-9_-]+(?:::[^\]]+)?\s*\]", "", desc)
    clean = re.sub(r"\.\s*\.", ".", clean)
    clean = re.sub(r"\s+([.,;:!?])", r"\1", clean)
    return " ".join(clean.split())

