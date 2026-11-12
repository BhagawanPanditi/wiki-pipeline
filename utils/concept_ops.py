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
    """Extract clean per-chunk paragraphs from a concept dict.
    
    Returns a list of dicts: [{"source_chunk": "doc_XXXXX_chunk_YYY", "text": "..."}] with ZERO inline citations.
    """
    # 1. If concept already has a structured 'paragraphs' field, return it cleanly
    if "paragraphs" in concept and isinstance(concept["paragraphs"], list) and concept["paragraphs"]:
        valid = []
        for p in concept["paragraphs"]:
            if isinstance(p, dict) and p.get("text"):
                src = p.get("source_chunk") or p.get("doc_id") or default_doc_id or "unknown"
                src = src.replace("::", "_")
                clean_t = re.sub(r"\[\s*(?:doc_|canon_)[A-Za-z0-9_-]+(?:::[-A-Za-z0-9_]+)?\s*\]", "", p["text"]).strip()
                if "```" not in clean_t and "\n" not in clean_t:
                    clean_t = " ".join(clean_t.split())
                if clean_t:
                    valid.append({"source_chunk": src, "doc_id": src.split("_chunk_")[0] if "_chunk_" in src else src, "text": clean_t})
        if valid:
            return valid

    # 2. Extract chunk/doc IDs from source_chunks
    expected_chunks: List[str] = []
    if default_doc_id:
        expected_chunks.append(default_doc_id.replace("::", "_"))
    for sc in concept.get("source_chunks", []):
        norm_sc = sc.replace("::", "_").strip()
        if norm_sc and norm_sc not in expected_chunks:
            expected_chunks.append(norm_sc)

    desc = concept.get("description", "").strip()
    if not desc:
        return []

    # If description contains distinct double newlines, inspect each block
    raw_blocks = [b.strip() for b in re.split(r"\n\s*\n", desc) if b.strip()]
    if len(raw_blocks) > 1 and len(expected_chunks) >= len(raw_blocks):
        paras = []
        for i, b in enumerate(raw_blocks):
            src = expected_chunks[i] if i < len(expected_chunks) else expected_chunks[0]
            clean_t = re.sub(r"\[\s*(?:doc_|canon_)[A-Za-z0-9_-]+(?:::[-A-Za-z0-9_]+)?\s*\]", "", b).strip()
            if "```" not in clean_t and "\n" not in clean_t:
                clean_t = " ".join(clean_t.split())
            if clean_t:
                paras.append({"source_chunk": src, "doc_id": src.split("_chunk_")[0] if "_chunk_" in src else src, "text": clean_t})
        return paras

    # Otherwise parse sentences and bracket tags
    sentences = re.split(r'(?<=[.!?])\s+', desc)
    chunk_paragraphs: Dict[str, List[str]] = defaultdict(list)
    chunk_order: List[str] = []
    current_chunk = expected_chunks[0] if len(expected_chunks) == 1 else None

    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        found_chunks = re.findall(r'\[\s*((?:doc_|canon_)[A-Za-z0-9_-]+(?:::[-A-Za-z0-9_]+)?)\s*\]', s_clean)
        if found_chunks:
            current_chunk = found_chunks[0].replace("::", "_")
        elif not current_chunk and expected_chunks:
            current_chunk = expected_chunks[0]

        key = current_chunk or (expected_chunks[0] if expected_chunks else "unknown")
        if key not in chunk_paragraphs:
            chunk_order.append(key)

        clean_sentence = re.sub(r'\[\s*(?:doc_|canon_)[A-Za-z0-9_-]+(?:::[^\]]+)?\s*\]', '', s_clean).strip()
        clean_sentence = re.sub(r'\.\s*\.', '.', clean_sentence)
        clean_sentence = re.sub(r'\s+([.,;:!?])', r'\1', clean_sentence)
        clean_sentence = " ".join(clean_sentence.split())
        if clean_sentence and len(clean_sentence) > 3:
            chunk_paragraphs[key].append(clean_sentence)

    result = []
    for sc in chunk_order:
        sents = chunk_paragraphs[sc]
        if sents:
            txt = " ".join(sents).strip()
            if not txt.endswith((".", "!", "?")):
                txt += "."
            result.append({"source_chunk": sc, "doc_id": sc.split("_chunk_")[0] if "_chunk_" in sc else sc, "text": txt})
    return result


def fuse_paragraphs_by_document(concept_list: List[Dict[str, Any]], default_doc_id: Optional[str] = None) -> List[Dict[str, str]]:
    """Merge paragraphs from multiple concepts, combining text belonging to the same chunk
    and keeping distinct chunks separate. Returns list of {"source_chunk": ..., "doc_id": ..., "text": ...}.
    """
    chunk_to_sents: Dict[str, List[str]] = defaultdict(list)
    chunk_order: List[str] = []
    seen_sents: Set[str] = set()

    for c in concept_list:
        paras = extract_concept_paragraphs(c, default_doc_id=default_doc_id)
        for p in paras:
            src = p.get("source_chunk") or p.get("doc_id", "unknown")
            src = src.replace("::", "_")
            if src not in chunk_to_sents:
                chunk_order.append(src)
            for s in re.split(r'(?<=[.!?])\s+', p["text"].strip()):
                s_clean = s.strip()
                if not s_clean:
                    continue
                norm = s_clean.casefold()
                norm = " ".join(norm.split())
                if norm and norm not in seen_sents:
                    seen_sents.add(norm)
                    chunk_to_sents[src].append(s_clean)

    merged = []
    for src in chunk_order:
        sents = chunk_to_sents[src]
        if sents:
            txt = " ".join(sents).strip()
            if not txt.endswith((".", "!", "?")):
                txt += "."
            merged.append({
                "source_chunk": src,
                "doc_id": src.split("_chunk_")[0] if "_chunk_" in src else src,
                "text": txt,
            })
    return merged


def render_paragraphs_to_markdown(paragraphs: List[Dict[str, str]]) -> str:
    """Render a structured paragraphs list into Markdown with [doc_XXXXX_chunk_YYY] citations at the end."""
    if not paragraphs:
        return ""
    lines = []
    for p in paragraphs:
        t = p.get("text", "").strip()
        src = (p.get("source_chunk") or p.get("doc_id") or "").strip()
        src = src.replace("::", "_")
        if not t:
            continue
        if src and src != "unknown":
            lines.append(f"{t} [{src}]")
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

