#!/usr/bin/env python3
"""
Step 5: Export Knowledge Graph to Obsidian / Markdown Wiki Format
-----------------------------------------------------------------
Generates clean Markdown concept files in `concepts_md/` directory:
- Filename: Sluggified canonical concept title (e.g. `closed_eye_hallucination.md`)
- YAML Frontmatter:
    title: str
    aliases: list
    brief: str
    entities: list
    sources: list (Document-level canonical document IDs, e.g. `canon_72994d7c713a`)
- Body:
    The comprehensive concept description paragraphs cited at the document level:
    "... factual statements ... [canon_72994d7c713a]"
    Stub resolution sentences ("Additionally, it is associated with...") separated
    into a distinct, dedicated final paragraph with a gap.
- # Related Section:
    - [[target_slug|Canonical Title]] - 1-2 sentence description of the relationship.
"""

import re
import json
from pathlib import Path
from typing import List
from tqdm import tqdm

from utils.atomic_io import atomic_write_json
from utils.cli import build_parser

STAGE = "export-markdown"
# Lists the pages the last export wrote, so the next export removes exactly those.
EXPORT_MANIFEST = ".export_manifest.json"
from utils.citation_utils import (
    clean_brief_text,
    clean_chunk_citations as normalize_chunk_citations,
)

def sluggify(title: str) -> str:
    """Creates a clean, filesystem-safe filename for Markdown files."""
    s = title.strip().lower()
    s = re.sub(r'[^a-z0-9\s_-]', '', s)
    s = re.sub(r'[\s-]+', '_', s)
    s = s.strip('_')
    return s[:120] if s else "unnamed_concept"

def escape_yaml_string(s: str) -> str:
    """Escapes strings for YAML frontmatter."""
    escaped = s.replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'

def extract_doc_id(sc: str) -> str:
    """Extract a document ID from a source chunk (canon_123::chunk_001 -> canon_123)."""
    if not isinstance(sc, str) or not sc.strip():
        return ""
    return sc.split("::", 1)[0].strip()

def clean_brief(brief: str) -> str:
    """Remove provenance IDs and flatten the brief to one line."""
    return clean_brief_text(brief)

def clean_chunk_citations(text: str) -> str:
    """Convert chunk-level citations such as [canon_x::chunk_001] to [canon_x]."""
    return normalize_chunk_citations(text)

def deduplicate_paragraph_citations(text: str) -> str:
    """Deduplicates repeated citations like [canon_1] [canon_1] [canon_1] -> [canon_1] at paragraph end."""
    trailing_cites = list(re.finditer(r'((?:\[[a-zA-Z0-9_-]+\]\s*)+)$', text))
    if trailing_cites:
        c_span = trailing_cites[0].span()
        cite_blob = text[c_span[0]:c_span[1]]
        doc_ids = sorted(list(set(re.findall(r'\[([a-zA-Z0-9_-]+)\]', cite_blob))))
        body_before = text[:c_span[0]].strip()
        dedup_cites_str = " ".join(f"[{d}]" for d in doc_ids)
        return f"{body_before} {dedup_cites_str}".strip() if body_before else dedup_cites_str
    return text

def format_description_paragraphs(description: str, source_docs: List[str]) -> str:
    """
    Format description by grouping text by source document.

    All paragraphs from a single document are concatenated with a space (no newlines),
    ending with that document's citation [canon_xxx]. Distinct documents form separate
    paragraphs separated by double newlines.
    """
    clean_desc = clean_chunk_citations(description)
    raw_paras = clean_desc.split("\n\n")
    cleaned_paras = []
    additionally_sentences = []

    for p in raw_paras:
        p_str = p.strip()
        if not p_str:
            continue

        # Check for inline or appended "Additionally..." stub absorption sentences.
        add_matches = list(re.finditer(r'(?:^|\s+)(Additionally,\s+.*?)(?=(?:\s+Additionally,)|$)', p_str))
        if add_matches:
            first_add_idx = add_matches[0].start()
            main_text = p_str[:first_add_idx].strip()
            additionally_sentences.extend(m.group(1).strip() for m in add_matches)
        else:
            main_text = p_str

        if main_text:
            cleaned_paras.append(main_text)

    # Normalize known document IDs from source_chunks
    normalized_source_docs = sorted(list({
        extract_doc_id(sc)
        for sc in source_docs
        if extract_doc_id(sc)
    }))

    # Group text by its referenced document ID
    doc_to_sentences: Dict[str, List[str]] = {}
    uncited_sentences: List[str] = []

    for p in cleaned_paras:
        # Extract any doc citation present in this paragraph
        found_docs = sorted(list(set(re.findall(r'\[([a-zA-Z0-9_-]+)\]', p))))
        # Strip internal citations from the sentence body
        body_text = re.sub(r'\[[a-zA-Z0-9_-]+\]', '', p).strip()
        body_text = " ".join(body_text.split())
        if not body_text:
            continue

        target_doc = None
        for d in found_docs:
            if d in normalized_source_docs or d.startswith("canon_"):
                target_doc = d
                break

        if target_doc:
            doc_to_sentences.setdefault(target_doc, []).append(body_text)
        elif normalized_source_docs:
            # Fall back to first known source doc if none explicitly cited in paragraph
            doc_to_sentences.setdefault(normalized_source_docs[0], []).append(body_text)
        else:
            uncited_sentences.append(body_text)

    # Build one concatenated paragraph per document, ending with [doc_id]
    result_paragraphs = []
    for doc_id, sents in doc_to_sentences.items():
        joined_text = " ".join(sents).strip()
        result_paragraphs.append(f"{joined_text} [{doc_id}]")

    if uncited_sentences and not doc_to_sentences:
        fallback_cite = " ".join(f"[{d}]" for d in normalized_source_docs)
        joined_text = " ".join(uncited_sentences).strip()
        if fallback_cite:
            joined_text = f"{joined_text} {fallback_cite}"
        result_paragraphs.append(joined_text)

    # Append resolve-stubs "Additionally..." sentences in a trailing paragraph
    if additionally_sentences:
        clean_add = [re.sub(r'\s+', ' ', s).strip() for s in additionally_sentences]
        result_paragraphs.append(" ".join(clean_add))

    return "\n\n".join(result_paragraphs) if result_paragraphs else clean_desc.strip()

def export_wiki_markdown(domain_dir: Path, output_dir: Path = None):
    """Write one Markdown page per concept.

    Reads the most complete graph available (refined, then stubs-resolved, then
    global) and writes an Obsidian-style page for each concept. The brief stays on a
    single line, and each body paragraph ends with a citation to its source document.

    Args:
        domain_dir: Run directory the graph was read from.
        output_dir: Folder to write into. Defaults to ``<run>/concepts_md`` when
            ``None``.

    Returns:
        None. Prints how many pages were written."""
    refined_file = domain_dir / "global_concepts_refined.json"
    if not refined_file.exists():
        stubs_file = domain_dir / "global_concepts_stubs_resolved.json"
        if stubs_file.exists():
            refined_file = stubs_file
        else:
            refined_file = domain_dir / "global_concepts.json"

    if not refined_file.exists():
        raise FileNotFoundError(f"No concept graph file found in {domain_dir}")

    if output_dir is None:
        output_dir = domain_dir / "concepts_md"

    print(f"Loading concepts from {refined_file}...")
    with open(refined_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        concepts = data.get("concepts", [])

    print(f"Total concepts to export: {len(concepts)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove only the pages the previous export wrote, as listed in its manifest.
    # Other files in the folder are never touched, so a mistyped --output_dir
    # cannot destroy unrelated notes.
    manifest_path = output_dir / EXPORT_MANIFEST
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", [])
        for name in previous:
            (output_dir / name).unlink(missing_ok=True)
    written_files = []

    # Map of lowercased titles to canonical titles for wikilink normalization
    title_lookup = {c["title"].lower().strip(): c["title"].strip() for c in concepts}

    exported_count = 0
    for c in tqdm(concepts, desc="Exporting Markdown concepts"):
        title = c.get("title", "").strip()
        if not title:
            continue

        aliases = c.get("aliases", [])
        brief = clean_brief(str(c.get("brief", "") or ""))
        entities = c.get("entities", [])
        raw_source_chunks = c.get("source_chunks", []) or []
        description = str(c.get("description", "") or "").strip()
        outbound_relations = c.get("outbound_relations", [])

        # Deduplicate source documents at the document level
        source_docs = sorted({
            extract_doc_id(sc)
            for sc in raw_source_chunks
            if isinstance(sc, str) and sc.strip()
        } - {""})

        # Format body paragraphs with document-level citations & separate stub resolution paragraph
        description_body = format_description_paragraphs(description, source_docs)

        # Construct YAML Frontmatter
        frontmatter_lines = [
            "---",
            f"title: {escape_yaml_string(title)}",
            "aliases:"
        ]
        if aliases:
            for a in aliases:
                frontmatter_lines.append(f"  - {escape_yaml_string(a)}")
        else:
            frontmatter_lines.append("  []")

        frontmatter_lines.append(f"brief: {escape_yaml_string(brief)}")

        frontmatter_lines.append("entities:")
        if entities:
            for ent in entities:
                frontmatter_lines.append(f"  - {escape_yaml_string(ent)}")
        else:
            frontmatter_lines.append("  []")

        frontmatter_lines.append("sources:")
        if source_docs:
            for sd in source_docs:
                frontmatter_lines.append(f"  - {escape_yaml_string(sd)}")
        else:
            frontmatter_lines.append("  []")

        frontmatter_lines.append("---")
        frontmatter_str = "\n".join(frontmatter_lines)

        # Construct # Related Section
        related_lines = ["\n# Related"]
        seen_targets = set()

        for rel in outbound_relations:
            raw_target = rel.get("target_concept", "").strip()
            if not raw_target:
                continue

            # STRICT NO-STUBS GUARANTEE: Target MUST exist in canonical title_lookup!
            if raw_target.lower() not in title_lookup:
                continue

            canonical_target = title_lookup[raw_target.lower()]
            if canonical_target.lower() == title.lower() or canonical_target.lower() in seen_targets:
                continue

            seen_targets.add(canonical_target.lower())
            target_slug = sluggify(canonical_target)
            rel_desc = rel.get("relation", "").strip()
            if not rel_desc:
                rel_desc = "Functionally associated entity in the knowledge graph."

            # Format link: [[target_slug|Canonical Title]]
            related_lines.append(f"- [[{target_slug}|{canonical_target}]] - {rel_desc}")

        if len(seen_targets) == 0:
            related_lines.append("*No outbound relations recorded.*")

        related_str = "\n".join(related_lines)

        # Full Document Content
        full_content = f"{frontmatter_str}\n\n{description_body}\n{related_str}\n"

        slug = sluggify(title)
        md_file = output_dir / f"{slug}.md"
        counter = 1
        while md_file.exists():
            md_file = output_dir / f"{slug}_{counter}.md"
            counter += 1

        written_files.append(md_file.name)
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(full_content)

        exported_count += 1

    atomic_write_json(manifest_path, {"files": written_files})
    print(f"\nSuccessfully exported {exported_count} Markdown wiki files to: {output_dir}/")

def main() -> None:
    """Render the refined concept graph as one Markdown file per concept.

    Reads ``global_concepts_refined.json`` from the run directory and writes an
    Obsidian-style Markdown page per concept. Citations keep the canonical
    ``canon_<hash>`` document ids; ``doc_mapping.json`` in the run directory maps
    each original document id to its canonical id.

    Returns:
        None. Writes to ``--output_dir`` if given, otherwise
        ``<wiki_base_dir>/<domain>/concepts_md/``.
    """
    # This stage makes no LLM calls. It uses the shared parser anyway so one
    # uniform command line works for every stage; the server and retry flags are
    # accepted and ignored.
    parser = build_parser(
        stage=STAGE,
        description="Export concepts to Obsidian Markdown format (no LLM calls)",
        include_max_tokens=False,
        include_model=False)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom output directory for Markdown files. "
                             "Defaults to <wiki_base_dir>/<domain>/concepts_md/.")

    args = parser.parse_args()
    domain_dir = Path(args.wiki_base_dir) / args.domain
    output_dir = Path(args.output_dir) if args.output_dir else (domain_dir / "concepts_md")

    export_wiki_markdown(domain_dir, output_dir)

if __name__ == "__main__":
    main()
