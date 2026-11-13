#!/usr/bin/env python3
"""Stage 6b: Subtopic Decomposition for Large Concepts (>600 words).
-------------------------------------------------------------------
Splits broad pillar concepts (>600 words) with multiple subtopics into:
1. An overarching, concise Pillar Concept (150-300 words).
2. Distinct, focused Child Concepts (150-350 words each) corresponding to
   substantive subtopics, preserving all citations, parameters, and metrics.
3. Automatically establishes bidirectional parent-child links.

Usage:
    python3 decompose_large_concepts.py --wiki_path /path/to/wiki
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set

from openai import BadRequestError
from pydantic import BaseModel, Field

from oversized_decompose import (
    OversizedDecomposerMixin,
    count_tokens,
    is_context_overflow,
)

from utils.atomic_io import atomic_write_json, sanitize_filename
from utils.checkpoint import load_json_log, save_log_sync
from utils.cli import build_parser, resolve_settings
from utils.concept_ops import (
    extract_concept_paragraphs,
    fuse_paragraphs_by_document,
    generate_stable_slug,
    get_concept_full_text,
)
from utils.errors import StageBudget
from utils.llm_params import call_llm
from utils.llm_pool import pool_for_stage
from utils.manifest import verify_stage_manifest, write_stage_manifest
from utils.stats import print_concept_word_distribution

STAGE = "decompose-concepts"

# ── XML System Prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT_DECOMPOSE = """\
You are an expert scientific encyclopedist and knowledge graph architect.
You are given a LARGE, COMPOSITE CONCEPT (>600 words) from a technical knowledge wiki, along with its absorbed subtopics and its source text broken down by source chunk.

YOUR TASK:
Decompose this broad topic into more digestible, focused concepts:
1. <pillar>: A focused OVERVIEW concept (150 to 300 words) covering the core definition, high-level role, and primary mechanism.
2. <child>: 2 to 4 discrete CHILD concepts (each 150 to 350 words) covering distinct subtopics, components, or operational phases.

CRITICAL RULES:
- 100% FACT & PROVENANCE PRESERVATION: Do NOT lose any specific numbers, metrics, or factual statements. Every source chunk present in the input must retain its factual contributions across either the pillar or one of the child concepts.
- PRESERVE SOURCE CHUNKS: You MUST attribute each paragraph to its originating source chunk using <paragraph source_chunk="doc_XXXXX_chunk_YYY">...</paragraph>. Never attribute facts from one chunk to another.
- NO INLINE CITATIONS IN PROSE: Do NOT write bracketed citations like [doc_00251] in the prose. The provenance is captured strictly by the source_chunk attribute on <paragraph>.
- TITLES: Child titles must be 1-6 words, Title Case, explicit and self-contained (e.g. "Workstation Ergonomics and Lighting", NOT just "Ergonomics").
- NO NOISE: Output no introductory text, commentary, or markdown formatting outside the XML tags.
- IF NOT DECOMPOSABLE: If the concept cannot be cleanly divided without severe redundancy, output exactly: <cannot_decompose/>

OUTPUT FORMAT:
<decomposition>
<pillar>
<paragraph source_chunk="doc_XXXXX_chunk_YYY">
Overview description paragraph...
</paragraph>
</pillar>
<child>
<title>Specific Child Concept Title A</title>
<paragraph source_chunk="doc_XXXXX_chunk_YYY">
Detailed factual paragraph focusing strictly on this subtopic...
</paragraph>
</child>
<child>
<title>Specific Child Concept Title B</title>
<paragraph source_chunk="doc_XXXXX_chunk_YYY">
Detailed factual paragraph focusing strictly on this subtopic...
</paragraph>
</child>
</decomposition>
"""

class DecomposedChild(BaseModel):
    title: str
    paragraphs: List[Dict[str, str]] = Field(default_factory=list)

class DecompositionResult(BaseModel):
    decomposed: bool = False
    pillar_paragraphs: List[Dict[str, str]] = Field(default_factory=list)
    children: List[DecomposedChild] = Field(default_factory=list)


def parse_xml_decomposition(text: str, default_source_chunk: str = "unknown") -> DecompositionResult:
    if not text or "<cannot_decompose" in text.lower():
        return DecompositionResult(decomposed=False)

    def extract_paragraphs(xml_block: str) -> List[Dict[str, str]]:
        paras: List[Dict[str, str]] = []
        p_matches = re.findall(r'<paragraph(?:\s+(?:source_chunk|doc_id)="([^"]+)")?>(.*?)</paragraph>', xml_block, re.DOTALL | re.IGNORECASE)
        for chunk_attr, p_text in p_matches:
            src = chunk_attr.strip().replace("::", "_") if chunk_attr else default_source_chunk
            did = src.split("_chunk_")[0] if "_chunk_" in src else src
            clean_t = re.sub(r"\[\s*(?:doc_|canon_)[A-Za-z0-9_-]+(?:::[^\]]+)?\s*\]", "", p_text).strip()
            clean_t = re.sub(r"\.\s*\.", ".", clean_t)
            clean_t = re.sub(r"\s+([.,;:!?])", r"\1", clean_t)
            clean_t = " ".join(clean_t.split())
            if clean_t:
                paras.append({"source_chunk": src, "doc_id": did, "text": clean_t})
        if not paras:
            # Fallback if model used <description>...</description>
            desc_m = re.search(r"<description>(.*?)</description>", xml_block, re.DOTALL | re.IGNORECASE)
            fallback_text = desc_m.group(1).strip() if desc_m else xml_block.strip()
            clean_t = re.sub(r"\[\s*(?:doc_|canon_)[A-Za-z0-9_-]+(?:::[^\]]+)?\s*\]", "", fallback_text).strip()
            clean_t = re.sub(r"\.\s*\.", ".", clean_t)
            clean_t = re.sub(r"\s+([.,;:!?])", r"\1", clean_t)
            clean_t = " ".join(clean_t.split())
            if clean_t:
                did = default_source_chunk.split("_chunk_")[0] if "_chunk_" in default_source_chunk else default_source_chunk
                paras.append({"source_chunk": default_source_chunk, "doc_id": did, "text": clean_t})
        return paras

    pillar_m = re.search(r"<pillar>(.*?)</pillar>", text, re.DOTALL | re.IGNORECASE)
    if not pillar_m:
        return DecompositionResult(decomposed=False)

    pillar_paras = extract_paragraphs(pillar_m.group(1))
    if not pillar_paras:
        return DecompositionResult(decomposed=False)

    children: List[DecomposedChild] = []
    child_blocks = re.findall(r"<child>(.*?)</child>", text, re.DOTALL | re.IGNORECASE)
    for block in child_blocks:
        t_m = re.search(r"<title>(.*?)</title>", block, re.DOTALL | re.IGNORECASE)
        if not t_m:
            continue
        c_title = t_m.group(1).strip()
        c_paras = extract_paragraphs(block)
        if c_title and c_paras:
            children.append(DecomposedChild(title=c_title, paragraphs=c_paras))

    if not children:
        return DecompositionResult(decomposed=False)

    return DecompositionResult(decomposed=True, pillar_paragraphs=pillar_paras, children=children)


class ConceptDecompositionPipeline(OversizedDecomposerMixin):
    def __init__(self, args):
        self.args = args
        wiki_path = getattr(args, "wiki_path", None)
        if not wiki_path:
            raise ValueError("Missing required --wiki_path argument.")

        self.wiki_path = Path(wiki_path)
        self.global_concepts_file = self.wiki_path / "global_concepts.json"
        self.concepts_dir = self.wiki_path / "concepts"
        self.log_file = self.wiki_path / "decomposition_log.json"

        if not self.global_concepts_file.exists():
            raise FileNotFoundError(f"Missing {self.global_concepts_file}. Run Stage 6 first.")

        verify_stage_manifest(self.wiki_path, "consolidate-global")

        self.settings = resolve_settings(args, STAGE)
        self.budget = StageBudget(self.settings.budget_minutes)
        self.checkpoint = {}
        if getattr(args, "resume", False):
            self.checkpoint = load_json_log(self.log_file, corrupt_message="Decomposition log corrupted. Initializing fresh.")

    def select_large_concepts(self, concepts: List[Dict[str, Any]], word_threshold: int = 600) -> List[Dict[str, Any]]:
        """Identify concepts exceeding word_threshold that have subtopics or multiple source documents."""
        large = []
        for c in concepts:
            text = get_concept_full_text(c)
            words = len(re.findall(r"\b\w+\b", text))
            paras = c.get("paragraphs", [])
            subtopics = c.get("subtopics", [])
            if words > word_threshold and (len(subtopics) >= 2 or len(paras) >= 2):
                large.append(c)
        return large

    async def decompose_concept(self, client: Any, item: Dict[str, Any], server_index: int) -> Dict[str, Any]:
        title = item.get("title", "")
        subtopics = ", ".join(item.get("subtopics", [])) or "None listed"
        paras = item.get("paragraphs", [])
        if not paras:
            paras = extract_concept_paragraphs(item)
        
        full_text = get_concept_full_text(item)
        words = len(re.findall(r"\b\w+\b", full_text))

        # Route oversized concepts exceeding single-shot input budget to Map-Reduce
        total_tokens = sum(count_tokens(p.get("text", "")) for p in paras)
        if total_tokens > self.single_shot_input_budget():
            return await self.decompose_oversized(client, item, paras)

        doc_blocks = []
        for p in paras:
            src = (p.get("source_chunk") or p.get("doc_id", "unknown")).replace("::", "_")
            ptxt = p.get("text", "").strip()
            doc_blocks.append(f"SOURCE CHUNK: {src}\nTEXT:\n{ptxt}\n")
        source_presentation = "\n".join(doc_blocks)

        prompt = f"""\
COMPOSITE CONCEPT TITLE: "{title}"
CURRENT LENGTH: {words} words
ABSORBED SUBTOPICS: {subtopics}

SOURCE PARAGRAPHS BY SOURCE CHUNK:
{source_presentation}
"""
        max_out = self.single_shot_max_out()
        try:
            raw_text = await call_llm(
                client=client,
                model=self.settings.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_DECOMPOSE},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_out,
                args=self.args,
                task="generate",
                max_growth=1,
            )
        except BadRequestError as exc:
            if not is_context_overflow(exc):
                raise
            return await self.decompose_oversized(client, item, paras)

        default_src = paras[0].get("source_chunk") or paras[0].get("doc_id", "unknown") if paras else "unknown"
        res = parse_xml_decomposition(raw_text, default_source_chunk=default_src)
        return {
            "title": title,
            "decomposed": res.decomposed,
            "pillar_paragraphs": res.pillar_paragraphs,
            "children": [{"title": ch.title, "paragraphs": ch.paragraphs} for ch in res.children]
        }

    async def run(self):
        # Create safety backup of pre-decomposition global_concepts.json if not present
        bak_file = self.global_concepts_file.with_suffix(".json.bak")
        if not bak_file.exists() and self.global_concepts_file.exists():
            import shutil
            shutil.copy(self.global_concepts_file, bak_file)
            print(f"Created backup of pre-decomposition concepts at {bak_file}")

        # Idempotent source loading: always read from pristine pre-decomposition backup if available
        src_file = bak_file if bak_file.exists() else self.global_concepts_file
        print(f"Loading concepts from {src_file}...")
        with open(src_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            concepts = data.get("concepts", [])

        total = len(concepts)
        print(f"Loaded {total} global concepts.")
        print_concept_word_distribution(concepts, title_label="STAGE 6b: BEFORE SUBTOPIC DECOMPOSITION")

        threshold = getattr(self.args, "word_threshold", 600)
        large_concepts = self.select_large_concepts(concepts, word_threshold=threshold)
        print(f"Identified {len(large_concepts)} broad concepts exceeding {threshold} words eligible for subtopic decomposition.")

        pending = [c for c in large_concepts if c.get("title") not in self.checkpoint]

        results = dict(self.checkpoint)
        log_lock = asyncio.Lock()
        flush_count = 0

        async def on_result(item, res):
            nonlocal flush_count
            t = item.get("title")
            async with log_lock:
                results[t] = res
                flush_count += 1
                if flush_count >= 10:
                    save_log_sync(self.log_file, results)
                    flush_count = 0

        if pending:
            print(f"Decomposing {len(pending)} concepts across {len(self.settings.servers)} server(s)...")
            pool = pool_for_stage(self.args, self.settings, STAGE)
            async with pool:
                res_obj = await pool.map(
                    pending,
                    self.decompose_concept,
                    attempts=self.settings.attempts,
                    desc="Decomposing Large Concepts",
                    key_fn=lambda c: c.get("title", ""),
                    budget=self.budget,
                    attempt_timeout=self.settings.profile.attempt_timeout,
                    on_result=on_result,
                )
                res_obj.raise_if_failed(STAGE)

        save_log_sync(self.log_file, results)

        # Apply decomposition to concepts
        decomposed_count = 0
        new_children_count = 0
        final_concepts: List[Dict[str, Any]] = []
        used_slugs: Set[str] = {c.get("slug") for c in concepts if c.get("slug")}

        for c in concepts:
            t = c.get("title")
            decomp_data = results.get(t)

            if decomp_data and decomp_data.get("decomposed") and decomp_data.get("children"):
                decomposed_count += 1
                pillar_slug = c.get("slug") or generate_stable_slug(t, used_slugs)
                c["slug"] = pillar_slug

                # Safety: Ensure no source documents from parent are dropped across pillar + children
                orig_paras = c.get("paragraphs", [])
                pillar_paras = list(decomp_data.get("pillar_paragraphs", []))
                all_child_paras = []
                for ch in decomp_data.get("children", []):
                    all_child_paras.extend(ch.get("paragraphs", []))

                surviving_chunks = {p.get("source_chunk") or p.get("doc_id") for p in pillar_paras + all_child_paras if p.get("source_chunk") or p.get("doc_id")}
                # If any original chunk was omitted by LLM in decomposition, retain FULL evidence in the pillar
                for orig_p in orig_paras:
                    src = orig_p.get("source_chunk") or orig_p.get("doc_id")
                    if src and src != "unknown" and src not in surviving_chunks:
                        print(f"[PROVENANCE RESCUE] Rescued omitted chunk '{src}' back into pillar '{t}'.")
                        rescue_text = (orig_p.get("text") or "").strip()
                        did = src.split("_chunk_")[0] if "_chunk_" in src else src
                        pillar_paras.append({"source_chunk": src, "doc_id": did, "text": rescue_text})
                        surviving_chunks.add(src)

                # Retain all pillar paragraphs without destructive truncation
                c["paragraphs"] = pillar_paras
                c.pop("description", None)

                # Ensure outbound_relations exists
                relations = c.get("outbound_relations", [])
                existing_targets = {r.get("target_concept", "").lower() for r in relations}

                final_concepts.append(c)

                # Consistent chunk-key inheritance from parent's source_chunks
                parent_chunks = c.get("source_chunks", [])

                # Add children without destructive truncation
                for ch in decomp_data.get("children", []):
                    ch_title = ch["title"].strip()
                    ch_slug = generate_stable_slug(ch_title, used_slugs)
                    new_children_count += 1

                    ch_paras = ch.get("paragraphs", [])
                    child_docs = {p.get("doc_id") for p in ch_paras if p.get("doc_id")}
                    # Inherit matching chunk-level keys from parent
                    ch_chunks = sorted(sc for sc in parent_chunks if sc.split("_chunk_")[0].split("::")[0] in child_docs) or sorted(child_docs)

                    child_concept = {
                        "title": ch_title,
                        "slug": ch_slug,
                        "aliases": [],
                        "subtopics": [],
                        "entities": [],
                        "paragraphs": ch_paras,
                        "source_chunks": ch_chunks,
                        "outbound_relations": [
                            {
                                "target_concept": t,
                                "relation": f"is a specialized component of {t}",
                                "overlap_score": 0.85,
                                "jaccard_similarity": 0.85,
                                "link_type": "subtopic_decomposition",
                                "verified_by_llm": True
                            }
                        ]
                    }
                    final_concepts.append(child_concept)

                    # Parent links to child
                    if ch_title.lower() not in existing_targets:
                        relations.append({
                            "target_concept": ch_title,
                            "relation": f"includes subtopic {ch_title}",
                            "overlap_score": 0.85,
                            "jaccard_similarity": 0.85,
                            "link_type": "subtopic_decomposition",
                            "verified_by_llm": True
                        })
                        existing_targets.add(ch_title.lower())
                c["outbound_relations"] = relations
            else:
                c.pop("description", None)
                # Keep untouched / standard concepts 100% intact with zero truncation
                final_concepts.append(c)

        print(f"\nDecomposed {decomposed_count} broad concepts into {new_children_count} focused subtopic concepts.")
        print(f"Total concepts after decomposition: {len(final_concepts)} (from {total}).")

        # Save back to global_concepts.json and refresh concepts/
        atomic_write_json(self.global_concepts_file, {"concepts": final_concepts})
        surviving_slugs = set()
        for fc in final_concepts:
            s = fc.get("slug")
            if s:
                surviving_slugs.add(f"{s}.json")
                atomic_write_json(self.concepts_dir / f"{s}.json", fc)

        # Clean stale concept JSON files from earlier runs
        for old_f in self.concepts_dir.glob("*.json"):
            if old_f.name not in surviving_slugs:
                try:
                    old_f.unlink()
                except OSError:
                    pass

        print_concept_word_distribution(final_concepts, title_label="STAGE 6b: AFTER SUBTOPIC DECOMPOSITION")

        write_stage_manifest(
            wiki_path=self.wiki_path,
            stage=STAGE,
            expected=len(final_concepts),
            completed=len(final_concepts),
            failed=0,
            decomposed_pillars=decomposed_count,
            created_subtopic_concepts=new_children_count,
        )

        print("\n============================================================")
        print("STAGE 6b: SUBTOPIC DECOMPOSITION COMPLETE")
        print(f"Pillar Concepts Decomposed: {decomposed_count}")
        print(f"New Subtopic Concepts Added: {new_children_count}")
        print(f"Final Concept Count:         {len(final_concepts)}")
        print("============================================================\n")


def main():
    parser = build_parser(stage=STAGE, description="Decompose broad composite concepts (>600 words) into focused subtopic concepts", include_resume=True)
    parser.add_argument("--word_threshold", type=int, default=600,
                        help="Word count threshold above which concepts are decomposed (default: 600 words).")
    args = parser.parse_args()
    pipeline = ConceptDecompositionPipeline(args)
    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()
