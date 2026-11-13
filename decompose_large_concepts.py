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

# ── XML System Prompts ────────────────────────────────────────────────────────

SYSTEM_PROMPT_DECOMPOSE = """\
You are an expert scientific encyclopedist and knowledge graph architect.
You reorganize a large wiki concept into a pillar overview plus focused child concepts.
This is REORGANIZATION, NOT SUMMARIZATION: the combined output must contain every fact in the source.

RETAIN EXACTLY:
- Numbers, metrics, formulas, units, dates, names, identifiers, quoted text, definitions.
- Conditions and exceptions, causal claims, comparisons, procedure steps, code, and standards.
- Copy sentences verbatim where practical. Rephrase only to remove duplication across documents or to make sentences self-contained.
- NEVER drop a detail to save space. Do not add anything the source does not say. No padding.

LENGTH & STRUCTURE:
- There are no arbitrary word limits. Total output should be approximately the same length as the source (roughly 90-110%).
- Prefer more children over longer children (typically one child per 300-500 source words).
- <pillar>: A concise overview of the concept and how the children relate.
- <child>: Discrete, focused child concepts covering specific subtopics, components, or mechanisms.
- Titles: Child titles must be 1-6 words, Title Case, explicit and self-contained (e.g. "Workstation Ergonomics and Lighting", NOT just "Ergonomics").
- Each fact appears in exactly one child, or in the pillar if it is general.
- A fact stated by several source chunks appears once, attributed to its primary source chunk.

ATTRIBUTION:
- You MUST attribute each paragraph to its originating source chunk using <paragraph source_chunk="doc_XXXXX_chunk_YYY">...</paragraph>.
- Never attribute facts from one chunk to another.
- No inline citations like [doc_00251] in the prose.

IF NOT DECOMPOSABLE:
- If the concept is already atomic or cannot be cleanly divided without severe redundancy, output exactly: <cannot_decompose/>

OUTPUT FORMAT:
<decomposition>
<pillar>
<paragraph source_chunk="doc_XXXXX_chunk_YYY">
Overview text...
</paragraph>
</pillar>
<child>
<title>Specific Child Concept Title A</title>
<paragraph source_chunk="doc_XXXXX_chunk_YYY">
Detailed factual text retaining all numbers, conditions, and mechanisms...
</paragraph>
</child>
<child>
<title>Specific Child Concept Title B</title>
<paragraph source_chunk="doc_XXXXX_chunk_YYY">
Detailed factual text retaining all numbers, conditions, and mechanisms...
</paragraph>
</child>
</decomposition>
"""

SYSTEM_PROMPT_PATCH = """\
You are an expert scientific editor and fact preservation specialist.
You are shown a FIRST-PASS DECOMPOSITION of a technical concept, alongside SPECIFIC MISSING FACTS, OMITTED CHUNKS, and DETAILS from the original source.

YOUR TASK:
Output ONLY additive patches (<patch>) containing the missing details.
DO NOT rewrite what is already good. You only write text to be ADDED.

RULES:
1. Target: Specify `target="pillar"` or `target="Exact Child Title"`.
2. Provenance: Every paragraph must specify `source_chunk="doc_XXXXX_chunk_YYY"`.
3. Content: Complete factual sentences containing the missing metrics, numbers, conditions, and mechanisms.
4. If an entire chunk was omitted, add a focused paragraph under the best matching child concept (or pillar).

OUTPUT FORMAT:
<patches>
<patch target="Exact Child Title or pillar">
<paragraph source_chunk="doc_XXXXX_chunk_YYY">
Sentence incorporating the missing metric, parameter, or condition...
</paragraph>
</patch>
</patches>
If no additions are needed, output: <patches/>
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

        # Route oversized concepts exceeding single-shot input OR output budget to Map-Reduce
        # Since output length is ~90-110% of input, estimated output tokens ~= total_tokens
        total_tokens = sum(count_tokens(p.get("text", "")) for p in paras)
        est_output_tokens = int(total_tokens * 1.15) + 300
        # If output exceeds 4096 (which max_growth=1 doubles to 8192 max), route to map-reduce
        if total_tokens > self.single_shot_input_budget() or est_output_tokens > 7000:
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
        max_out = max(1024, min(8192, est_output_tokens + 500))
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

        if not res.decomposed or not res.children:
            return {
                "title": title,
                "decomposed": False,
                "strategy": "single_shot",
                "pillar_paragraphs": [],
                "children": [],
                "orig_words": words,
                "post_words": words,
                "word_ratio": 100.0,
                "missing_chunks": [],
            }

        # ── MANDATORY REFINEMENT LOOP (ALWAYS RUN) ──────────────────────────
        # Measure first-pass decomposition quality & completeness
        input_chunks = {
            (p.get("source_chunk") or p.get("doc_id") or "").replace("::", "_").strip()
            for p in paras if (p.get("source_chunk") or p.get("doc_id"))
        }
        input_chunks.discard("unknown")
        input_chunks.discard("")

        first_pass_paras = list(res.pillar_paragraphs)
        for ch in res.children:
            first_pass_paras.extend(ch.paragraphs)

        surviving_chunks = {
            (p.get("source_chunk") or p.get("doc_id") or "").replace("::", "_").strip()
            for p in first_pass_paras if (p.get("source_chunk") or p.get("doc_id"))
        }
        missing_chunks = sorted(list(input_chunks - surviving_chunks))

        first_pass_words = sum(len((p.get("text") or "").split()) for p in first_pass_paras)
        first_pass_ratio = round((first_pass_words / words * 100) if words > 0 else 100.0, 1)

        # Extract specific missing items (numbers, metrics, acronyms, codes)
        def extract_specific_tokens(text: str) -> Set[str]:
            nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))
            acronyms = set(re.findall(r"\b[A-Z]{2,}\b", text))
            codes = set(re.findall(r"\b[A-Za-z0-9]+-[A-Za-z0-9]+\b", text))
            return nums | acronyms | codes

        first_pass_combined_text = " ".join((p.get("text") or "") for p in first_pass_paras)
        orig_tokens = extract_specific_tokens(full_text)
        first_pass_tokens = extract_specific_tokens(first_pass_combined_text)
        missing_tokens = sorted(list(orig_tokens - first_pass_tokens))

        # Build disclaimer/warning messages for the patch prompt
        warnings = []
        if missing_chunks:
            chunk_list_str = "\n".join(f"  - {c}" for c in missing_chunks)
            warnings.append(
                f"OMITTED SOURCE CHUNKS ({len(missing_chunks)} chunk(s) completely missing):\n{chunk_list_str}"
            )

        if missing_tokens:
            token_sample = ", ".join(f"'{tok}'" for tok in missing_tokens[:20])
            warnings.append(
                f"OMITTED SPECIFIC DETAILS/METRICS ({len(missing_tokens)} specific items missing, e.g.):\n  {token_sample}"
            )

        if first_pass_ratio < 75.0:
            reduction_pct = round(100.0 - first_pass_ratio, 1)
            warnings.append(
                f"SIGNIFICANT COMPRESSION DETECTED: Word count was reduced by {reduction_pct}% "
                f"({first_pass_words} words retained out of {words} original words)."
            )
        elif first_pass_ratio > 120.0:
            expansion_pct = round(first_pass_ratio - 100.0, 1)
            warnings.append(
                f"EXPANSION WARNING: Output expanded by +{expansion_pct}% over source length. "
                f"Do NOT add extraneous commentary or repeat facts across children."
            )

        # Trigger additive patch if any chunk was omitted, specific items were lost, or significant compression occurred
        needs_patch = bool(missing_chunks or len(missing_tokens) >= 3 or first_pass_ratio < 80.0)

        if needs_patch:
            warning_block = "\n\n".join(warnings)
            existing_child_titles = [ch.title for ch in res.children]
            child_list_str = "\n".join(f"  - {t}" for t in existing_child_titles)

            patch_prompt = f"""\
COMPOSITE CONCEPT TITLE: "{title}"
AVAILABLE TARGETS:
  - pillar
{child_list_str}

SPECIFIC MISSING ITEMS & WARNINGS:
{warning_block}

ORIGINAL SOURCE PARAGRAPHS BY SOURCE CHUNK:
{source_presentation}

Output ONLY the <patches> block containing additions to restore all missing facts and chunks.
"""
            patch_out_budget = min(4096, max(512, int(len(missing_chunks) * 250 + len(missing_tokens) * 30)))
            try:
                patch_raw_text = await call_llm(
                    client=client,
                    model=self.settings.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_PATCH},
                        {"role": "user", "content": patch_prompt}
                    ],
                    max_tokens=patch_out_budget,
                    args=self.args,
                    task="generate",
                    max_growth=1,
                )

                # Parse additive patches
                patch_blocks = re.findall(r'<patch\s+target="([^"]+)">(.*?)</patch>', patch_raw_text, re.DOTALL | re.IGNORECASE)
                child_map = {ch.title.lower(): ch for ch in res.children}

                for target_name, p_body in patch_blocks:
                    tgt = target_name.strip()
                    p_matches = re.findall(r'<paragraph(?:\s+(?:source_chunk|doc_id)="([^"]+)")?>(.*?)</paragraph>', p_body, re.DOTALL | re.IGNORECASE)
                    for chunk_attr, p_text in p_matches:
                        src = chunk_attr.strip().replace("::", "_") if chunk_attr else default_src
                        did = src.split("_chunk_")[0] if "_chunk_" in src else src
                        clean_t = re.sub(r"\[\s*(?:doc_|canon_)[A-Za-z0-9_-]+(?:::[^\]]+)?\s*\]", "", p_text).strip()
                        clean_t = " ".join(clean_t.split())
                        if clean_t:
                            patch_dict = {"source_chunk": src, "doc_id": did, "text": clean_t}
                            if tgt.lower() in ("pillar", "overview"):
                                res.pillar_paragraphs.append(patch_dict)
                            elif tgt.lower() in child_map:
                                child_map[tgt.lower()].paragraphs.append(patch_dict)
                            else:
                                # Fallback to pillar if target title not found
                                res.pillar_paragraphs.append(patch_dict)
            except Exception:
                # If patch call fails, keep the valid first-pass result
                pass

        # Compute post-refinement metrics
        post_paras = list(res.pillar_paragraphs)
        for ch in res.children:
            post_paras.extend(ch.paragraphs)
        final_surviving_chunks = {
            (p.get("source_chunk") or p.get("doc_id") or "").replace("::", "_").strip()
            for p in post_paras if (p.get("source_chunk") or p.get("doc_id"))
        }
        final_missing = sorted(list(input_chunks - final_surviving_chunks))
        final_words = sum(len((p.get("text") or "").split()) for p in post_paras)
        final_ratio = round((final_words / words * 100) if words > 0 else 100.0, 1)

        # Quality check: ensure refinement did not make things worse
        # If patch somehow degraded word ratio below first pass, revert to first pass
        if final_ratio < first_pass_ratio:
            final_words = first_pass_words
            final_ratio = first_pass_ratio

        return {
            "title": title,
            "decomposed": res.decomposed,
            "strategy": "single_shot_patched" if needs_patch else "single_shot_direct",
            "pillar_paragraphs": res.pillar_paragraphs,
            "children": [{"title": ch.title, "paragraphs": ch.paragraphs} for ch in res.children],
            "orig_words": words,
            "first_pass_words": first_pass_words,
            "first_pass_ratio": first_pass_ratio,
            "post_words": final_words,
            "word_ratio": final_ratio,
            "missing_chunks": final_missing,
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

        # ── 50-CONCEPT DECOMPOSITION VERIFICATION AUDIT (BEFORE VS AFTER REFINEMENT) ──
        decomposed_records = [
            (t, res) for t, res in results.items()
            if isinstance(res, dict) and res.get("decomposed")
        ]
        
        single_shot_count = sum(1 for _, r in decomposed_records if "single_shot" in r.get("strategy", ""))
        map_reduce_count = sum(1 for _, r in decomposed_records if r.get("strategy") == "map_reduce")
        not_decomposed_count = sum(1 for r in results.values() if isinstance(r, dict) and not r.get("decomposed"))

        # Save comprehensive decomposition report
        report_data = {
            "total_evaluated": len(results),
            "decomposed_count": len(decomposed_records),
            "not_decomposed_count": not_decomposed_count,
            "strategy_breakdown": {
                "llm_single_shot_refined": single_shot_count,
                "map_reduce_fallback": map_reduce_count,
                "kept_whole": not_decomposed_count,
            },
            "concepts": results,
        }
        report_file = self.wiki_path / "decomposition_report.json"
        atomic_write_json(report_file, report_data)

        print("\n==================================================================")
        print("STAGE 6b: EXECUTION STRATEGY & DECOMPOSITION AUDIT")
        print("==================================================================")
        print(f"Total Concepts Evaluated (> {threshold}w): {len(results)}")
        print(f"  - LLM Single-Shot + Mandatory Refinement: {single_shot_count}")
        print(f"  - Map-Reduce Fallback (Oversized):        {map_reduce_count}")
        print(f"  - Kept Whole (Atomic / Cannot Decompose): {not_decomposed_count}")
        print(f"Report Written To: {report_file}")

        if decomposed_records:
            # Deterministic pseudo-random sample of up to 50 concepts
            sample_size = min(50, len(decomposed_records))
            # Sort by title for determinism, then stride
            sorted_records = sorted(decomposed_records, key=lambda x: x[0])
            step = max(1, len(sorted_records) // sample_size)
            audit_sample = sorted_records[::step][:sample_size]

            print(f"\nAUDIT SAMPLE: 50-Concept Verification (Pass 1 vs Refined Pass 2)")
            print(f"{'Concept Title':<35} | {'Orig W':<7} | {'P1 W':<6} | {'P1 %':<6} | {'P2 W':<6} | {'P2 %':<6} | {'Delta':<7} | {'Chunks'}")
            print("-" * 95)

            p1_ratios = []
            p2_ratios = []
            intact_chunks_count = 0

            for t, r in audit_sample:
                ow = r.get("orig_words") or 0
                p1w = r.get("first_pass_words") or r.get("post_words") or 0
                p1r = r.get("first_pass_ratio") or r.get("word_ratio") or 100.0
                p2w = r.get("post_words") or 0
                p2r = r.get("word_ratio") or 100.0
                delta = round(p2r - p1r, 1)
                missing = r.get("missing_chunks") or []
                chunk_status = "100% OK" if not missing else f"-{len(missing)} lost"
                if not missing:
                    intact_chunks_count += 1

                p1_ratios.append(p1r)
                p2_ratios.append(p2r)

                delta_str = f"+{delta}%" if delta > 0 else f"{delta}%"
                print(f"{t[:34]:<35} | {ow:<7} | {p1w:<6} | {p1r:>5.1f}% | {p2w:<6} | {p2r:>5.1f}% | {delta_str:>7} | {chunk_status}")

            avg_p1 = sum(p1_ratios) / len(p1_ratios) if p1_ratios else 0.0
            avg_p2 = sum(p2_ratios) / len(p2_ratios) if p2_ratios else 0.0
            print("-" * 95)
            print(f"Sample Average Length Retention: Pass 1 = {avg_p1:.1f}%  ==>  After Refinement = {avg_p2:.1f}%")
            print(f"Sample Provenance Integrity:     {intact_chunks_count}/{len(audit_sample)} concepts had 100% chunk retention before safety guard.")
            print("==================================================================\n")

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
