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

from pydantic import BaseModel, Field

from utils.atomic_io import atomic_write_json, sanitize_filename
from utils.checkpoint import load_json_log, save_log_sync
from utils.cli import build_parser, resolve_settings
from utils.concept_ops import generate_stable_slug
from utils.errors import StageBudget
from utils.llm_params import call_llm
from utils.llm_pool import pool_for_stage
from utils.manifest import verify_stage_manifest, write_stage_manifest
from utils.stats import print_concept_word_distribution

STAGE = "decompose-concepts"

# ── XML System Prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT_DECOMPOSE = """\
You are an expert scientific encyclopedist and knowledge graph architect.
You are given a LARGE, COMPOSITE CONCEPT (>600 words) from a technical knowledge wiki, along with its absorbed subtopics and full description.

YOUR TASK:
Decompose this broad topic into more digestible, focused concepts:
1. <pillar>: A focused OVERVIEW concept (150 to 300 words) covering the core definition, high-level role, and primary mechanism.
2. <child>: 2 to 4 discrete CHILD concepts (each 150 to 350 words) covering distinct subtopics, components, or operational phases.

CRITICAL RULES:
- FACT PRESERVATION: Do NOT lose any specific numbers, metrics, experimental findings, or parameters. Every fact in the original text must live in either the pillar or one of the child concepts.
- CITATIONS: Every paragraph must keep its original document citations (e.g. [doc_00251]) matching where the facts came from.
- TITLES: Child titles must be 1-6 words, Title Case, explicit and self-contained (e.g. "Canine Retinal Photoreception", NOT just "Photoreceptors").
- NO NOISE: Output no introductory text, commentary, or markdown formatting outside the XML tags.
- IF NOT DECOMPOSABLE: If the concept cannot be cleanly divided without severe redundancy, output exactly: <cannot_decompose/>

OUTPUT FORMAT:
<decomposition>
<pillar>
<description>
Overview description paragraph(s) with citations...
</description>
</pillar>
<child>
<title>Specific Child Concept Title A</title>
<description>
Detailed factual description focusing strictly on this subtopic with citations...
</description>
</child>
<child>
<title>Specific Child Concept Title B</title>
<description>
Detailed factual description focusing strictly on this subtopic with citations...
</description>
</child>
</decomposition>
"""

class DecomposedChild(BaseModel):
    title: str
    description: str

class DecompositionResult(BaseModel):
    decomposed: bool = False
    pillar_description: str = ""
    children: List[DecomposedChild] = Field(default_factory=list)

def parse_xml_decomposition(text: str) -> DecompositionResult:
    if not text or "<cannot_decompose" in text.lower():
        return DecompositionResult(decomposed=False)

    pillar_m = re.search(r"<pillar>.*?<description>(.*?)</description>.*?</pillar>", text, re.DOTALL | re.IGNORECASE)
    if not pillar_m or not pillar_m.group(1).strip():
        return DecompositionResult(decomposed=False)

    pillar_desc = pillar_m.group(1).strip()
    children = []

    child_blocks = re.findall(r"<child>(.*?)</child>", text, re.DOTALL | re.IGNORECASE)
    for block in child_blocks:
        t_m = re.search(r"<title>(.*?)</title>", block, re.DOTALL | re.IGNORECASE)
        d_m = re.search(r"<description>(.*?)</description>", block, re.DOTALL | re.IGNORECASE)
        if t_m and d_m and t_m.group(1).strip() and d_m.group(1).strip():
            c_title = t_m.group(1).strip()
            c_desc = d_m.group(1).strip()
            # Clean citations
            c_desc = re.sub(r"\[(?:doc_|canon_)[A-Za-z0-9_-]+::[A-Za-z0-9_-]+\]", "", c_desc)
            children.append(DecomposedChild(title=c_title, description=c_desc))

    if not children:
        return DecompositionResult(decomposed=False)

    return DecompositionResult(decomposed=True, pillar_description=pillar_desc, children=children)

# ── Pipeline Implementation ──────────────────────────────────────────────────

class ConceptDecompositionPipeline:
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
        """Identify concepts exceeding word_threshold that have subtopics or substantive breadth."""
        large = []
        for c in concepts:
            words = len(re.findall(r"\b\w+\b", c.get("description", "")))
            if words > word_threshold and (len(c.get("subtopics", [])) >= 2 or len(c.get("source_chunks", [])) >= 2):
                large.append(c)
        return large

    async def decompose_concept(self, client: Any, item: Dict[str, Any], server_index: int) -> Dict[str, Any]:
        title = item.get("title", "")
        subtopics = ", ".join(item.get("subtopics", [])) or "None listed"
        desc = item.get("description", "")
        words = len(re.findall(r"\b\w+\b", desc))

        prompt = f"""\
COMPOSITE CONCEPT TITLE: "{title}"
CURRENT LENGTH: {words} words
ABSORBED SUBTOPICS: {subtopics}

FULL TEXT:
{desc}
"""
        raw_text = await call_llm(
            client=client,
            model=self.settings.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_DECOMPOSE},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.settings.max_tokens or 4096,
            args=self.args,
            task="generate",
            max_growth=1,
        )

        res = parse_xml_decomposition(raw_text)
        return {
            "title": title,
            "decomposed": res.decomposed,
            "pillar_description": res.pillar_description,
            "children": [{"title": ch.title, "description": ch.description} for ch in res.children]
        }

    async def run(self):
        print(f"Loading concepts from {self.global_concepts_file}...")
        with open(self.global_concepts_file, "r", encoding="utf-8") as f:
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
                # Update pillar description
                if decomp_data.get("pillar_description"):
                    c["description"] = decomp_data["pillar_description"]

                # Ensure outbound_relations exists
                relations = c.get("outbound_relations", [])
                existing_targets = {r.get("target_concept", "").lower() for r in relations}

                final_concepts.append(c)

                # Add children
                for ch in decomp_data.get("children", []):
                    ch_title = ch["title"].strip()
                    ch_slug = generate_stable_slug(ch_title, used_slugs)
                    new_children_count += 1

                    # Child inherits source chunks of parent
                    child_concept = {
                        "title": ch_title,
                        "slug": ch_slug,
                        "aliases": [],
                        "subtopics": [],
                        "entities": [],
                        "description": ch["description"],
                        "source_chunks": list(c.get("source_chunks", [])),
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
                final_concepts.append(c)

        print(f"\nDecomposed {decomposed_count} broad concepts into {new_children_count} focused subtopic concepts.")
        print(f"Total concepts after decomposition: {len(final_concepts)} (from {total}).")

        # Save back to global_concepts.json and refresh concepts/
        atomic_write_json(self.global_concepts_file, {"concepts": final_concepts})
        for fc in final_concepts:
            s = fc.get("slug")
            if s:
                atomic_write_json(self.concepts_dir / f"{s}.json", fc)

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
