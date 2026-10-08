#!/usr/bin/env python3
"""
Concept Brief Generator Pass
----------------------------
1. Loads `global_concepts_refined.json`.
2. Generates a condensed, 2-3 sentence (~40-60 word / 4-5 line) factual `brief` for each concept.
3. Automatically reuses descriptions if they are already short (<= 40 words).
4. Updates `global_concepts_refined.json` and all individual concept files in `concepts/`.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from utils.atomic_io import atomic_write_json, sanitize_filename
from utils.cli import build_parser, resolve_settings
from utils.event_log import EventLog
from utils.text import first_sentences
from utils.layout import log_path
from utils.errors import StageBudget
from utils.json_repair import clean_json_string
from utils.llm_pool import pool_for_stage
from utils.citation_utils import clean_brief_text

STAGE = "write-briefs"

# ── Data Schema ──────────────────────────────────────────────────────────────
class ConceptBriefSchema(BaseModel):
    brief: str = Field(description="A 2-3 sentence dense factual summary (40-60 words) capturing the core mechanism, function, and metrics of the concept.")

# ── System Prompt for Brief Generation ──────────────────────────────────────
SYSTEM_PROMPT_BRIEF = """\
You are an expert technical knowledge graph agent.
Your task is to summarize a long technical concept description into a condensed BRIEF.

RULES FOR BRIEF GENERATION:
1. LENGTH: Exactly 2 to 3 dense, well-formed complete sentences (~40 to 60 words).
2. CONTENT: Capture the core definition, primary technical mechanism, key parameters, or biological function.
3. RETENTION: Retain exact specific numbers, metrics, or biological/technical codes where relevant.
4. NO CITATIONS: SILENTLY OMIT all paper citations, DOIs, author names, and internal provenance IDs such as [canon_abc123] or [canon_abc123::chunk_001]. The brief must contain no canon_ IDs or chunk IDs.
5. NO NOISE: SILENTLY OMIT conversational introductory filler (e.g. "This concept is...") and return the brief as a single line.

OUTPUT FORMAT:
Return valid JSON:
{
  "brief": "First dense sentence explaining the core concept. Second sentence covering primary mechanism or key quantitative finding."
}
"""

# ── Helper Utilities ─────────────────────────────────────────────────────────

# ── LLM Async Worker ─────────────────────────────────────────────────────────
async def generate_brief_llm(client, concept, settings) -> str:
    """One brief. Raises on any unusable response so the pool can classify and
    retry it; the caller falls back to a truncated description if retries run
    out, which is why this stage never aborts a run."""
    title = concept.get("title", "")
    description = concept.get("description", "")
    user_prompt = f"CONCEPT TITLE: {title}\n\nFULL DESCRIPTION:\n{description}"

    response = await client.chat.completions.create(
        model=settings.model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_BRIEF},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=settings.max_tokens,
        response_format={"type": "json_object"}
    )
    raw = response.choices[0].message.content
    parsed = ConceptBriefSchema.model_validate_json(clean_json_string(raw))
    cleaned_brief = clean_brief_text(parsed.brief)
    if len(cleaned_brief) <= 10:
        raise ValueError(f"brief too short ({len(cleaned_brief)} chars) for '{title}'")
    return cleaned_brief

# ── Main Pipeline ────────────────────────────────────────────────────────────
class ConceptBriefPipeline:
    def __init__(self, args):
        """Prepare a write-briefs run. Nothing is read or written yet.

        Args:
            args: Parsed CLI namespace with ``domain``, ``wiki_base_dir`` and the shared
                stage flags.

        Sets:
            The resolved stage settings, the run directory, the refined-graph path, the
            debug log and the time budget."""
        self.args = args
        self.settings = resolve_settings(args, STAGE)
        self.domain_dir = Path(args.wiki_base_dir) / args.domain
        self.refined_file = self.domain_dir / "global_concepts_refined.json"
        self.concepts_dir = self.domain_dir / "concepts"
        self.debug = EventLog(log_path(self.domain_dir, STAGE, "debug"),
                                 label=STAGE)
        self.budget = StageBudget(self.settings.budget_minutes)

        self.concepts: List[Dict[str, Any]] = []

    def load_data(self):
        """Load the refined concept graph into ``self.concepts``.

        Raises:
            FileNotFoundError: If ``global_concepts_refined.json`` does not exist yet."""
        if not self.refined_file.exists():
            raise FileNotFoundError(f"Missing refined graph file: {self.refined_file}")

        print(f"Loading concepts from {self.refined_file}...")
        with open(self.refined_file, "r", encoding="utf-8") as f:
            self.concepts = json.load(f).get("concepts", [])
        print(f"Loaded {len(self.concepts)} total concepts.")

    async def run(self):
        """Give every concept a brief, then rewrite the refined graph.

        A concept that already has a brief keeps it. The rest go to the LLM through the
        server pool. Any that still fail after retries get a deterministic fallback made
        from their first two sentences, so one bad response never aborts the run.

        Returns:
            None. Writes ``global_concepts_refined.json`` in place."""
        self.load_data()

        # Identify concepts that need LLM generation vs. simple reuse
        tasks_to_generate = []
        already_briefed = 0

        for concept in self.concepts:
            desc = concept.get("description", "").strip()
            words = desc.split()

            # Rule: If description is <= 40 words or <= 2 sentences, reuse description directly (with citations stripped)
            sentences = [s for s in re.split(r'(?<=[.!?])\s+', desc) if s.strip()]
            if len(words) <= 40 or len(sentences) <= 2:
                concept["brief"] = clean_brief_text(desc)
                already_briefed += 1
            else:
                tasks_to_generate.append(concept)

        print(f"Directly set brief for {already_briefed} short descriptions.")
        print(f"Queued {len(tasks_to_generate)} long concept descriptions for LLM brief synthesis.")

        if tasks_to_generate and self.settings.servers:
            settings = self.settings
            print(f"Spreading {len(tasks_to_generate)} brief(s) across "
                  f"{len(settings.servers)} server(s) at "
                  f"{settings.concurrency_per_server} concurrent request(s) each "
                  f"(item i -> server i % N).")
            print(f"Per-item attempts: {settings.attempts} | Stage budget: "
                  f"{settings.budget_minutes:g} min | Debug log: {self.debug.path}")

            async def handler(client, concept, server_index):
                """Request one brief from the LLM for one concept.

                Args:
                    client: The ``AsyncOpenAI`` client bound to one server.
                    concept: The concept to describe.
                    server_index: Index of the serving server, used only for accounting.

                Returns:
                    The brief text from :func:`generate_brief_llm`.

                Raises:
                    Whatever :func:`generate_brief_llm` raises, so the pool can retry it."""
                return await generate_brief_llm(client, concept, settings)

            async def apply_brief(concept, brief_text):
                """Attach a finished brief to its concept. Called in completion order.

                Args:
                    concept: The concept dict that was briefed.
                    brief_text: The brief to store under ``brief``.

                Returns:
                    None."""
                concept["brief"] = brief_text

            pool = pool_for_stage(self.args, settings, STAGE)
            try:
                async with pool:
                    result = await pool.map(
                        tasks_to_generate,
                        handler,
                        attempts=settings.attempts,
                        desc="Synthesizing Concept Briefs",
                        key_fn=lambda c: c.get("title", ""),
                        budget=self.budget,
                        debug=self.debug,
                        attempt_timeout=settings.profile.attempt_timeout,
                        on_result=apply_brief,
                    )
            finally:
                self.debug.close()

            # This stage has never aborted a run over a missing brief: anything
            # that exhausted its retries falls back to the first two sentences of
            # the description, exactly as before.
            fallbacks = 0
            for failure in result.failures:
                concept = tasks_to_generate[failure["index"]]
                concept["brief"] = clean_brief_text(
                    first_sentences(concept.get("description", ""), max_sentences=2))
                fallbacks += 1
            if fallbacks:
                print(f"\u26a0\ufe0f  {fallbacks} brief(s) fell back to a truncated description.")

            print("Requests served per server:")
            for url, count in sorted(result.per_server.items(), key=lambda kv: -kv[1]):
                print(f"  {url:<40} {count}")

        # Reorder JSON fields so "brief" appears right after "title" and "aliases"
        reordered_concepts = []
        for c in self.concepts:
            ordered = {
                "title": c.get("title", ""),
                "aliases": c.get("aliases", []),
                "brief": c.get("brief", ""),
                "description": c.get("description", ""),
                "entities": c.get("entities", []),
                "source_chunks": c.get("source_chunks", []),
                "outbound_relations": c.get("outbound_relations", [])
            }
            reordered_concepts.append(ordered)

        # Save to global_concepts_refined.json
        print(f"\nSaving updated concepts to {self.refined_file}...")
        atomic_write_json(self.refined_file, {"concepts": reordered_concepts})

        # Repopulate concepts/ directory
        print(f"Updating individual concept JSON files in {self.concepts_dir}/...")
        if self.concepts_dir.exists():
            for old_file in self.concepts_dir.glob("*.json"):
                try: old_file.unlink()
                except Exception: pass
        self.concepts_dir.mkdir(parents=True, exist_ok=True)

        for gc in reordered_concepts:
            safe_name = sanitize_filename(gc["title"])
            if not safe_name: safe_name = "unnamed_concept"

            file_path = self.concepts_dir / f"{safe_name}.json"
            counter = 1
            while file_path.exists():
                file_path = self.concepts_dir / f"{safe_name}_{counter}.json"
                counter += 1

            atomic_write_json(file_path, gc)

        print("\n==================================================")
        print("CONCEPT BRIEF GENERATION COMPLETE")
        print("==================================================")
        print(f"Total Concepts Processed:  {len(reordered_concepts)}")
        print(f"Short Descriptions Reused: {already_briefed}")
        print(f"LLM Briefs Generated:       {len(tasks_to_generate)}")
        print(f"Outputs written to:\n  - {self.refined_file}\n  - {self.concepts_dir}/")

# ── CLI Entrypoint ────────────────────────────────────────────────────────────
def main():
    """Parse the command line and run brief generation.

    Returns:
        None."""
    parser = build_parser(
        stage=STAGE,
        description="Generate 2-3 sentence briefs for knowledge graph concepts")
    args = parser.parse_args()
    pipeline = ConceptBriefPipeline(args)
    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()
