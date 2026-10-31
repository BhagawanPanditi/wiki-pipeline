#!/usr/bin/env python3
"""Stage 7: Concept Brief & Key Entity Generation.
------------------------------------------------
Generates a 2-3 sentence overview brief and extracts important, domain-salient
entities for each consolidated concept in a single direct LLM call.

Usage:
    python3 write_briefs_and_entities.py --wiki_path /path/to/wiki
"""


import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from utils.atomic_io import atomic_write_json, sanitize_filename
from utils.checkpoint import load_bool_log, save_log_sync
from utils.cli import build_parser, resolve_settings
from utils.errors import StageBudget
from utils.llm_params import call_llm
from utils.llm_pool import pool_for_stage
from utils.manifest import verify_stage_manifest, write_stage_manifest

STAGE = "write-briefs"

# ── XML System Prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT_BRIEF_ENTITIES = """\
You are an expert knowledge agent.
Your task is to analyze a comprehensive concept description and produce two vital outputs in ONE pass:
1. A condensed BRIEF (1-3 sentences, ~25-60 words).
2. A list of 5 to 12 IMPORTANT ENTITIES mentioned in the text.

RULES FOR BRIEF:
- 1-3 complete sentences, about 25-60 words. If the description is under ~60 words, write ONE accurate sentence; never pad or infer.
- Keep exact numbers, metrics, and identifiers. Silently omit internal provenance IDs like [doc_...].
- No noise: do not add conversational filler.

RULES FOR ENTITIES (5-12):
- Specific named things in the text: proper names, standard terms of art, identifiers, quantities with units, named methods, or rules.
- Use the standard singular, unqualified surface form.
- Exclude words so vague they would fit almost any concept in the corpus.
- Each entity 1-4 words in Title Case.

OUTPUT FORMAT:
Output ONLY the <brief>...</brief> and <entities>...</entities> XML blocks:
<brief>
First dense sentence defining the core concept and its primary role. Second sentence explaining the primary mechanism, rule, or parameter.
</brief>
<entities>
- Primary Entity A
- Method or Rule B
- Tool or System C
- Component or Metric D
</entities>
"""

class BriefAndEntities(BaseModel):
    brief: str = Field(default="")
    entities: List[str] = Field(default_factory=list)

# ── XML Parsing Helper ────────────────────────────────────────────────────────

def parse_xml_brief_and_entities(text: str) -> BriefAndEntities:
    """Parse <brief> and <entities> XML blocks directly. Fails loud if tags are missing."""
    if not text:
        raise ValueError("Empty response received from LLM")

    brief_m = re.search(r"<brief>(.*?)</brief>", text, re.DOTALL | re.IGNORECASE)
    if not brief_m or not brief_m.group(1).strip():
        raise ValueError("Missing or empty <brief> block in LLM response")

    raw_brief = brief_m.group(1).strip()
    # Strip citation brackets like [doc_...::chunk_...]
    raw_brief = re.sub(r"\[(?:doc_|canon_)[^\]]+\]", "", raw_brief)
    brief = " ".join(raw_brief.split())

    entities = []
    entities_m = re.search(r"<entities>(.*?)</entities>", text, re.DOTALL | re.IGNORECASE)
    if not entities_m:
        raise ValueError("Missing <entities> block in LLM response")

    raw_entities = entities_m.group(1).strip().splitlines()
    for line in raw_entities:
        cleaned = re.sub(r"^[-*•\d\.\s]+", "", line).strip()
        if cleaned and len(cleaned) > 2 and len(cleaned) < 80:
            entities.append(cleaned)

    if not entities:
        raise ValueError("Zero valid entities found in <entities> block")

    return BriefAndEntities(brief=brief, entities=entities)

# ── Pipeline Implementation ──────────────────────────────────────────────────

class BriefAndEntitiesPipeline:
    def __init__(self, args):
        self.args = args
        wiki_path = getattr(args, "wiki_path", None)
        if not wiki_path:
            raise ValueError("Missing required --wiki_path argument.")

        self.domain_dir = Path(wiki_path)
        self.global_concepts_file = self.domain_dir / "global_concepts.json"
        self.concepts_dir = self.domain_dir / "concepts"
        self.log_file = self.domain_dir / "briefs_entities_log.json"

        if not self.global_concepts_file.exists():
            raise FileNotFoundError(
                f"Missing {self.global_concepts_file}. Complete Stage 6 (consolidate_global.py) first."
            )

        verify_stage_manifest(self.domain_dir, "consolidate-global")

        self.concepts_dir.mkdir(parents=True, exist_ok=True)
        if getattr(args, "resume", False):
            self.checkpoint = load_bool_log(self.log_file, corrupt_message="Log corrupted. Initializing fresh.")
            # Also populate checkpoint from disk if concepts/ already has enriched files
            for c_file in self.concepts_dir.glob("*.json"):
                try:
                    with open(c_file, "r", encoding="utf-8") as cf:
                        disk_data = json.load(cf)
                        t = disk_data.get("title")
                        if t and disk_data.get("brief") and "entities" in disk_data:
                            self.checkpoint[t] = True
                except Exception:
                    pass
        else:
            self.checkpoint = {}
        self.settings = resolve_settings(args, STAGE)
        self.budget = StageBudget(self.settings.budget_minutes)
        self.flush_count = 0
        self.log_lock = asyncio.Lock()

    async def generate_brief_and_entities(self, client: Any, item: Dict[str, Any], server_index: int) -> BriefAndEntities:
        title = item.get("title", "")
        aliases = ", ".join(item.get("aliases", []))
        description = item.get("description", "")

        user_prompt = f"CONCEPT TITLE: \"{title}\"\n"
        if aliases:
            user_prompt += f"ALIASES: {aliases}\n"
        user_prompt += f"\nFULL DESCRIPTION:\n{description}\n"

        max_model_len = getattr(self.args, "max_model_len", None) or (self.cfg.get("max_model_len") if hasattr(self, "cfg") else None)
        raw_text = await call_llm(
            client=client,
            model=self.settings.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_BRIEF_ENTITIES},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=self.settings.max_tokens or 1024,
            args=self.args,
            cfg=getattr(self, "cfg", None),
            task="generate",
            max_growth=0,
            max_model_len=max_model_len,
        )
        try:
            return parse_xml_brief_and_entities(raw_text)
        except Exception as e:
            if getattr(self.args, "debug", False):
                print(f"\n[DEBUG Stage 7] Parse failed for concept '{title}'. Exception: {e}")
                print(f"[DEBUG Stage 7] Raw LLM Output:\n{raw_text}\n{'='*60}")
            raise

    async def persist_concept_brief(self, item: Dict[str, Any], res: BriefAndEntities):
        item["brief"] = res.brief
        item["entities"] = res.entities

        # Update individual concept JSON in concepts/ using stable slug (F-12)
        slug = item.get("slug")
        if not slug:
            slug = sanitize_filename(item["title"]) or "unnamed_concept"
        file_path = self.concepts_dir / f"{slug}.json"
        atomic_write_json(file_path, item)

        async with self.log_lock:
            self.checkpoint[item["title"]] = True
            self.flush_count += 1
            if self.flush_count >= 20:
                save_log_sync(self.log_file, self.checkpoint)
                self.flush_count = 0

    def assemble_and_save(self, concepts: List[Dict[str, Any]]):
        """Assemble enriched concepts from disk and update global_concepts.json atomically.
        Fails loud if any concept is missing its LLM-generated brief.
        """
        print(f"Assembling enriched concepts from {self.concepts_dir}...")
        assembled_concepts = []
        missing_briefs = []

        for c in concepts:
            slug = c.get("slug")
            if not slug:
                slug = sanitize_filename(c.get("title", "")) or "unnamed_concept"
            c_file = self.concepts_dir / f"{slug}.json"
            if c_file.exists():
                try:
                    with open(c_file, "r", encoding="utf-8") as cf:
                        disk_data = json.load(cf)
                        if disk_data.get("brief"):
                            c["brief"] = disk_data["brief"]
                        if disk_data.get("entities"):
                            c["entities"] = disk_data["entities"]
                except Exception:
                    pass

            if not c.get("brief"):
                missing_briefs.append(c.get("title", slug))
            assembled_concepts.append(c)

        if missing_briefs:
            raise RuntimeError(
                f"Stage 7 incomplete: {len(missing_briefs)} concept(s) missing LLM brief. "
                f"Examples: {missing_briefs[:10]}. Not writing global_concepts.json."
            )

        print(f"Updating {self.global_concepts_file} with briefs and entities...")
        atomic_write_json(self.global_concepts_file, {"concepts": assembled_concepts})

        write_stage_manifest(
            wiki_path=self.domain_dir,
            stage=STAGE,
            expected=len(concepts),
            completed=len(assembled_concepts),
            failed=0,
        )

    async def run(self):
        print(f"Loading concepts from {self.global_concepts_file}...")
        with open(self.global_concepts_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            concepts = data.get("concepts", [])

        total = len(concepts)
        pending = [c for c in concepts if not self.checkpoint.get(c.get("title", ""), False)]

        print(f"Total concepts: {total} | Pending brief & entity extraction: {len(pending)}")
        settings = self.settings

        try:
            if pending:
                print(f"Distributing across {len(settings.servers)} server(s).")
                print(f"Per-item attempts: {settings.attempts} | Stage budget: {settings.budget_minutes:g} min")

                pool = pool_for_stage(self.args, settings, STAGE)
                async with pool:
                    res = await pool.map(
                        pending,
                        self.generate_brief_and_entities,
                        attempts=settings.attempts,
                        desc="Generating Briefs & Entities",
                        key_fn=lambda c: c.get("title", ""),
                        budget=self.budget,
                        attempt_timeout=settings.profile.attempt_timeout,
                        on_result=self.persist_concept_brief,
                    )
                    res.raise_if_failed(STAGE)
        finally:
            save_log_sync(self.log_file, self.checkpoint)
            self.assemble_and_save(concepts)

        print("\n============================================================")
        print("STAGE 7: BRIEF & IMPORTANT ENTITIES EXTRACTION COMPLETE")
        print(f"Total Concepts Processed: {total}")
        print(f"Output saved directly to {self.global_concepts_file}")
        print("============================================================\n")

def main():
    parser = build_parser(stage=STAGE, description="Extract condensed brief (<5 lines) and vital domain entities", include_resume=True)
    args = parser.parse_args()

    pipeline = BriefAndEntitiesPipeline(args)
    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()
