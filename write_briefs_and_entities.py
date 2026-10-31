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
from utils.llm_pool import pool_for_stage

STAGE = "write-briefs"

# ── XML System Prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT_BRIEF_ENTITIES = """\
You are an expert knowledge agent.
Your task is to analyze a comprehensive concept description and produce two vital outputs in ONE pass:
1. A condensed BRIEF (< 5 lines, 2-3 dense factual sentences, ~40-60 words).
2. A list of IMPORTANT ENTITIES mentioned in the text.

RULES FOR BRIEF:
- Length: Exactly 2 to 3 dense, well-formed complete sentences (< 5 lines, ~40-60 words).
- Content: State the core definition, primary role, key parameters, or functional purpose.
- Exactness: Retain exact numbers, metrics, or specifications where relevant.
- No Citations: SILENTLY OMIT all paper citations, DOIs, and internal provenance IDs like [doc_...].
- No Noise: Silently omit conversational filler and output a clean paragraph.

RULES FOR IMPORTANT ENTITIES:
- Extract salient, specific entities, terms, identifiers, tools, components, metrics, or methods mentioned in the text.
- Strictly EXCLUDE:
  * Overly generic empty words (e.g. "thing", "stuff", "data", "info", "system", "process", "result", "aspect").
  * Trivial conversational filler or generic single common nouns without domain specificity.
- Keep each entity to 1-4 words in Title Case.

OUTPUT FORMAT:
Output ONLY the <brief>...</brief> and <entities>...</entities> XML blocks with no introductory or concluding text:
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
    """Parse <brief> and <entities> XML blocks directly."""
    if not text:
        return BriefAndEntities()

    brief = ""
    brief_m = re.search(r"<brief>(.*?)</brief>", text, re.DOTALL | re.IGNORECASE)
    if brief_m:
        raw_brief = brief_m.group(1).strip()
        # Strip citation brackets like [doc_...::chunk_...] or [canon_...]
        raw_brief = re.sub(r"\[(?:doc_|canon_)[^\]]+\]", "", raw_brief)
        brief = " ".join(raw_brief.split())

    entities = []
    entities_m = re.search(r"<entities>(.*?)</entities>", text, re.DOTALL | re.IGNORECASE)
    if entities_m:
        raw_entities = entities_m.group(1).strip().splitlines()
        for line in raw_entities:
            cleaned = re.sub(r"^[-*•\d\.\s]+", "", line).strip()
            if cleaned and len(cleaned) > 2 and len(cleaned) < 80:
                entities.append(cleaned)

    # Fallback to JSON if model emitted json
    if not brief and "{" in clean_text:
        try:
            m = re.search(r"\{.*\}", clean_text, re.DOTALL)
            if m:
                payload = json.loads(m.group(0))
                brief = payload.get("brief", "").strip()
                entities = payload.get("entities", [])
        except Exception:
            pass

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

        extra_body: Dict[str, Any] = {}
        model_kwargs: Dict[str, Any] = {}
        if getattr(self.args, "disable_thinking", False):
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
            model_kwargs = {
                "temperature": 0.7,
                "top_p": 0.80,
                "presence_penalty": 1.5,
                "extra_body": extra_body,
            }
        else:
            model_kwargs = {
                "temperature": 0.1,
            }

        response = await client.chat.completions.create(
            model=self.settings.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_BRIEF_ENTITIES},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=self.settings.max_tokens or 2048,
            **model_kwargs
        )
        raw_text = response.choices[0].message.content or ""
        result = parse_xml_brief_and_entities(raw_text)

        # Fallback if brief is missing: take first 2 sentences from description
        if not result.brief:
            sentences = re.split(r'(?<=[.!?])\s+', description.strip())
            result.brief = " ".join(sentences[:2])

        return result

    async def persist_concept_brief(self, item: Dict[str, Any], res: BriefAndEntities):
        item["brief"] = res.brief
        item["entities"] = res.entities

        # Update individual concept JSON in concepts/
        safe_name = sanitize_filename(item["title"]) or "unnamed_concept"
        file_path = self.concepts_dir / f"{safe_name}.json"
        counter = 1
        while file_path.exists() and not file_path.is_file():
            file_path = self.concepts_dir / f"{safe_name}_{counter}.json"
            counter += 1

        atomic_write_json(file_path, item)

        async with self.log_lock:
            self.checkpoint[item["title"]] = True
            self.flush_count += 1
            if self.flush_count >= 20:
                save_log_sync(self.log_file, self.checkpoint)
                self.flush_count = 0

    def assemble_and_save(self, concepts: List[Dict[str, Any]]):
        """Assemble enriched concepts from disk and update global_concepts.json atomically."""
        print(f"Assembling enriched concepts from {self.concepts_dir}...")
        assembled_concepts = []
        for c in concepts:
            safe_name = sanitize_filename(c.get("title", "")) or "unnamed_concept"
            c_file = self.concepts_dir / f"{safe_name}.json"
            if c_file.exists():
                try:
                    with open(c_file, "r", encoding="utf-8") as cf:
                        disk_data = json.load(cf)
                        if "brief" in disk_data:
                            c["brief"] = disk_data["brief"]
                        if "entities" in disk_data:
                            c["entities"] = disk_data["entities"]
                except Exception:
                    pass
            # Fallback if brief is still missing: derive from first 2 sentences
            if not c.get("brief"):
                sentences = re.split(r'(?<=[.!?])\s+', c.get("description", "").strip())
                c["brief"] = " ".join(sentences[:2])
            if "entities" not in c:
                c["entities"] = []
            assembled_concepts.append(c)

        print(f"Updating {self.global_concepts_file} with briefs and entities...")
        atomic_write_json(self.global_concepts_file, {"concepts": assembled_concepts})

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
                    await pool.map(
                        pending,
                        self.generate_brief_and_entities,
                        attempts=settings.attempts,
                        desc="Generating Briefs & Entities",
                        key_fn=lambda c: c.get("title", ""),
                        budget=self.budget,
                        attempt_timeout=settings.profile.attempt_timeout,
                        on_result=self.persist_concept_brief,
                    )
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\nInterrupted! Flushing progress and assembling enriched concepts...")
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
