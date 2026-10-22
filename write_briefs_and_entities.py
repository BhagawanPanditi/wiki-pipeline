#!/usr/bin/env python3
"""
Combined Concept Brief & Important Entities Generator Pass
-----------------------------------------------------------
Reads `global_concepts.json` directly (no stub-resolving or thin concept folding needed).
In a single fast LLM call per concept:
1. Generates a condensed, 2-3 sentence (< 5 lines, ~40-60 words) factual `brief`.
2. Extracts domain-vital `entities` (genes, enzymes, proteins, chemicals, algorithms, tools, diseases)
   while strictly excluding generic nouns, common adjectives, and high-level proper names.
Updates `global_concepts.json` and writes individual concept files to `concepts/<title>.json`.
Uses pure XML tags, sandboxed <think> tags, non-thinking mode toggle, and no disk debug logs.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from utils.atomic_io import atomic_write_json, sanitize_filename
from utils.checkpoint import load_bool_log, save_log_async
from utils.cli import build_parser, resolve_settings
from utils.errors import StageBudget
from utils.llm_pool import pool_for_stage

STAGE = "write-briefs"

# ── XML System Prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT_BRIEF_ENTITIES = """\
You are an expert technical knowledge graph agent.
Your task is to analyze a comprehensive concept description and produce two vital outputs in ONE pass:
1. A condensed BRIEF (< 5 lines, 2-3 dense factual sentences, ~40-60 words).
2. A list of IMPORTANT, DOMAIN-VITAL ENTITIES mentioned in the text.

RULES FOR BRIEF:
- Length: Exactly 2 to 3 dense, well-formed complete sentences (< 5 lines, ~40-60 words).
- Content: State the core definition, primary technical mechanism, key parameters, or biological function.
- Exactness: Retain exact numbers, metrics, or technical codes where relevant.
- No Citations: SILENTLY OMIT all paper citations, DOIs, and internal provenance IDs like [canon_...].
- No Noise: Silently omit conversational filler (e.g. "This concept is...") and output a clean paragraph.

RULES FOR IMPORTANT ENTITIES:
- Extract ONLY vital, specific technical/scientific entities: genes, enzymes, proteins, chemicals, pathogens, specific algorithms, protocols, formal software tools, cellular components, or specific medical conditions.
- Strictly EXCLUDE:
  * Generic common nouns or categories (e.g., "cell", "protein", "study", "process", "system", "research", "method", "data").
  * General geographic places, general dates/years, or author names.
  * Trivial qualifiers or non-technical proper nouns.
- Keep each entity to 1-4 words in Title Case.

THINKING PROCESS (SANDBOXED):
- If you need to reason or select entities, put all thinking strictly inside `<think>...</think>` at the very start of your response.
- Once `</think>` is closed, output ONLY the `<brief>...</brief>` and `<entities>...</entities>` XML blocks.
- Absolutely NO commentary or notes outside `<think>...</think>`.

OUTPUT FORMAT:
<think>
[Key mechanism summary and vital domain entity selection]
</think>
<brief>
First dense sentence defining the core concept and its primary role. Second sentence explaining the primary technical mechanism or key parameter.
</brief>
<entities>
- Specific Enzyme A
- Gene B
- Technical Tool C
- Formal Pathway D
</entities>
"""

class BriefAndEntities(BaseModel):
    brief: str = Field(default="")
    entities: List[str] = Field(default_factory=list)

# ── XML Parsing Helper ────────────────────────────────────────────────────────

def parse_xml_brief_and_entities(text: str) -> BriefAndEntities:
    """Parse <brief> and <entities> XML blocks, stripping any thinking tags."""
    if not text:
        return BriefAndEntities()

    clean_text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
    clean_text = re.sub(r"<thought>.*?(?:</thought>|$)", "", clean_text, flags=re.DOTALL | re.IGNORECASE)

    brief = ""
    brief_m = re.search(r"<brief>(.*?)</brief>", clean_text, re.DOTALL | re.IGNORECASE)
    if brief_m:
        raw_brief = brief_m.group(1).strip()
        # Strip citation brackets like [canon_...::chunk_...]
        raw_brief = re.sub(r"\[canon_[^\]]+\]", "", raw_brief)
        brief = " ".join(raw_brief.split())

    entities = []
    entities_m = re.search(r"<entities>(.*?)</entities>", clean_text, re.DOTALL | re.IGNORECASE)
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
        self.checkpoint = load_bool_log(self.log_file, corrupt_message="Log corrupted. Initializing fresh.")
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
                await save_log_async(self.log_file, self.checkpoint)
                self.flush_count = 0

    async def run(self):
        print(f"Loading concepts from {self.global_concepts_file}...")
        with open(self.global_concepts_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            concepts = data.get("concepts", [])

        total = len(concepts)
        pending = [c for c in concepts if not self.checkpoint.get(c.get("title", ""), False)]

        print(f"Total concepts: {total} | Pending brief & entity extraction: {len(pending)}")
        settings = self.settings

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

        await save_log_async(self.log_file, self.checkpoint)

        # Overwrite global_concepts.json with enriched briefs and entities
        print(f"Updating {self.global_concepts_file} with briefs and entities...")
        atomic_write_json(self.global_concepts_file, {"concepts": concepts})

        print("\n============================================================")
        print("STAGE 7: BRIEF & IMPORTANT ENTITIES EXTRACTION COMPLETE")
        print(f"Total Concepts Processed: {total}")
        print(f"Output saved directly to {self.global_concepts_file}")
        print("============================================================\n")

def main():
    parser = build_parser(stage=STAGE, description="Extract condensed brief (<5 lines) and vital domain entities")
    args = parser.parse_args()

    pipeline = BriefAndEntitiesPipeline(args)
    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()
