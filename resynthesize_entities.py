#!/usr/bin/env python3
"""
Entity Re-synthesizer Pass
-------------------------
Re-extracts clean, high-precision named entities (max 10 key entities)
for every concept in a domain using the LLM pool.

Replaces the bloated/noisy entity lists from legacy extraction passes.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from utils.atomic_io import atomic_write_json
from utils.cli import build_parser, resolve_settings
from utils.event_log import EventLog
from utils.layout import log_path
from utils.errors import StageBudget
from utils.json_repair import clean_json_string
from utils.llm_pool import pool_for_stage

STAGE = "resynthesize-entities"

# ── Data Schema ──────────────────────────────────────────────────────────────
class ConceptEntitiesSchema(BaseModel):
    entities: List[str] = Field(
        default_factory=list,
        description="At most 10 key named entities, tools, genes, datasets, proteins, dates, or specific technical terms mentioned in the description."
    )

# ── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT_ENTITIES = """\
You are an expert technical knowledge extractor.
Analyze the concept title and its description, and extract at most 10 KEY NAMED ENTITIES.

RULES:
1. QUANTITY: Extract at most 10 of the most essential named entities directly relevant to this concept.
2. TYPES: Flat list of specific tools, software, algorithms, gene symbols, protein names, dataset codes, formal benchmarks, biological pathways, study authors, or precise technical terms mentioned in the description.
3. CONCISENESS: Each entity must be a concise string identifier (e.g., "CRISPR-Cas9", "HG19", "PyTorch", "Hippocampus", "SOX2"). Do NOT include full sentences or explanatory clauses.
4. NO INVENTIONS: Extract ONLY terms present or explicitly described in the provided description.

OUTPUT FORMAT:
Return valid JSON matching this exact structure:
{
  "entities": ["Entity 1", "Entity 2", "Entity 3"]
}
"""

async def extract_concept_entities(client, concept: Dict[str, Any], settings) -> List[str]:
    """Extract at most 10 entities for one concept."""
    user_prompt = f"Concept Title: {concept.get('title', '')}\n\nDescription:\n{concept.get('description', '')}"

    resp = await client.chat.completions.create(
        model=settings.model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_ENTITIES},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=256,
        response_format={"type": "json_object"}
    )
    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        raise ValueError("Empty completion for entity extraction")

    try:
        cleaned_json = clean_json_string(raw, allow_bare_object=True)
        parsed = ConceptEntitiesSchema.model_validate_json(cleaned_json)
        entities = [e.strip() for e in parsed.entities if e and isinstance(e, str) and e.strip()]
        return entities[:10]
    except Exception as e:
        raise ValueError(f"Failed to parse entity completion: {e}")

async def process_domain_entities(domain_dir: Path, args):
    # Target graph file to update
    target_files = [
        domain_dir / "global_concepts_refined.json",
        domain_dir / "global_concepts.json",
    ]
    graph_file = None
    for f in target_files:
        if f.is_file():
            graph_file = f
            break

    if not graph_file:
        raise FileNotFoundError(f"No concept graph file found in {domain_dir}")

    print(f"Loading concepts from {graph_file}...")
    with open(graph_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        concepts = data.get("concepts", [])

    print(f"Loaded {len(concepts)} concepts for entity re-synthesis.")
    settings = resolve_settings(args, "extract-concepts")
    debug = EventLog(log_path(domain_dir, STAGE, "debug"), label=STAGE)
    budget = StageBudget(settings.budget_minutes)

    save_lock = asyncio.Lock()
    completed_count = 0

    async def handler(client, concept, server_index):
        return await extract_concept_entities(client, concept, settings)

    async def on_result(concept, entities_list):
        nonlocal completed_count
        concept["entities"] = entities_list
        async with save_lock:
            completed_count += 1
            if completed_count % 100 == 0:
                atomic_write_json(graph_file, {"concepts": concepts})

    pool = pool_for_stage(args, settings, STAGE)
    try:
        async with pool:
            await pool.map(
                concepts,
                handler,
                attempts=settings.attempts,
                desc="Re-extracting Entities",
                key_fn=lambda c: c.get("title", ""),
                budget=budget,
                debug=debug,
                attempt_timeout=settings.profile.attempt_timeout,
                on_result=on_result,
            )
    finally:
        debug.close()

    # Final write
    print(f"Writing updated concepts with clean entities to {graph_file}...")
    atomic_write_json(graph_file, {"concepts": concepts})
    print("Entity re-synthesis complete!")

def main():
    parser = build_parser(stage="extract-concepts", description="Re-synthesize at most 10 key entities per concept")
    args = parser.parse_args()
    domain_dir = Path(args.wiki_base_dir) / args.domain
    asyncio.run(process_domain_entities(domain_dir, args))

if __name__ == "__main__":
    main()
