#!/usr/bin/env python3
"""
Concept Brief Generator Pass
----------------------------
1. Loads `global_concepts_refined.json`.
2. Generates a condensed, 2-3 sentence (~40-60 word / 4-5 line) factual `brief` for each concept.
3. Automatically reuses descriptions if they are already short (<= 40 words).
4. Updates `global_concepts_refined.json` and all individual concept files in `concepts/`.
"""

import re
import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import httpx
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from citation_utils import clean_brief_text
from atomic_io import atomic_write_json

# ── Server Configuration ─────────────────────────────────────────────────────
from pipeline_config import load_global_config
DEFAULT_SERVERS, DEFAULT_CONCURRENCY_PER_SERVER, GLOBAL_MODEL = load_global_config()

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
def clean_json_string(raw: str) -> str:
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if match: return match.group(1).strip()
    return raw.strip()

def sanitize_filename(name: str) -> str:
    keepchars = (' ', '.', '_', '-')
    sanitized = "".join(c for c in name if c.isalnum() or c in keepchars).rstrip()
    return sanitized.replace(" ", "_")[:100]

def get_sentence_fallback(text: str, max_sentences: int = 2) -> str:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    if len(sentences) <= max_sentences:
        return text.strip()
    return " ".join(sentences[:max_sentences])

# ── LLM Async Worker ─────────────────────────────────────────────────────────
async def generate_brief_llm(
    client: AsyncOpenAI, 
    title: str, 
    description: str, 
    model_name: str, 
    max_tokens: int
) -> str:
    user_prompt = f"CONCEPT TITLE: {title}\n\nFULL DESCRIPTION:\n{description}"
    
    for attempt in range(10):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_BRIEF},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"}
                ),
                timeout=45.0  # 45s timeout per call
            )
            raw = response.choices[0].message.content
            parsed = ConceptBriefSchema.model_validate_json(clean_json_string(raw))
            cleaned_brief = clean_brief_text(parsed.brief)
            if len(cleaned_brief) > 10:
                return cleaned_brief
        except Exception:
            if attempt == 9:
                return clean_brief_text(get_sentence_fallback(description, max_sentences=2))
            await asyncio.sleep(min(1.5 * (attempt + 1), 8.0))

    return clean_brief_text(get_sentence_fallback(description, max_sentences=2))

# ── Main Pipeline ────────────────────────────────────────────────────────────
class ConceptBriefPipeline:
    def __init__(self, args):
        self.domain_dir = Path(args.wiki_base_dir) / args.domain
        self.refined_file = self.domain_dir / "global_concepts_refined.json"
        self.concepts_dir = self.domain_dir / "concepts"
        
        self.model_name = args.model
        self.max_tokens = args.max_tokens
        self.servers = args.servers
        self.concurrency = args.concurrency_per_server

        self.concepts: List[Dict[str, Any]] = []

    def load_data(self):
        if not self.refined_file.exists():
            raise FileNotFoundError(f"Missing refined graph file: {self.refined_file}")

        print(f"Loading concepts from {self.refined_file}...")
        with open(self.refined_file, "r", encoding="utf-8") as f:
            self.concepts = json.load(f).get("concepts", [])
        print(f"Loaded {len(self.concepts)} total concepts.")

    async def run(self):
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

        if tasks_to_generate and self.servers:
            queue = asyncio.Queue()
            for c in tasks_to_generate:
                await queue.put(c)

            effective_concurrency = min(self.concurrency, 20)
            total_workers = len(self.servers) * effective_concurrency
            for _ in range(total_workers):
                await queue.put(None)

            pbar = tqdm(total=len(tasks_to_generate), desc="Synthesizing Concept Briefs")

            async def worker(client: AsyncOpenAI):
                while True:
                    concept = await queue.get()
                    if concept is None:
                        queue.task_done()
                        break

                    title = concept.get("title", "")
                    desc = concept.get("description", "")
                    brief_text = await generate_brief_llm(client, title, desc, self.model_name, self.max_tokens)
                    concept["brief"] = brief_text

                    pbar.update(1)
                    queue.task_done()

            tasks, http_clients = [], []
            for server_url in self.servers:
                hc = httpx.AsyncClient(
                    limits=httpx.Limits(max_connections=effective_concurrency + 10),
                    timeout=httpx.Timeout(40.0)
                )
                http_clients.append(hc)
                client = AsyncOpenAI(base_url=server_url, api_key="EMPTY", http_client=hc, max_retries=0)
                for _ in range(effective_concurrency):
                    tasks.append(asyncio.create_task(worker(client)))

            try:
                await asyncio.gather(*tasks)
            finally:
                pbar.close()
                for hc in http_clients:
                    await hc.aclose()

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

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(gc, f, indent=2)

        print(f"\n==================================================")
        print(f"CONCEPT BRIEF GENERATION COMPLETE")
        print(f"==================================================")
        print(f"Total Concepts Processed:  {len(reordered_concepts)}")
        print(f"Short Descriptions Reused: {already_briefed}")
        print(f"LLM Briefs Generated:       {len(tasks_to_generate)}")
        print(f"Outputs written to:\n  - {self.refined_file}\n  - {self.concepts_dir}/")

# ── CLI Entrypoint ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate 2-3 sentence briefs for knowledge graph concepts")
    parser.add_argument("--domain", type=str, required=True, help="Target domain (e.g. biology)")
    parser.add_argument("--wiki_base_dir", type=str, default="/home/compiling-ganesh/24m0829/long_document/wiki")
    parser.add_argument("--model", type=str, default=GLOBAL_MODEL)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--servers", nargs="+", default=DEFAULT_SERVERS)
    parser.add_argument("--concurrency_per_server", type=int, default=DEFAULT_CONCURRENCY_PER_SERVER)

    args = parser.parse_args()
    pipeline = ConceptBriefPipeline(args)
    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()