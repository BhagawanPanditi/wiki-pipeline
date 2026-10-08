#!/usr/bin/env python3
"""
Step 4A: Dedicated Stub Resolution Pipeline
--------------------------------------------
1. Resolves Major Stubs (> 2 source chunks) via LLM extraction per chunk.
2. Absorbs Minor Stubs (<= 2 source chunks or downgraded) directly into 
   pointing parent concepts' descriptions and entities arrays.
3. Preserves citation provenances (source_chunks) for all absorptions.
4. Outputs: `domain_dir/global_concepts_stubs_resolved.json`.
"""

import re
import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from tqdm.asyncio import tqdm
from pydantic import BaseModel, Field
import httpx
from openai import AsyncOpenAI

# ── Configuration ────────────────────────────────────────────────────────────
from pipeline_config import load_global_config
from atomic_io import atomic_write_json
DEFAULT_SERVERS, DEFAULT_CONCURRENCY_PER_SERVER, GLOBAL_MODEL = load_global_config()

class StubRelation(BaseModel):
    target_concept: str
    relation: str

class StubConceptExtraction(BaseModel):
    is_valid_concept: bool = Field(description="True if enough context exists to define this, False if too vague.")
    title: str
    aliases: List[str] = Field(default_factory=list)
    description: str
    entities: List[str] = Field(default_factory=list)
    outbound_relations: List[StubRelation] = Field(default_factory=list)

SYSTEM_PROMPT = """\
You are an expert technical knowledge graph agent tasked with defining a missing concept (a 'stub').
A stub is a concept that was referenced by other entities, but was never explicitly extracted.

YOUR MISSION:
Analyze the provided RAW TEXT chunk (with context before/after) and the EXISTING RELATIONS pointing to the stub.
Using ONLY this provided context, extract the stub as a fully fleshed-out concept.

RULES:
1. is_valid_concept: If the text provides enough clues to write a meaningful, factual description, set to true. If generic/noise, set to false.

FACT vs. ENTITY STRICT SEPARATION:
1. DESCRIPTIONS HOLD ALL FACTS: Write a single, dense, factual paragraph detailing what the stub is. ALL quantitative metrics, ranges, parameters, and experimental findings MUST be written in full complete sentences inside `description`.
2. ENTITIES LIST: `entities` is STRICTLY a flat list of short named string identifiers (e.g., tools, gene symbols, datasets).

EXHAUSTIVE FACTUAL RETENTION MANDATE:
Descriptions MUST retain exact detail. You MUST preserve:
1. Exact numbers, statistical findings, and percentages.
2. Specific versions, taxonomies, and formal classifications.
3. Specific operational bounds, environmental constraints, and dates.

OUTBOUND RELATIONSHIPS:
- Describe how the stub relates outward to Target Concept B in 1 concise sentence.
- DO NOT hallucinate targets. ONLY link to concepts explicitly named in the provided context. If none, return [].

OUTPUT FORMAT:
Return valid JSON:
{
  "is_valid_concept": true,
  "title": "Exact Name of Stub",
  "aliases": ["Alt Name"],
  "description": "Dense factual paragraph with metrics...",
  "entities": ["Tool", "Gene"],
  "outbound_relations": [{"target_concept": "Target Title", "relation": "1 sentence relation"}]
}
"""

def clean_json_string(raw: str) -> str:
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if match: return match.group(1).strip()
    return raw.strip()

class StubResolver:
    def __init__(self, args):
        self.domain_dir = Path(args.wiki_base_dir) / args.domain
        self.staging_dir = self.domain_dir / "staging"
        self.cache_dir = self.domain_dir / "cache"
        self.global_file = self.domain_dir / "global_concepts.json"
        self.output_file = self.domain_dir / "global_concepts_stubs_resolved.json"
        self.log_file = self.domain_dir / "global_consolidation.json"
        
        self.model_name = args.model
        self.max_tokens = args.max_tokens
        self.servers = args.servers
        self.concurrency = args.concurrency_per_server

        self.global_concepts = []
        self.canonical_titles = set()
        self.chunk_text_map: Dict[str, dict] = {}

    def load_data(self):
        print("Loading global concepts & log data...")
        if not self.global_file.exists():
            raise FileNotFoundError(f"Missing {self.global_file}. Run global_concept_consolidation.py first.")

        with open(self.global_file, "r", encoding="utf-8") as f:
            self.global_concepts = json.load(f).get("concepts", [])
        
        for c in self.global_concepts:
            if c.get("title"):
                self.canonical_titles.add(c["title"].lower())

        with open(self.log_file, "r", encoding="utf-8") as f:
            self.log_data = json.load(f)
            
        self.stubs = self.log_data.get("unresolved_stubs", [])
        
        print("Loading chunk text cache...")
        cache_file = self.cache_dir / "chunks_cache.json"
        if not cache_file.exists():
            raise FileNotFoundError(f"Missing text cache: {cache_file}")
            
        with open(cache_file, "r", encoding="utf-8") as f:
            for c in json.load(f):
                self.chunk_text_map[f"{c['doc_id']}::{c['chunk_id']}"] = c

        print(f"Loaded {len(self.global_concepts)} concepts, {len(self.stubs)} unresolved stubs, and {len(self.chunk_text_map)} text chunks.")

    def build_single_chunk_prompt(self, stub_name: str, sc: str) -> str:
        if "::" not in sc: return ""
        doc_id, chunk_id = sc.split("::", 1)
        stub_lower = stub_name.lower()
        cached_text = self.chunk_text_map.get(sc)
        relations_text = []
        
        concepts_file = self.staging_dir / doc_id / chunk_id / "concepts.json"
        if concepts_file.exists():
            try:
                with open(concepts_file, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                for c in cdata.get("concepts", []):
                    c_title = c.get("title", "")
                    for rel in c.get("outbound_relations", []):
                        if rel.get("target_concept", "").strip().lower() == stub_lower:
                            rel_desc = rel.get("relation", "").strip()
                            relations_text.append(f"- '{c_title}' links to this stub: {rel_desc}")
            except Exception: pass

        if not cached_text and not relations_text: return ""

        context_lines = [f"\n====================\n--- SOURCE: {sc} ---"]
        if cached_text:
            context_lines.append(f"CONTEXT BEFORE:\n{cached_text.get('context_before', '')}\n")
            context_lines.append(f"CURRENT CHUNK:\n{cached_text.get('current_chunk', '')}\n")
            context_lines.append(f"CONTEXT AFTER:\n{cached_text.get('context_after', '')}\n")
            
        if relations_text:
            context_lines.append("EXISTING RELATIONS POINTING TO THIS STUB:")
            context_lines.extend(relations_text)
            
        return "\n".join(context_lines)

    async def resolve_major_stub(self, client: AsyncOpenAI, stub: dict) -> dict:
        stub_name = stub["stub_name"]
        source_chunks = stub.get("source_chunks", [])
        sem = asyncio.Semaphore(5)
        failed_chunks = []

        async def process_chunk(sc: str):
            async with sem:
                context_str = self.build_single_chunk_prompt(stub_name, sc)
                if not context_str.strip():
                    failed_chunks.append(sc)
                    return sc, None
                user_prompt = f"TARGET STUB TO DEFINE: {stub_name}\n\n{context_str}"
                
                for attempt in range(10):
                    try:
                        response = await asyncio.wait_for(
                            client.chat.completions.create(
                                model=self.model_name,
                                messages=[
                                    {"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": user_prompt}
                                ],
                                temperature=0.1, max_tokens=self.max_tokens, response_format={"type": "json_object"}
                            ),
                            timeout=60.0
                        )
                        data = StubConceptExtraction.model_validate_json(clean_json_string(response.choices[0].message.content))
                        return sc, data
                    except Exception:
                        if attempt == 9:
                            failed_chunks.append(sc)
                            return sc, None
                        await asyncio.sleep(min(1.5 * (attempt + 1), 8.0))

        results = await asyncio.gather(*(process_chunk(sc) for sc in source_chunks))
        valid_extractions = [(sc, data) for sc, data in results if data and data.is_valid_concept]

        if failed_chunks:
            return {"status": "failed", "stub": stub_name, "failed_chunks": sorted(set(failed_chunks))}
        
        if not valid_extractions:
            return {"status": "downgraded", "stub": stub_name}
        
        merged_aliases, merged_descriptions, seen_sentences, merged_entities = set(), [], set(), set()
        merged_relations_map = {}
        
        for sc, data in valid_extractions:
            merged_aliases.update(data.aliases)
            merged_entities.update(data.entities)
            for sent in re.split(r'(?<=[.!?])\s+', data.description.strip()):
                clean_sent = sent.strip()
                if clean_sent and clean_sent.lower() not in seen_sentences:
                    seen_sentences.add(clean_sent.lower())
                    merged_descriptions.append(clean_sent)
            
            for rel in data.outbound_relations:
                if rel.target_concept.lower() in self.canonical_titles:
                    tgt, rel_txt = rel.target_concept, rel.relation.strip()
                    key = (tgt, rel_txt)
                    merged_relations_map.setdefault(key, set()).add(sc)
                    
        final_relations = [
            {"target_concept": tgt, "relation": rel_txt, "source_chunks": sorted(list(sc_set))}
            for (tgt, rel_txt), sc_set in merged_relations_map.items()
        ]
            
        return {
            "status": "success",
            "stub": stub_name,
            "concept": {
                "title": stub_name,
                "aliases": sorted(list(merged_aliases)),
                "description": " ".join(merged_descriptions),
                "entities": sorted(list(merged_entities)),
                "source_chunks": sorted([sc for sc, _ in valid_extractions]),
                "outbound_relations": final_relations
            }
        }

    async def run(self):
        self.load_data()
        major_stubs = [s for s in self.stubs if len(s.get("source_chunks", [])) > 2]
        minor_stubs_set = {s["stub_name"] for s in self.stubs if len(s.get("source_chunks", [])) <= 2}
        
        new_concepts = []
        failed_stubs = []
        if major_stubs and not self.servers:
            raise RuntimeError("Major stub resolution requires at least one configured LLM server.")
        if major_stubs:
            print(f"Resolving {len(major_stubs)} Major Stubs (>2 chunks) via LLM pool...")
            queue = asyncio.Queue()
            for s in major_stubs: await queue.put(s)
            
            effective_concurrency = min(self.concurrency, 15)
            for _ in range(len(self.servers) * effective_concurrency): await queue.put(None)

            pbar = tqdm(total=len(major_stubs), desc="LLM Major Stub Resolution")
            async def worker(client: AsyncOpenAI):
                while True:
                    stub = await queue.get()
                    if stub is None: queue.task_done(); break
                    res = await self.resolve_major_stub(client, stub)
                    if res["status"] == "success":
                        new_concepts.append(res["concept"])
                    elif res["status"] == "failed":
                        failed_stubs.append({"stub": res["stub"], "failed_chunks": res.get("failed_chunks", [])})
                    else:
                        minor_stubs_set.add(res["stub"])
                    pbar.update(1)
                    queue.task_done()

            tasks, http_clients = [], []
            for url in self.servers:
                hc = httpx.AsyncClient(limits=httpx.Limits(max_connections=effective_concurrency + 10), timeout=httpx.Timeout(45.0))
                http_clients.append(hc)
                client = AsyncOpenAI(base_url=url, api_key="EMPTY", http_client=hc, max_retries=0)
                for _ in range(effective_concurrency): tasks.append(asyncio.create_task(worker(client)))

            await asyncio.gather(*tasks)
            pbar.close()
            for hc in http_clients: await hc.aclose()

            if failed_stubs:
                raise RuntimeError(
                    f"Major stub resolution failed for {len(failed_stubs)} stub(s) after retries. "
                    f"Examples: {failed_stubs[:5]}"
                )

        minor_stubs_lower_map = {s.lower(): s for s in minor_stubs_set}
        print(f"Absorbing {len(minor_stubs_lower_map)} Minor Stubs directly into parent concepts...")
        
        for concept in tqdm(self.global_concepts, desc="Absorbing Minor Stubs"):
            retained_relations = []
            seen_targets = set()
            for rel in concept.get("outbound_relations", []):
                tgt = rel.get("target_concept", "").strip()
                if tgt.lower() in minor_stubs_lower_map:
                    canonical_stub_name = minor_stubs_lower_map[tgt.lower()]
                    rel_text = rel.get("relation", "").strip()
                    desc = concept.get("description", "").strip()
                    addition = f"Additionally, it is associated with {canonical_stub_name}: {rel_text}" if rel_text else f"Additionally, it is related to {canonical_stub_name}."
                    
                    if desc:
                        concept["description"] = f"{desc}\n\n{addition}"
                    else:
                        concept["description"] = addition
                    concept_sources = set(concept.get("source_chunks", []))
                    concept_sources.update(rel.get("source_chunks", []))
                    concept["source_chunks"] = sorted(list(concept_sources))
                    
                    if canonical_stub_name not in concept.get("entities", []):
                        concept.setdefault("entities", []).append(canonical_stub_name)
                        concept["entities"].sort()

                    # Check if stub can be repointed to an existing canonical concept
                    if tgt.lower() in self.canonical_titles and tgt.lower() != concept["title"].lower():
                        if tgt.lower() not in seen_targets:
                            seen_targets.add(tgt.lower())
                            retained_relations.append(rel)
                else:
                    if tgt.lower() not in seen_targets:
                        seen_targets.add(tgt.lower())
                        retained_relations.append(rel)
            concept["outbound_relations"] = retained_relations

        for nc in new_concepts:
            self.global_concepts.append(nc)

        print(f"Saving resolved concepts to {self.output_file}...")
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.output_file, {"concepts": self.global_concepts})

        self.log_data["stub_resolution_summary"] = {
            "major_stubs_synthesized": len(new_concepts),
            "minor_stubs_absorbed": len(minor_stubs_set),
            "concepts_after_stub_pass": len(self.global_concepts)
        }
        atomic_write_json(self.log_file, self.log_data)

        print(f"\n==================================================")
        print(f"STUB RESOLUTION COMPLETE")
        print(f"==================================================")
        print(f"Major Stubs Synthesized via LLM: +{len(new_concepts)}")
        print(f"Minor Stubs Absorbed into Parents: {len(minor_stubs_set)}")
        print(f"Total Concepts in Graph:         {len(self.global_concepts)}")
        print(f"Output: {self.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4A: Resolve Stubs")
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--wiki_base_dir", type=str, default="/home/compiling-ganesh/24m0829/long_document/wiki")
    parser.add_argument("--model", type=str, default=GLOBAL_MODEL)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--servers", nargs="+", default=DEFAULT_SERVERS)
    parser.add_argument("--concurrency_per_server", type=int, default=DEFAULT_CONCURRENCY_PER_SERVER)
    
    args = parser.parse_args()
    resolver = StubResolver(args)
    asyncio.run(resolver.run())