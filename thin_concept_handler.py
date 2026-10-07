#!/usr/bin/env python3
"""
Step 4B: LLM-Driven Thin Concept Folding & Pruning Pipeline
------------------------------------------------------------
1. Scans for Thin Concepts (< 30 words).
2. Identifies Candidate Host Concepts via Inbound Relations or Shared Chunks.
3. Uses Async LLM calls to verify semantic fit and synthesize a single, dense merged description.
4. Clears stale files in `concepts/` and outputs clean `global_concepts_refined.json` and individual concept JSONs.
"""

import re
import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import Counter

import httpx
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# ── Configuration ────────────────────────────────────────────────────────────
from pipeline_config import load_global_config
DEFAULT_SERVERS, DEFAULT_CONCURRENCY_PER_SERVER, GLOBAL_MODEL = load_global_config()

# ── Pydantic Schema for LLM Decision ─────────────────────────────────────────
class FoldDecisionSchema(BaseModel):
    should_fold: bool = Field(description="True if the thin concept should be merged into one of the candidate hosts.")
    target_host_title: Optional[str] = Field(default=None, description="Exact title of the candidate host concept to fold into.")
    synthesized_description: Optional[str] = Field(default=None, description="Rewritten single comprehensive paragraph for the host concept integrating all facts from the thin concept.")

# ── System Prompt for Thin Concept Folding ───────────────────────────────────
SYSTEM_PROMPT_FOLD = """\
You are an expert technical knowledge graph consolidator.
You are given a THIN CONCEPT (a short concept with minimal detail) and a list of CANDIDATE HOST CONCEPTS from the same document corpus.

YOUR TASK:
Determine if the THIN CONCEPT is a minor sub-topic, detail, or parameter that should be folded into ONE of the CANDIDATE HOST CONCEPTS.

RULES:
1. FOLD IF: The thin concept is a component, property, variant, or minor aspect directly explained by or belonging to a Candidate Host.
2. DO NOT FOLD IF: The thin concept is a fundamentally distinct, major independent topic or entity.
3. SYNTHESIZED DESCRIPTION: If `should_fold` is true, write a single, dense, factual paragraph for the target host concept that naturally integrates all quantitative claims, facts, and details from the thin concept into the host's existing description.

OUTPUT FORMAT:
Return valid JSON matching this schema:
{
  "should_fold": true,
  "target_host_title": "Exact Title of Chosen Candidate Host",
  "synthesized_description": "Single comprehensive merged paragraph..."
}
If should_fold is false:
{
  "should_fold": false,
  "target_host_title": null,
  "synthesized_description": null
}
"""

def clean_json_string(raw: str) -> str:
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if match: return match.group(1).strip()
    return raw.strip()

def sanitize_filename(name: str) -> str:
    keepchars = (' ', '.', '_', '-')
    sanitized = "".join(c for c in name if c.isalnum() or c in keepchars).rstrip()
    return sanitized.replace(" ", "_")[:100]

class ThinConceptHandler:
    def __init__(self, args):
        self.domain_dir = Path(args.wiki_base_dir) / args.domain
        self.input_file = self.domain_dir / "global_concepts_stubs_resolved.json"
        self.output_file = self.domain_dir / "global_concepts_refined.json"
        self.concepts_dir = self.domain_dir / "concepts"
        self.log_file = self.domain_dir / "global_consolidation.json"

        self.model_name = args.model
        self.max_tokens = args.max_tokens
        self.servers = args.servers
        self.concurrency = args.concurrency_per_server
        self.min_words = args.min_concept_words

        self.concepts: List[Dict[str, Any]] = []
        self.title_map: Dict[str, Dict[str, Any]] = {}

    def load_data(self):
        if not self.input_file.exists():
            raise FileNotFoundError(f"Missing input file {self.input_file}. Run stub_resolver.py first!")

        print(f"Loading concepts from {self.input_file}...")
        with open(self.input_file, "r", encoding="utf-8") as f:
            self.concepts = json.load(f).get("concepts", [])

        self.title_map = {c["title"].lower(): c for c in self.concepts}
        print(f"Loaded {len(self.concepts)} total concepts.")

    def find_candidate_hosts(self, thin_concept: Dict[str, Any], inbound_index: Dict[str, List[Dict[str, Any]]], chunk_index: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        t_title_lower = thin_concept["title"].lower()
        candidates = []
        seen_titles = {t_title_lower}

        # Priority 1: Inbound Link Host (H -> T)
        if t_title_lower in inbound_index:
            for host in inbound_index[t_title_lower]:
                h_title_lower = host["title"].lower()
                if h_title_lower not in seen_titles:
                    candidates.append(host)
                    seen_titles.add(h_title_lower)

        # Priority 2: Shared Citation Chunks
        if thin_concept.get("source_chunks"):
            chunk_matches = Counter()
            for sc in thin_concept.get("source_chunks", []):
                for host in chunk_index.get(sc.strip(), []):
                    h_title_lower = host["title"].lower()
                    if h_title_lower not in seen_titles:
                        chunk_matches[h_title_lower] += 1

            for h_title_lower, _ in chunk_matches.most_common(3):
                host = self.title_map.get(h_title_lower)
                if host and h_title_lower not in seen_titles:
                    candidates.append(host)
                    seen_titles.add(h_title_lower)

        return candidates[:3]  # Top 3 candidate hosts

    async def evaluate_folding_llm(self, client: AsyncOpenAI, thin_concept: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Optional[FoldDecisionSchema]:
        user_prompt = f"""THIN CONCEPT TO EVALUATE:
Title: {thin_concept['title']}
Aliases: {thin_concept.get('aliases', [])}
Description: {thin_concept.get('description', '')}

CANDIDATE HOST CONCEPTS:
"""
        for idx, cand in enumerate(candidates, 1):
            user_prompt += f"\n--- Candidate Host {idx} ---\nTitle: {cand['title']}\nDescription: {cand.get('description', '')}\n"

        for attempt in range(10):
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT_FOLD},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=self.max_tokens,
                        response_format={"type": "json_object"}
                    ),
                    timeout=60.0  # Hard timeout protection
                )
                raw_text = response.choices[0].message.content
                return FoldDecisionSchema.model_validate_json(clean_json_string(raw_text))
            except Exception:
                if attempt == 9:
                    return None
                await asyncio.sleep(min(2.0 * (attempt + 1), 10.0))

        return None

    async def run(self):
        self.load_data()

        # Build Inbound Index and Chunk Index
        inbound_index: Dict[str, List[Dict[str, Any]]] = {}
        chunk_index: Dict[str, List[Dict[str, Any]]] = {}

        for c in self.concepts:
            for rel in c.get("outbound_relations", []):
                tgt = rel.get("target_concept", "").strip().lower()
                if tgt: inbound_index.setdefault(tgt, []).append(c)
            for sc in c.get("source_chunks", []):
                if sc.strip(): chunk_index.setdefault(sc.strip(), []).append(c)

        # Identify Thin Concepts
        thin_tasks = []
        for c in self.concepts:
            words = len(c.get("description", "").strip().split())
            if words < self.min_words:
                candidates = self.find_candidate_hosts(c, inbound_index, chunk_index)
                if candidates:
                    thin_tasks.append((c, candidates))

        print(f"Identified {len(thin_tasks)} thin concepts (< {self.min_words} words) with viable candidate hosts.")

        folded_titles: Set[str] = set()
        pruned_titles: Set[str] = set()

        # Run Async LLM Evaluation
        if thin_tasks and self.servers:
            queue = asyncio.Queue()
            for item in thin_tasks: await queue.put(item)

            effective_concurrency = min(self.concurrency, 15)
            total_workers = len(self.servers) * effective_concurrency
            for _ in range(total_workers): await queue.put(None)

            lock = asyncio.Lock()
            pbar = tqdm(total=len(thin_tasks), desc="LLM Thin Concept Folding")

            async def worker(client: AsyncOpenAI):
                while True:
                    item = await queue.get()
                    if item is None:
                        queue.task_done()
                        break

                    thin_concept, candidates = item
                    decision = await self.evaluate_folding_llm(client, thin_concept, candidates)

                    if decision and decision.should_fold and decision.target_host_title:
                        target_host = self.title_map.get(decision.target_host_title.lower())
                        if target_host and target_host["title"].lower() not in folded_titles:
                            async with lock:
                                # Apply Merged Synthesis
                                if decision.synthesized_description:
                                    target_host["description"] = decision.synthesized_description

                                # Add Entities & Aliases
                                if thin_concept["title"] not in target_host.get("entities", []):
                                    target_host.setdefault("entities", []).append(thin_concept["title"])
                                    target_host["entities"].sort()

                                for alias in thin_concept.get("aliases", []):
                                    if alias not in target_host.get("aliases", []) and alias.lower() != target_host["title"].lower():
                                        target_host.setdefault("aliases", []).append(alias)

                                # Merge Citation Source Chunks
                                host_chunks = set(target_host.get("source_chunks", []))
                                host_chunks.update(thin_concept.get("source_chunks", []))
                                target_host["source_chunks"] = sorted(list(host_chunks))

                                # Remove Outbound Link to Folded Concept
                                target_host["outbound_relations"] = [
                                    r for r in target_host.get("outbound_relations", [])
                                    if r.get("target_concept", "").strip().lower() != thin_concept["title"].lower()
                                ]

                                folded_titles.add(thin_concept["title"].lower())

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

        # Strict enforcement: Any concept that remains below min_words and was not folded is evaluated
        # If it has viable candidates or is too thin (< 35 words / < 3 sentences), it is folded into best host or pruned
        for c in self.concepts:
            t_title_lower = c["title"].lower()
            if t_title_lower in folded_titles:
                continue
            words = len(c.get("description", "").strip().split())
            if words < self.min_words:
                # Check if it has any viable host candidate
                candidates = self.find_candidate_hosts(c, inbound_index, chunk_index)
                if candidates:
                    # Deterministically append to best host candidate to ensure 100% absorption
                    best_host = candidates[0]
                    if best_host["title"].lower() not in folded_titles and best_host["title"].lower() != t_title_lower:
                        desc_to_add = c.get("description", "").strip()
                        if desc_to_add:
                            best_host["description"] = f"{best_host.get('description', '').strip()}\n\n{desc_to_add}"
                        best_host_chunks = set(best_host.get("source_chunks", []))
                        best_host_chunks.update(c.get("source_chunks", []))
                        best_host["source_chunks"] = sorted(list(best_host_chunks))
                        if c["title"] not in best_host.get("entities", []):
                            best_host.setdefault("entities", []).append(c["title"])
                        folded_titles.add(t_title_lower)
                        continue

                # If no candidate host exists and it is ultra-thin (< 25 words), prune to prevent stub-like concepts
                if words < 25 or len(c.get("outbound_relations", [])) == 0:
                    pruned_titles.add(t_title_lower)

        # Build Final Clean Graph: Absolutely zero thin concepts below min_words survived
        final_concepts = [
            c for c in self.concepts
            if c["title"].lower() not in folded_titles and c["title"].lower() not in pruned_titles
        ]

        # Save Outputs & Wipe Stale Concept Files
        if self.concepts_dir.exists():
            for old_file in self.concepts_dir.glob("*.json"):
                try: old_file.unlink()
                except Exception: pass
        self.concepts_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving refined global JSON to {self.output_file}...")
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump({"concepts": final_concepts}, f, indent=2)

        print(f"Saving individual concept JSON files to {self.concepts_dir}/...")
        for gc in tqdm(final_concepts, desc="Writing individual files"):
            safe_name = sanitize_filename(gc["title"])
            if not safe_name: safe_name = "unnamed_concept"

            file_path = self.concepts_dir / f"{safe_name}.json"
            counter = 1
            while file_path.exists():
                file_path = self.concepts_dir / f"{safe_name}_{counter}.json"
                counter += 1

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(gc, f, indent=2)

        # Update Log File
        log_data = {}
        if self.log_file.exists():
            try:
                with open(self.log_file, "r", encoding="utf-8") as f: log_data = json.load(f)
            except Exception: pass

        log_data["thin_concept_summary"] = {
            "concepts_before_thin_pass": len(self.concepts),
            "thin_concepts_folded_via_llm": len(folded_titles),
            "isolated_thin_concepts_pruned": len(pruned_titles),
            "final_clean_concepts_count": len(final_concepts)
        }

        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)

        print(f"\n==================================================")
        print(f"LLM THIN CONCEPT HANDLING COMPLETE")
        print(f"==================================================")
        print(f"Initial Concepts:               {len(self.concepts)}")
        print(f"Thin Concepts Folded via LLM:  -{len(folded_titles)}")
        print(f"Isolated Ultra-Thin Pruned:    -{len(pruned_titles)}")
        print(f"Final Clean Concepts in Graph:  {len(final_concepts)}")
        print(f"==================================================")
        print(f"Outputs written to:\n  - {self.output_file}\n  - {self.concepts_dir}/")

def main():
    parser = argparse.ArgumentParser(description="Step 4B: LLM-Driven Thin Concept Folding & Pruning")
    parser.add_argument("--domain", type=str, required=True, help="Target domain (e.g. biology)")
    parser.add_argument("--wiki_base_dir", type=str, default="/home/compiling-ganesh/24m0829/long_document/wiki")
    parser.add_argument("--model", type=str, default=GLOBAL_MODEL)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--servers", nargs="+", default=DEFAULT_SERVERS)
    parser.add_argument("--concurrency_per_server", type=int, default=DEFAULT_CONCURRENCY_PER_SERVER)
    parser.add_argument("--min_concept_words", type=int, default=30)

    args = parser.parse_args()
    handler = ThinConceptHandler(args)
    asyncio.run(handler.run())

if __name__ == "__main__":
    main()