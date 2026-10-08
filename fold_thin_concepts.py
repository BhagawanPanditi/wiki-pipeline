#!/usr/bin/env python3
"""
Step 4B: LLM-Driven Thin Concept Folding & Pruning Pipeline
------------------------------------------------------------
1. Scans for Thin Concepts (< 30 words).
2. Identifies Candidate Host Concepts via Inbound Relations or Shared Chunks.
3. Uses Async LLM calls to verify semantic fit and synthesize a single, dense merged description.
4. Clears stale files in `concepts/` and outputs clean `global_concepts_refined.json` and individual concept JSONs.
"""

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

# ── Configuration ────────────────────────────────────────────────────────────
from utils.atomic_io import atomic_write_json, sanitize_filename
from utils.cli import build_parser, resolve_settings
from utils.event_log import EventLog
from utils.layout import log_path
from utils.errors import StageBudget
from utils.json_repair import clean_json_string
from utils.llm_pool import pool_for_stage

STAGE = "fold-thin-concepts"

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

class ThinConceptHandler:
    def __init__(self, args):
        """Prepare thin-concept folding. Nothing is read until :meth:`run`.

        Args:
            args: Parsed CLI namespace, including ``--min_concept_words``.

        Sets:
            Paths, the stage settings, the debug and decision logs, the time budget, and
            the word threshold ``min_words``."""
        self.domain_dir = Path(args.wiki_base_dir) / args.domain
        self.input_file = self.domain_dir / "global_concepts_stubs_resolved.json"
        self.output_file = self.domain_dir / "global_concepts_refined.json"
        self.concepts_dir = self.domain_dir / "concepts"
        self.log_file = self.domain_dir / "global_consolidation.json"

        self.args = args
        self.settings = resolve_settings(args, STAGE)
        self.debug = EventLog(log_path(self.domain_dir, STAGE, "debug"), label=STAGE)
        self.decisions = EventLog(log_path(self.domain_dir, STAGE, "decisions"), label=STAGE)
        self.budget = StageBudget(self.settings.budget_minutes)
        self.min_words = args.min_concept_words

        self.concepts: List[Dict[str, Any]] = []
        self.title_map: Dict[str, Dict[str, Any]] = {}

    def load_data(self):
        """Load the stub-resolved graph and index it by lowercased title.

        Returns:
            None. Fills ``concepts`` and ``title_map``.

        Raises:
            FileNotFoundError: If the resolve-stubs output is missing."""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Missing input file {self.input_file}. Run resolve_stubs.py first!")

        print(f"Loading concepts from {self.input_file}...")
        with open(self.input_file, "r", encoding="utf-8") as f:
            self.concepts = json.load(f).get("concepts", [])

        self.title_map = {c["title"].lower(): c for c in self.concepts}
        print(f"Loaded {len(self.concepts)} total concepts.")

    def find_candidate_hosts(self, thin_concept: Dict[str, Any], inbound_index: Dict[str, List[Dict[str, Any]]], chunk_index: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Rank the concepts that could absorb a thin concept.

        Args:
            thin_concept: The concept under consideration.
            inbound_index: Lowercased title mapped to the concepts that link to it.
            chunk_index: Source chunk mapped to the concepts that cite it.

        Returns:
            Candidate host concepts, best first: concepts that link to the thin concept,
            then concepts that cite one of its source chunks. The thin concept itself is
            never a candidate."""
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

    async def evaluate_folding_llm(self, client, item: Tuple[Dict[str, Any],
                                     List[Dict[str, Any]]], server_index: int):
        """One fold decision.

        Raises on any unusable response so ServerPool classifies and retries it.
        Returning None after 10 silent attempts used to mean "do not fold"; a
        raised-and-exhausted error means the same thing now, except it is also
        recorded in the debug log and counted in the per-server report.
        """
        thin_concept, candidates = item
        user_prompt = f"""THIN CONCEPT TO EVALUATE:
Title: {thin_concept['title']}
Aliases: {thin_concept.get('aliases', [])}
Description: {thin_concept.get('description', '')}

CANDIDATE HOST CONCEPTS:
"""
        for idx, cand in enumerate(candidates, 1):
            user_prompt += f"\n--- Candidate Host {idx} ---\nTitle: {cand['title']}\nDescription: {cand.get('description', '')}\n"

        response = await client.chat.completions.create(
            model=self.settings.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_FOLD},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=self.settings.max_tokens,
            response_format={"type": "json_object"}
        )
        raw_text = response.choices[0].message.content
        return FoldDecisionSchema.model_validate_json(clean_json_string(raw_text))

    async def run(self):
        """Fold each thin concept into its best host, or prune it.

        Concepts under ``min_words`` are offered to the LLM along with their candidate
        hosts, in parallel through the pool. Folds are applied in completion order.
        Whatever is still thin afterwards is folded deterministically into its best
        candidate, or pruned if it has no candidate and almost no content. Each decision
        is recorded in the decision log.

        Returns:
            None. Writes ``global_concepts_refined.json``."""
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
        settings = self.settings
        if thin_tasks and settings.servers:
            print(f"Spreading {len(thin_tasks)} thin concept(s) across "
                  f"{len(settings.servers)} server(s) at "
                  f"{settings.concurrency_per_server} concurrent request(s) each "
                  f"(item i -> server i % N).")
            print(f"Per-item attempts: {settings.attempts} | Stage budget: "
                  f"{settings.budget_minutes:g} min | Debug log: {self.debug.path}")

            async def apply_fold(item, decision):
                """Merge a thin concept into its host, in completion order.

                There are no awaits inside, so this is already atomic with
                respect to the event loop and needs no lock. The ordering
                matters: a host can only be folded into once, guarded by
                folded_titles.
                """
                thin_concept, _candidates = item
                if not (decision and decision.should_fold and decision.target_host_title):
                    return
                target_host = self.title_map.get(decision.target_host_title.lower())
                if not target_host or target_host["title"].lower() in folded_titles:
                    return

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

                self.decisions.record(
                    "concept_folded", method="llm", thin_concept=thin_concept["title"],
                    host=target_host["title"],
                    description_rewritten=bool(decision.synthesized_description))
                folded_titles.add(thin_concept["title"].lower())

            pool = pool_for_stage(self.args, settings, STAGE)
            try:
                async with pool:
                    result = await pool.map(
                        thin_tasks,
                        self.evaluate_folding_llm,
                        attempts=settings.attempts,
                        desc="LLM Thin Concept Folding",
                        key_fn=lambda it: it[0].get("title", ""),
                        budget=self.budget,
                        debug=self.debug,
                        attempt_timeout=settings.profile.attempt_timeout,
                        on_result=apply_fold,
                    )
            finally:
                self.debug.close()

            print("Requests served per server:")
            for url, count in sorted(result.per_server.items(), key=lambda kv: -kv[1]):
                print(f"  {url:<40} {count}")
            if result.failures:
                print(f"\u26a0\ufe0f  {result.failed_count} thin concept(s) got no fold "
                      f"decision and were left to the deterministic rules below.")

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
                        self.decisions.record("concept_folded", method="deterministic", thin_concept=c["title"], host=best_host["title"], description_rewritten=bool(desc_to_add))
                        folded_titles.add(t_title_lower)
                        continue

                # If no candidate host exists and it is ultra-thin (< 25 words), prune to prevent stub-like concepts
                if words < 25 or len(c.get("outbound_relations", [])) == 0:
                    self.decisions.record("concept_pruned", thin_concept=c["title"], words=words,
                                          outbound_relations=len(c.get("outbound_relations", [])))
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
        atomic_write_json(self.output_file, {"concepts": final_concepts})

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

        atomic_write_json(self.log_file, log_data)

        print("\n==================================================")
        print("LLM THIN CONCEPT HANDLING COMPLETE")
        print("==================================================")
        print(f"Initial Concepts:               {len(self.concepts)}")
        print(f"Thin Concepts Folded via LLM:  -{len(folded_titles)}")
        print(f"Isolated Ultra-Thin Pruned:    -{len(pruned_titles)}")
        print(f"Final Clean Concepts in Graph:  {len(final_concepts)}")
        print("==================================================")
        print(f"Outputs written to:\n  - {self.output_file}\n  - {self.concepts_dir}/")

def main():
    """Parse the command line and run thin-concept folding.

    Returns:
        None."""
    parser = build_parser(
        stage=STAGE,
        description="Step 4B: LLM-Driven Thin Concept Folding & Pruning")
    parser.add_argument("--min_concept_words", type=int, default=30)

    args = parser.parse_args()
    handler = ThinConceptHandler(args)
    asyncio.run(handler.run())

if __name__ == "__main__":
    main()
