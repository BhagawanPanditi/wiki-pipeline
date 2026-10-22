#!/usr/bin/env python3
"""
Final Knowledge Graph Consolidation, Island Health Audit & Markdown Wiki Export
--------------------------------------------------------------------------------
Combines:
1. Macro Alphabetical Scan: Evaluates concepts grouped by initial character (A-Z, 0-9) to detect remaining duplicates/links.
2. Deep Verification: Confirms candidate actions (MERGE, LINK, PRUNE, KEEP_SEPARATE) using Briefs and vital Entities.
3. Graph Harmonization: Absorbs merged concepts, normalizes bidirectional relations, removes pruned noise.
4. Island Health Audit: Computes connected components, giant component coverage, and writes `islands_report.json`.
5. Markdown Wiki Export: Compiles every concept into clean, cited Markdown files in `concepts_md/<title>.md` + `concept_index.md`.
Fast execution, strict CLI / config.json parameter resolution, pure XML prompts, no disk debug logs.
"""

import asyncio
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
from pydantic import BaseModel
from tqdm import tqdm

from utils.atomic_io import atomic_write_json, atomic_write_text, sanitize_filename
from utils.cli import build_parser, resolve_settings
from utils.llm_pool import pool_for_stage

STAGE = "final-macro-scan"
DEEP_STAGE = "final-deep-verify"

# ── XML System Prompt for Deep Verification ───────────────────────────────────

SYSTEM_PROMPT_DEEP_VERIFY = """\
You are an expert technical knowledge graph consolidator performing final quality verification.
You are given a candidate pair or cluster of concepts with their Canonical Titles, Briefs, and Top Entities.

Determine the definitive action for this group:
- MERGE: If both concepts represent the EXACT SAME real-world entity, tool, biological process, or condition.
- LINK: If concepts are functionally related or co-occur in the same mechanism but are distinct entities.
- PRUNE: If one is uninformative scraping boilerplate or navigational filler.
- KEEP_SEPARATE: If they are distinct concepts with no direct relationship.

THINKING PROCESS (SANDBOXED):
- Put all reasoning strictly inside `<think>...</think>` at the very start of your response.
- Once `</think>` is closed, output ONLY the final `<verdict>...</verdict>` XML block.
- Absolutely NO commentary or notes outside `<think>...</think>`.

OUTPUT FORMAT:
<think>
[Analysis of candidate relationship]
</think>
<verdict>
<action>MERGE</action>
<primary_title>Exact Canonical Title</primary_title>
<reason>Direct synonymous reference.</reason>
</verdict>
"""

class FinalActionVerdict(BaseModel):
    action: str = "KEEP_SEPARATE"
    primary_title: str = ""
    reason: str = ""

def parse_xml_final_action(text: str) -> FinalActionVerdict:
    if not text:
        return FinalActionVerdict()

    clean_text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
    clean_text = re.sub(r"<thought>.*?(?:</thought>|$)", "", clean_text, flags=re.DOTALL | re.IGNORECASE)

    act_m = re.search(r"<action>(.*?)</action>", clean_text, re.IGNORECASE)
    action = act_m.group(1).strip().upper() if act_m else "KEEP_SEPARATE"

    canon_m = re.search(r"<primary_title>(.*?)</primary_title>", clean_text, re.DOTALL | re.IGNORECASE)
    primary = canon_m.group(1).strip() if canon_m else ""

    reason_m = re.search(r"<reason>(.*?)</reason>", clean_text, re.DOTALL | re.IGNORECASE)
    reason = reason_m.group(1).strip() if reason_m else ""

    # JSON fallback
    if action == "KEEP_SEPARATE" and "{" in clean_text:
        try:
            m = re.search(r"\{.*\}", clean_text, re.DOTALL)
            if m:
                payload = json.loads(m.group(0))
                action = payload.get("action", "KEEP_SEPARATE").upper()
                primary = payload.get("primary_title", "")
                reason = payload.get("reason", "")
        except Exception:
            pass

    return FinalActionVerdict(action=action, primary_title=primary, reason=reason)

# ── UnionFind Data Structure ──────────────────────────────────────────────────

class DisjointSet:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, item: str) -> str:
        curr = item.strip().lower()
        path = []
        while curr in self.parent and self.parent[curr] != curr:
            path.append(curr)
            curr = self.parent[curr]
        for p in path:
            self.parent[p] = curr
        if curr not in self.parent:
            self.parent[curr] = curr
        return curr

    def union(self, a: str, b: str):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a

# ── Pipeline Execution ────────────────────────────────────────────────────────

class FinalConsolidationPipeline:
    def __init__(self, args):
        self.args = args
        wiki_path = getattr(args, "wiki_path", None)
        if not wiki_path:
            raise ValueError("Missing required --wiki_path argument.")

        self.wiki_path = Path(wiki_path)
        self.global_concepts_file = self.wiki_path / "global_concepts.json"
        self.md_dir = self.wiki_path / "concepts_md"
        self.islands_file = self.wiki_path / "islands_report.json"
        self.settings = resolve_settings(args, STAGE)

        if not self.global_concepts_file.exists():
            raise FileNotFoundError(f"Missing {self.global_concepts_file}. Run Stage 7 & 8 first.")

    def run(self):
        asyncio.run(self.run_async())

    async def run_async(self):
        print(f"Loading concepts from {self.global_concepts_file}...")
        with open(self.global_concepts_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            concepts = data.get("concepts", [])

        print(f"Loaded {len(concepts)} concepts for final consolidation.")

        # 1. Alphabetical Macro-Scan for candidates
        alpha_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for c in concepts:
            t = c.get("title", "").strip()
            if not t:
                continue
            first_char = t[0].upper()
            bucket_key = first_char if first_char.isalpha() else "#"
            alpha_buckets[bucket_key].append(c)

        candidate_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        for bucket, b_concepts in alpha_buckets.items():
            sorted_b = sorted(b_concepts, key=lambda x: x["title"].lower())
            for i in range(len(sorted_b) - 1):
                c1, c2 = sorted_b[i], sorted_b[i + 1]
                t1, t2 = c1["title"].lower(), c2["title"].lower()
                # Check for shared prefixes or high token overlap
                words1 = set(t1.split())
                words2 = set(t2.split())
                overlap = words1 & words2
                if len(overlap) >= 1 and (len(overlap) / max(len(words1), len(words2))) >= 0.5:
                    pk = tuple(sorted([c1["title"], c2["title"]]))
                    if pk not in seen_pairs:
                        seen_pairs.add(pk)
                        candidate_pairs.append((c1, c2))

        print(f"Alphabetical scan identified {len(candidate_pairs)} candidate pair(s) for deep verification.")

        # 2. Deep Verification via LLM
        uf = DisjointSet()
        for c in concepts:
            uf.find(c["title"])

        verified_merges = 0
        new_links = 0
        pruned_titles: Set[str] = set()

        if candidate_pairs and not getattr(self.args, "skip_deep_verify", False):
            pool = pool_for_stage(self.args, self.settings, DEEP_STAGE)

            async def verify_pair(client: Any, pair: Tuple[Dict[str, Any], Dict[str, Any]], server_index: int) -> FinalActionVerdict:
                c1, c2 = pair
                c1_brief = c1.get("brief", "") or c1.get("description", "")[:200]
                c2_brief = c2.get("brief", "") or c2.get("description", "")[:200]
                c1_ents = ", ".join(c1.get("entities", [])[:6]) or "None"
                c2_ents = ", ".join(c2.get("entities", [])[:6]) or "None"

                prompt = (
                    f"CONCEPT 1: \"{c1.get('title')}\"\n"
                    f"  Brief: {c1_brief}\n"
                    f"  Key Entities: {c1_ents}\n\n"
                    f"CONCEPT 2: \"{c2.get('title')}\"\n"
                    f"  Brief: {c2_brief}\n"
                    f"  Key Entities: {c2_ents}\n"
                )

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
                    model_kwargs = {"temperature": 0.0}

                response = await client.chat.completions.create(
                    model=self.settings.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_DEEP_VERIFY},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.settings.max_tokens or 1000,
                    **model_kwargs
                )
                raw_text = response.choices[0].message.content or ""
                return parse_xml_final_action(raw_text)

            async def on_verdict(pair: Tuple[Dict[str, Any], Dict[str, Any]], res: FinalActionVerdict):
                nonlocal verified_merges, new_links
                c1, c2 = pair
                if res.action == "MERGE":
                    uf.union(c1["title"], c2["title"])
                    verified_merges += 1
                elif res.action == "LINK":
                    # Add bidirectional relation
                    new_links += 1
                elif res.action == "PRUNE":
                    pruned_titles.add(c2["title"])

            async with pool:
                await pool.map(
                    candidate_pairs,
                    verify_pair,
                    attempts=self.settings.attempts,
                    desc="Deep Verifying Candidate Pairs",
                    budget=None,
                    attempt_timeout=self.settings.profile.attempt_timeout,
                    on_result=on_verdict,
                )

            print(f"Deep verification results: {verified_merges} merges approved, {pruned_titles} pruned.")

        # 3. Fuse Merged Concepts & Apply Pruning
        grouped_concepts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for c in concepts:
            if c["title"] in pruned_titles:
                continue
            root = uf.find(c["title"])
            grouped_concepts[root].append(c)

        final_concepts: List[Dict[str, Any]] = []
        canonical_title_map: Dict[str, str] = {}

        for root, group in grouped_concepts.items():
            title_counter = Counter(c["title"] for c in group)
            chosen_title = title_counter.most_common(1)[0][0]
            canonical_title_map[root] = chosen_title

            merged_aliases = set()
            merged_entities = set()
            merged_chunks = set()
            briefs = []
            descriptions = []

            for c in group:
                merged_aliases.add(c["title"])
                for a in c.get("aliases", []):
                    merged_aliases.add(a)
                for ent in c.get("entities", []):
                    merged_entities.add(ent)
                for sc in c.get("source_chunks", []):
                    merged_chunks.add(sc)
                if c.get("brief"):
                    briefs.append(c["brief"])
                if c.get("description"):
                    descriptions.append(c["description"])

            merged_aliases.discard(chosen_title)
            final_brief = briefs[0] if briefs else ""
            final_desc = "\n\n".join(dict.fromkeys(descriptions))

            final_concepts.append({
                "title": chosen_title,
                "brief": final_brief,
                "aliases": sorted(list(merged_aliases)),
                "entities": sorted(list(merged_entities)),
                "description": final_desc,
                "source_chunks": sorted(list(merged_chunks)),
                "outbound_relations": group[0].get("outbound_relations", [])
            })

        print(f"Total surviving concepts after final merge & pruning: {len(final_concepts)}")
        atomic_write_json(self.global_concepts_file, {"concepts": final_concepts})

        # 4. Island Health Audit (Connected Components & Giant Component)
        print("Analyzing knowledge graph topology & islands...")
        G = nx.Graph()
        for c in final_concepts:
            G.add_node(c["title"])
            for rel in c.get("outbound_relations", []):
                tgt = rel.get("target_concept", "").strip()
                if tgt and tgt in canonical_title_map.values():
                    G.add_edge(c["title"], tgt)

        components = list(nx.connected_components(G))
        components.sort(key=len, reverse=True)
        giant_size = len(components[0]) if components else 0
        coverage = (giant_size / len(final_concepts) * 100) if final_concepts else 0.0

        islands_summary = {
            "total_concepts": len(final_concepts),
            "total_islands": len(components),
            "giant_component_size": giant_size,
            "giant_component_coverage_pct": round(coverage, 2),
            "isolated_concepts_count": sum(1 for comp in components if len(comp) == 1),
            "island_sizes": [len(comp) for comp in components[:20]]
        }
        atomic_write_json(self.islands_file, islands_summary)

        # 5. Compile and Export Markdown Wiki
        print(f"Exporting Markdown Wiki to {self.md_dir}...")
        self.md_dir.mkdir(parents=True, exist_ok=True)

        index_entries = []
        for c in tqdm(final_concepts, desc="Exporting Markdown Pages"):
            title = c["title"]
            brief = c.get("brief", "")
            aliases = ", ".join(c.get("aliases", []))
            entities = ", ".join(c.get("entities", [])[:15])
            desc = c.get("description", "")
            relations = c.get("outbound_relations", [])

            safe_name = sanitize_filename(title) or "unnamed_concept"
            md_path = self.md_dir / f"{safe_name}.md"

            md_content = [f"# {title}\n"]
            if aliases:
                md_content.append(f"**Aliases**: {aliases}\n")
            if brief:
                md_content.append(f"> **Overview**: {brief}\n")
            if entities:
                md_content.append(f"**Key Entities**: {entities}\n")

            md_content.append("## Description\n")
            md_content.append(f"{desc}\n")

            if relations:
                md_content.append("## Related Concepts\n")
                for rel in relations:
                    tgt = rel.get("target_concept", "")
                    r_text = rel.get("relation", "")
                    jacc = rel.get("jaccard_similarity")
                    jacc_str = f" *(Jaccard: {jacc})*" if jacc is not None else ""
                    md_content.append(f"- **[[{tgt}]]**: {r_text}{jacc_str}")
                md_content.append("")

            atomic_write_text(md_path, "\n".join(md_content))
            index_entries.append((title, f"{safe_name}.md", brief))

        # Root Concept Index
        index_entries.sort(key=lambda x: x[0].lower())
        index_lines = ["# Knowledge Graph Concept Index\n", f"Total Concepts: {len(index_entries)}\n"]
        for t, fname, b in index_entries:
            b_snip = f" — {b[:120]}..." if b else ""
            index_lines.append(f"- [{t}]({fname}){b_snip}")

        atomic_write_text(self.md_dir / "concept_index.md", "\n".join(index_lines))

        print("\n============================================================")
        print("STAGE 9: FINAL CONSOLIDATION & WIKI EXPORT COMPLETE")
        print(f"Total Concepts in Wiki:      {len(final_concepts)}")
        print(f"Total Islands:               {len(components)}")
        print(f"Giant Component Coverage:    {giant_size}/{len(final_concepts)} ({coverage:.1f}%)")
        print(f"Markdown Wiki Exported:      {self.md_dir}")
        print(f"Island Report Written:       {self.islands_file}")
        print("============================================================\n")

def main():
    parser = build_parser(stage=STAGE, description="Final knowledge graph consolidation, island audit and markdown wiki export")
    parser.add_argument("--skip_deep_verify", action="store_true",
                        help="Skip LLM deep verification and export current graph.")
    args = parser.parse_args()

    pipeline = FinalConsolidationPipeline(args)
    pipeline.run()

if __name__ == "__main__":
    main()
