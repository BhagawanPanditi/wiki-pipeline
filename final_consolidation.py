#!/usr/bin/env python3
"""
Final Knowledge Graph Consolidation, Island Health Audit & Markdown Wiki Export
--------------------------------------------------------------------------------
1. Alphabetical Candidate Scan: Groups concepts with high character/token overlap.
2. Parallel LLM Decision: Evaluates candidate pairs in parallel across cluster (MERGE, LINK, PRUNE, KEEP).
3. Sequential Execution Phase:
   - PRUNE executed first: Drops noise/boilerplate nodes.
   - MERGE executed second:
       * Unified DisjointSet maps aliases/titles into single canonical root.
       * Provenance sources concatenated & deduplicated.
       * Descriptions merged with unique sentence preservation.
       * Most comprehensive brief selected.
       * Relations remapped to new canonical targets with SELF-LOOPS removed.
   - LINK executed third: Adds bidirectional relations with Jaccard tracking.
4. Island Health Audit: Computes connected components, giant component size & coverage %, writes `islands_report.json`.
5. Markdown Wiki Export: Compiles into `concepts_md/<title>.md` (with Jaccard similarity displayed in relations) + `concept_index.md`.
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
from utils.citation_utils import clean_chunk_citations
from utils.cli import build_parser, resolve_settings
from utils.llm_pool import pool_for_stage

STAGE = "final-macro-scan"
DEEP_STAGE = "final-deep-verify"

def print_link_distribution(concepts: List[Dict[str, Any]], title_label: str = "STAGE 9: FINAL CONSOLIDATED GRAPH"):
    """Compute and print histogram distribution of related links across all concepts with dynamic bins (bin width = 5)."""
    if not concepts:
        print(f"\n[{title_label}] No concepts found.")
        return

    counts = [len(c.get("outbound_relations", [])) for c in concepts]
    total = len(counts)
    avg_links = sum(counts) / total if total else 0.0
    max_count = max(counts) if counts else 0

    # Build dynamic bins with step=5 up to max_count
    bins = [("0 links", 0, 0)]
    step = 5
    curr = 1
    while curr <= max_count:
        upper = curr + step - 1
        bins.append((f"{curr} - {upper} links", curr, upper))
        curr += step

    bin_counts = {label: 0 for label, _, _ in bins}
    for count in counts:
        for label, low, high in bins:
            if low <= count <= high:
                bin_counts[label] += 1
                break

    print(f"\n============================================================")
    print(f"RELATED LINKS DISTRIBUTION [{title_label}] (Total Concepts: {total})")
    print(f"Average Links per Concept: {avg_links:.2f} | Max Links: {max_count} | Disconnected Concepts (0 links): {bin_counts['0 links']} ({bin_counts['0 links'] / total * 100:.1f}%)")
    print(f"------------------------------------------------------------")
    print(f"{'Links Range':<20} {'Concept Count':<15} {'Percentage':<12}")
    print(f"------------------------------------------------------------")
    for label, _, _ in bins:
        c_count = bin_counts[label]
        pct = (c_count / total * 100) if total else 0.0
        bar = "█" * int(pct / 2)
        print(f"{label:<20} {c_count:<15} {pct:>5.1f}%   {bar}")
    print(f"============================================================\n")

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

# ── DisjointSet ───────────────────────────────────────────────────────────────

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
                words1 = set(t1.split())
                words2 = set(t2.split())
                overlap = words1 & words2
                if len(overlap) >= 1 and (len(overlap) / max(len(words1), len(words2))) >= 0.5:
                    pk = tuple(sorted([c1["title"], c2["title"]]))
                    if pk not in seen_pairs:
                        seen_pairs.add(pk)
                        candidate_pairs.append((c1, c2))

        print(f"Alphabetical scan identified {len(candidate_pairs)} candidate pair(s) for deep verification.")

        # 2. Parallel LLM Decisions (Cluster Execution)
        decisions: List[Tuple[Tuple[Dict[str, Any], Dict[str, Any]], FinalActionVerdict]] = []

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
                decisions.append((pair, res))

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

        # 3. Sequential Deterministic Application: PRUNE -> MERGE -> LINK
        pruned_titles: Set[str] = set()
        merge_uf = DisjointSet()
        for c in concepts:
            merge_uf.find(c["title"])

        additional_links: List[Tuple[str, str]] = []

        # (A) Collect PRUNE
        for (c1, c2), verdict in decisions:
            if verdict.action == "PRUNE":
                # Prune candidate with shorter description
                victim = c2["title"] if len(c2.get("description", "")) <= len(c1.get("description", "")) else c1["title"]
                pruned_titles.add(victim)

        # (B) Collect MERGE
        verified_merges = 0
        for (c1, c2), verdict in decisions:
            if verdict.action == "MERGE" and c1["title"] not in pruned_titles and c2["title"] not in pruned_titles:
                merge_uf.union(c1["title"], c2["title"])
                verified_merges += 1

        # (C) Collect LINK
        for (c1, c2), verdict in decisions:
            if verdict.action == "LINK" and c1["title"] not in pruned_titles and c2["title"] not in pruned_titles:
                additional_links.append((c1["title"], c2["title"]))

        print(f"Decisions collected: {len(pruned_titles)} pruned, {verified_merges} merged, {len(additional_links)} linked.")

        # 4. Group by Canonical Root
        grouped_concepts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for c in concepts:
            if c["title"] in pruned_titles:
                continue
            root = merge_uf.find(c["title"])
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
            seen_sentences = set()
            description_sentences = []
            briefs = []

            for c in group:
                merged_aliases.add(c["title"])
                for a in c.get("aliases", []):
                    if a.strip():
                        merged_aliases.add(a.strip())
                for ent in c.get("entities", []):
                    if ent.strip():
                        merged_entities.add(ent.strip())
                for sc in c.get("source_chunks", []):
                    if sc.strip():
                        merged_chunks.add(sc.strip())

                if c.get("brief"):
                    briefs.append(c["brief"])

                # De-duplicate sentences across chunk descriptions
                desc = c.get("description", "").strip()
                if desc:
                    for sent in re.split(r'(?<=[.!?])\s+', desc):
                        s_clean = sent.strip()
                        if s_clean and s_clean.lower() not in seen_sentences:
                            seen_sentences.add(s_clean.lower())
                            description_sentences.append(s_clean)

            merged_aliases.discard(chosen_title)
            # Pick longest / most complete brief
            chosen_brief = max(briefs, key=len) if briefs else ""
            fused_desc = clean_chunk_citations(" ".join(description_sentences))

            # Collect raw outbound relations
            raw_relations = []
            for c in group:
                raw_relations.extend(c.get("outbound_relations", []))

            final_concepts.append({
                "title": chosen_title,
                "brief": chosen_brief,
                "aliases": sorted(list(merged_aliases)),
                "entities": sorted(list(merged_entities)),
                "description": fused_desc,
                "source_chunks": sorted(list(merged_chunks)),
                "_raw_relations": raw_relations,
            })

        # Canonical title lookup
        concept_by_name = {c["title"].lower(): c for c in final_concepts}

        # 5. Remap Outbound Relations & REMOVE SELF-LOOPS
        print("Remapping relations and removing self-loops...")
        for c in final_concepts:
            current_title = c["title"]
            fused_relations: Dict[str, Dict[str, Any]] = {}

            for rel in c.get("_raw_relations", []):
                tgt_raw = rel.get("target_concept", "").strip()
                if not tgt_raw:
                    continue
                root_tgt = merge_uf.find(tgt_raw)
                canonical_target = canonical_title_map.get(root_tgt, tgt_raw)

                # FORBID SELF-LOOPS
                if canonical_target.lower() == current_title.lower():
                    continue

                # FORBID PRUNED TARGETS
                if canonical_target in pruned_titles or canonical_target.lower() not in concept_by_name:
                    continue

                if canonical_target not in fused_relations:
                    fused_relations[canonical_target] = {
                        "target_concept": canonical_target,
                        "relation": rel.get("relation", ""),
                        "jaccard_similarity": rel.get("jaccard_similarity"),
                        "shared_entities": rel.get("shared_entities", []),
                        "source_chunks": list(rel.get("source_chunks", []))
                    }
                else:
                    # Merge relation text if complementary
                    existing_text = fused_relations[canonical_target]["relation"]
                    new_text = rel.get("relation", "")
                    if new_text and new_text not in existing_text:
                        fused_relations[canonical_target]["relation"] = f"{existing_text} {new_text}".strip()
                    # Keep highest jaccard score
                    j1 = fused_relations[canonical_target].get("jaccard_similarity") or 0.0
                    j2 = rel.get("jaccard_similarity") or 0.0
                    if j2 > j1:
                        fused_relations[canonical_target]["jaccard_similarity"] = j2

            c["outbound_relations"] = list(fused_relations.values())
            del c["_raw_relations"]

        # Apply Additional Verified Links
        for t1, t2 in additional_links:
            root1 = canonical_title_map.get(merge_uf.find(t1), t1)
            root2 = canonical_title_map.get(merge_uf.find(t2), t2)
            if root1.lower() != root2.lower() and root1.lower() in concept_by_name and root2.lower() in concept_by_name:
                c1_obj = concept_by_name[root1.lower()]
                c2_obj = concept_by_name[root2.lower()]
                targets1 = {r["target_concept"].lower() for r in c1_obj["outbound_relations"]}
                targets2 = {r["target_concept"].lower() for r in c2_obj["outbound_relations"]}

                if root2.lower() not in targets1:
                    c1_obj["outbound_relations"].append({
                        "target_concept": root2,
                        "relation": "Functionally related in technical domain context.",
                        "jaccard_similarity": None,
                        "shared_entities": [],
                        "source_chunks": []
                    })
                if root1.lower() not in targets2:
                    c2_obj["outbound_relations"].append({
                        "target_concept": root1,
                        "relation": "Functionally related in technical domain context.",
                        "jaccard_similarity": None,
                        "shared_entities": [],
                        "source_chunks": []
                    })

        print(f"Total surviving concepts after final merge & pruning: {len(final_concepts)}")
        atomic_write_json(self.global_concepts_file, {"concepts": final_concepts})

        # Print detailed link distribution statistics
        print_link_distribution(final_concepts, title_label="STAGE 9: SURVIVING CONCEPTS GRAPH")

        # Sync individual concept files if concepts directory exists
        concepts_dir = self.wiki_path / "concepts"
        if concepts_dir.is_dir():
            surviving_files = set()
            for c in final_concepts:
                c_fname = f"{sanitize_filename(c['title'])}.json"
                surviving_files.add(c_fname)
                try:
                    atomic_write_json(concepts_dir / c_fname, c, fsync_dir=False)
                except Exception:
                    pass
            # Remove any concept files that were pruned or merged away
            for old_file in concepts_dir.glob("*.json"):
                if old_file.name not in surviving_files:
                    try:
                        old_file.unlink()
                    except OSError:
                        pass

        # 6. Island Health Audit (Connected Components & Giant Component)
        print("Analyzing knowledge graph topology & islands...")
        G = nx.Graph()
        for c in final_concepts:
            G.add_node(c["title"])
            for rel in c.get("outbound_relations", []):
                tgt = rel.get("target_concept", "").strip()
                if tgt and tgt in [con["title"] for con in final_concepts]:
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

        # 7. Compile and Export Markdown Wiki
        print(f"Exporting Markdown Wiki to {self.md_dir}...")
        self.md_dir.mkdir(parents=True, exist_ok=True)

        index_entries = []
        for c in tqdm(final_concepts, desc="Exporting Markdown Pages"):
            title = c["title"]
            brief = c.get("brief", "")
            aliases = ", ".join(c.get("aliases", []))
            entities = ", ".join(c.get("entities", [])[:15])
            desc = clean_chunk_citations(c.get("description", ""))
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
                primary_cap = getattr(self.args, "max_primary_links", 15)
                primary_relations = relations[:primary_cap]
                extended_relations = relations[primary_cap:]

                for rel in primary_relations:
                    tgt = rel.get("target_concept", "")
                    r_text = rel.get("relation", "")
                    jacc = rel.get("jaccard_similarity")
                    jacc_str = f" *(Jaccard similarity: {jacc})*" if jacc is not None else ""
                    md_content.append(f"- **[[{tgt}]]**: {r_text}{jacc_str}")

                if extended_relations:
                    md_content.append("\n### Extended Context & Mentions\n")
                    compact_links = [f"[[{r.get('target_concept', '')}]]" for r in extended_relations if r.get('target_concept')]
                    md_content.append(", ".join(compact_links))
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
    parser = build_parser(stage=STAGE, description="Final knowledge graph consolidation, island audit and markdown wiki export", include_resume=True)
    parser.add_argument("--skip_deep_verify", action="store_true",
                        help="Skip LLM deep verification and export current graph.")
    parser.add_argument("--max_primary_links", type=int, default=15,
                        help="Maximum primary related links with full explanations on Markdown pages (remainder listed compactly).")
    args = parser.parse_args()

    pipeline = FinalConsolidationPipeline(args)
    pipeline.run()

if __name__ == "__main__":
    main()
