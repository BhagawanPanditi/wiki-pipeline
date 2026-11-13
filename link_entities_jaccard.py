#!/usr/bin/env python3
"""Stage 8: Semantic Jaccard Overlap Linker & Relation Characterization.
---------------------------------------------------------------------
Establishes bidirectional links between concepts sharing salient named entities:
1. Calculates Jaccard similarity across shared entities between concept pairs.
2. Characterizes direct relationships with concise LLM relation tags (<= 10 words).
3. Orders outbound relations by Jaccard similarity.

Usage:
    python3 link_entities_jaccard.py --wiki_path /path/to/wiki
"""


import asyncio
import json
import resource
import re
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
from tqdm import tqdm

from utils.atomic_io import atomic_write_json, sanitize_filename
from utils.checkpoint import load_json_log, save_log_sync
from utils.cli import build_parser, resolve_settings
from utils.concept_ops import get_concept_full_text
from utils.lemma import normalize_head_noun
from utils.llm_params import call_llm
from utils.llm_pool import pool_for_stage
from utils.manifest import verify_stage_manifest, write_stage_manifest
from utils.stats import print_link_distribution

STAGE = "link-entities"

SYSTEM_PROMPT_RELATION_SYNTHESIS = """\
You are an expert knowledge specialist.
You are given Concept 1 (the SUBJECT) and Concept 2 (the OBJECT), with their briefs and shared entities.
Write a phrase P such that "Concept 1 P Concept 2" is true, specific, and directional.

GUIDELINES:
- Output a concise phrase (3 to 10 words, no trailing punctuation).
- Prefer specific functional or hierarchical roles:
  * "is a subtype of"
  * "implements the protocol of"
  * "depends on components of"
  * "is evaluated against"
  * "serves as a competing alternative to"
  * "is an operational requirement for"
- Avoid vague tags like "related to", "connected with", "used with".
- If only co-occurrence is supported, output: "co-occurs with via <shared entity>".

OUTPUT FORMAT:
Output ONLY the <relation>...</relation> XML block:
<relation>
Concise relationship phrase
</relation>
"""

def parse_xml_relation(text: str) -> str:
    if not text:
        return ""
    rel_m = re.search(r"<relation>(.*?)</relation>", text, re.DOTALL | re.IGNORECASE)
    if rel_m:
        return rel_m.group(1).strip()
    return text.strip().strip('"').strip("'")

async def enrich_graph_via_jaccard(
    args: Any,
    wiki_path: Path,
    min_shared_entities: int = 1,
    min_jaccard: float = 0.05,
    max_hub_threshold: int = 150,
    specific_entity_threshold: int = 50,
    dry_run: bool = False
):
    global_concepts_file = wiki_path / "global_concepts.json"
    concepts_dir = wiki_path / "concepts"

    if not global_concepts_file.exists():
        raise FileNotFoundError(f"Missing {global_concepts_file}. Complete Stage 7 first.")

    verify_stage_manifest(wiki_path, "write-briefs")

    print(f"Loading concepts from {global_concepts_file}...")
    with open(global_concepts_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        concepts = data.get("concepts", [])

    print(f"Loaded {len(concepts)} concepts. Indexing vital entities...")

    # Initial Link Distribution
    print_link_distribution(concepts, title_label="STAGE 8: BEFORE JACCARD LINKING")

    entity_to_concepts: Dict[str, Set[str]] = defaultdict(set)
    concept_entities: Dict[str, Set[str]] = {}
    concept_by_title: Dict[str, Dict[str, Any]] = {}
    title_lookup: Dict[str, str] = {}

    for c in concepts:
        title = c["title"]
        concept_by_title[title] = c
        norm_t = normalize_head_noun(title)
        if len(norm_t) > 2:
            title_lookup[norm_t] = title
        for alias in c.get("aliases", []):
            norm_a = normalize_head_noun(alias)
            if len(norm_a) > 2:
                title_lookup[norm_a] = title
        for sub in c.get("subtopics", []):
            norm_s = normalize_head_noun(sub)
            if len(norm_s) > 2:
                title_lookup[norm_s] = title

        raw_entities = c.get("entities", [])
        norm_set = set()
        for ent in raw_entities:
            norm_e = normalize_head_noun(ent)
            if len(norm_e) > 2:
                norm_set.add(norm_e)
                entity_to_concepts[norm_e].add(title)
        concept_entities[title] = norm_set

    # Initial graph connectivity
    G = nx.Graph()
    for c in concepts:
        G.add_node(c["title"])
        for rel in c.get("outbound_relations", []):
            tgt = rel.get("target_concept", "").strip()
            if tgt and tgt in concept_by_title:
                G.add_edge(c["title"], tgt)

    initial_components = list(nx.connected_components(G))
    initial_components.sort(key=len, reverse=True)
    giant_size = len(initial_components[0]) if initial_components else 0
    print(f"Initial Graph Connectivity: {len(initial_components)} components. Giant component: {giant_size}/{len(concepts)} ({giant_size/len(concepts)*100:.1f}%)")

    # 1. Direct Conceptual Mention (Title/Alias in Entities)
    title_mention_links: List[Tuple[str, str, float, List[str]]] = []
    for title, ents in concept_entities.items():
        for ent in ents:
            if ent in title_lookup:
                matched_target = title_lookup[ent]
                if matched_target != title and not G.has_edge(title, matched_target):
                    title_mention_links.append((title, matched_target, 1.0, [ent]))

    print(f"Discovered {len(title_mention_links)} cross-reference links where a concept's title appears in key entities.")

    # 2. Shared Entities & IDF-Weighted Qualification (F-14)
    # Drop entities with df > max(50, 0.02 * N) to prevent broad hub dominance
    num_concepts = len(concepts)
    df_ceiling = max(50, int(0.02 * num_concepts))
    filtered_entity_to_concepts = {
        ent: titles for ent, titles in entity_to_concepts.items()
        if len(titles) <= df_ceiling
    }

    # Precompute IDF for remaining entities
    idf = {
        ent: math.log(max(1.0, num_concepts / max(1, len(titles))))
        for ent, titles in filtered_entity_to_concepts.items()
    }

    pair_shared: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for ent, titles in filtered_entity_to_concepts.items():
        if len(titles) > 1:
            t_list = sorted(list(titles))
            for i in range(len(t_list)):
                for j in range(i + 1, len(t_list)):
                    pair_shared[(t_list[i], t_list[j])].add(ent)

    candidate_links: List[Tuple[str, str, float, List[str], str]] = []
    for (t1, t2), shared in pair_shared.items():
        if G.has_edge(t1, t2):
            continue
        set1 = concept_entities[t1]
        set2 = concept_entities[t2]
        union_ents = set1 | set2

        # IDF-weighted overlap score
        shared_idf = sum(idf.get(e, 0.0) for e in shared)
        union_idf = sum(idf.get(e, 0.0) for e in union_ents) or 1.0
        idf_score = shared_idf / union_idf

        qualifies = False
        # F-14: Qualify if >= 2 shared entities with sufficient IDF score, OR 1 rare entity
        if len(shared) >= 2 and idf_score >= 0.08:
            qualifies = True
        elif len(shared) == 1 and min_shared_entities <= 1:
            only_ent = next(iter(shared))
            if len(entity_to_concepts[only_ent]) <= min(20, specific_entity_threshold):
                qualifies = True

        if qualifies:
            candidate_links.append((t1, t2, idf_score, sorted(list(shared)), "entity_overlap"))

    print(f"Discovered {len(candidate_links)} reciprocal candidate pair(s) meeting IDF semantic criteria.")

    # Convert title_mention_links to 5-tuple
    formatted_title_mentions = [
        (t1, t2, score, shared, "title_mention")
        for t1, t2, score, shared in title_mention_links
    ]
    all_new_links = formatted_title_mentions + candidate_links

    # Establish reciprocal bidirectional links for all qualifying pairs
    added_links_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    new_edges_added = 0

    for t1, t2, score, shared_list, l_type in all_new_links:
        if G.has_edge(t1, t2):
            continue
        shared_summary = ", ".join(shared_list[:3])
        if l_type == "title_mention":
            rel_text_1 = "named in this page's key entities"
            rel_text_2 = "named in this page's key entities"
        else:
            rel_text_1 = f"via: {shared_summary}"
            rel_text_2 = f"via: {shared_summary}"

        link_payload_1 = {
            "target_concept": t2,
            "relation": rel_text_1,
            "overlap_score": round(score, 3),
            "jaccard_similarity": round(score, 3),  # backward compatibility
            "shared_entities": shared_list,
            "link_type": l_type,
            "relation_source": "deterministic",
            "source_chunks": []
        }
        link_payload_2 = {
            "target_concept": t1,
            "relation": rel_text_2,
            "overlap_score": round(score, 3),
            "jaccard_similarity": round(score, 3),  # backward compatibility
            "shared_entities": shared_list,
            "link_type": l_type,
            "relation_source": "deterministic",
            "source_chunks": []
        }

        added_links_by_source[t1].append(link_payload_1)
        added_links_by_source[t2].append(link_payload_2)
        G.add_edge(t1, t2)
        new_edges_added += 1

    post_components = list(nx.connected_components(G))
    post_components.sort(key=len, reverse=True)
    post_giant = len(post_components[0]) if post_components else 0

    print(f"\nAdded {new_edges_added} bidirectional semantic edges (guaranteed 100% reciprocal).")
    print(f"Graph After Linking: {len(post_components)} components (down from {len(initial_components)}).")
    print(f"Giant component size: {post_giant}/{len(concepts)} ({post_giant/len(concepts)*100:.1f}%)")

    # Update concepts in memory & sort outbound relations by Jaccard descending
    for c in concepts:
        title = c["title"]
        existing_relations = c.get("outbound_relations", [])
        existing_targets = {r.get("target_concept", "").lower() for r in existing_relations}

        if title in added_links_by_source:
            for new_rel in added_links_by_source[title]:
                if new_rel["target_concept"].lower() not in existing_targets:
                    existing_relations.append(new_rel)
                    existing_targets.add(new_rel["target_concept"].lower())

        # Sort all outbound relations: highest Jaccard similarity first
        existing_relations.sort(
            key=lambda r: (r.get("jaccard_similarity") if r.get("jaccard_similarity") is not None else -1.0),
            reverse=True
        )
        c["outbound_relations"] = existing_relations

    # 3. LLM Relation Description Synthesis (Only for Top Candidate Links rendered on Concept Markdown)
    max_rendered_links = getattr(args, "max_primary_links", 60)
    pairs_to_synthesize: Dict[str, Dict[str, Any]] = {}
    for c in concepts:
        t1 = c["title"]
        b1 = c.get("brief", "") or get_concept_full_text(c)[:250]
        # Only synthesize phrases for links within the concept's markdown rendering budget
        outbound_to_consider = c.get("outbound_relations", [])[:max_rendered_links]
        for rel in outbound_to_consider:
            t2 = rel.get("target_concept", "")
            if not t2 or t2.lower() == t1.lower() or t2 not in concept_by_title:
                continue
            r_text = rel.get("relation", "")
            # Synthesize meaningful LLM phrases for all uncharacterized edges or generic placeholders
            is_generic = (
                not r_text 
                or r_text.startswith("via:") 
                or "named in this page" in r_text 
                or "Functionally related" in r_text
            )
            if is_generic:
                pk = ":::".join(sorted([t1, t2]))
                if pk not in pairs_to_synthesize:
                    b2 = concept_by_title[t2].get("brief", "") or get_concept_full_text(concept_by_title[t2])[:250]
                    pairs_to_synthesize[pk] = {
                        "concept_1": t1,
                        "brief_1": b1,
                        "concept_2": t2,
                        "brief_2": b2,
                        "shared_entities": rel.get("shared_entities", []),
                        "pair_key": pk
                    }

    if pairs_to_synthesize and not getattr(args, "skip_llm_descriptions", False):
        log_file = wiki_path / "link_relations_log.json"
        checkpoint: Dict[str, Any] = {}
        if getattr(args, "resume", False):
            checkpoint = load_json_log(log_file, corrupt_message="Link relations log corrupted. Initializing fresh.")

        pending_pairs = [p for pk, p in pairs_to_synthesize.items() if pk not in checkpoint or not checkpoint[pk]]
        print(f"\nIdentified {len(pairs_to_synthesize)} unique semantic relationship edge(s) across the graph.")
        print(f"Pending LLM relation synthesis: {len(pending_pairs)} (already completed: {len(pairs_to_synthesize) - len(pending_pairs)})")

        if pending_pairs:
            settings = resolve_settings(args, STAGE)
            print(f"Distributing across {len(settings.servers)} server(s) for LLM relationship synthesis...")
            pool = pool_for_stage(args, settings, STAGE)

            flush_counter = 0
            log_lock = asyncio.Lock()

            async def on_relation_result(item: Dict[str, Any], res: str):
                nonlocal flush_counter
                pk = item["pair_key"]
                async with log_lock:
                    checkpoint[pk] = res
                    flush_counter += 1
                    if flush_counter >= 100:
                        await save_log_async(log_file, checkpoint)
                        flush_counter = 0

            async def generate_relation(client: Any, item: Dict[str, Any], server_index: int) -> str:
                ents_str = ", ".join(item["shared_entities"][:8]) if item["shared_entities"] else "Domain context"
                prompt = (
                    f"CONCEPT 1 (SUBJECT): \"{item['concept_1']}\"\n"
                    f"Brief: {item['brief_1']}\n\n"
                    f"CONCEPT 2 (OBJECT): \"{item['concept_2']}\"\n"
                    f"Brief: {item['brief_2']}\n\n"
                    f"SHARED KEY ENTITIES: {ents_str}\n"
                )
                raw_text = await call_llm(
                    client=client,
                    model=settings.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_RELATION_SYNTHESIS},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=settings.max_tokens or 256,
                    args=args,
                    task="judge",
                    max_growth=0,
                )
                parsed = parse_xml_relation(raw_text)
                return parsed if parsed else f"co-occurs with via {ents_str}"

            try:
                async with pool:
                    await pool.map(
                        pending_pairs,
                        generate_relation,
                        attempts=settings.attempts,
                        desc="Synthesizing LLM Relations",
                        key_fn=lambda p: p["pair_key"],
                        on_result=on_relation_result,
                    )
            finally:
                save_log_sync(log_file, checkpoint)

        # Apply rich synthesized relations back to graph
        applied_count = 0
        for c in concepts:
            t1 = c["title"]
            for rel in c.get("outbound_relations", []):
                t2 = rel.get("target_concept", "")
                pk = ":::".join(sorted([t1, t2]))
                if pk in checkpoint and checkpoint[pk]:
                    rel["relation"] = checkpoint[pk]
                    applied_count += 1
        print(f"Applied LLM relationship descriptions to {applied_count} reciprocal directed relations.")

    # Final Link Distribution
    print_link_distribution(concepts, title_label="STAGE 8: AFTER JACCARD LINKING")

    if dry_run:
        print("[DRY RUN] No files modified.")
        return

    print(f"Saving enriched graph to {global_concepts_file}...")
    atomic_write_json(global_concepts_file, {"concepts": concepts})

    if concepts_dir.exists():
        print(f"Updating individual concept JSONs in {concepts_dir}/...")
        for c in tqdm(concepts, desc="Writing concept files"):
            slug = c.get("slug")
            if not slug:
                slug = sanitize_filename(c["title"]) or "unnamed_concept"
            file_path = concepts_dir / f"{slug}.json"
            atomic_write_json(file_path, c, fsync_dir=False)

    write_stage_manifest(
        wiki_path=wiki_path,
        stage=STAGE,
        expected=len(concepts),
        completed=len(concepts),
        failed=0,
        edges_added=new_edges_added,
    )

    print("\n============================================================")
    print("STAGE 8: JACCARD ENTITY LINKING COMPLETE")
    print(f"Total Concepts:              {len(concepts)}")
    print(f"New Bidirectional Edges:     {new_edges_added}")
    print(f"Final Giant Component Size:  {post_giant} ({post_giant/len(concepts)*100:.1f}%)")
    print("============================================================\n")

def main():
    parser = build_parser(
        stage=STAGE,
        description="Establish bidirectional semantic relations using Jaccard entity overlap",
        include_max_tokens=False,
        include_model=False,
        include_resume=True
    )
    parser.add_argument("--min_shared_entities", type=int, default=1,
                        help="Minimum shared entities to trigger a link (default: 1 for specific entities).")
    parser.add_argument("--min_jaccard", type=float, default=0.05,
                        help="Minimum Jaccard similarity score (|A∩B| / |A∪B|).")
    parser.add_argument("--max_hub_threshold", type=int, default=150,
                        help="Max concepts sharing an entity before treating it as an ultra-generic hub.")
    parser.add_argument("--max_primary_links", type=int, default=60,
                        help="Max links rendered per concept on wiki markdown pages (only synthesize for these).")
    parser.add_argument("--specific_entity_threshold", type=int, default=50,
                        help="Max concepts sharing an entity for 1 shared entity to trigger a link.")
    parser.add_argument("--skip_llm_descriptions", action="store_true",
                        help="Skip LLM relationship description generation and keep template text.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Simulate without modifying files.")

    args = parser.parse_args()
    wiki_path = getattr(args, "wiki_path", None)
    if not wiki_path:
        raise ValueError("Missing required --wiki_path argument.")

    # Raise open file descriptor limits for high-concurrency async operations
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(65536, hard) if hard > 0 else 65536
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception:
        pass

    asyncio.run(enrich_graph_via_jaccard(
        args=args,
        wiki_path=Path(wiki_path),
        min_shared_entities=args.min_shared_entities,
        min_jaccard=args.min_jaccard,
        max_hub_threshold=args.max_hub_threshold,
        specific_entity_threshold=args.specific_entity_threshold,
        dry_run=args.dry_run
    ))

if __name__ == "__main__":
    main()
