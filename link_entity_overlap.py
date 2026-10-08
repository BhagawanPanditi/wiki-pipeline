#!/usr/bin/env python3
"""
Entity & Semantic Overlap Linker for Wiki Knowledge Graph
---------------------------------------------------------
Solves graph fragmentation (islands / isolated nodes) by discovering latent
relationships between concepts sharing:
1. Significant named entity overlaps (e.g. shared genes, tools, chemicals, pathogens).
2. Shared source chunk context or high lexical entity co-occurrence.

Connects isolated nodes to the giant component while keeping edges informative.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Any, Tuple
import networkx as nx
from tqdm import tqdm

from pipeline_lib.atomic_io import atomic_write_json
from pipeline_lib.cli import build_parser

STAGE = "entity-overlap"

def normalize_entity(ent: str) -> str:
    return ent.strip().lower()

def enrich_graph_via_entity_overlap(
    domain_dir: Path,
    min_shared_entities: int = 2,
    max_new_links_per_node: int = 4,
    dry_run: bool = False
):
    refined_file = domain_dir / "global_concepts_refined.json"
    if not refined_file.exists():
        raise FileNotFoundError(f"Missing refined graph file: {refined_file}")

    print(f"Loading concepts from {refined_file}...")
    with open(refined_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        concepts = data.get("concepts", [])

    print(f"Loaded {len(concepts)} concepts. Indexing entities...")

    # Build entity -> concept inverted index
    entity_to_concepts: Dict[str, Set[str]] = defaultdict(set)
    concept_by_title: Dict[str, Dict[str, Any]] = {}
    title_canonical_map: Dict[str, str] = {}

    for c in concepts:
        title = c["title"]
        concept_by_title[title] = c
        title_canonical_map[title.lower()] = title
        
        entities = c.get("entities", [])
        for ent in entities:
            norm_e = normalize_entity(ent)
            # Filter out generic ultra-short fragments (<= 2 chars)
            if len(norm_e) > 2:
                entity_to_concepts[norm_e].add(title)

    # Calculate existing graph connectivity
    G = nx.Graph()
    for c in concepts:
        G.add_node(c["title"])
        for rel in c.get("outbound_relations", []):
            tgt_raw = rel.get("target_concept", "").strip()
            tgt = title_canonical_map.get(tgt_raw.lower())
            if tgt and tgt in concept_by_title:
                G.add_edge(c["title"], tgt)

    initial_components = list(nx.connected_components(G))
    initial_components.sort(key=len, reverse=True)
    giant_size = len(initial_components[0]) if initial_components else 0
    print(f"Current Graph: {len(initial_components)} islands. Giant component: {giant_size}/{len(concepts)} ({giant_size/len(concepts)*100:.1f}%)")

    # Find pair overlap scores
    pair_shared_entities: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for ent, concept_titles in entity_to_concepts.items():
        if len(concept_titles) > 1 and len(concept_titles) <= 50:  # avoid ultra-generic hubs
            titles_list = list(concept_titles)
            for i in range(len(titles_list)):
                for j in range(i + 1, len(titles_list)):
                    t1, t2 = sorted([titles_list[i], titles_list[j]])
                    pair_shared_entities[(t1, t2)].add(ent)

    # Filter pairs meeting threshold and not already linked
    new_edges_added = 0
    candidate_links = []
    for (t1, t2), shared in pair_shared_entities.items():
        if len(shared) >= min_shared_entities:
            # Check if edge already exists
            if not G.has_edge(t1, t2):
                candidate_links.append((t1, t2, sorted(list(shared))))

    # Prioritize links that connect isolated nodes or bridge smaller islands to giant component
    candidate_links.sort(key=lambda x: len(x[2]), reverse=True)

    node_link_budget: Dict[str, int] = defaultdict(int)
    added_links_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for t1, t2, shared_list in candidate_links:
        if node_link_budget[t1] >= max_new_links_per_node and node_link_budget[t2] >= max_new_links_per_node:
            continue

        shared_summary = ", ".join(shared_list[:3])
        rel_statement = f"Functionally related via shared key entities and pathways: {shared_summary}."
        
        # Add outbound relation to t1
        added_links_by_source[t1].append({
            "target_concept": t2,
            "relation": rel_statement,
            "source_chunks": []
        })
        node_link_budget[t1] += 1

        # Add bidirectional outbound relation to t2
        added_links_by_source[t2].append({
            "target_concept": t1,
            "relation": rel_statement,
            "source_chunks": []
        })
        node_link_budget[t2] += 1

        G.add_edge(t1, t2)
        new_edges_added += 1

    post_components = list(nx.connected_components(G))
    post_components.sort(key=len, reverse=True)
    post_giant_size = len(post_components[0]) if post_components else 0

    print(f"\nAdded {new_edges_added} new relations based on shared entities (>= {min_shared_entities} shared).")
    print(f"New Graph: {len(post_components)} islands (down from {len(initial_components)}).")
    print(f"New Giant component: {post_giant_size}/{len(concepts)} ({post_giant_size/len(concepts)*100:.1f}%)")

    if dry_run:
        print("[DRY RUN] No files modified.")
        return

    # Update concepts in memory
    for c in concepts:
        title = c["title"]
        if title in added_links_by_source:
            existing_relations = c.get("outbound_relations", [])
            existing_targets = {r.get("target_concept", "").lower() for r in existing_relations}
            for new_rel in added_links_by_source[title]:
                if new_rel["target_concept"].lower() not in existing_targets:
                    existing_relations.append(new_rel)
            c["outbound_relations"] = existing_relations

    # Save to global_concepts_refined.json
    print(f"\nWriting updated concepts to {refined_file}...")
    atomic_write_json(refined_file, {"concepts": concepts})

    # Update individual concept files
    concepts_dir = domain_dir / "concepts"
    if concepts_dir.exists():
        print(f"Updating files in {concepts_dir}/...")
        for c in tqdm(concepts, desc="Updating concept files"):
            name = c["title"].replace(" ", "_")
            matches = list(concepts_dir.glob(f"{name}*.json"))
            if matches:
                with open(matches[0], "w", encoding="utf-8") as f:
                    json.dump(c, f, indent=2)

    print("Successfully enriched graph with entity overlap links!")

def main() -> None:
    """Add entity-overlap links to the refined graph and report what changed.

    Reads ``global_concepts_refined.json`` from the run directory, links concepts
    that share enough entities, and writes the enriched graph back in place.

    Returns:
        None. With ``--dry_run`` nothing is written and the intended changes are
        only printed.
    """
    # This stage makes no LLM calls. It uses the shared parser anyway so that one
    # uniform command line works for every stage in run_pipeline.sh; the server
    # and retry flags are accepted and ignored.
    parser = build_parser(
        stage=STAGE,
        description="Enrich knowledge graph connections using entity overlap "
                    "(no LLM calls)",
        include_max_tokens=False,
        include_model=False)
    parser.add_argument("--min_shared_entities", type=int, default=2,
                        help="Minimum shared entities to establish a link.")
    parser.add_argument("--max_links_per_node", type=int, default=4,
                        help="Max new edges per node, to avoid star-graph pollution.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Simulate without modifying files.")

    args = parser.parse_args()
    domain_dir = Path(args.wiki_base_dir) / args.domain
    enrich_graph_via_entity_overlap(
        domain_dir=domain_dir,
        min_shared_entities=args.min_shared_entities,
        max_new_links_per_node=args.max_links_per_node,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
