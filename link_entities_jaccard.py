#!/usr/bin/env python3
"""
Entity Jaccard-Based Semantic Overlap Linker
--------------------------------------------
Solves graph fragmentation by establishing reciprocal, bidirectional relations
between concepts that share vital named entities:
- Uses Jaccard similarity: |E_A ∩ E_B| / |E_A ∪ E_B|.
- Links concept pairs with Jaccard similarity >= min_jaccard (default 0.15) OR >= min_shared (default 2 entities).
- Ignores high-frequency hub entities shared by > 50 concepts to prevent star graphs.
- Enforces max_links_per_node (default 4) to preserve graph sparsity.
- Saves the computed Jaccard similarity and shared entity names inside each relation edge.
- Updates `global_concepts.json` and all individual JSON files in `concepts/`.
- Fast, 100% deterministic, zero LLM calls, zero debug logging.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
from tqdm import tqdm

from utils.atomic_io import atomic_write_json, sanitize_filename
from utils.cli import build_parser

STAGE = "link-entities"

def normalize_entity(ent: str) -> str:
    """Normalize entity text for exact matching."""
    return ent.strip().lower()

def enrich_graph_via_jaccard(
    wiki_path: Path,
    min_shared_entities: int = 2,
    min_jaccard: float = 0.15,
    max_links_per_node: int = 4,
    dry_run: bool = False
):
    global_concepts_file = wiki_path / "global_concepts.json"
    concepts_dir = wiki_path / "concepts"

    if not global_concepts_file.exists():
        raise FileNotFoundError(f"Missing {global_concepts_file}. Complete Stage 7 first.")

    print(f"Loading concepts from {global_concepts_file}...")
    with open(global_concepts_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        concepts = data.get("concepts", [])

    print(f"Loaded {len(concepts)} concepts. Indexing vital entities...")

    entity_to_concepts: Dict[str, Set[str]] = defaultdict(set)
    concept_entities: Dict[str, Set[str]] = {}
    concept_by_title: Dict[str, Dict[str, Any]] = {}

    for c in concepts:
        title = c["title"]
        concept_by_title[title] = c
        raw_entities = c.get("entities", [])
        norm_set = set()
        for ent in raw_entities:
            norm_e = normalize_entity(ent)
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

    # Find candidate pairs and compute Jaccard similarity
    pair_shared: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for ent, titles in entity_to_concepts.items():
        if 1 < len(titles) <= 50:  # ignore ultra-generic hubs
            t_list = sorted(list(titles))
            for i in range(len(t_list)):
                for j in range(i + 1, len(t_list)):
                    pair_shared[(t_list[i], t_list[j])].add(ent)

    candidate_links: List[Tuple[str, str, float, List[str]]] = []
    for (t1, t2), shared in pair_shared.items():
        if G.has_edge(t1, t2):
            continue
        set1 = concept_entities[t1]
        set2 = concept_entities[t2]
        union_len = len(set1 | set2)
        jaccard = len(shared) / union_len if union_len > 0 else 0.0

        if len(shared) >= min_shared_entities or jaccard >= min_jaccard:
            candidate_links.append((t1, t2, jaccard, sorted(list(shared))))

    # Prioritize higher Jaccard similarity
    candidate_links.sort(key=lambda x: (x[2], len(x[3])), reverse=True)
    print(f"Discovered {len(candidate_links)} candidate pair(s) meeting Jaccard threshold >= {min_jaccard} or shared >= {min_shared_entities}.")

    node_link_budget: Dict[str, int] = defaultdict(int)
    added_links_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    new_edges_added = 0

    for t1, t2, jaccard, shared_list in candidate_links:
        if node_link_budget[t1] >= max_links_per_node and node_link_budget[t2] >= max_links_per_node:
            continue

        shared_summary = ", ".join(shared_list[:4])
        rel_text = f"Functionally related via shared key entities: {shared_summary}."

        link_payload_1 = {
            "target_concept": t2,
            "relation": rel_text,
            "jaccard_similarity": round(jaccard, 3),
            "shared_entities": shared_list,
            "source_chunks": []
        }
        link_payload_2 = {
            "target_concept": t1,
            "relation": rel_text,
            "jaccard_similarity": round(jaccard, 3),
            "shared_entities": shared_list,
            "source_chunks": []
        }

        added_links_by_source[t1].append(link_payload_1)
        added_links_by_source[t2].append(link_payload_2)
        node_link_budget[t1] += 1
        node_link_budget[t2] += 1
        G.add_edge(t1, t2)
        new_edges_added += 1

    post_components = list(nx.connected_components(G))
    post_components.sort(key=len, reverse=True)
    post_giant = len(post_components[0]) if post_components else 0

    print(f"\nAdded {new_edges_added} bidirectional semantic edges.")
    print(f"Graph After Linking: {len(post_components)} components (down from {len(initial_components)}).")
    print(f"Giant component size: {post_giant}/{len(concepts)} ({post_giant/len(concepts)*100:.1f}%)")

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

    print(f"Saving enriched graph to {global_concepts_file}...")
    atomic_write_json(global_concepts_file, {"concepts": concepts})

    if concepts_dir.exists():
        print(f"Updating individual concept JSONs in {concepts_dir}/...")
        for c in tqdm(concepts, desc="Writing concept files"):
            safe_name = sanitize_filename(c["title"]) or "unnamed_concept"
            file_path = concepts_dir / f"{safe_name}.json"
            counter = 1
            while file_path.exists() and not file_path.is_file():
                file_path = concepts_dir / f"{safe_name}_{counter}.json"
                counter += 1
            atomic_write_json(file_path, c)

    print("\n============================================================")
    print("STAGE 8: JACCARD ENTITY LINKING COMPLETE")
    print(f"Total Concepts:              {len(concepts)}")
    print(f"New Bidirectional Links:     {new_edges_added}")
    print(f"Final Giant Component Size:  {post_giant} ({post_giant/len(concepts)*100:.1f}%)")
    print("============================================================\n")

def main():
    parser = build_parser(
        stage=STAGE,
        description="Establish bidirectional semantic relations using Jaccard entity overlap",
        include_max_tokens=False,
        include_model=False
    )
    parser.add_argument("--min_shared_entities", type=int, default=2,
                        help="Minimum shared entities to trigger a link.")
    parser.add_argument("--min_jaccard", type=float, default=0.15,
                        help="Minimum Jaccard similarity score (|A∩B| / |A∪B|).")
    parser.add_argument("--max_links_per_node", type=int, default=4,
                        help="Max new edges per node to maintain graph sparsity.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Simulate without modifying files.")

    args = parser.parse_args()
    wiki_path = getattr(args, "wiki_path", None)
    if not wiki_path:
        raise ValueError("Missing required --wiki_path argument.")

    enrich_graph_via_jaccard(
        wiki_path=Path(wiki_path),
        min_shared_entities=args.min_shared_entities,
        min_jaccard=args.min_jaccard,
        max_links_per_node=args.max_links_per_node,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
